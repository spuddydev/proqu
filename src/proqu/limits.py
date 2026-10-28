"""Hard limits on a running job.

The supervisor asks its monitor about the job once a poll. A monitor answers
with the reason the job must be stopped, or None while it is within limits.

These limits stop mistakes, not deliberate evasion. A job can escape them, for
example by running setsid with an empty environment so the process list no
longer finds it, by calling systemd-run itself to leave its scope, or in watch
mode by filling tmpfs shared memory that is not counted as its own.

macOS has no cgroups, so there a job is always watched. Its RAM is the memory
footprint of its processes, which also holds what they put on the GPU.
"""

import contextlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from proqu import amd, darwin, machine, paths, procs
from proqu.store import Job
from proqu.units import format_size

# how long a cgroup probe result is trusted
PROBE_AGE = 86400
# how long systemd-run may take to put a job in its scope
SCOPE_WAIT = 5.0
# how often GPU use is checked, the GPU tools are too slow to run every poll
GPU_EVERY = 5.0
# the longest wait between GPU checks while the GPU query keeps failing
GPU_BACKOFF_MAX = 60.0

# variables systemd-run needs to reach the user's manager
SCOPE_VARS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")

OWN_LIMIT = "this was the job's own limit"

# how each vendor's jobs are told which GPUs are theirs, and how to measure their use
VISIBLE = {"nvidia": "CUDA_VISIBLE_DEVICES", "amd": "ROCR_VISIBLE_DEVICES"}
MEASURE = {"nvidia": "nvidia-smi", "amd": "amd-smi or rocm-smi"}


def ram_reason(limit: int, peak: int) -> str:
    # the kernel stops a job at the limit, so a sampled peak below it only lags behind
    seen = f" (peak {format_size(peak)})" if peak > limit else " (stopped by the kernel)"
    return (
        f"used more than its RAM limit of {format_size(limit)}{seen}: "
        f"{OWN_LIMIT}, reduce its memory use or resubmit with a larger --ram if it truly needs it"
    )


class Monitor:
    """Watches RAM by summing the memory of every process in the job, and GPU use."""

    def __init__(self, job: Job, session: int, gpu_uuids: tuple[str, ...] = ()):
        self.job = job
        self.session = session
        self.gpu_uuids = gpu_uuids
        self.peak = 0
        self.vendor = machine.gpu_vendor()
        # a GPU that uses the RAM shows its memory in the RAM, and cannot be told apart per job
        if self.vendor == "amd":
            self.watch_gpus = True
        elif self.vendor == "apple":
            self.watch_gpus = False
        else:
            self.watch_gpus = shutil.which("nvidia-smi") is not None
        self.next_gpu_check = time.monotonic()
        self.gpu_query_failed = False
        self.gpu_wait = GPU_EVERY

    def check(self) -> tuple[str, str] | None:
        return self.check_ram() or self.check_gpus()

    def check_ram(self) -> tuple[str, str] | None:
        used = sum(memory_of(pid) for pid in procs.job_members(self.job.id, self.session))
        self.peak = max(self.peak, used)
        if used > self.job.ram:
            return "killed", ram_reason(self.job.ram, used)
        return None

    def check_gpus(self) -> tuple[str, str] | None:
        if not self.watch_gpus or time.monotonic() < self.next_gpu_check:
            return None
        apps = gpu_apps()
        # a failing GPU query is asked less and less often
        self.gpu_wait = GPU_EVERY if apps is not None else min(self.gpu_wait * 2, GPU_BACKOFF_MAX)
        self.next_gpu_check = time.monotonic() + self.gpu_wait
        if apps is None:
            if not self.gpu_query_failed:
                print("proqu: could not query GPU use, GPU limits are not checked", flush=True)
            self.gpu_query_failed = True
            return None
        members = set(procs.job_members(self.job.id, self.session))
        mine = [app for app in apps if app[0] in members]
        return gpu_breach(self.job, self.gpu_uuids, mine, self.vendor or "nvidia")

    def kill_all(self) -> None:
        """Stop anything of the job the process list missed. Only a cgroup can do this."""

    def finish(self, code: int | None = None) -> tuple[str, str] | None:
        """Evidence found only once the job has ended, given its exit code."""
        # a short job can start and end between two samples, the kernel remembers its peak
        # in KiB on Linux and in bytes on macOS
        scale = 1 if sys.platform == "darwin" else 1024
        largest = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
        self.peak = max(self.peak, largest)
        return None


def gpu_breach(
    job: Job,
    assigned: tuple[str, ...],
    apps: list[tuple[int, str, int | None]],
    vendor: str = "nvidia",
) -> tuple[str, str] | None:
    """Why the job's GPU use breaks its request, given its processes on GPUs."""
    if not apps:
        return None
    if not job.gpus:
        return "killed", (
            f"used a GPU without asking for one: {OWN_LIMIT}, "
            "resubmit with --gpus and --vram if it needs a GPU"
        )
    used: dict[str, int] = {}
    for _, uuid, memory in apps:
        if uuid not in assigned:
            return "killed", (
                f"used GPU {uuid}, which was not given to it: {OWN_LIMIT}, "
                f"use the GPUs in {VISIBLE.get(vendor, VISIBLE['nvidia'])} only"
            )
        used[uuid] = used.get(uuid, 0) + (memory or 0)
    worst = max(used.values())
    if worst > job.vram:
        return "killed", (
            f"used {format_size(worst)} of GPU memory, more than its --vram of "
            f"{format_size(job.vram)}: {OWN_LIMIT}, measure it with "
            f"{MEASURE.get(vendor, MEASURE['nvidia'])} and resubmit"
        )
    return None


def gpu_apps() -> list[tuple[int, str, int | None]] | None:
    """Processes using GPUs as (pid, GPU uuid, bytes), or None if the query failed."""
    vendor = machine.gpu_vendor()
    if vendor == "amd":
        return amd.apps()
    if vendor == "apple":
        # no process can be seen using it, its memory is counted in the RAM
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    apps = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        memory = int(fields[2]) * 1024**2 if fields[2].isdigit() else None
        apps.append((int(fields[0]), fields[1], memory))
    return apps


def memory_of(pid: int) -> int:
    """Proportional memory of a process, so shared pages are not counted twice.

    macOS has no proportional count, so there it is the memory footprint, in which
    pages shared between a job's processes are counted for each of them.
    """
    if sys.platform == "darwin":
        return darwin.footprint(pid)
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class CgroupMonitor(Monitor):
    """Reads what the kernel measured in the job's own cgroup.

    The kernel enforces the limit itself. The scope's files vanish as soon as
    its last process ends, which after an out of memory kill takes only
    milliseconds, so a thread samples them often while the job runs.
    """

    SAMPLE = 0.02

    def __init__(self, job: Job, session: int, path: Path, gpu_uuids: tuple[str, ...] = ()):
        super().__init__(job, session, gpu_uuids)
        self.path = path
        self.oom_kills = 0
        # if the limit was not applied, fall back to watching
        # the kernel rounds the limit up to a whole page
        page = os.sysconf("SC_PAGE_SIZE")
        self.applied = -(-job.ram // page) * page
        self.enforced = _read_int(path / "memory.max") == self.applied
        self.tampered = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()

    def check_ram(self) -> tuple[str, str] | None:
        if self.tampered:
            return "killed", (
                f"changed the memory limit of its own cgroup: {OWN_LIMIT}, "
                "a job must not change its own limits"
            )
        return None if self.enforced else super().check_ram()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.SAMPLE):
            if not self.sample():
                return

    def sample(self) -> bool:
        """Read the scope's counters, returning False once the scope is gone."""
        peak = _read_int(self.path / "memory.peak") or _read_int(self.path / "memory.current")
        self.peak = max(self.peak, peak or 0)
        if self.enforced and _read_int(self.path / "memory.max") != self.applied:
            # a vanished file reads as None too, which is only a tamper while the scope exists
            self.tampered = self.tampered or (self.path / "memory.events").exists()
        try:
            for line in (self.path / "memory.events").read_text().splitlines():
                name, value = line.split()
                if name == "oom_kill":
                    self.oom_kills = max(self.oom_kills, int(value))
        except (OSError, ValueError):
            return False
        return True

    def kill_all(self) -> None:
        # unlike the process list, this also catches a process that hid its job id
        with contextlib.suppress(OSError):
            (self.path / "cgroup.kill").write_text("1")

    def finish(self, code: int | None = None) -> tuple[str, str] | None:
        self._stop.set()
        self._thread.join()
        self.sample()
        # a kill we did not send, at the edge of the limit, is the kernel's
        near_limit = self.peak >= self.job.ram * 0.9
        if self.oom_kills or (code == -signal.SIGKILL and near_limit):
            return "killed", ram_reason(self.job.ram, self.peak)
        return None


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def cgroups_available() -> bool:
    """Whether jobs can be put in their own kernel enforced cgroup.

    Needs systemd-run for the user, a delegated memory controller, and linger,
    because without linger the user's scopes die at the last logout. macOS has none.
    """
    if sys.platform == "darwin":
        return False
    key = f"{os.getuid()}:{procs.boot_id()}:{os.environ.get('XDG_RUNTIME_DIR', '')}"
    path = paths.state_dir() / "cgroups.json"
    try:
        cached = json.loads(path.read_text())
        if cached["key"] == key and time.time() - cached["at"] < PROBE_AGE:
            return bool(cached["ok"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    ok = _probe()
    _remember(ok)
    return ok


def mark_unavailable() -> None:
    """Forget the probe result after systemd-run failed for a job, so the next job probes again."""
    with contextlib.suppress(OSError):
        (paths.state_dir() / "cgroups.json").unlink()


def _remember(ok: bool) -> None:
    key = f"{os.getuid()}:{procs.boot_id()}:{os.environ.get('XDG_RUNTIME_DIR', '')}"
    path = paths.state_dir() / "cgroups.json"
    tmp = path.with_name(f"cgroups.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"key": key, "ok": ok, "at": time.time()}))
        os.replace(tmp, path)
    except OSError:
        pass


def _probe() -> bool:
    if not shutil.which("systemd-run") or not lingering():
        return False
    script = 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.max"'
    command = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M"]
    try:
        result = subprocess.run(
            [*command, "--", "sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=SCOPE_WAIT,
            env=scope_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == str(64 * 1024 * 1024)


def lingering() -> bool:
    """Whether the user's systemd manager outlives their last login."""
    try:
        name = Path(f"/proc/{os.getpid()}").owner()
    except (OSError, KeyError):
        return False
    return Path("/var/lib/systemd/linger", name).exists()


def scope_env(env: dict[str, str]) -> dict[str, str]:
    """The environment systemd-run needs to reach the user's manager."""
    runtime = env.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    runtime = runtime or f"/run/user/{os.getuid()}"
    bus = env.get("DBUS_SESSION_BUS_ADDRESS") or os.environ.get("DBUS_SESSION_BUS_ADDRESS")
    return {
        **env,
        "XDG_RUNTIME_DIR": runtime,
        "DBUS_SESSION_BUS_ADDRESS": bus or f"unix:path={runtime}/bus",
    }


def scope_command(job: Job, marker: Path) -> list[str]:
    """The job's command inside its own memory and CPU capped scope.

    A small shell step writes the marker before running the job, so the
    supervisor can tell a job that ran from a scope that was never made. It
    also drops the variables that only systemd-run needed, so the job sees the
    environment it was submitted with.
    """
    added = [name for name in SCOPE_VARS if name not in job.env]
    unset = f"unset {' '.join(added)}; " if added else ""
    # by full path, because the job's own PATH may not hold it
    return [
        shutil.which("systemd-run") or "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--unit",
        f"proqu-{job.id}",
        "-p",
        f"MemoryMax={job.ram}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"CPUQuota={job.cpus * 100}%",
        "-p",
        "OOMPolicy=kill",
        "--",
        "/bin/sh",
        "-c",
        f'printf started > "$0" && {{ {unset}exec "$@"; }}',
        str(marker),
        *job.argv,
    ]


def await_scope(proc: subprocess.Popen, marker: Path) -> tuple[bool, Path | None]:
    """Whether the job started inside its scope, and the scope's cgroup if known."""
    deadline = time.monotonic() + SCOPE_WAIT
    while time.monotonic() < deadline:
        if marker.exists():
            return True, cgroup_of(proc.pid)
        if proc.poll() is not None:
            return marker.exists(), None
        time.sleep(0.01)
    return marker.exists(), None


def cgroup_of(pid: int) -> Path | None:
    try:
        with open(f"/proc/{pid}/cgroup") as handle:
            for line in handle:
                if line.startswith("0::"):
                    return Path("/sys/fs/cgroup") / line.strip()[3:].lstrip("/")
    except OSError:
        pass
    return None
