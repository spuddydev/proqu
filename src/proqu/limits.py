"""Hard limits on a running job.

The supervisor asks its monitor about the job once a poll. A monitor answers
with the reason the job must be stopped, or None while it is within limits.
"""

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from proqu import paths, procs
from proqu.store import Job
from proqu.units import format_size

# how long a cgroup probe result is trusted
PROBE_AGE = 86400
# how long systemd-run may take to put a job in its scope
SCOPE_WAIT = 5.0

OWN_LIMIT = "this was the job's own limit"


def ram_reason(limit: int, peak: int) -> str:
    return (
        f"used more than its RAM limit of {format_size(limit)} (peak {format_size(peak)}): "
        f"{OWN_LIMIT}, reduce its memory use or resubmit with a larger --ram if it truly needs it"
    )


class Monitor:
    """Watches RAM by summing the memory of every process in the job."""

    def __init__(self, job: Job, session: int):
        self.job = job
        self.session = session
        self.peak = 0

    def check(self) -> tuple[str, str] | None:
        used = sum(memory_of(pid) for pid in procs.job_members(self.job.id, self.session))
        self.peak = max(self.peak, used)
        if used > self.job.ram:
            return "killed", ram_reason(self.job.ram, used)
        return None

    def finish(self, code: int | None = None) -> tuple[str, str] | None:
        """Evidence found only once the job has ended, given its exit code."""
        return None


def memory_of(pid: int) -> int:
    """Proportional memory of a process, so shared pages are not counted twice."""
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

    def __init__(self, job: Job, session: int, path: Path):
        super().__init__(job, session)
        self.path = path
        self.oom_kills = 0
        # if the limit was not applied, fall back to watching
        self.enforced = _read_int(path / "memory.max") == job.ram
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()

    def check(self) -> tuple[str, str] | None:
        return None if self.enforced else super().check()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.SAMPLE):
            if not self.sample():
                return

    def sample(self) -> bool:
        """Read the scope's counters, returning False once the scope is gone."""
        peak = _read_int(self.path / "memory.peak") or _read_int(self.path / "memory.current")
        self.peak = max(self.peak, peak or 0)
        try:
            for line in (self.path / "memory.events").read_text().splitlines():
                name, value = line.split()
                if name == "oom_kill":
                    self.oom_kills = max(self.oom_kills, int(value))
        except (OSError, ValueError):
            return False
        return True

    def finish(self, code: int | None = None) -> tuple[str, str] | None:
        self._stop.set()
        self._thread.join()
        self.sample()
        # a kill we did not send, at the edge of the limit, is the kernel's
        near_limit = self.peak >= self.job.ram * 0.9
        if self.oom_kills or (code == -signal.SIGKILL and near_limit):
            return "killed", ram_reason(self.job.ram, max(self.peak, self.job.ram))
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
    because without linger the user's scopes die at the last logout.
    """
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
    """Stop using cgroups for a day after systemd-run failed for a job."""
    _remember(False)


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
    supervisor can tell a job that ran from a scope that was never made.
    """
    return [
        "systemd-run",
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
        'printf started > "$0" && exec "$@"',
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
