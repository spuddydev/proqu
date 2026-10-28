"""Detection of the resources this machine offers to jobs."""

import contextlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from proqu import amd, darwin, paths

# how long a detected machine is trusted before it is detected again
SNAPSHOT_AGE = 3600
# the free GPU memory is read while the queue is locked, so it may not take long
FREE_VRAM_WAIT = 5


VENDORS = ("nvidia", "amd", "apple")


@dataclass(frozen=True)
class GpuInfo:
    """One GPU. Every GPU of a machine has the same vendor, the first one found.

    vram 0 means the GPU has no memory of its own and uses the RAM, as on Apple silicon.
    """

    index: int
    uuid: str
    vram: int
    # a snapshot cached before vendors were known only ever held NVIDIA GPUs
    vendor: str = "nvidia"


@dataclass(frozen=True)
class Machine:
    ram: int
    cpu_ids: tuple[int, ...]
    gpus: tuple[GpuInfo, ...]


def detect(previous: Machine | None = None) -> Machine:
    """Detect the machine. A failed GPU query keeps the previous GPU list."""
    if _macos():
        cpu_ids = tuple(range(darwin.cpu_count()))
        return Machine(ram=darwin.total_ram(), cpu_ids=cpu_ids, gpus=apple_gpus())
    gpus = detect_gpus()
    if gpus is None:
        gpus = previous.gpus if previous else ()
    return Machine(ram=detect_ram(), cpu_ids=detect_cpus(), gpus=gpus)


def _macos() -> bool:
    return sys.platform == "darwin"


def available_cpus() -> set[int]:
    """The CPU ids this process may run on now. macOS cannot pin, so that is every CPU."""
    if hasattr(os, "sched_getaffinity"):
        return os.sched_getaffinity(0)
    return set(range(darwin.cpu_count()))


def gpu_vendor(seen: Machine | None = None) -> str | None:
    """The vendor of a machine's GPUs, by default the cached snapshot's, or None without any."""
    seen = seen or cached()
    return seen.gpus[0].vendor if seen and seen.gpus else None


def snapshot(refresh: bool = False) -> Machine:
    """The machine as the queue sees it, cached for an hour.

    Inside a job the caller sits in the job's own cgroup and CPU pinning, which
    would shrink what is detected. A job therefore never refreshes or replaces
    the cached snapshot, and only detects for itself when there is none.
    """
    path = paths.machine_path()
    cached = _load(path)
    in_job = bool(os.environ.get("PROQU_JOB_ID"))
    if cached and (in_job or (not refresh and _age(path) < SNAPSHOT_AGE)):
        return cached
    machine = detect(cached)
    if not in_job:
        _save(path, machine)
    return machine


def cached() -> Machine | None:
    """The cached snapshot of any age, or None. Never detects, so it is always quick."""
    return _load(paths.machine_path())


def _age(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return float("inf")


def _save(path: Path, machine: Machine) -> None:
    """Write the snapshot atomically. Failing to cache it is not an error."""
    try:
        handle, tmp = tempfile.mkstemp(dir=path.parent, prefix="machine.", suffix=".tmp")
    except OSError:
        return
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(asdict(machine)))
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def _load(path: Path) -> Machine | None:
    """The cached snapshot, or None when it is missing or does not make sense."""
    try:
        data = json.loads(path.read_text())
        machine = Machine(
            ram=data["ram"],
            cpu_ids=tuple(data["cpu_ids"]),
            gpus=tuple(GpuInfo(**gpu) for gpu in data["gpus"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    numbers = [machine.ram, *machine.cpu_ids]
    for gpu in machine.gpus:
        numbers += [gpu.index, gpu.vram]
    sane = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in numbers)
    if not sane or machine.ram == 0 or not machine.cpu_ids:
        return None
    if not all(isinstance(gpu.uuid, str) and gpu.vendor in VENDORS for gpu in machine.gpus):
        return None
    return machine


def detect_ram(proc: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Total RAM, lowered by any memory limit on the cgroup proqu runs in."""
    total = 0
    for line in (proc / "meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
            break
    limit = _cgroup_memory_limit(proc, cgroup_root)
    return min(total, limit) if limit else total


def free_ram(proc: Path | None = None) -> int | None:
    """Memory the kernel could give new work now, MemAvailable, or None if it cannot be read.

    Read fresh on every call, never cached, as it changes from moment to moment. On
    macOS, which has no MemAvailable, it is an estimate from the free and reusable pages.
    """
    if proc is None:
        if _macos():
            return darwin.free_ram()
        proc = Path("/proc")
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_memory_limit(proc: Path, cgroup_root: Path) -> int | None:
    """The tightest memory limit on this process's cgroup or its ancestors."""
    limits = []
    for version, path, _ in _cgroups(proc, "memory"):
        if version == 2:
            limits += [_read_limit(node / "memory.max") for node in _ancestors(cgroup_root, path)]
        else:
            v1_root = cgroup_root / "memory"
            limits += [_read_limit(n / "memory.limit_in_bytes") for n in _ancestors(v1_root, path)]
    found = [limit for limit in limits if limit]
    return min(found) if found else None


def _cgroup_cpu_limit(proc: Path, cgroup_root: Path) -> int | None:
    """The tightest whole CPU count allowed by a cgroup CPU quota."""
    counts = []
    for version, path, controllers in _cgroups(proc, "cpu"):
        if version == 2:
            nodes, quota_file, period_file = _ancestors(cgroup_root, path), "cpu.max", None
        else:
            nodes = _ancestors(cgroup_root / controllers, path)
            quota_file, period_file = "cpu.cfs_quota_us", "cpu.cfs_period_us"
        for node in nodes:
            try:
                if period_file:
                    quota = (node / quota_file).read_text().strip()
                    period = (node / period_file).read_text().strip()
                else:
                    quota, period = (node / quota_file).read_text().split()
            except (OSError, ValueError):
                continue
            # an unlimited quota reads "max" on v2 and "-1" on v1
            if quota.isdigit() and period.isdigit() and int(period) > 0:
                counts.append(max(1, math.ceil(int(quota) / int(period))))
    return min(counts) if counts else None


def _cgroups(proc: Path, controller: str) -> list[tuple[int, str, str]]:
    """The cgroup paths of this process, as (version, path, controllers)."""
    try:
        lines = (proc / "self" / "cgroup").read_text().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            found.append((2, path, controllers))
        elif controller in controllers.split(","):
            found.append((1, path, controllers))
    return found


def _ancestors(root: Path, path: str) -> list[Path]:
    """The cgroup directory for path and every parent up to the mount root.

    In a private cgroup namespace the path may be missing under the mount, or
    climb above it with "..", so only the mount root itself is read.
    """
    node = root / path.lstrip("/")
    if ".." in Path(path).parts or not node.is_dir():
        return [root]
    nodes = [node]
    while node != root and root in node.parents:
        node = node.parent
        nodes.append(node)
    return nodes


def _read_limit(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text.isdigit():
        return None
    value = int(text)
    # cgroup v1 reports "no limit" as a huge page aligned number
    return value if value < 2**60 else None


def detect_cpus(
    proc: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> tuple[int, ...]:
    """CPU ids this process may run on, trimmed to any cgroup CPU quota."""
    cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
    quota = _cgroup_cpu_limit(proc, cgroup_root)
    return cpu_ids[:quota] if quota else cpu_ids


def detect_gpus() -> tuple[GpuInfo, ...] | None:
    """The GPUs of one vendor, NVIDIA first, then AMD. None when a query failed.

    A machine with GPUs of both is offered only its NVIDIA ones.
    """
    nvidia = detect_nvidia()
    if nvidia != ():
        return nvidia
    return detect_amd()


def detect_amd(sys_root: Path = Path("/sys")) -> tuple[GpuInfo, ...] | None:
    """AMD GPUs from sysfs: empty when there are none, None when they cannot be read."""
    found = amd.detect(sys_root)
    if found is None:
        return None
    return tuple(GpuInfo(gpu.index, amd.key(gpu), gpu.vram, "amd") for gpu in found)


def detect_nvidia() -> tuple[GpuInfo, ...] | None:
    """NVIDIA GPUs from nvidia-smi: empty when it is missing, None when it fails."""
    if not shutil.which("nvidia-smi"):
        return ()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_gpus(result.stdout)


def parse_gpus(text: str) -> tuple[GpuInfo, ...]:
    """Parse nvidia-smi rows, skipping unreadable ones and GPUs in MIG mode."""
    gpus = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or not fields[0].isdigit() or not fields[2].isdigit():
            continue
        if fields[3].lower() == "enabled":
            continue
        gpus.append(GpuInfo(index=int(fields[0]), uuid=fields[1], vram=int(fields[2]) * 1024**2))
    return tuple(gpus)


def apple_gpus() -> tuple[GpuInfo, ...]:
    """The GPU of Apple silicon as one slot, or none on any other Mac.

    Its memory is the RAM, so vram is 0 and a job counts GPU memory in its RAM.
    Intel Macs are offered no GPU, as nothing there could keep a job to its GPU.
    """
    if platform.machine() != "arm64":
        return ()
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        name = result.stdout
    except (OSError, subprocess.SubprocessError):
        name = ""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug.startswith("apple"):
        slug = f"apple-{slug}" if slug else "apple-gpu"
    return (GpuInfo(index=0, uuid=slug, vram=0, vendor="apple"),)


def free_vram() -> dict[int, int] | None:
    """Free memory on each GPU by index, or None when it cannot be read."""
    if gpu_vendor() == "amd":
        return amd.free_vram()
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=FREE_VRAM_WAIT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_free_vram(result.stdout)


def parse_free_vram(text: str) -> dict[int, int]:
    """Parse nvidia-smi rows of index and free MiB, skipping unreadable ones."""
    free = {}
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            free[int(fields[0])] = int(fields[1]) * 1024**2
    return free
