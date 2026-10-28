"""Detection of the resources this machine offers to jobs."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    vram: int


@dataclass(frozen=True)
class Machine:
    ram: int
    cpu_ids: tuple[int, ...]
    gpus: tuple[GpuInfo, ...]


def detect() -> Machine:
    return Machine(ram=detect_ram(), cpu_ids=detect_cpus(), gpus=detect_gpus())


def detect_ram(proc: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Total RAM, lowered by any memory limit on the cgroup proqu runs in."""
    total = 0
    for line in (proc / "meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
            break
    limit = _cgroup_memory_limit(proc, cgroup_root)
    return min(total, limit) if limit else total


def _cgroup_memory_limit(proc: Path, cgroup_root: Path) -> int | None:
    """The tightest memory limit on this process's cgroup or its ancestors."""
    try:
        lines = (proc / "self" / "cgroup").read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            # cgroup v2: every ancestor may carry its own memory.max
            node = cgroup_root / path.lstrip("/")
            while True:
                limits.append(_read_limit(node / "memory.max"))
                if node == cgroup_root or cgroup_root not in node.parents:
                    break
                node = node.parent
        elif "memory" in controllers.split(","):
            limits.append(_read_limit(cgroup_root / "memory" / "memory.limit_in_bytes"))
    found = [limit for limit in limits if limit]
    return min(found) if found else None


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


def detect_cpus() -> tuple[int, ...]:
    return tuple(sorted(os.sched_getaffinity(0)))


def detect_gpus() -> tuple[GpuInfo, ...]:
    """NVIDIA GPUs from nvidia-smi, or none when it is missing or fails."""
    if not shutil.which("nvidia-smi"):
        return ()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return parse_gpus(result.stdout)


def parse_gpus(text: str) -> tuple[GpuInfo, ...]:
    gpus = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            continue
        gpus.append(GpuInfo(index=int(fields[0]), uuid=fields[1], vram=int(fields[2]) * 1024**2))
    return tuple(gpus)
