"""AMD GPU support read from sysfs, with fdinfo and amd-smi as fallbacks for per process use.

Nothing here has been run on real AMD hardware. Every function swallows errors and
reports an unknown shape as a failed query, so a surprise never takes the scheduler down.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

AMD_VENDOR = "0x1002"
# APUs expose a small carve out of system memory, which is no use as a compute device.
MIN_VRAM = 1024**3
SMI_WAIT = 5
UNITS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
}


@dataclass(frozen=True)
class AmdGpu:
    index: int
    uuid: str
    vram: int
    gpu_id: int
    pci: str


def key(gpu: AmdGpu) -> str:
    """The name a GPU is known by in app listings: its uuid, else its index."""
    return gpu.uuid or f"AMD-{gpu.index}"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _properties(path: Path) -> dict[str, int]:
    """Parse a KFD properties file of "name value" lines, skipping odd lines."""
    text = _read(path)
    props: dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return props


def detect(sys_root: Path = Path("/sys")) -> tuple[AmdGpu, ...] | None:
    """AMD GPUs a job may be given: empty with no KFD tree, None when it cannot be read."""
    nodes = sys_root / "class/kfd/kfd/topology/nodes"
    if not nodes.is_dir():
        return ()
    try:
        names = sorted((n.name for n in nodes.iterdir() if n.name.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    position = 0
    for name in names:
        props = _properties(nodes / name / "properties")
        gpu_id = props.get("gpu_id") or _read_int(nodes / name / "gpu_id") or 0
        if props.get("simd_count", 0) <= 0 or gpu_id == 0:
            continue
        # Skipped GPUs still take an index. ROCR_VISIBLE_DEVICES numbers every GPU the
        # runtime sees, but that this matches KFD node order is unverified on hardware.
        index = position
        position += 1
        gpu = _describe(sys_root, props, index, gpu_id)
        if gpu is not None:
            gpus.append(gpu)
    return tuple(gpus)


def _describe(sys_root: Path, props: dict[str, int], index: int, gpu_id: int) -> AmdGpu | None:
    """Build the GPU for a KFD node, or None when it is not a usable AMD compute device."""
    minor = props.get("drm_render_minor")
    if not minor:
        return None
    device = sys_root / f"class/drm/renderD{minor}/device"
    if _read(device / "vendor") != AMD_VENDOR:
        return None
    vram = _read_int(device / "mem_info_vram_total")
    if vram is None or vram < MIN_VRAM:
        return None
    # A partitioned GPU is like NVIDIA MIG, which proqu skips. The file name and the
    # "SPX" value for an unpartitioned card are unverified.
    partition = _read(device / "current_compute_partition")
    if partition is not None and partition != "SPX":
        return None
    try:
        pci = device.resolve().name
    except OSError:
        return None
    unique = props.get("unique_id", 0)
    uuid = f"GPU-{unique:016x}" if unique > 0 else ""
    return AmdGpu(index=index, uuid=uuid, vram=vram, gpu_id=gpu_id, pci=pci)


def _device_of(sys_root: Path, gpu: AmdGpu) -> Path:
    """The sysfs directory of a GPU's PCI device, found by address."""
    return sys_root / "bus/pci/devices" / gpu.pci


def free_vram(sys_root: Path = Path("/sys")) -> dict[int, int] | None:
    """Free memory on each AMD GPU by ROCr index, or None when it cannot be read."""
    gpus = detect(sys_root)
    if gpus is None:
        return None
    free = {}
    for gpu in gpus:
        device = _device_of(sys_root, gpu)
        total = _read_int(device / "mem_info_vram_total")
        used = _read_int(device / "mem_info_vram_used")
        if total is None or used is None:
            return None
        free[gpu.index] = max(total - used, 0)
    return free


def apps(
    sys_root: Path = Path("/sys"), proc_root: Path = Path("/proc")
) -> list[tuple[int, str, int | None]] | None:
    """Processes using AMD GPUs as (pid, GPU key, bytes), or None if no source could tell.

    Only memory in use counts. The ROCm runtime opens every GPU when it starts, even
    one hidden from it, so a process holding a GPU with nothing on it is not using it
    and must not be killed as if it were.

    Sources, first available wins: KFD per process sysfs, DRM fdinfo of this user's
    processes, then amd-smi. When nothing finds a user the answer is an empty list only if
    there are GPUs and fdinfo was readable, so a quiet GPU is told apart from a blind query.
    """
    gpus = detect(sys_root)
    if gpus is None:
        return None
    kfd = sys_root / "class/kfd/kfd/proc"
    if kfd.is_dir():
        return _in_use(_kfd_apps(kfd, gpus))
    found, readable = _fdinfo_apps(proc_root, gpus)
    if found:
        return _in_use(found)
    smi = _smi_apps(gpus)
    if smi is not None:
        return _in_use(smi)
    return [] if gpus and readable else None


def _in_use(found: list[tuple[int, str, int | None]]) -> list[tuple[int, str, int | None]]:
    return [app for app in found if app[2]]


def _pids(root: Path) -> list[tuple[int, Path]]:
    try:
        entries = [(int(e.name), e) for e in root.iterdir() if e.name.isdigit()]
    except OSError:
        return []
    return sorted(entries)


def _kfd_apps(kfd: Path, gpus: tuple[AmdGpu, ...]) -> list[tuple[int, str, int | None]]:
    """Per process VRAM from KFD. A readable vram file means the process opened the GPU."""
    found = []
    for pid, directory in _pids(kfd):
        for gpu in gpus:
            used = _read_int(directory / f"vram_{gpu.gpu_id}")
            if used is not None:
                found.append((pid, key(gpu), used))
    return found


def _fdinfo_apps(
    proc_root: Path, gpus: tuple[AmdGpu, ...]
) -> tuple[list[tuple[int, str, int | None]], bool]:
    """Per process VRAM from DRM fdinfo, and whether any fdinfo could be read at all."""
    by_pci = {gpu.pci: gpu for gpu in gpus}
    found = []
    seen = set()
    readable = False
    uid = os.getuid()
    for pid, directory in _pids(proc_root):
        try:
            if directory.stat().st_uid != uid:
                continue
            fds = sorted((directory / "fdinfo").iterdir())
        except OSError:
            continue
        for fd in fds:
            text = _read(fd)
            if text is None:
                continue
            readable = True
            fields = _fdinfo_fields(text)
            if fields.get("drm-driver") != "amdgpu":
                continue
            gpu = by_pci.get(fields.get("drm-pdev", ""))
            client = fields.get("drm-client-id")
            if gpu is None or client is None or (gpu.pci, client) in seen:
                continue
            # Forked processes share one client, so only the first pid seen gets it.
            seen.add((gpu.pci, client))
            found.append((pid, key(gpu), _size(fields.get("drm-memory-vram", "0"))))
    return found, readable


def _fdinfo_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if sep:
            fields[name.strip()] = value.strip()
    return fields


def _size(text: str) -> int | None:
    """Bytes from "N KiB", "N MiB" or a plain byte count, else None."""
    parts = text.split()
    if not parts or not parts[0].isdigit() or len(parts) > 2:
        return None
    unit = parts[1].lower() if len(parts) == 2 else "b"
    return int(parts[0]) * UNITS[unit] if unit in UNITS else None


def _smi_apps(gpus: tuple[AmdGpu, ...]) -> list[tuple[int, str, int | None]] | None:
    """Per process VRAM from amd-smi, or None when it is missing, fails or is not understood.

    Without sudo it shows only this user's processes, which is all proqu needs to explain
    its own jobs, but other users' use of a GPU stays hidden.
    """
    if not shutil.which("amd-smi"):
        return None
    try:
        result = subprocess.run(
            ["amd-smi", "process", "--json"],
            capture_output=True,
            text=True,
            timeout=SMI_WAIT,
            check=True,
        )
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return _parse_smi(data, gpus)


def _parse_smi(data: object, gpus: tuple[AmdGpu, ...]) -> list[tuple[int, str, int | None]] | None:
    """Read amd-smi process JSON. The shape varies by ROCm release, so any surprise is None."""
    if not isinstance(data, list):
        return None
    by_index = {gpu.index: gpu for gpu in gpus}
    found = []
    for entry in data:
        if not isinstance(entry, dict) or not _is_int(entry.get("gpu")):
            return None
        processes = entry.get("process_list")
        if not isinstance(processes, list):
            return None
        # That amd-smi's gpu number equals the ROCr index is unverified.
        gpu = by_index.get(entry["gpu"])
        for item in processes:
            if not isinstance(item, dict):
                return None
            info = item.get("process_info")
            info = info if isinstance(info, dict) else item
            if not _is_int(info.get("pid")):
                return None
            if gpu is not None:
                found.append((info["pid"], key(gpu), _smi_memory(info)))
    return found


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _smi_memory(info: dict) -> int | None:
    """VRAM bytes from a process entry, where the value may be an int, a string or an object."""
    usage = info.get("memory_usage", info.get("mem_usage"))
    if not isinstance(usage, dict):
        return None
    value = usage.get("vram_mem")
    if isinstance(value, dict):
        number, unit = value.get("value"), value.get("unit", "B")
        if _is_int(number) and isinstance(unit, str) and unit.lower() in UNITS:
            return number * UNITS[unit.lower()]
        return None
    if isinstance(value, str):
        return _size(value)
    return value if _is_int(value) else None
