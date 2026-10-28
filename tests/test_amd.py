import json
import os
import stat
from pathlib import Path

import pytest

from proqu import amd

GIB = 1024**3


def node(
    sys_root: Path,
    n: int,
    simd: int = 0,
    gpu_id: int = 0,
    minor: int = 0,
    unique: int | None = None,
) -> None:
    directory = sys_root / "class/kfd/kfd/topology/nodes" / str(n)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"simd_count {simd}", f"drm_render_minor {minor}"]
    if unique is not None:
        lines.append(f"unique_id {unique}")
    (directory / "properties").write_text("\n".join(lines) + "\n")
    (directory / "gpu_id").write_text(f"{gpu_id}\n")


def device(
    sys_root: Path,
    minor: int,
    pci: str,
    vram: int = 16 * GIB,
    used: int = 0,
    vendor: str = "0x1002",
    partition: str | None = None,
) -> None:
    real = sys_root / "bus/pci/devices" / pci
    real.mkdir(parents=True, exist_ok=True)
    (real / "vendor").write_text(vendor + "\n")
    (real / "mem_info_vram_total").write_text(f"{vram}\n")
    (real / "mem_info_vram_used").write_text(f"{used}\n")
    if partition is not None:
        (real / "current_compute_partition").write_text(partition + "\n")
    link = sys_root / "class/drm" / f"renderD{minor}"
    link.mkdir(parents=True, exist_ok=True)
    (link / "device").symlink_to(real)


@pytest.fixture
def tree(tmp_path):
    """CPU node, discrete GPU, APU, partitioned GPU, foreign vendor, discrete GPU."""
    root = tmp_path / "sys"
    node(root, 0)
    node(root, 1, simd=64, gpu_id=1111, minor=128, unique=0xABCDEF)
    device(root, 128, "0000:03:00.0", used=4 * GIB)
    node(root, 2, simd=8, gpu_id=2222, minor=129)
    device(root, 129, "0000:04:00.0", vram=512 * 1024**2)
    node(root, 3, simd=64, gpu_id=3333, minor=130)
    device(root, 130, "0000:05:00.0", partition="CPX")
    node(root, 4, simd=64, gpu_id=4444, minor=131)
    device(root, 131, "0000:06:00.0", vendor="0x10de")
    node(root, 5, simd=64, gpu_id=5555, minor=132)
    device(root, 132, "0000:07:00.0", vram=8 * GIB, partition="SPX")
    return root


def test_detect_order_and_indices(tree):
    gpus = amd.detect(tree)
    assert [(g.index, g.gpu_id, g.pci) for g in gpus] == [
        (0, 1111, "0000:03:00.0"),
        (4, 5555, "0000:07:00.0"),
    ]
    assert gpus[0].vram == 16 * GIB


def test_detect_counts_skipped_gpus_in_index(tree):
    # The APU, the partitioned card and the foreign vendor still take indices 1 to 3.
    assert [g.index for g in amd.detect(tree)] == [0, 4]


def test_uuid_formatting_and_missing(tree):
    gpus = amd.detect(tree)
    assert gpus[0].uuid == "GPU-0000000000abcdef"
    assert gpus[1].uuid == ""


def test_numeric_node_order(tmp_path):
    root = tmp_path
    for n in (2, 10, 1):
        node(root, n, simd=4, gpu_id=100 + n, minor=120 + n)
        device(root, 120 + n, f"0000:0{n % 10}:00.0")
    assert [g.gpu_id for g in amd.detect(root)] == [101, 102, 110]


def test_no_kfd_tree(tmp_path):
    assert amd.detect(tmp_path) == ()
    assert amd.free_vram(tmp_path) == {}


def test_unreadable_topology(tree):
    nodes = tree / "class/kfd/kfd/topology/nodes"
    nodes.chmod(0)
    try:
        if os.access(nodes, os.R_OK):
            pytest.skip("permissions are not enforced")
        assert amd.detect(tree) is None
        assert amd.free_vram(tree) is None
    finally:
        nodes.chmod(stat.S_IRWXU)


def test_free_vram(tree):
    assert amd.free_vram(tree) == {0: 12 * GIB, 4: 8 * GIB}


def test_free_vram_failure(tree):
    (tree / "bus/pci/devices/0000:07:00.0/mem_info_vram_used").unlink()
    assert amd.free_vram(tree) is None


def test_key(tree):
    gpus = amd.detect(tree)
    assert amd.key(gpus[0]) == "GPU-0000000000abcdef"
    assert amd.key(gpus[1]) == "AMD-4"


def kfd_proc(root: Path, pid: int, **vram: int) -> None:
    directory = root / "class/kfd/kfd/proc" / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in vram.items():
        (directory / name).write_text(f"{value}\n")


def fdinfo(
    proc: Path, pid: int, fd: int, pdev: str, client: int, memory: str, driver: str = "amdgpu"
) -> None:
    directory = proc / str(pid) / "fdinfo"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / str(fd)).write_text(
        f"pos:\t0\ndrm-driver:\t{driver}\ndrm-pdev:\t{pdev}\n"
        f"drm-client-id:\t{client}\ndrm-memory-vram:\t{memory}\n"
    )


def test_kfd_proc_readings(tree, tmp_path):
    kfd_proc(tree, 42, vram_1111=3 * GIB, vram_5555=0)
    kfd_proc(tree, 43, vram_9999=5)
    kfd_proc(tree, 44, vram_5555=7)
    # a GPU opened with nothing on it is not in use, the runtime opens every GPU
    assert amd.apps(tree, tmp_path / "noproc") == [
        (42, "GPU-0000000000abcdef", 3 * GIB),
        (44, "AMD-4", 7),
    ]


def test_kfd_proc_empty(tree, tmp_path):
    (tree / "class/kfd/kfd/proc").mkdir(parents=True)
    assert amd.apps(tree, tmp_path / "noproc") == []


def test_kfd_wins_over_fdinfo(tree, tmp_path):
    (tree / "class/kfd/kfd/proc").mkdir(parents=True)
    proc = tmp_path / "proc"
    fdinfo(proc, 5, 3, "0000:03:00.0", 1, "1 KiB")
    assert amd.apps(tree, proc) == []


def test_fdinfo_dedupe_and_units(tree, tmp_path):
    proc = tmp_path / "proc"
    fdinfo(proc, 10, 3, "0000:03:00.0", 1, "2048 KiB")
    fdinfo(proc, 10, 4, "0000:03:00.0", 1, "2048 KiB")
    fdinfo(proc, 11, 3, "0000:03:00.0", 1, "2048 KiB")
    fdinfo(proc, 12, 3, "0000:07:00.0", 9, "3 MiB")
    fdinfo(proc, 13, 3, "0000:07:00.0", 8, "1000")
    fdinfo(proc, 14, 3, "0000:07:00.0", 7, "5 KiB", driver="i915")
    fdinfo(proc, 15, 3, "0000:99:00.0", 6, "5 KiB")
    assert amd.apps(tree, proc) == [
        (10, "GPU-0000000000abcdef", 2048 * 1024),
        (12, "AMD-4", 3 * 1024**2),
        (13, "AMD-4", 1000),
    ]


def test_fdinfo_other_users_skipped(tree, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    fdinfo(proc, 10, 3, "0000:03:00.0", 1, "1 KiB")
    other = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: other)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert amd.apps(tree, proc) is None


def fake_smi(directory: Path, output: str = "", code: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "amd-smi"
    payload = directory / "payload.json"
    payload.write_text(output)
    script.write_text(f"#!/bin/sh\ncat {payload}\nexit {code}\n")
    script.chmod(0o755)


def smi_path(monkeypatch, directory: Path) -> None:
    monkeypatch.setenv("PATH", f"{directory}:/usr/bin:/bin")


def quiet_proc(tmp_path: Path) -> Path:
    proc = tmp_path / "proc"
    fdinfo(proc, 1, 0, "0000:03:00.0", 1, "1 KiB", driver="i915")
    return proc


def test_smi_new_shape(tree, tmp_path, monkeypatch):
    data = [
        {
            "gpu": 0,
            "process_list": [
                {
                    "process_info": {
                        "pid": 77,
                        "memory_usage": {"vram_mem": {"value": 2, "unit": "GB"}},
                    }
                }
            ],
        },
        {"gpu": 4, "process_list": []},
    ]
    fake_smi(tmp_path / "bin", json.dumps(data))
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, quiet_proc(tmp_path)) == [(77, "GPU-0000000000abcdef", 2 * GIB)]


def test_smi_old_string_shape(tree, tmp_path, monkeypatch):
    data = [
        {
            "gpu": 4,
            "process_list": [{"pid": 78, "mem_usage": {"vram_mem": "16368 MB"}}],
        }
    ]
    fake_smi(tmp_path / "bin", json.dumps(data))
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, quiet_proc(tmp_path)) == [(78, "AMD-4", 16368 * 1024**2)]


def test_smi_bare_int_shape(tree, tmp_path, monkeypatch):
    data = [
        {
            "gpu": 0,
            "process_list": [{"process_info": {"pid": 79, "mem_usage": {"vram_mem": 12345}}}],
        }
    ]
    fake_smi(tmp_path / "bin", json.dumps(data))
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, quiet_proc(tmp_path)) == [(79, "GPU-0000000000abcdef", 12345)]


def test_smi_unknown_memory_is_not_counted_as_use(tree, tmp_path, monkeypatch):
    data = [{"gpu": 0, "process_list": [{"pid": 80}]}]
    fake_smi(tmp_path / "bin", json.dumps(data))
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, quiet_proc(tmp_path)) == []


@pytest.mark.parametrize("output", ["not json", '{"gpu": 0}', '[{"gpu": "x"}]', '[{"gpu": 0}]'])
def test_smi_malformed_then_blind(tree, tmp_path, monkeypatch, output):
    fake_smi(tmp_path / "bin", output)
    smi_path(monkeypatch, tmp_path / "bin")
    # fdinfo was readable and there are GPUs, so no users is the answer
    assert amd.apps(tree, quiet_proc(tmp_path)) == []
    # with no readable fdinfo the query has simply failed
    assert amd.apps(tree, tmp_path / "noproc") is None


def test_smi_nonzero_exit(tree, tmp_path, monkeypatch):
    fake_smi(tmp_path / "bin", "[]", code=3)
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, tmp_path / "noproc") is None


def test_smi_missing_and_no_fdinfo(tree, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert amd.apps(tree, tmp_path / "noproc") is None
    assert amd.apps(tree, quiet_proc(tmp_path)) == []


def test_fdinfo_wins_over_smi(tree, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    fdinfo(proc, 10, 3, "0000:03:00.0", 1, "1 KiB")
    fake_smi(tmp_path / "bin", json.dumps([{"gpu": 0, "process_list": [{"pid": 99}]}]))
    smi_path(monkeypatch, tmp_path / "bin")
    assert amd.apps(tree, proc) == [(10, "GPU-0000000000abcdef", 1024)]


def test_apps_without_gpus_and_no_source(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert amd.apps(tmp_path, quiet_proc(tmp_path)) is None


def test_apps_unreadable_topology(tree, tmp_path):
    nodes = tree / "class/kfd/kfd/topology/nodes"
    nodes.chmod(0)
    try:
        if os.access(nodes, os.R_OK):
            pytest.skip("permissions are not enforced")
        assert amd.apps(tree, tmp_path) is None
    finally:
        nodes.chmod(stat.S_IRWXU)
