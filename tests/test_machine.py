import os

from proqu import machine


def fake_proc(tmp_path, cgroup_line, mem_kib=16 * 1024**2):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemTotal:       {mem_kib} kB\nMemFree: 1 kB\n")
    (proc / "self" / "cgroup").write_text(cgroup_line + "\n")
    return proc


def test_ram_without_cgroup_limit(tmp_path):
    proc = fake_proc(tmp_path, "0::/user.slice")
    root = tmp_path / "cg"
    (root / "user.slice").mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text("max\n")
    assert machine.detect_ram(proc, root) == 16 * 1024**3


def test_ram_lowered_by_ancestor_cgroup(tmp_path):
    proc = fake_proc(tmp_path, "0::/outer/inner")
    root = tmp_path / "cg"
    (root / "outer" / "inner").mkdir(parents=True)
    (root / "outer" / "memory.max").write_text(str(4 * 1024**3))
    (root / "outer" / "inner" / "memory.max").write_text("max")
    assert machine.detect_ram(proc, root) == 4 * 1024**3


def test_ram_cgroup_v1_limit(tmp_path):
    proc = fake_proc(tmp_path, "4:memory:/docker/abc")
    root = tmp_path / "cg"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text(str(2 * 1024**3))
    assert machine.detect_ram(proc, root) == 2 * 1024**3


def test_ram_cgroup_v1_unlimited(tmp_path):
    proc = fake_proc(tmp_path, "4:memory:/")
    root = tmp_path / "cg"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712")
    assert machine.detect_ram(proc, root) == 16 * 1024**3


def test_cpus_match_affinity():
    assert machine.detect_cpus() == tuple(sorted(os.sched_getaffinity(0)))


def test_parse_gpus_skips_unreadable_rows():
    text = "0, GPU-aaa, 24576\n1, GPU-bbb, [N/A]\ngarbage\n"
    assert machine.parse_gpus(text) == (machine.GpuInfo(0, "GPU-aaa", 24576 * 1024**2),)


def fake_nvidia_smi(tmp_path, monkeypatch, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


def test_gpus_from_nvidia_smi(tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "echo '0, GPU-aaa, 1024'")
    assert machine.detect_gpus() == (machine.GpuInfo(0, "GPU-aaa", 1024**3),)


def test_gpus_when_nvidia_smi_fails(tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "exit 9")
    assert machine.detect_gpus() == ()


def test_gpus_when_nvidia_smi_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert machine.detect_gpus() == ()
