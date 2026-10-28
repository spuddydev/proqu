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


def test_ram_in_private_cgroup_namespace(tmp_path):
    proc = fake_proc(tmp_path, "0::/not/visible/here")
    root = tmp_path / "cg"
    root.mkdir()
    (root / "memory.max").write_text(str(3 * 1024**3))
    assert machine.detect_ram(proc, root) == 3 * 1024**3


def test_cpus_match_affinity(tmp_path):
    proc = fake_proc(tmp_path, "0::/")
    assert machine.detect_cpus(proc, tmp_path / "none") == tuple(sorted(os.sched_getaffinity(0)))


def test_cpus_trimmed_by_cgroup_quota(tmp_path):
    proc = fake_proc(tmp_path, "0::/box")
    root = tmp_path / "cg"
    (root / "box").mkdir(parents=True)
    (root / "box" / "cpu.max").write_text("150000 100000\n")
    (root / "cpu.max").write_text("max 100000\n")
    assert machine.detect_cpus(proc, root) == tuple(sorted(os.sched_getaffinity(0)))[:2]


def test_parse_gpus_skips_unreadable_and_mig_rows():
    text = (
        "0, GPU-aaa, 24576, [N/A]\r\n"
        "1, GPU-bbb, [N/A], Disabled\n"
        "2, GPU-ccc, 40960, Enabled\n"
        "garbage\n\n"
        "3, GPU-ddd, 8192, Disabled\n"
    )
    assert machine.parse_gpus(text) == (
        machine.GpuInfo(0, "GPU-aaa", 24576 * 1024**2),
        machine.GpuInfo(3, "GPU-ddd", 8192 * 1024**2),
    )


def fake_nvidia_smi(tmp_path, monkeypatch, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


def test_gpus_from_nvidia_smi(tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "echo '0, GPU-aaa, 1024, Disabled'")
    assert machine.detect_gpus() == (machine.GpuInfo(0, "GPU-aaa", 1024**3),)


def test_gpus_when_nvidia_smi_fails(tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "exit 9")
    assert machine.detect_gpus() is None


def test_gpus_when_nvidia_smi_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert machine.detect_gpus() == ()


def test_failed_gpu_query_keeps_previous_gpus(tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "exit 9")
    previous = machine.Machine(1, (0,), (machine.GpuInfo(0, "GPU-aaa", 1024**3),))
    assert machine.detect(previous).gpus == previous.gpus


def test_snapshot_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    first = machine.snapshot()
    calls = []
    monkeypatch.setattr(machine, "detect", lambda previous=None: calls.append(1) or first)
    assert machine.snapshot() == first
    assert calls == []
    machine.snapshot(refresh=True)
    assert calls == [1]


def test_snapshot_never_refreshes_inside_a_job(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    first = machine.snapshot()
    monkeypatch.setenv("PROQU_JOB_ID", "abc123")
    monkeypatch.setattr(machine, "detect", lambda previous=None: 1 / 0)
    assert machine.snapshot(refresh=True) == first
