import json
import multiprocessing
import os
import time

import pytest

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


def test_snapshot_inside_a_job_without_cache_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PROQU_JOB_ID", "abc123")
    machine.snapshot()
    assert not (tmp_path / "state" / "machine.json").exists()


def test_snapshot_expires(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    first = machine.snapshot()
    old = time.time() - machine.SNAPSHOT_AGE - 10
    os.utime(tmp_path / "state" / "machine.json", (old, old))
    seen = []
    monkeypatch.setattr(machine, "detect", lambda previous=None: seen.append(previous) or first)
    machine.snapshot()
    assert seen == [first]


def test_corrupt_snapshot_is_detected_again(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "machine.json").write_text("{torn")
    assert machine.snapshot() == machine.detect()


def _refresh_many(home, count):
    os.environ["PROQU_HOME"] = home
    os.environ.pop("PROQU_JOB_ID", None)
    for _ in range(count):
        machine.snapshot(refresh=True)


def test_concurrent_snapshots_do_not_crash(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=_refresh_many, args=(str(tmp_path), 50)) for _ in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_cgroup_limit_above_physical_ram_is_ignored(tmp_path):
    proc = fake_proc(tmp_path, "0::/box")
    root = tmp_path / "cg"
    (root / "box").mkdir(parents=True)
    (root / "box" / "memory.max").write_text(str(64 * 1024**3))
    assert machine.detect_ram(proc, root) == 16 * 1024**3


def test_cgroup_v1_reads_only_the_memory_controller(tmp_path):
    proc = fake_proc(tmp_path, "5:cpu,cpuacct:/x\n4:memory:/")
    root = tmp_path / "cg"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text(str(2 * 1024**3))
    # a decoy limit where the unfiltered code would look for the cpu line's path
    (root / "memory" / "x").mkdir(parents=True)
    (root / "memory" / "x" / "memory.limit_in_bytes").write_text(str(1024**3))
    assert machine.detect_ram(proc, root) == 2 * 1024**3


def test_cpu_quota_trims_a_known_cpu_set(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7})
    proc = fake_proc(tmp_path, "0::/box")
    root = tmp_path / "cg"
    (root / "box").mkdir(parents=True)
    (root / "box" / "cpu.max").write_text("250000 100000\n")
    assert machine.detect_cpus(proc, root) == (0, 1, 2)


def test_absent_nvidia_smi_clears_previous_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    previous = machine.Machine(1, (0,), (machine.GpuInfo(0, "GPU-aaa", 1024**3),))
    assert machine.detect(previous).gpus == ()


def test_snapshot_passes_previous_gpus_on(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    first = machine.snapshot()
    seen = []
    monkeypatch.setattr(machine, "detect", lambda previous=None: seen.append(previous) or first)
    machine.snapshot(refresh=True)
    assert seen == [first]


def test_cpu_quota_on_cgroup_v1(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    proc = fake_proc(tmp_path, "3:cpu,cpuacct:/box")
    node = tmp_path / "cg" / "cpu,cpuacct" / "box"
    node.mkdir(parents=True)
    (node / "cpu.cfs_quota_us").write_text("200000\n")
    (node / "cpu.cfs_period_us").write_text("100000\n")
    assert machine.detect_cpus(proc, tmp_path / "cg") == (0, 1)


def test_unlimited_cgroup_v1_cpu_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    proc = fake_proc(tmp_path, "3:cpu,cpuacct:/")
    node = tmp_path / "cg" / "cpu,cpuacct"
    node.mkdir(parents=True)
    (node / "cpu.cfs_quota_us").write_text("-1\n")
    (node / "cpu.cfs_period_us").write_text("100000\n")
    assert machine.detect_cpus(proc, tmp_path / "cg") == (0, 1)


def test_cgroup_path_above_the_mount_reads_the_root(tmp_path):
    proc = fake_proc(tmp_path, "0::/../../outside")
    root = tmp_path / "a" / "b" / "cg"
    root.mkdir(parents=True)
    (tmp_path / "a" / "outside").mkdir()
    (tmp_path / "a" / "outside" / "memory.max").write_text(str(1024**3))
    (root / "memory.max").write_text(str(2 * 1024**3))
    assert machine.detect_ram(proc, root) == 2 * 1024**3


@pytest.mark.parametrize(
    "data",
    [
        {"ram": "1G", "cpu_ids": [0], "gpus": []},
        {"ram": -1, "cpu_ids": [0], "gpus": []},
        {"ram": 1, "cpu_ids": [], "gpus": []},
        {"ram": 1, "cpu_ids": ["0"], "gpus": []},
        {"ram": 1, "cpu_ids": [0], "gpus": [{"index": 0, "uuid": 5, "vram": 1}]},
    ],
)
def test_nonsense_snapshot_is_detected_again(tmp_path, monkeypatch, data):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "machine.json").write_text(json.dumps(data))
    assert machine.snapshot() == machine.detect()


def test_unwritable_snapshot_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    state = machine.paths.state_dir()

    def refuse(*args, **kwargs):
        raise OSError("read only")

    monkeypatch.setattr(machine.tempfile, "mkstemp", refuse)
    assert machine.snapshot() == machine.detect()
    assert not (state / "machine.json").exists()
