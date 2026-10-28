"""macOS and AMD wiring, run on any machine against fakes of the kernel and the GPUs."""

import os
import resource
import subprocess
import sys
import time

import pytest

from proqu import amd, cli, darwin, limits, machine, procs, runner
from proqu.cli import main
from proqu.store import Job

from .test_runner import ended, log_of, starting_job

G = 1024**3


@pytest.fixture
def macos(monkeypatch):
    """The process helpers and detection take their macOS path."""
    monkeypatch.setattr(procs, "_macos", lambda: True)
    monkeypatch.setattr(machine, "_macos", lambda: True)
    monkeypatch.setattr(procs, "_job_ids", {})


def test_process_checks_ask_the_kernel_on_macos(macos, monkeypatch):
    table = {10: (os.getuid(), 500, False), 11: (os.getuid(), 600, True)}
    monkeypatch.setattr(darwin, "process", table.get)
    monkeypatch.setattr(darwin, "start_time", lambda pid: table[pid][1] if pid in table else None)
    monkeypatch.setattr(darwin, "boot_id", lambda: "boot-1")
    assert procs.start_ticks(10) == 500
    assert procs.is_alive(10, 500)
    assert not procs.is_alive(10, 501)
    # a zombie has exited
    assert not procs.is_alive(11, 600)
    assert not procs.is_alive(12, 1)
    assert procs.boot_id() == "boot-1"


def test_job_members_on_macos_by_session_and_by_marker(macos, monkeypatch):
    leader = subprocess.Popen(["sleep", "5"], start_new_session=True)
    marked = subprocess.Popen(["sleep", "5"], start_new_session=True)
    stranger = subprocess.Popen(["sleep", "5"], start_new_session=True)
    others = 999_999_999
    try:
        listed = [leader.pid, marked.pid, stranger.pid, others]
        monkeypatch.setattr(darwin, "pids", lambda: listed)
        uid = os.getuid()

        def process(pid):
            return (uid + 1, 1, False) if pid == others else (uid, 1, False)

        def environ(pid):
            return [b"A=1", b"PROQU_JOB_ID=job123"] if pid == marked.pid else [b"A=1"]

        monkeypatch.setattr(darwin, "process", process)
        monkeypatch.setattr(darwin, "environ", environ)
        found = procs.members_of({"job123": leader.pid, "other": None})
        assert found == {"job123": [leader.pid, marked.pid], "other": []}
        assert procs.session_of(leader.pid) == leader.pid
    finally:
        for child in (leader, marked, stranger):
            child.kill()
            child.wait()


def test_each_process_environment_is_read_once_on_macos(macos, monkeypatch):
    uid = os.getuid()
    starts = {100: 1, 101: 1}
    reads = []
    monkeypatch.setattr(darwin, "pids", lambda: list(starts))
    monkeypatch.setattr(darwin, "process", lambda pid: (uid, starts[pid], False))
    monkeypatch.setattr(procs, "session_of", lambda pid: None)

    def environ(pid):
        reads.append(pid)
        return [b"PROQU_JOB_ID=job1"] if pid == 100 else []

    monkeypatch.setattr(darwin, "environ", environ)
    for _ in range(3):
        assert procs.members_of({"job1": None}) == {"job1": [100]}
    assert sorted(reads) == [100, 101]
    # a reused pid is a new process, so it is read again
    starts[101] = 2
    procs.members_of({"job1": None})
    assert sorted(reads) == [100, 101, 101]


def test_machine_detection_on_macos(macos, monkeypatch):
    monkeypatch.setattr(darwin, "total_ram", lambda: 32 * G)
    monkeypatch.setattr(darwin, "cpu_count", lambda: 10)
    monkeypatch.setattr(darwin, "free_ram", lambda: 20 * G)
    apple = (machine.GpuInfo(0, "apple-m2", 0, "apple"),)
    monkeypatch.setattr(machine, "apple_gpus", lambda: apple)
    assert machine.detect() == machine.Machine(32 * G, tuple(range(10)), apple)
    assert machine.free_ram() == 20 * G


def test_cpus_are_counted_where_they_cannot_be_pinned(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(darwin, "cpu_count", lambda: 6)
    assert machine.available_cpus() == set(range(6))


def test_memory_and_enforcement_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(darwin, "footprint", lambda pid: 123 * G)
    assert limits.memory_of(os.getpid()) == 123 * G
    assert limits.cgroups_available() is False


def test_peak_after_the_job_is_read_in_bytes_on_macos(home, monkeypatch):
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    job = starting_job(["true"])
    monitor = limits.Monitor(job, os.getpid())
    monkeypatch.setattr(sys, "platform", "darwin")
    monitor.finish(0)
    assert monitor.peak == usage.ru_maxrss
    monkeypatch.setattr(sys, "platform", "linux")
    monitor.finish(0)
    assert monitor.peak == usage.ru_maxrss * 1024


def test_job_runs_unpinned_where_cpus_cannot_be_pinned(home, monkeypatch):
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    job = starting_job(["sh", "-c", "echo ran"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert log_of(job).strip() == "ran"


def test_the_mac_is_kept_awake_while_a_job_runs(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "args"
    (bin_dir / "caffeinate").write_text(f'#!/bin/sh\necho "$@" > {record}\n')
    (bin_dir / "caffeinate").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    runner.keep_awake(4242)
    time.sleep(0.2)
    assert not record.exists()
    monkeypatch.setattr(sys, "platform", "darwin")
    runner.keep_awake(4242)
    deadline = time.monotonic() + 5
    while not record.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert record.read_text().split() == ["-i", "-w", "4242"]


def test_limits_on_macos_say_how_jobs_are_watched(home, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    assert main(["limits"]) == 0
    out = capsys.readouterr().out
    assert "four times a second" in out
    assert "cannot pin a job to CPUs" in out


# AMD

AMD_MACHINE = machine.Machine(
    ram=16 * G,
    cpu_ids=(0, 1, 2, 3),
    gpus=(
        machine.GpuInfo(0, "GPU-00000000000000aa", 24 * G, "amd"),
        machine.GpuInfo(2, "AMD-2", 16 * G, "amd"),
    ),
)


def amd_job(**changes) -> Job:
    script = "echo [$ROCR_VISIBLE_DEVICES] [$HIP_VISIBLE_DEVICES] [$CUDA_VISIBLE_DEVICES]"
    return starting_job(["sh", "-c", script], **changes)


@pytest.fixture
def amd_box(home, monkeypatch):
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: AMD_MACHINE)
    monkeypatch.setattr(machine, "cached", lambda: AMD_MACHINE)
    monkeypatch.setattr(amd, "apps", lambda: [])
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def test_amd_job_sees_only_its_gpus(amd_box):
    job = amd_job(env=dict(os.environ), gpus=2, vram=G, gpu_ids=[0, 2])
    runner.supervise(job.id)
    # the other filters would apply again to the GPUs left visible, so they are dropped
    assert log_of(job).strip() == "[GPU-00000000000000aa,2] [] []"


def test_amd_job_without_a_gpu_sees_none(amd_box):
    job = amd_job(env=dict(os.environ))
    runner.supervise(job.id)
    assert log_of(job).strip() == "[-1] [-1] [-1]"


def test_amd_gpu_use_is_watched_and_named_for_amd(amd_box, monkeypatch):
    job = amd_job(env=dict(os.environ), gpus=1, vram=G, gpu_ids=[0])
    monitor = limits.Monitor(job, os.getpid(), ("GPU-00000000000000aa",))
    assert monitor.watch_gpus
    breach = limits.gpu_breach(job, ("GPU-00000000000000aa",), [(1, "AMD-2", 5)], "amd")
    assert breach[1].endswith("use the GPUs in ROCR_VISIBLE_DEVICES only")
    breach = limits.gpu_breach(job, ("GPU-00000000000000aa",), [(1, "GPU-00000000000000aa", 2 * G)])
    assert "measure it with nvidia-smi" in breach[1]
    breach = limits.gpu_breach(
        job, ("GPU-00000000000000aa",), [(1, "GPU-00000000000000aa", 2 * G)], "amd"
    )
    assert "measure it with amd-smi or rocm-smi" in breach[1]


def test_amd_queries_go_to_sysfs(amd_box, monkeypatch):
    monkeypatch.setattr(amd, "apps", lambda: [(5, "AMD-2", 3)])
    monkeypatch.setattr(amd, "free_vram", lambda: {0: G, 2: 2 * G})
    assert limits.gpu_apps() == [(5, "AMD-2", 3)]
    assert machine.free_vram() == {0: G, 2: 2 * G}


def test_apple_gpu_is_not_watched_and_has_no_variable(home, monkeypatch):
    apple = machine.Machine(16 * G, (0, 1), (machine.GpuInfo(0, "apple-m2", 0, "apple"),))
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: apple)
    monkeypatch.setattr(machine, "cached", lambda: apple)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    job = starting_job(["sh", "-c", "echo [$CUDA_VISIBLE_DEVICES]"], gpus=1, gpu_ids=[0])
    assert not limits.Monitor(job, os.getpid()).watch_gpus
    assert limits.gpu_apps() == []
    runner.supervise(job.id)
    assert log_of(job).strip() == "[]"


def test_nvidia_comes_first_then_amd(monkeypatch):
    found = (amd.AmdGpu(1, "", 8 * G, 7, "0000:03:00.0"),)
    monkeypatch.setattr(amd, "detect", lambda sys_root=None: found)
    monkeypatch.setattr(machine, "detect_nvidia", lambda: ())
    assert machine.detect_gpus() == (machine.GpuInfo(1, "AMD-1", 8 * G, "amd"),)
    nvidia = (machine.GpuInfo(0, "GPU-a", G),)
    monkeypatch.setattr(machine, "detect_nvidia", lambda: nvidia)
    assert machine.detect_gpus() == nvidia
    # a failed NVIDIA query keeps the previous GPUs rather than trying AMD
    monkeypatch.setattr(machine, "detect_nvidia", lambda: None)
    assert machine.detect_gpus() is None
    monkeypatch.setattr(machine, "detect_nvidia", lambda: ())
    monkeypatch.setattr(amd, "detect", lambda sys_root=None: None)
    assert machine.detect_gpus() is None
