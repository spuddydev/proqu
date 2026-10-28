import os
import sys
import time

import pytest

from proqu import limits, runner
from proqu.store import Store

from .test_runner import ended, log_of, starting_job

M = 1024**2

GROW = (
    "import time\n"
    "blocks = []\n"
    "for _ in range(80):\n"
    "    blocks.append(bytearray(10 * 1024 * 1024))\n"
    "    time.sleep(0.05)\n"
    "time.sleep(30)\n"
)


def test_memory_of_this_process_is_counted():
    assert limits.memory_of(os.getpid()) > M
    assert limits.memory_of(2**22 + 12345) == 0


def test_growing_job_is_killed_at_its_ram_limit(home):
    job = starting_job([sys.executable, "-c", GROW], ram=200 * M)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used more than its RAM limit of 200.0M")
    assert "this was the job's own limit" in done.reason
    assert done.peak_ram > 200 * M
    assert time.monotonic() - started < 20


def test_peak_ram_is_recorded_for_a_job_within_limits(home):
    code = "import time\nx = bytearray(60 * 1024 * 1024)\ntime.sleep(2.5)\n"
    job = starting_job([sys.executable, "-c", code], ram=500 * M)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "succeeded"
    assert 60 * M < done.peak_ram < 500 * M


def test_children_count_towards_the_limit(home):
    child = "import time\nx = bytearray(150 * 1024 * 1024)\ntime.sleep(30)\n"
    script = f"{sys.executable} -c '{child}' & {sys.executable} -c '{child}' & wait"
    job = starting_job(["sh", "-c", script], ram=250 * M)
    runner.supervise(job.id)
    assert ended(job).state == "killed"
    assert Store().get(job.id).peak_ram > 250 * M


REAL_PROBE = limits.cgroups_available


@pytest.fixture
def cgroups(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    if not REAL_PROBE():
        pytest.skip("this machine cannot give jobs their own cgroup")


@pytest.mark.parametrize("attempt", range(5))
def test_kernel_kills_a_job_over_its_ram_in_a_scope(cgroups, attempt):
    job = starting_job([sys.executable, "-c", GROW], ram=200 * M)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used more than its RAM limit of 200.0M")


def test_scope_carries_the_cpu_quota_and_peak(cgroups):
    script = 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/cpu.max"; sleep 1.5'
    job = starting_job(["sh", "-c", script], cpus=2, cpu_ids=[0, 1])
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "succeeded"
    assert log_of(job).split()[:2] == ["200000", "100000"]
    assert done.peak_ram


def test_fast_job_in_a_scope_runs_once(cgroups, tmp_path):
    counter = tmp_path / "count"
    job = starting_job(["sh", "-c", f"echo x >> {counter}"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert counter.read_text() == "x\n"


def test_failing_systemd_run_falls_back_to_watching(home, monkeypatch, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemd-run").write_text("#!/bin/sh\necho no bus >&2\nexit 1\n")
    (fake / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setattr(limits, "cgroups_available", lambda: True)
    forgotten = []
    monkeypatch.setattr(limits, "mark_unavailable", lambda: forgotten.append(1))
    counter = tmp_path / "count"
    job = starting_job(["sh", "-c", f"echo x >> {counter}"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert counter.read_text() == "x\n"
    assert forgotten == [1]


def test_probe_result_is_cached(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    calls = []
    monkeypatch.setattr(limits, "_probe", lambda: calls.append(1) or True)
    assert limits.cgroups_available() and limits.cgroups_available()
    assert calls == [1]
    limits.mark_unavailable()
    assert not limits.cgroups_available()
    assert calls == [1]
