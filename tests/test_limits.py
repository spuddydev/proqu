import os
import sys
import time

from proqu import limits, runner
from proqu.store import Store

from .test_runner import ended, starting_job

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
