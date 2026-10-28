"""The queue end to end: real passes, real supervisors, real processes."""

import os
import time

import pytest

from proqu import machine, procs, tick
from proqu.cli import main
from proqu.store import Store

G = 1024**3
CPUS = tuple(sorted(os.sched_getaffinity(0)))[:2]


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PROQU_SESSION", "s1")
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    small = machine.Machine(ram=4 * G, cpu_ids=CPUS, gpus=())
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: small)
    yield tmp_path
    for job in Store().active():
        tick.kill_leftovers([job.id])


def submit(*extra, command=("sleep", "1")):
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "2s", "--max", "30s"]
    assert main([*request, *extra, "--", *command]) == 0
    return max(Store().for_session("s1"), key=lambda job: job.submitted_at)


def settle(job_ids, timeout=30):
    store = Store()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = [store.get(job_id) for job_id in job_ids]
        if all(job.state not in ("queued", "starting", "running") for job in jobs):
            return jobs
        time.sleep(0.1)
    raise AssertionError(f"jobs still active: {[(j.id, j.state) for j in jobs]}")


@pytest.mark.skipif(len(CPUS) < 2, reason="needs two CPUs")
def test_jobs_start_when_they_fit_and_the_rest_follow(queue):
    first = submit("--ram", "3G")
    second = submit("--ram", "2G")
    third = submit("--ram", "1G")
    store = Store()
    assert store.get(first.id).state in ("starting", "running")
    assert store.get(second.id).state == "queued"
    assert store.get(third.id).state in ("starting", "running")
    done = settle([first.id, second.id, third.id])
    assert [job.state for job in done] == ["succeeded"] * 3
    assert done[1].started_at >= min(done[0].ended_at, done[2].ended_at)


def test_dependant_of_a_failed_job_fails_and_is_refunded(queue):
    parent = submit(command=("sh", "-c", "sleep 1; exit 4"))
    child = submit("--after", parent.id, "--bid", "100")
    assert Store().bid_spent("s1") == 100
    parent_done, child_done = settle([parent.id, child.id])
    assert parent_done.reason == "exited with code 4"
    assert child_done.state == "failed"
    assert child_done.reason == f"job {parent.id}, which it waits for, ended failed"
    assert Store().bid_spent("s1") == 0


def test_dependant_waits_even_when_it_fits(queue):
    parent = submit()
    child = submit("--after", parent.id, command=("true",))
    assert Store().get(child.id).state == "queued"
    parent_done, child_done = settle([parent.id, child.id])
    assert child_done.state == "succeeded"
    assert child_done.started_at >= parent_done.ended_at


def test_lost_supervisor_is_noticed_by_the_next_command(queue, capsys):
    job = submit(command=("sleep", "30"))
    store = Store()
    deadline = time.monotonic() + 10
    while store.get(job.id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    running = store.get(job.id)
    os.kill(running.supervisor_pid, 9)
    time.sleep(0.3)
    assert main(["status"]) == 0
    lost = store.get(job.id)
    assert lost.state == "failed"
    assert lost.reason == tick.LOST
    time.sleep(0.2)
    assert not procs.job_members(job.id, None)


def test_restart_loses_running_jobs(queue):
    job = submit(command=("sleep", "30"))
    store = Store()
    (queue / "state" / "boot_id").write_text("an-earlier-boot\n")
    tick.tick(store)
    assert store.get(job.id).reason == tick.LOST


def test_cancelled_job_that_lost_its_supervisor_reads_cancelled(queue):
    job = submit(command=("sleep", "30"))
    store = Store()
    with store.locked():
        store.update(job.id, cancel_requested=True, supervisor_pid=1, supervisor_start=0)
    tick.tick(store)
    assert store.get(job.id).state == "cancelled"


def test_passes_inside_a_job_do_not_start_jobs(queue, monkeypatch):
    monkeypatch.setenv("PROQU_JOB_ID", "outer1")
    job = submit(command=("true",))
    assert Store().get(job.id).state == "queued"
    monkeypatch.delenv("PROQU_JOB_ID")
    tick.tick()
    assert settle([job.id])[0].state == "succeeded"
