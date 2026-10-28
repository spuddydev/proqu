import os
import sys
import time

import pytest

from proqu import procs, runner
from proqu.store import Job, Store


def starting_job(argv, cwd="/", **changes):
    store = Store()
    values = dict(
        id=store.new_id(),
        session="s1",
        name=None,
        argv=argv,
        cwd=str(cwd),
        env=dict(os.environ),
        ram=1024**3,
        cpus=1,
        gpus=0,
        vram=0,
        est=5,
        max=60,
        bid=0,
        after=[],
        submitted_at=time.time(),
        state="starting",
        cpu_ids=[sorted(os.sched_getaffinity(0))[0]],
    )
    values.update(changes)
    job = Job(**values)
    with store.locked():
        store.insert(job)
    return job


def log_of(job):
    from proqu import paths

    return paths.log_path(job.id).read_text()


def ended(job):
    return Store().get(job.id)


def test_successful_job_runs_in_its_cwd_with_its_env(home, tmp_path):
    job = starting_job(
        ["sh", "-c", "pwd; echo $PROQU_JOB_ID; echo [$CUDA_VISIBLE_DEVICES]"], tmp_path
    )
    runner.supervise(job.id)
    done = ended(job)
    assert (done.state, done.exit_code, done.reason) == ("succeeded", 0, None)
    assert done.started_at and done.ended_at >= done.started_at
    assert log_of(job).split() == [str(tmp_path), job.id, "[]"]


def test_job_is_pinned_to_its_cpus(home):
    cpu = sorted(os.sched_getaffinity(0))[-1]
    job = starting_job(["sh", "-c", "grep Cpus_allowed_list /proc/self/status"], cpu_ids=[cpu])
    runner.supervise(job.id)
    assert log_of(job).split()[-1] == str(cpu)


def test_failing_job_records_its_exit_code(home):
    job = starting_job(["sh", "-c", "exit 3"])
    runner.supervise(job.id)
    done = ended(job)
    assert (done.state, done.exit_code, done.reason) == ("failed", 3, "exited with code 3")


def test_job_killed_by_a_signal(home):
    job = starting_job(["sh", "-c", "kill -SEGV $$"])
    runner.supervise(job.id)
    assert ended(job).reason == "terminated by signal SIGSEGV"


def test_job_that_cannot_start(home):
    job = starting_job(["/no/such/program"])
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "failed"
    assert done.reason.startswith("could not start: No such file or directory")


def test_job_with_a_missing_cwd(home, tmp_path):
    job = starting_job(["true"], tmp_path / "gone")
    runner.supervise(job.id)
    assert ended(job).reason.startswith("could not start")


def test_leftover_processes_are_stopped_before_the_job_ends(home):
    job = starting_job(["sh", "-c", "sleep 30 & echo $! ; setsid sleep 30 & echo $!"])
    runner.supervise(job.id)
    pids = [int(pid) for pid in log_of(job).split()]
    assert ended(job).state == "succeeded"
    for pid in pids:
        state = procs._stat_fields(pid)
        assert state is None or state[0] in ("Z", "X")


def test_only_starting_jobs_are_run(home):
    job = starting_job(["true"], state="queued")
    runner.supervise(job.id)
    assert ended(job).state == "queued"


def test_job_cancelled_before_launch_never_runs(home, tmp_path):
    marker = tmp_path / "ran"
    job = starting_job(["touch", str(marker)], cancel_requested=True)
    runner.supervise(job.id)
    done = ended(job)
    assert (done.state, done.reason) == ("cancelled", "cancelled by you")
    assert not marker.exists()


@pytest.mark.parametrize(
    ("code", "outcome"), [(0, ("succeeded", None)), (2, ("failed", "exited with code 2"))]
)
def test_classify(code, outcome):
    assert runner.classify(code) == outcome


def test_spawned_supervisor_runs_the_job_detached(home):
    job = starting_job([sys.executable, "-c", "print('hello')"])
    store = Store()
    pid, ticks = runner.spawn_supervisor(job.id)
    assert procs.start_ticks(pid) in (ticks, None)
    deadline = time.monotonic() + 20
    while store.get(job.id).state in ("starting", "running") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert store.get(job.id).state == "succeeded"
    assert "hello" in log_of(job)
    time.sleep(0.2)
    assert not procs.is_alive(pid, ticks)


def test_job_is_stopped_at_its_max_runtime(home):
    job = starting_job(["sleep", "30"], max=1)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("still running at its max runtime of 1s")
    assert time.monotonic() - started < 10


def test_stubborn_job_is_killed_after_the_grace(home, monkeypatch):
    from proqu import config

    monkeypatch.setattr(config, "load", lambda path: {"kill_grace": 1})
    job = starting_job(["sh", "-c", "trap '' TERM; sleep 30"], max=1)
    started = time.monotonic()
    runner.supervise(job.id)
    assert ended(job).state == "killed"
    assert 2 <= time.monotonic() - started < 10


def test_job_that_exits_cleanly_after_the_polite_kill_still_counts_as_killed(home):
    job = starting_job(["sh", "-c", "trap 'exit 0' TERM; sleep 30 & wait"], max=1)
    runner.supervise(job.id)
    assert ended(job).state == "killed"


def test_running_job_stops_on_cancel(home):
    import threading

    job = starting_job(["sleep", "30"])

    def cancel_soon():
        store = Store()
        while store.get(job.id).state != "running":
            time.sleep(0.05)
        with store.locked():
            store.update(job.id, cancel_requested=True)

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert (done.state, done.reason) == ("cancelled", "cancelled by you")
    assert time.monotonic() - started < 10
