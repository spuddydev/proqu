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


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
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
    deadline = time.monotonic() + 20
    while procs.is_alive(pid, ticks) and time.monotonic() < deadline:
        time.sleep(0.1)
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


def cancel_after(job, delay):
    import threading

    def cancel():
        store = Store()
        while store.get(job.id).state != "running":
            time.sleep(0.01)
        time.sleep(delay)
        with store.locked():
            store.update(job.id, cancel_requested=True)

    threading.Thread(target=cancel, daemon=True).start()


def test_cancel_counts_when_the_job_ends_between_looks(home):
    job = starting_job(["sleep", "0.4"])
    cancel_after(job, 0.1)
    runner.supervise(job.id)
    assert ended(job).state == "cancelled"


def test_job_cancelled_before_launch_gets_its_bid_back(home):
    store = Store()
    job = starting_job(["true"], bid=300, cancel_requested=True)
    with store.locked():
        store.add_bid_spent("s1", 300)
    runner.supervise(job.id)
    assert store.bid_spent("s1") == 0


def test_job_that_could_not_start_gets_its_bid_back(home):
    store = Store()
    job = starting_job(["/no/such/program"], bid=300)
    with store.locked():
        store.add_bid_spent("s1", 300)
    runner.supervise(job.id)
    assert store.bid_spent("s1") == 0


def test_job_that_ran_keeps_its_bid_spent(home):
    store = Store()
    job = starting_job(["true"], bid=300)
    with store.locked():
        store.add_bid_spent("s1", 300)
    runner.supervise(job.id)
    assert store.bid_spent("s1") == 300


def test_unavailable_cpus_are_reported_plainly(home):
    job = starting_job(["true"], cpu_ids=[4096])
    runner.supervise(job.id)
    assert (
        ended(job).reason
        == "could not start: its CPUs [4096] are no longer available on this machine"
    )


def test_assigned_gpus_are_visible_by_uuid(home):
    job = starting_job(["sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], gpus=1, vram=1, gpu_ids=[1])
    runner.supervise(job.id)
    assert log_of(job).strip() == "GPU-b"


def test_first_writer_wins_over_the_supervisor(home):
    job = starting_job(["sleep", "0.5"])
    import threading

    def lose():
        store = Store()
        while store.get(job.id).state != "running":
            time.sleep(0.01)
        with store.locked():
            store.transition(job.id, ("running",), "failed", reason="lost")

    threading.Thread(target=lose, daemon=True).start()
    runner.supervise(job.id)
    assert (ended(job).state, ended(job).reason) == ("failed", "lost")


def test_supervisor_error_still_stops_the_job(home, monkeypatch):
    def broken(*args):
        raise RuntimeError("supervisor bug")

    monkeypatch.setattr(runner, "watch", broken)
    job = starting_job(["sh", "-c", "sleep 30 & sleep 30"])
    with pytest.raises(RuntimeError):
        runner.supervise(job.id)
    assert procs.job_members(job.id, None) == []


def test_supervisor_ignores_modules_in_the_callers_directory(home, tmp_path, monkeypatch):
    (tmp_path / "json.py").write_text("raise SystemExit('shadowed')\n")
    (tmp_path / "sqlite3.py").write_text("raise SystemExit('shadowed')\n")
    monkeypatch.chdir(tmp_path)
    job = starting_job([sys.executable, "-c", "print('fine')"])
    runner.spawn_supervisor(job.id)
    store = Store()
    deadline = time.monotonic() + 20
    while store.get(job.id).state in ("starting", "running") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert store.get(job.id).state == "succeeded"


def test_job_without_a_path_is_found_in_the_default_path(home):
    job = starting_job(["true"], env={})
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"


def test_file_that_is_not_executable_is_reported_as_denied(home, tmp_path):
    (tmp_path / "tool").write_text("#!/bin/sh\n")
    job = starting_job(["./tool"], tmp_path)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "failed"
    assert done.reason == "could not start: Permission denied ./tool"
    on_path = starting_job(["tool"], tmp_path, env={"PATH": str(tmp_path)})
    runner.supervise(on_path.id)
    assert ended(on_path).reason == "could not start: Permission denied tool"
