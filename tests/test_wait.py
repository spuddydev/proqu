import os
import signal
import sqlite3
import time

import pytest

from proqu import cli, paths
from proqu.cli import main
from proqu.store import Store

BASE = ["submit", "--ram", "2G", "--cpus", "2", "--est", "1m", "--max", "5m"]


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(cli, "WAIT_POLL", 0.05)


def submit(*extra):
    assert main([*BASE, *extra, "--", "true"]) == 0
    return max(Store().for_session(os.environ["PROQU_SESSION"]), key=lambda job: job.submitted_at)


def end(job_id, **changes):
    store = Store()
    with store.locked():
        store.update(job_id, ended_at=time.time(), **changes)


def test_wait_reports_success_and_marks_it_seen(home, capsys):
    job = submit("--name", "train")
    end(job.id, state="succeeded", started_at=time.time() - 5, exit_code=0)
    paths.log_path(job.id).write_text("hello\n")
    capsys.readouterr()
    assert main(["wait", job.id]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Job {job.id} (train) succeeded after 5s\n")
    assert out.endswith("hello\n")
    store = Store()
    assert store.get(job.id).notified
    assert store.waiters(job.id) == []


def test_wait_exits_1_for_any_other_end(home, capsys):
    job = submit()
    end(job.id, state="cancelled", reason="cancelled by you")
    assert main(["wait", job.id]) == 1
    assert f"Job {job.id} cancelled before it started: cancelled by you" in capsys.readouterr().out


def test_wait_times_out_on_a_queued_job(home, capsys):
    job = submit("--cpus", "4")
    second = submit("--cpus", "4")
    capsys.readouterr()
    assert main(["wait", second.id, "--timeout", "1s"]) == cli.TIMED_OUT
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        f"Job {second.id} is queued, position 2 in the queue, "
        "expected to start in 1m and finish in 2m."
    )
    assert lines[-1].startswith(f"Stopped waiting after 1s: job {second.id} is still queued.")
    assert Store().waiters(second.id) == []
    assert not Store().get(second.id).notified
    assert job.id not in lines[0]


def test_wait_sees_a_job_end_while_waiting(home, capsys, monkeypatch):
    job = submit()
    seen = []

    def poll(seconds):
        # the waiter is registered while waiting, and the job ends later
        seen.append(Store().waiters(job.id))
        end(job.id, state="failed", reason="exited with code 2", started_at=time.time())

    monkeypatch.setattr(cli.time, "sleep", poll)
    assert main(["wait", job.id]) == 1
    assert len(seen) == 1 and len(seen[0]) == 1 and seen[0][0][0] == os.getpid()
    assert Store().waiters(job.id) == []


def test_wait_on_another_sessions_job_is_refused(home, capsys, monkeypatch):
    job = submit()
    monkeypatch.setenv("PROQU_SESSION", "s2")
    assert main(["wait", job.id]) == 1
    assert f"no job {job.id} in this session" in capsys.readouterr().err
    assert Store().waiters(job.id) == []


def test_bad_timeout_is_refused(home, capsys):
    job = submit()
    assert main(["wait", job.id, "--timeout", "soon"]) == 2
    assert "--timeout" in capsys.readouterr().err


def test_wait_on_a_queued_job_stops_when_the_config_is_broken(home, capsys):
    submit("--cpus", "4")
    second = submit("--cpus", "4")
    paths.config_path().write_text("{bad")
    capsys.readouterr()
    assert main(["wait", second.id]) == cli.CONFIG_BROKEN
    out = capsys.readouterr().out
    assert f"job {second.id} is still queued and nothing can start" in out
    assert "not valid JSON" in out and f"run `proqu wait {second.id}` again" in out
    assert Store().waiters(second.id) == []


def test_wait_stops_when_the_config_breaks_while_waiting(home, capsys, monkeypatch):
    submit("--cpus", "4")
    second = submit("--cpus", "4")
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: paths.config_path().write_text("{bad"))
    capsys.readouterr()
    assert main(["wait", second.id]) == cli.CONFIG_BROKEN
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"Job {second.id} is queued, position 2")
    assert "nothing can start" in lines[-1]


@pytest.mark.parametrize("error", [cli.LockTimeout("busy"), sqlite3.OperationalError("disk")])
def test_wait_report_stands_when_marking_it_seen_fails(home, capsys, monkeypatch, error):
    job = submit()
    end(job.id, state="succeeded", started_at=time.time() - 5, exit_code=0)

    def refuse(self, job_id, **changes):
        raise error

    monkeypatch.setattr(Store, "update", refuse)
    capsys.readouterr()
    assert main(["wait", job.id]) == 0
    assert f"Job {job.id} succeeded after 5s" in capsys.readouterr().out


def test_terminated_after_the_report_is_not_a_stopped_wait(home, capsys, monkeypatch):
    job = submit()
    end(job.id, state="failed", reason="exited with code 2", started_at=time.time())

    def terminate(self, job_id, **changes):
        raise cli.Terminated

    monkeypatch.setattr(Store, "update", terminate)
    capsys.readouterr()
    assert main(["wait", job.id]) == 1
    out = capsys.readouterr().out
    assert f"Job {job.id} failed after" in out
    assert "Stopped waiting" not in out


def interrupt_wait_with(monkeypatch, signum):
    def stop(store, job_id, timeout):
        os.kill(os.getpid(), signum)
        raise AssertionError("the signal should have ended the wait")

    monkeypatch.setattr(cli, "wait_for", stop)


@pytest.mark.parametrize(("signum", "code"), [(signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_a_stopping_signal_ends_the_wait_and_restores_the_handlers(
    home, capsys, monkeypatch, signum, code
):
    job = submit()
    before = {sig: signal.getsignal(sig) for sig in cli.STOPPING}
    interrupt_wait_with(monkeypatch, signum)
    capsys.readouterr()
    assert main(["wait", job.id]) == code
    assert "Stopped waiting" in capsys.readouterr().out
    assert {sig: signal.getsignal(sig) for sig in cli.STOPPING} == before
    assert Store().waiters(job.id) == []


def test_a_second_signal_cannot_interrupt_the_cleanup(home, monkeypatch):
    job = submit()
    real = cli.remove_waiter

    def remove(job_id, pid):
        # a hangup arriving while the terminate is being handled
        os.kill(os.getpid(), signal.SIGHUP)
        real(job_id, pid)

    monkeypatch.setattr(cli, "remove_waiter", remove)
    interrupt_wait_with(monkeypatch, signal.SIGTERM)
    assert main(["wait", job.id]) == 143
    assert Store().waiters(job.id) == []


def test_ctrl_c_stops_the_wait_with_130(home, capsys, monkeypatch):
    job = submit()

    def stop(store, job_id, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "wait_for", stop)
    capsys.readouterr()
    assert main(["wait", job.id]) == 130
    assert f"Stopped waiting for job {job.id} before it ended." in capsys.readouterr().out
    assert Store().waiters(job.id) == []


def test_wait_runs_scheduling_passes_while_it_waits(home, monkeypatch):
    job = submit()
    passes = []
    monkeypatch.setattr(cli, "WAIT_TICK", 0.0)
    monkeypatch.setattr(cli.ticking, "tick", lambda store=None: passes.append(1))
    assert cli.wait_for(Store(), job.id, 0.3) == cli.TIMED_OUT
    assert len(passes) >= 2


@pytest.mark.parametrize("error", [cli.LockTimeout("busy"), sqlite3.OperationalError("locked")])
def test_a_failed_scheduling_pass_does_not_end_the_wait(home, monkeypatch, error):
    job = submit()

    def refuse(store=None):
        raise error

    monkeypatch.setattr(cli, "WAIT_TICK", 0.0)
    monkeypatch.setattr(cli.ticking, "tick", refuse)
    assert cli.wait_for(Store(), job.id, 0.2) == cli.TIMED_OUT


def test_wait_never_sleeps_past_its_timeout(home, monkeypatch):
    job = submit()
    sleeps = []
    real = time.sleep

    def sleep(seconds):
        sleeps.append(seconds)
        real(seconds)

    monkeypatch.setattr(cli, "WAIT_POLL", 5.0)
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli.wait_for(Store(), job.id, 0.2) == cli.TIMED_OUT
    assert sleeps and max(sleeps) <= 0.2
