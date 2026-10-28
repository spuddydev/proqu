import os
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
