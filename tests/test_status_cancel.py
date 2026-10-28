import os
import time

from proqu.cli import main
from proqu.store import Store

BASE = ["submit", "--ram", "2G", "--cpus", "2", "--est", "1m", "--max", "5m"]


def submit(*extra):
    assert main([*BASE, *extra, "--", "true"]) == 0
    return max(Store().for_session(os.environ["PROQU_SESSION"]), key=lambda job: job.submitted_at)


def test_status_lists_only_this_sessions_jobs(home, capsys, monkeypatch):
    mine = submit("--name", "mine")
    monkeypatch.setenv("PROQU_SESSION", "s2")
    theirs = submit("--name", "theirs", "--bid", "50")
    monkeypatch.setenv("PROQU_SESSION", "s1")
    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert f"{mine.id} mine: queued" in out
    assert theirs.id not in out
    assert "Bid budget: 11000 of 11000 left" in out


def test_status_of_one_job(home, capsys):
    job = submit("--name", "train", "--bid", "5")
    capsys.readouterr()
    assert main(["status", job.id]) == 0
    out = capsys.readouterr().out
    assert "command: true" in out
    assert "bid: 5" in out
    assert "Bid budget: 10995 of 11000 left" in out


def test_status_of_missing_and_foreign_jobs_read_the_same(home, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "s2")
    theirs = submit()
    monkeypatch.setenv("PROQU_SESSION", "s1")
    capsys.readouterr()
    assert main(["status", theirs.id]) == 1
    foreign = capsys.readouterr().err.replace(theirs.id, "ID")
    assert main(["status", "ffffff"]) == 1
    missing = capsys.readouterr().err.replace("ffffff", "ID")
    assert foreign == missing


def test_status_hides_old_finished_jobs(home, capsys):
    store = Store()
    for index in range(12):
        job = submit()
        with store.locked():
            store.update(job.id, state="succeeded", started_at=index, ended_at=index + 1)
    capsys.readouterr()
    main(["status"])
    assert "2 older finished jobs hidden" in capsys.readouterr().out
    main(["status", "--all"])
    assert "hidden" not in capsys.readouterr().out


def test_status_shows_kill_reason(home, capsys):
    job = submit()
    store = Store()
    with store.locked():
        store.update(
            job.id,
            state="killed",
            reason="exceeded RAM limit 2.0G",
            started_at=time.time() - 90,
            ended_at=time.time(),
        )
    capsys.readouterr()
    main(["status"])
    assert "killed after 1m30s: exceeded RAM limit 2.0G" in capsys.readouterr().out


def test_cancel_queued_job_refunds_bid(home, capsys):
    job = submit("--bid", "300")
    assert Store().bid_spent("s1") == 300
    assert main(["cancel", job.id]) == 0
    assert "bid of 300 refunded" in capsys.readouterr().out
    cancelled = Store().get(job.id)
    assert (cancelled.state, cancelled.reason) == ("cancelled", "cancelled by you")
    assert Store().bid_spent("s1") == 0


def test_cancel_twice_is_an_error(home, capsys):
    job = submit("--bid", "300")
    main(["cancel", job.id])
    assert main(["cancel", job.id]) == 1
    assert "already cancelled" in capsys.readouterr().err
    assert Store().bid_spent("s1") == 0


def test_cancel_running_job_only_flags_it(home, capsys):
    job = submit("--bid", "300")
    store = Store()
    with store.locked():
        store.update(job.id, state="running", started_at=time.time())
    assert main(["cancel", job.id]) == 0
    assert "will be stopped shortly" in capsys.readouterr().out
    flagged = Store().get(job.id)
    assert (flagged.state, flagged.cancel_requested) == ("running", True)
    assert Store().bid_spent("s1") == 300


def test_cannot_cancel_another_sessions_job(home, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "s2")
    theirs = submit()
    monkeypatch.setenv("PROQU_SESSION", "s1")
    assert main(["cancel", theirs.id]) == 1
    assert Store().get(theirs.id).state == "queued"
