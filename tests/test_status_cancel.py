import itertools
import os
import re
import time
from types import SimpleNamespace

import pytest

from proqu import cli
from proqu.cli import main
from proqu.store import Store

BASE = ["submit", "--ram", "2G", "--cpus", "2", "--est", "1m", "--max", "5m"]


NOW = 1_800_000_000.0


@pytest.fixture
def frozen(monkeypatch):
    """The command's own clock moves a millisecond per reading, so durations are exact.

    It still moves, so that jobs submitted one after another keep their order.
    """
    ticks = itertools.count()
    clock = SimpleNamespace(
        time=lambda: NOW + next(ticks) / 1000, sleep=time.sleep, monotonic=time.monotonic
    )
    monkeypatch.setattr(cli, "time", clock)


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


def test_status_hides_the_oldest_finished_jobs(home, capsys):
    store = Store()
    ids = []
    for index in range(12):
        job = submit()
        ids.append(job.id)
        with store.locked():
            store.update(job.id, state="succeeded", started_at=index, ended_at=index + 1)
    capsys.readouterr()
    main(["status"])
    out = capsys.readouterr().out
    assert ids[0] not in out and ids[1] not in out
    assert ids[2] in out and ids[11] in out


def test_status_of_running_and_cancelling_jobs(home, frozen, capsys):
    running = submit()
    late = submit()
    overdue = submit()
    cancelling = submit()
    store = Store()
    now = NOW
    with store.locked():
        store.update(running.id, state="running", started_at=now - 20)
        store.update(late.id, state="running", started_at=now - 90)
        store.update(overdue.id, state="running", started_at=now - 600)
        store.update(cancelling.id, state="running", started_at=now, cancel_requested=True)
    capsys.readouterr()
    main(["status"])
    out = capsys.readouterr().out
    assert (
        f"{running.id}: running for 20s, expected to finish in about 40s, "
        "killed in 4m40s if still running"
    ) in out
    assert f"{late.id}: running for 1m30s, past its 1m estimate, killed in 3m30s" in out
    assert f"{overdue.id}: running for 10m, past its max runtime" in out
    assert f"{cancelling.id}: running, stopping because you cancelled it" in out


def test_cancel_sets_end_time(home):
    job = submit()
    main(["cancel", job.id])
    assert Store().get(job.id).ended_at is not None


def test_cancel_of_starting_job_twice_is_fine(home, capsys):
    job = submit()
    store = Store()
    with store.locked():
        store.update(job.id, state="starting")
    assert main(["cancel", job.id]) == 0
    assert main(["cancel", job.id]) == 0
    assert Store().get(job.id).cancel_requested


def test_text_after_double_dash_is_refused_outside_submit(home, capsys):
    job = submit()
    with pytest.raises(SystemExit):
        main(["cancel", job.id, "--", "oops"])
    assert Store().get(job.id).state == "queued"


def test_detail_lines(home, capsys):
    parent = submit()
    child = submit("--after", parent.id)
    store = Store()
    with store.locked():
        store.update(child.id, state="failed", exit_code=3, peak_ram=1024**3)
    capsys.readouterr()
    main(["status", child.id])
    out = capsys.readouterr().out
    assert f"waits for: {parent.id}" in out
    assert "exit code: 3" in out
    assert "peak RAM: 1.0G" in out


def test_status_shows_places_of_queued_jobs(home, frozen, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "s2")
    theirs = submit("--cpus", "4", "--name", "theirs", "--bid", "77")
    monkeypatch.setenv("PROQU_SESSION", "s1")
    first = submit("--cpus", "4")
    second = submit("--cpus", "4")
    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "position 1 in the queue" not in out
    assert "position 2 in the queue, expected to start in 1m and finish in 2m" in out
    assert "position 3 in the queue, expected to start in 2m and finish in 3m" in out
    assert main(["status", second.id]) == 0
    detail = capsys.readouterr().out
    assert "state: queued for 0s, position 3 in the queue, expected to start in 2m" in detail
    for text in (out, detail):
        # ids are hex, so a bare 77 can only be the other session's bid
        assert theirs.id not in text and "theirs" not in text
        assert not re.search(r"\b77\b", text)
    assert first.id in out


def test_status_of_a_job_with_no_projection(home, frozen, capsys):
    job = submit()
    store = Store()
    with store.locked():
        store.update(job.id, after=["ffffff"])
    capsys.readouterr()
    main(["status"])
    out = capsys.readouterr().out
    assert (
        f"{job.id}: queued for 0s, position 1 in the queue, no start time can be projected" in out
    )


def test_starting_job_shows_no_expected_finish(home, capsys):
    job = submit()
    store = Store()
    with store.locked():
        store.update(job.id, state="starting")
    capsys.readouterr()
    main(["status"])
    assert f"{job.id}: starting\n" in capsys.readouterr().out
