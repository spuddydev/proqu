import multiprocessing
import sqlite3
import time

import pytest

from proqu import store as store_module
from proqu.store import Job, LockTimeout, Store, StoreError


def make_job(store, session="s1", bid=0, **changes):
    values = dict(
        id=store.new_id(),
        session=session,
        name=None,
        argv=["true"],
        cwd="/",
        env={"A": "1"},
        ram=1024,
        cpus=1,
        gpus=0,
        vram=0,
        est=60,
        max=120,
        bid=bid,
        after=[],
        submitted_at=time.time(),
    )
    values.update(changes)
    return Job(**values)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    opened = Store()
    yield opened
    opened.close()


def test_insert_and_get_round_trip(store):
    job = make_job(store, name="train", after=["abc123"])
    with store.locked():
        store.insert(job)
    assert store.get(job.id) == job


def test_sessions_only_see_their_own_jobs(store):
    mine = make_job(store, session="s1")
    theirs = make_job(store, session="s2")
    with store.locked():
        store.insert(mine)
        store.insert(theirs)
    assert store.get(theirs.id, session="s1") is None
    assert [job.id for job in store.for_session("s1")] == [mine.id]
    assert {job.id for job in store.active()} == {mine.id, theirs.id}


def test_update_refuses_unknown_fields(store):
    with pytest.raises(KeyError):
        store.update("abc", colour="red")


def test_transition_only_from_expected_states(store):
    job = make_job(store)
    with store.locked():
        store.insert(job)
        assert store.transition(job.id, ("queued",), "starting", cpu_ids=[0])
        assert not store.transition(job.id, ("queued",), "cancelled")
    moved = store.get(job.id)
    assert (moved.state, moved.cpu_ids) == ("starting", [0])


def test_states_of_listed_jobs(store):
    job = make_job(store)
    with store.locked():
        store.insert(job)
    assert store.states([job.id, "missing"]) == {job.id: "queued"}


def test_bid_spent_accumulates(store):
    with store.locked():
        store.add_bid_spent("s1", 300)
        store.add_bid_spent("s1", -100)
    assert store.bid_spent("s1") == 200
    assert store.bid_spent("s2") == 0


def test_waiters(store):
    with store.locked():
        store.add_waiter("abc", 10, 99)
        store.add_waiter("abc", 11, 98)
        store.remove_waiter("abc", 10)
    assert store.waiters("abc") == [(11, 98)]


def test_failed_block_rolls_back(store):
    job = make_job(store)
    with pytest.raises(RuntimeError), store.locked():
        store.insert(job)
        raise RuntimeError
    assert store.get(job.id) is None


def _hold_lock(home, ready, release):
    import os

    os.environ["PROQU_HOME"] = home
    held = Store()
    with held.locked():
        ready.set()
        release.wait(10)


def test_lock_times_out_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    waiting = Store(lock_timeout=0.3)
    holder = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(LockTimeout, match="holding the queue lock"), waiting.locked():
            pass
    finally:
        release.set()
        holder.join()


def _submit_many(home, count):
    import os

    os.environ["PROQU_HOME"] = home
    store = Store()
    for _ in range(count):
        with store.locked():
            store.insert(make_job(store, bid=7))
            store.add_bid_spent("s1", 7)


def test_concurrent_writers_never_collide(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    Store().close()
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=_submit_many, args=(str(tmp_path), 5)) for _ in range(20)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0
    store = Store()
    jobs = store.for_session("s1")
    assert len({job.id for job in jobs}) == 100
    assert store.bid_spent("s1") == 700


def test_database_uses_wal(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_half_made_schema_is_impossible(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    broken = "CREATE TABLE jobs (id TEXT); CREATE TABLE nonsense"
    monkeypatch.setattr(store_module, "SCHEMA", broken)
    with pytest.raises(StoreError):
        Store()
    monkeypatch.undo()
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    Store().close()


def test_newer_database_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    Store().close()
    conn = sqlite3.connect(tmp_path / "proqu.db")
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(StoreError, match="update proqu"):
        Store()


def test_damaged_database_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    (tmp_path / "proqu.db").write_bytes(b"this is not a database at all" * 100)
    with pytest.raises(StoreError, match="damaged"):
        Store()


def test_opening_needs_no_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    holder = Store()
    with holder.locked():
        Store(lock_timeout=0.1).close()


def test_active_skips_finished_jobs_in_order(store):
    first = make_job(store, submitted_at=1)
    second = make_job(store, submitted_at=2)
    done = make_job(store, submitted_at=0, state="succeeded")
    with store.locked():
        for job in (second, done, first):
            store.insert(job)
        store.update(second.id, state="running")
    assert [job.id for job in store.active()] == [first.id, second.id]


def test_states_return_only_the_asked_ids(store):
    jobs = [make_job(store) for _ in range(3)]
    with store.locked():
        for job in jobs:
            store.insert(job)
        store.update(jobs[1].id, state="failed")
    assert store.states([jobs[0].id, jobs[1].id]) == {jobs[0].id: "queued", jobs[1].id: "failed"}


def test_waiters_belong_to_their_job(store):
    with store.locked():
        store.add_waiter("aaa", 10, 1)
        store.add_waiter("bbb", 10, 2)
        store.add_waiter("aaa", 10, 3)
    assert store.waiters("aaa") == [(10, 3)]
    assert store.waiters("bbb") == [(10, 2)]


def test_update_round_trips_every_kind_of_field(store):
    job = make_job(store)
    with store.locked():
        store.insert(job)
        store.update(job.id, cpu_ids=[2, 3], cancel_requested=True, notified=True, peak_ram=5)
        store.update(job.id)
    stored = store.get(job.id)
    assert stored.cpu_ids == [2, 3]
    assert stored.cancel_requested is True
    assert stored.notified is True
    assert stored.peak_ram == 5


def test_transition_and_refund_roll_back_together(store):
    job = make_job(store, bid=50)
    with store.locked():
        store.insert(job)
        store.add_bid_spent("s1", 50)
    with pytest.raises(RuntimeError), store.locked():
        assert store.transition(job.id, ("queued",), "cancelled")
        store.add_bid_spent("s1", -50)
        raise RuntimeError
    assert store.get(job.id).state == "queued"
    assert store.bid_spent("s1") == 50


def test_new_id_skips_taken_ids(store, monkeypatch):
    job = make_job(store, id="aaaaaa")
    with store.locked():
        store.insert(job)
    ids = iter(["aaaaaa", "bbbbbb"])
    monkeypatch.setattr(store_module.secrets, "token_hex", lambda n: next(ids))
    assert store.new_id() == "bbbbbb"


def test_first_release_database_is_brought_up_to_date(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path))
    store = Store()
    job = make_job(store)
    store.insert(job)
    store.close()
    conn = sqlite3.connect(tmp_path / "proqu.db")
    conn.execute("DROP TABLE held")
    conn.execute("PRAGMA user_version = 1")
    conn.close()
    store = Store()
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    assert store.get(job.id) == job
    store.mark_held(job.id, 5.0)
    assert store.held_since() == {job.id: 5.0}


def test_held_time_keeps_the_first_and_is_forgotten_once_not_queued(store):
    queued, started = make_job(store), make_job(store, state="running")
    store.insert(queued)
    store.insert(started)
    store.mark_held(queued.id, 1.0)
    store.mark_held(queued.id, 2.0)
    store.mark_held(started.id, 1.0)
    store.forget_held()
    assert store.held_since() == {queued.id: 1.0}
