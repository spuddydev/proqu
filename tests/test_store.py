import multiprocessing
import time

import pytest

from proqu.store import Job, LockTimeout, Store


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
