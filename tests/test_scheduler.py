import pytest

from proqu.scheduler import (
    Fail,
    Gpu,
    Limits,
    Plan,
    Projection,
    Queued,
    Request,
    Running,
    Start,
    order,
    plan,
    project,
)

G = 1 << 30
NOW = 100_000.0
L8 = Limits(8 * G, tuple(range(8)), ())
GPUS = Limits(64 * G, tuple(range(8)), (Gpu(0, "u0", 24 * G), Gpu(1, "u1", 12 * G)))


def q(id, ram=1, t=0.0, bid=0, est=60, mx=120, after=(), cpus=1, gpus=0, vram=0):
    """A queued job. Sizes are in GiB, times in seconds."""
    return Queued(id, Request(ram * G, cpus, gpus, vram * G), bid, t, est, mx, tuple(after))


def r(id, ram=1, start=NOW, est=60, mx=120, cpu_ids=(0,), gpu_ids=(), vram=0):
    """A running job. Sizes are in GiB, times in seconds."""
    req = Request(ram * G, len(cpu_ids), len(gpu_ids), vram * G)
    return Running(id, req, cpu_ids, gpu_ids, start, est, mx)


def run(limits, running, queued, deps=None):
    return plan(limits, running, queued, deps or {}, NOW, 10)


def started(p):
    return [s.id for s in p.start]


def ids(jobs):
    return [j.id for j in jobs]


# Ordering


def test_no_bids_is_first_in_first_out():
    assert ids(order([q("b", t=2), q("a", t=1), q("c", t=3)], 10)) == ["a", "b", "c"]


def test_ties_break_by_submit_time_then_id():
    assert ids(order([q("b", t=1), q("a", t=1)], 10)) == ["a", "b"]
    # Equal scores: bid 10 a minute later matches bid 0 at rate 10.
    assert ids(order([q("b", t=120, bid=10), q("a", t=60)], 10)) == ["a", "b"]


def test_higher_bid_goes_first():
    assert ids(order([q("a", t=1), q("b", t=2, bid=5)], 10)) == ["b", "a"]


def test_a_job_waiting_101_minutes_outranks_a_fresh_1000_bid():
    jobs = [q("new", t=NOW, bid=1000), q("old", t=NOW - 101 * 60)]
    assert ids(order(jobs, 10)) == ["old", "new"]


def test_a_1100_bid_100_minutes_later_ranks_after_a_1000_bid():
    jobs = [q("late", t=NOW + 100 * 60, bid=1100), q("early", t=NOW, bid=1000)]
    assert ids(order(jobs, 10)) == ["early", "late"]


# Fitting


def test_user_example_starts_4g_and_3g_while_5g_waits():
    p = run(L8, [], [q("a4", 4, t=1), q("b5", 5, t=2), q("c3", 3, t=3)])
    assert started(p) == ["a4", "c3"]
    assert p.fail == []


def test_freed_space_goes_to_the_first_job_in_order_that_fits():
    # 3G is free. Position 2 needs 3G, position 5 needs 1G and position 8 needs 2G.
    sizes = [4, 3, 4, 4, 1, 4, 4, 2]
    queue = [q(f"p{i}", ram, t=i) for i, ram in enumerate(sizes, 1)]
    p = run(L8, [r("r5", 5, start=NOW - 20)], queue)
    assert started(p) == ["p2"]


def test_cpu_ids_are_the_lowest_free():
    p = run(L8, [r("c", cpu_ids=(0, 2))], [q("a", cpus=3)])
    assert p.start == [Start("a", (1, 3, 4), ())]


def test_cpu_count_limits_starts():
    p = run(L8, [r("c", cpu_ids=tuple(range(7)))], [q("a", cpus=2), q("b")])
    assert started(p) == ["b"]


def test_gpu_choice_is_best_fit():
    queue = [q(x, t=i, gpus=1, vram=10) for i, x in enumerate("abc")]
    p = run(GPUS, [], queue)
    # 12G card first, then the 24G card is shared while vram lasts.
    assert [s.gpu_ids for s in p.start] == [(1,), (0,), (0,)]


def test_gpus_are_shared_while_vram_allows():
    p = run(GPUS, [r("g", gpu_ids=(0,), vram=10)], [q("a", gpus=2, vram=12)])
    assert p.start == [Start("a", (1,), (0, 1))]


def test_gpu_job_waits_when_no_card_has_room():
    p = run(GPUS, [r("g", gpu_ids=(0,), vram=20)], [q("a", gpus=2, vram=6)])
    assert p.start == []


def test_running_use_above_lowered_limits_blocks_starts():
    p = run(Limits(4 * G, (0, 1), ()), [r("x", 6, cpu_ids=(0,))], [q("a")])
    assert p.start == []


# Reservation and backfill


def test_backfill_that_ends_by_the_reservation_starts():
    running = [r("r2", 2, mx=100)]
    assert started(run(L8, running, [q("big", 8, t=1), q("s", 2, t=2, mx=100)])) == ["s"]


def test_backfill_that_runs_past_the_reservation_waits():
    running = [r("r2", 2, mx=100)]
    assert started(run(L8, running, [q("big", 8, t=1), q("s", 2, t=2, mx=101)])) == []


def test_backfill_that_fits_in_the_shadow_starts_until_it_is_used_up():
    # 6G is reserved at the hard end of the 4G job, leaving a 2G shadow.
    queue = [q("big", 6, t=1), q("s", 2, t=2, mx=1000), q("t", 2, t=3, mx=1000)]
    assert started(run(L8, [r("r4", 4, mx=100)], queue)) == ["s"]


def test_shadow_counts_every_job_ending_at_the_reserved_time():
    limits = Limits(100, (0, 1, 2, 3), ())
    running = [
        Running("r1", Request(30, 1), (0,), (), 0, 10, 100),
        Running("r2", Request(30, 1), (1,), (), 0, 10, 100),
    ]
    queue = [
        Queued("big", Request(70, 3), 0, 1, 10, 50),
        Queued("bf", Request(20, 1), 0, 2, 10, 500),
    ]
    assert started(plan(limits, running, queue, {}, 50, 10)) == ["bf"]


def test_shadow_is_per_gpu():
    running = [r("g", gpu_ids=(0,), vram=20, mx=100)]
    # Best fit puts the small job on card 0, which the reserved job needs.
    blocked = [q("big", gpus=1, vram=22, t=1), q("s", gpus=1, vram=3, t=2, mx=1000)]
    assert run(GPUS, running, blocked).start == []
    fits = [q("big", gpus=1, vram=20, t=1), q("s", gpus=1, vram=10, t=2, mx=1000)]
    assert run(GPUS, running, fits).start == [Start("s", (1,), (1,))]


def test_a_large_job_is_not_starved_by_small_ones():
    queue = [q("big", 8, t=1)] + [q(f"s{i}", 1, t=2 + i, mx=1000) for i in range(5)]
    assert started(run(L8, [r("r1", 1, mx=100)], queue)) == []
    # Once the running job ends, the large job goes first.
    assert started(run(L8, [], queue)) == ["big"]


def test_a_job_past_its_hard_end_still_holds_a_reservation():
    running = [r("x", 8, start=NOW - 500, mx=100)]
    assert started(run(L8, running, [q("a", 8, t=1), q("b", 1, t=2, mx=10)])) == []


# Dependencies and failures


@pytest.mark.parametrize("dep", ["running", "queued"])
def test_a_pending_dependency_blocks_even_when_the_job_fits(dep):
    running = [r("d")] if dep == "running" else []
    queue = [q("x", t=2, after=["d"])] + ([q("d", t=1)] if dep == "queued" else [])
    p = run(L8, running, queue)
    assert "x" not in started(p)
    assert p.fail == []


def test_a_succeeded_dependency_unblocks():
    assert started(run(L8, [], [q("x", after=["d"])], {"d": "succeeded"})) == ["x"]


@pytest.mark.parametrize("state", ["failed", "killed", "cancelled"])
def test_a_dependency_that_did_not_succeed_fails_the_dependant(state):
    p = run(L8, [], [q("x", after=["d"])], {"d": state})
    assert p == Plan([], [Fail("x", f"dependency d {state}")])


def test_an_unknown_dependency_fails_the_dependant():
    assert run(L8, [], [q("x", after=["zz"])]).fail == [Fail("x", "dependency zz unknown")]


def test_failure_spreads_to_dependants_in_the_same_pass():
    queue = [q("w", 9, t=0), q("y", t=1, after=["x"]), q("x", t=2, after=["w"])]
    assert run(L8, [], queue).fail == [
        Fail("w", f"needs ram {9 * G}, limit is {8 * G}"),
        Fail("y", "dependency x failed"),
        Fail("x", "dependency w failed"),
    ]


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        (q("x", 9), f"needs ram {9 * G}, limit is {8 * G}"),
        (q("x", cpus=9), "needs 9 cpus, limit is 8"),
        (q("x", gpus=3, vram=1), "needs 3 gpus, limit is 2"),
        (q("x", gpus=2, vram=16), f"needs 2 gpus with vram {16 * G}, only 1 have that much"),
    ],
)
def test_a_request_over_the_limits_fails_naming_the_resource(job, reason):
    assert run(GPUS if job.req.gpus else L8, [], [job]).fail == [Fail("x", reason)]


# Projection


def proj(limits, running, queued, deps=None):
    out = project(limits, running, queued, deps or {}, NOW, 10)
    return {k: (v.position, v.start - NOW, v.end - NOW) for k, v in out.items()}


def test_projection_follows_expected_ends():
    running = [r("r8", 8, start=NOW - 10, est=60)]
    queue = [q("a", 8, t=1, est=30, mx=60), q("b", 4, t=2, est=10), q("c", 4, t=3, est=50)]
    assert proj(L8, running, queue) == {
        "a": (1, 50, 80),
        "b": (2, 80, 90),
        "c": (3, 80, 130),
    }


def test_a_running_job_past_its_estimate_is_expected_to_end_now():
    running = [r("o", 8, start=NOW - 200, est=100, mx=300)]
    assert proj(L8, running, [q("a", 8, est=30)]) == {"a": (1, 0, 30)}


def test_a_running_job_past_its_hard_end_never_moves_time_back():
    running = [r("o", 8, start=NOW - 500, est=10, mx=50)]
    assert project(L8, running, [q("a", 8, est=5)], {}, NOW, 10) == {
        "a": Projection(1, NOW, NOW + 5)
    }


def test_projection_waits_for_dependencies_and_skips_doomed_jobs():
    queue = [
        q("d", 2, t=1, est=40),
        q("x", 2, t=2, est=10, after=["d"]),
        q("bad", t=3, after=["zz"]),
    ]
    assert proj(L8, [], queue) == {"d": (1, 0, 40), "x": (2, 40, 50)}


def test_projection_applies_backfill():
    queue = [
        q("big", 8, t=1, est=50, mx=100),
        q("s", 2, t=2, est=10, mx=50),
        q("l", 2, t=3, est=10, mx=1000),
    ]
    assert proj(L8, [r("r2", 2, est=60, mx=100)], queue) == {
        "s": (2, 0, 10),
        "big": (1, 60, 110),
        "l": (3, 110, 120),
    }


def test_a_dependency_cycle_is_neither_started_nor_projected():
    queue = [q("a", after=["b"]), q("b", after=["a"])]
    assert run(L8, [], queue).start == []
    assert proj(L8, [], queue) == {}


def test_empty_queue():
    assert run(L8, [], []).start == []
    assert proj(L8, [], []) == {}
