from dataclasses import replace

import pytest

from proqu.scheduler import (
    Fail,
    Free,
    Gpu,
    Limits,
    Plan,
    Projection,
    Queued,
    Request,
    Running,
    Share,
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
    # x is long past its hard end, yet b fits now and is refused only to protect a.
    running = [r("x", 4, start=NOW - 500, mx=100)]
    assert started(run(L8, running, [q("a", 8, t=1), q("b", 1, t=2, mx=10)])) == []
    assert started(run(L8, running, [q("b", 1, t=2, mx=10)])) == ["b"]


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
    assert p == Plan([], [Fail("x", f"job d, which it waits for, ended {state}")])


def test_an_unknown_dependency_fails_the_dependant():
    assert run(L8, [], [q("x", after=["zz"])]).fail == [
        Fail("x", "job zz, which it waits for, does not exist")
    ]


def test_failure_spreads_to_dependants_in_the_same_pass():
    queue = [q("w", 9, t=0), q("y", t=1, after=["x"]), q("x", t=2, after=["w"])]
    assert run(L8, [], queue).fail == [
        Fail(
            "w", "needs 9.0G RAM but the queue limits were lowered after it was submitted to 8.0G"
        ),
        Fail("y", "job x, which it waits for, can never start"),
        Fail("x", "job w, which it waits for, can never start"),
    ]


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        (
            q("x", 9),
            "needs 9.0G RAM but the queue limits were lowered after it was submitted to 8.0G",
        ),
        (
            q("x", cpus=9),
            "needs 9 CPUs but the queue limits were lowered after it was submitted to 8",
        ),
        (
            q("x", gpus=3, vram=1),
            "needs 3 GPUs but the queue limits were lowered after it was submitted to 2",
        ),
        (
            q("x", gpus=2, vram=16),
            "needs 2 GPUs with 16.0G each but the queue limits were lowered after it was "
            "submitted, only 1 allow that much",
        ),
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


# Bids and aging through plan and project

OLD_AND_BIDDING = [q("old", 8, t=NOW - 100 * 60), q("rich", 8, t=NOW, bid=500)]


def test_a_later_bid_starts_first():
    queue = [q("early", 8, t=1), q("rich", 8, t=2, bid=100)]
    assert started(run(L8, [], queue)) == ["rich"]


def test_a_later_bid_is_projected_first():
    queue = [q("early", 8, t=1, est=60), q("rich", 8, t=2, bid=100, est=30)]
    assert proj(L8, [], queue) == {"rich": (1, 0, 30), "early": (2, 30, 90)}


def test_plan_uses_the_aging_rate_it_is_given():
    # 100 minutes of waiting is worth 1000 at rate 10 and nothing at rate 0.
    assert started(plan(L8, [], OLD_AND_BIDDING, {}, NOW, 10)) == ["old"]
    assert started(plan(L8, [], OLD_AND_BIDDING, {}, NOW, 0)) == ["rich"]


def test_project_uses_the_aging_rate_it_is_given():
    aged = project(L8, [], OLD_AND_BIDDING, {}, NOW, 10)
    flat = project(L8, [], OLD_AND_BIDDING, {}, NOW, 0)
    assert (aged["old"].position, aged["rich"].position) == (1, 2)
    assert (flat["old"].position, flat["rich"].position) == (2, 1)
    assert aged["old"].start == NOW
    assert flat["rich"].start == NOW


def test_order_uses_the_aging_rate_it_is_given():
    assert ids(order(OLD_AND_BIDDING, 0)) == ["rich", "old"]
    assert ids(order(OLD_AND_BIDDING, 10)) == ["old", "rich"]
    assert ids(order(OLD_AND_BIDDING, 20)) == ["old", "rich"]


# Reservation details


@pytest.mark.parametrize("reverse", [False, True])
def test_the_reservation_waits_until_enough_jobs_have_ended(reverse):
    # The first release leaves 6G, which is not enough, so the reservation is at 200.
    running = [r("r1", 2, mx=100), r("r2", 2, mx=200, cpu_ids=(1,))]
    if reverse:
        running.reverse()
    queue = [q("big", 8, t=1), q("s", 4, t=2, mx=150)]
    assert started(run(L8, running, queue)) == ["s"]


def test_a_job_started_in_this_pass_holds_resources_until_its_hard_end():
    queue = [q("a", 2, t=1, mx=100), q("big", 8, t=2), q("long", 2, t=3, mx=1000)]
    assert started(run(L8, [], queue)) == ["a"]


def test_a_job_started_in_this_pass_is_released_at_its_maximum_not_its_estimate():
    queue = [q("a", 2, t=1, est=10, mx=100), q("big", 8, t=2), q("s", 2, t=3, mx=50)]
    assert started(run(L8, [], queue)) == ["a", "s"]


def test_only_the_first_blocked_job_reserves():
    # big1 reserves at 300. big2 alone could start at 100, but does not hold a reservation.
    running = [r("r1", 2, mx=100), r("r2", 2, mx=300, cpu_ids=(1,))]
    queue = [q("big1", 8, t=1), q("big2", 5, t=2), q("s", 2, t=3, mx=200)]
    assert started(run(L8, running, queue)) == ["s"]


def test_a_refused_backfill_does_not_use_up_the_shadow():
    # 6G is reserved at 100, leaving 2G. The 3G job is refused, so the 2G job still fits.
    queue = [q("big", 6, t=1), q("s3", 3, t=2, mx=1000), q("s2", 2, t=3, mx=1000)]
    assert started(run(L8, [r("r4", 4, mx=100)], queue)) == ["s2"]


def test_a_reservation_blocked_only_by_cpu_count_is_kept():
    running = [r("r6", cpu_ids=(0, 1, 2, 3, 4, 5), mx=100)]
    big = q("big", cpus=7, t=1)
    assert started(run(L8, running, [big, q("s", cpus=2, t=2, mx=1000)])) == []
    assert started(run(L8, running, [big, q("s", cpus=2, t=2, mx=100)])) == ["s"]


# Running jobs holding ids outside lowered limits

TWO_CPUS = Limits(8 * G, (0, 1), ())


def test_a_running_gpu_outside_the_limits_does_not_stop_starts():
    running = [r("g", gpu_ids=(5,), vram=1)]
    assert started(run(GPUS, running, [q("a")])) == ["a"]


def test_a_running_gpu_outside_the_limits_does_not_break_a_reservation():
    running = [r("g", 63, gpu_ids=(5,), vram=1, mx=100)]
    big = q("big", 64, t=1)
    assert started(run(GPUS, running, [big, q("s", 1, t=2, mx=101)])) == []
    assert started(run(GPUS, running, [big, q("s", 1, t=2, mx=100)])) == ["s"]


def test_a_running_cpu_outside_the_limits_is_not_given_back_as_free():
    # The running job holds cpu 0 and cpu 5. Only cpu 1 is free, now and at its end.
    running = [r("x", cpu_ids=(0, 5), mx=100)]
    big = q("big", cpus=2, t=1)
    assert started(run(TWO_CPUS, running, [big, q("s", t=2, mx=1000)])) == []
    assert started(run(TWO_CPUS, running, [big, q("s", t=2, mx=100)])) == ["s"]


# Projection details


def test_projection_starts_at_the_earliest_expected_end():
    running = [r("r1", 4, est=50, mx=120), r("r2", 4, est=100, mx=120, cpu_ids=(1,))]
    assert proj(L8, running, [q("a", 4, est=10)]) == {"a": (1, 50, 60)}


def test_a_running_estimate_above_the_hard_end_is_capped_at_the_hard_end():
    running = [r("o", 8, est=500, mx=100)]
    assert proj(L8, running, [q("a", 8, est=30)]) == {"a": (1, 100, 130)}


def test_projection_waits_for_gpu_vram():
    running = [r("g", gpu_ids=(0,), vram=20, est=50, mx=100)]
    queue = [q("g1", gpus=1, vram=10, t=1, est=100), q("g2", gpus=1, vram=10, t=2, est=5)]
    assert proj(GPUS, running, queue) == {"g1": (1, 0, 100), "g2": (2, 50, 55)}


def test_a_projected_gpu_job_holds_its_gpu():
    # The running job ends at 20, a replay step during which a is still running.
    running = [r("r1", est=20)]
    queue = [q("a", gpus=1, vram=24, t=1, est=100), q("b", gpus=1, vram=24, t=2, est=10)]
    assert proj(GPUS, running, queue) == {"a": (1, 0, 100), "b": (2, 100, 110)}


def test_a_projected_job_holds_its_cpus():
    # When the running job ends at 20 only cpu 0 is free, as a holds the rest.
    running = [r("r1", est=20)]
    queue = [q("a", cpus=7, t=1, est=100), q("b", cpus=2, t=2, est=10)]
    assert proj(L8, running, queue) == {"a": (1, 0, 100), "b": (2, 100, 110)}


def long_hard_end_scenario(backfill_max):
    # a starts at once and s could start when r1 ends at 20. big is blocked until a's hard
    # end at 100.
    running = [r("r1", 4, est=20, mx=20)]
    queue = [
        q("a", 4, t=1, est=30, mx=100),
        q("big", 8, t=2, est=10, mx=10),
        q("s", 4, t=3, est=10, mx=backfill_max),
    ]
    return proj(L8, running, queue)


def test_a_projected_job_reserves_until_its_hard_end_not_its_estimate():
    # s ends at 70, before big could start at a's hard end of 100.
    assert long_hard_end_scenario(50) == {"a": (1, 0, 30), "big": (2, 30, 40), "s": (3, 20, 30)}


def test_projection_checks_backfill_against_the_time_of_the_replay_step():
    # At 20, s would end at 110 which is past 100, though 90 from now is not.
    assert long_hard_end_scenario(90) == {"a": (1, 0, 30), "big": (2, 30, 40), "s": (3, 40, 50)}


# Smaller cases


@pytest.mark.parametrize("pending_first", [False, True])
def test_a_job_with_two_dependencies_waits_for_both(pending_first):
    after = ["pend", "done"] if pending_first else ["done", "pend"]
    queue = [q("pend", t=1), q("x", t=2, after=after)]
    p = run(L8, [], queue, {"done": "succeeded"})
    assert started(p) == ["pend"]
    assert p.fail == []
    both = {"done": "succeeded", "pend": "succeeded"}
    assert started(run(L8, [], [q("x", after=after)], both)) == ["x"]


def test_a_request_for_every_cpu_is_within_the_limits():
    p = run(L8, [], [q("x", cpus=8)])
    assert started(p) == ["x"]
    assert p.fail == []


def test_equal_scores_break_by_submit_time_before_id():
    assert ids(order([q("a", t=2), q("b", t=1)], 0)) == ["b", "a"]
    assert ids(order([q("a", t=120, bid=10), q("b", t=60)], 10)) == ["b", "a"]


def test_projection_positions_count_a_job_that_will_fail():
    queue = [q("bad", t=1, after=["zz"]), q("a", 2, t=2), q("b", 2, t=3)]
    assert proj(L8, [], queue) == {"a": (2, 0, 60), "b": (3, 0, 60)}


# Length delay

H = 3600
CAP = 4 * H


def test_long_job_ranks_after_a_later_short_one_of_equal_bid():
    jobs = [q("long", t=0, mx=10 * H), q("short", t=60, mx=600)]
    assert ids(order(jobs, 10, CAP)) == ["short", "long"]
    assert ids(order(jobs, 10)) == ["long", "short"]


def test_length_delay_is_a_tenth_of_max_so_close_lengths_keep_fifo():
    jobs = [q("a", t=0, mx=1000), q("b", t=20, mx=900)]
    assert ids(order(jobs, 10, CAP)) == ["a", "b"]


def test_length_delay_stops_at_the_cap():
    # both are past the cap, so only the submit time separates them
    jobs = [q("first", t=0, mx=200 * H), q("second", t=60, mx=100 * H)]
    assert ids(order(jobs, 10, CAP)) == ["first", "second"]
    # a cap of 3h delays both by 3h, so the submit time decides, with no cap in reach
    # the shorter job is delayed less and goes first
    below = [q("first", t=0, mx=100 * H), q("second", t=60, mx=30 * H)]
    assert ids(order(below, 10, 3 * H)) == ["first", "second"]
    assert ids(order(below, 10, 100 * H)) == ["second", "first"]


def test_a_cap_of_zero_turns_the_delay_off():
    jobs = [q("long", t=0, mx=10 * H), q("short", t=60, mx=600)]
    assert ids(order(jobs, 10, 0)) == ["long", "short"]


def test_a_bid_can_outweigh_the_length_delay():
    jobs = [q("long", t=0, mx=10 * H, bid=5000), q("short", t=60, mx=600)]
    assert ids(order(jobs, 10, CAP)) == ["long", "short"]


def test_plan_uses_the_length_delay():
    jobs = [q("long", ram=8, t=NOW - 600, mx=10 * H), q("short", ram=8, t=NOW - 60, mx=600)]
    assert started(plan(L8, [], jobs, {}, NOW, 10, CAP)) == ["short"]
    assert started(plan(L8, [], jobs, {}, NOW, 10, 0)) == ["long"]


def test_project_uses_the_length_delay():
    jobs = [q("long", ram=8, t=NOW - 600, mx=10 * H), q("short", ram=8, t=NOW - 60, mx=600)]
    delayed = project(L8, [], jobs, {}, NOW, 10, CAP)
    plain = project(L8, [], jobs, {}, NOW, 10, 0)
    assert (delayed["short"].position, delayed["long"].position) == (1, 2)
    assert delayed["short"].start == NOW
    assert (plain["long"].position, plain["short"].position) == (1, 2)
    assert plain["long"].start == NOW


def test_project_keeps_the_length_delay_when_replaying_later_steps():
    busy = [r("busy", ram=8, est=100, mx=200, cpu_ids=(0,))]
    jobs = [
        q("long", ram=8, t=NOW - 600, mx=10 * H, est=500),
        q("short", ram=8, t=NOW - 60, mx=600),
    ]
    out = project(L8, busy, jobs, {}, NOW, 10, CAP)
    assert out["short"].start == NOW + 100
    assert out["long"].start == NOW + 100 + 60


# Free memory


def free_run(limits, running, queued, free):
    return plan(limits, running, queued, {}, NOW, 10, free=free)


BIG = Limits(62 * G, tuple(range(8)), ())


def test_shared_machine_holds_a_job_other_programs_leave_no_room_for():
    # 62G machine, but other users leave only 8G free
    p = free_run(BIG, [], [q("big", ram=40)], Free(8 * G))
    assert started(p) == []
    assert p.held == {"big": "ram"}
    assert p.fail == []


def test_job_starts_once_the_memory_is_free():
    p = free_run(BIG, [], [q("big", ram=40)], Free(50 * G))
    assert started(p) == ["big"]
    assert p.held == {}


def test_without_free_memory_the_limits_alone_decide():
    assert started(run(BIG, [], [q("big", ram=40)])) == ["big"]


def test_jobs_started_in_one_pass_use_up_the_free_memory():
    p = free_run(L8, [], [q("a", ram=3, t=1), q("b", ram=3, t=2)], Free(5 * G))
    assert started(p) == ["a"]
    assert p.held == {"b": "ram"}


def test_smaller_job_may_pass_a_held_one_when_it_leaves_room_for_it():
    # big fits the limits but not the free memory, small fits both and leaves room
    p = free_run(L8, [], [q("big", ram=6, t=1), q("small", ram=1, t=2)], Free(3 * G))
    assert started(p) == ["small"]
    assert p.held == {"big": "ram"}


def test_held_job_keeps_its_room_in_the_limits():
    # small fits the free memory, but would leave the held job no room in the limits
    queued = [q("big", ram=6, t=1), q("small", ram=3, t=2)]
    p = free_run(L8, [], queued, Free(4 * G))
    assert started(p) == []
    assert p.held == {"big": "ram"}


def test_free_memory_never_starts_what_the_limits_refuse():
    p = free_run(L8, [r("run", ram=6)], [q("a", ram=4)], Free(60 * G))
    assert started(p) == []
    assert p.held == {}


def test_free_gpu_memory_chooses_the_gpu():
    # best fit by the limits is GPU 1, but only GPU 0 has the memory free
    job = q("g", gpus=1, vram=10)
    assert started(run(GPUS, [], [job])) == ["g"] and run(GPUS, [], [job]).start[0].gpu_ids == (1,)
    p = free_run(GPUS, [], [job], Free(60 * G, {0: 20 * G, 1: 4 * G}))
    assert [s.gpu_ids for s in p.start] == [(0,)]


def test_gpu_job_held_when_no_gpu_has_the_memory_free():
    p = free_run(GPUS, [], [q("g", gpus=1, vram=10)], Free(60 * G, {0: 4 * G, 1: 4 * G}))
    assert started(p) == []
    assert p.held == {"g": "gpu"}


def test_gpu_job_held_when_gpu_memory_cannot_be_read():
    queued = [q("g", gpus=1, vram=10, t=1), q("cpu", t=2)]
    p = free_run(GPUS, [], queued, Free(60 * G, None))
    # the job without a GPU is not held back by GPU memory
    assert started(p) == ["cpu"]
    assert p.held == {"g": "gpu-unread"}


def test_gpu_jobs_started_in_one_pass_use_up_free_gpu_memory():
    queued = [q("a", gpus=1, vram=10, t=1), q("b", gpus=1, vram=10, t=2)]
    p = free_run(GPUS, [], queued, Free(60 * G, {0: 15 * G, 1: 0}))
    assert started(p) == ["a"]
    assert p.held == {"b": "gpu"}


def test_projection_ignores_free_memory():
    # projections stay on the limits, the free memory only ever delays a start
    out = project(BIG, [], [q("big", ram=40)], {}, NOW, 10)
    assert out["big"].start == NOW


def test_gpu_chosen_by_free_memory_must_still_leave_room_for_the_reserved_job():
    two = Limits(64 * G, tuple(range(8)), (Gpu(0, "u0", 24 * G), Gpu(1, "u1", 14 * G)))
    # first needs both GPUs, so it is reserved for when the running job frees GPU 0
    running = [r("run", gpu_ids=(0,), vram=14, mx=100)]
    queued = [q("first", gpus=2, vram=14, t=1), q("long", gpus=1, vram=4, t=2, mx=10_000)]
    # by the limits alone, long goes on GPU 0 and leaves room for first
    assert started(run(two, running, queued)) == ["long"]
    # only GPU 1 has memory free, and long there would leave first no room
    p = free_run(two, running, queued, Free(60 * G, {0: 0, 1: 14 * G}))
    assert started(p) == []
    assert p.held == {}


# The share long jobs may hold

DAY = 86_400
SHARE = Share(12 * 3600, 75)


def share_run(limits, running, queued, share=SHARE, now=NOW):
    return plan(limits, running, queued, {}, now, 10, share=share)


def test_long_jobs_together_hold_at_most_the_share():
    # 8G and 8 cores, so long jobs may hold 6G and 6 cores
    running = [r("old", ram=4, mx=DAY, cpu_ids=(0,))]
    assert started(share_run(L8, running, [q("long", ram=3, mx=DAY)])) == []
    assert started(share_run(L8, running, [q("long", ram=2, mx=DAY)])) == ["long"]


def test_short_jobs_ignore_the_share():
    running = [r("old", ram=4, mx=DAY, cpu_ids=(0,))]
    assert started(share_run(L8, running, [q("short", ram=3, mx=3600)])) == ["short"]


def test_cores_count_against_the_share():
    running = [r("old", mx=DAY, cpu_ids=(0, 1, 2, 3, 4))]
    assert started(share_run(L8, running, [q("long", mx=DAY, cpus=2)])) == []
    assert started(share_run(L8, running, [q("long", mx=DAY, cpus=1)])) == ["long"]


def test_gpu_memory_counts_against_the_share_on_each_gpu():
    # GPU 0 has 24G, so long jobs may hold 18G of it
    running = [r("old", mx=DAY, gpu_ids=(0,), vram=12)]
    one = Limits(64 * G, tuple(range(8)), (Gpu(0, "u0", 24 * G),))
    assert started(share_run(one, running, [q("long", mx=DAY, gpus=1, vram=8)])) == []
    assert started(share_run(one, running, [q("long", mx=DAY, gpus=1, vram=6)])) == ["long"]


def test_a_long_job_larger_than_the_share_runs_alone():
    assert started(share_run(L8, [], [q("huge", ram=8, mx=DAY, cpus=8)])) == ["huge"]
    running = [r("old", ram=1, mx=DAY)]
    assert started(share_run(L8, running, [q("huge", ram=7, mx=DAY, cpus=7)])) == []


def test_a_long_job_larger_than_the_share_keeps_other_long_jobs_out():
    running = [r("huge", ram=7, mx=DAY, cpu_ids=(0,))]
    assert started(share_run(L8, running, [q("long", ram=1, mx=DAY)])) == []
    assert started(share_run(L8, running, [q("short", ram=1, mx=60)])) == ["short"]


def test_long_job_blocked_by_the_share_is_reserved_for_when_a_long_job_ends():
    # long jobs may hold 6 of the 8 cores, old holds 1, so long must wait for old to end
    running = [r("old", mx=DAY, cpu_ids=(0,))]
    queued = [q("long", cpus=6, mx=DAY, t=1), q("filler", cpus=2, mx=2 * DAY, t=2)]
    # filler fits the share now, but would take the share long needs once old ends
    assert started(share_run(L8, running, queued)) == []
    # one that ends before old does may still go ahead
    queued[1] = q("filler", cpus=2, mx=DAY - 1, t=2)
    assert started(share_run(L8, running, queued)) == ["filler"]


def test_projection_accounts_for_the_share():
    running = [r("old", ram=4, est=3600, mx=DAY, cpu_ids=(0,))]
    out = project(L8, running, [q("long", ram=3, mx=DAY)], {}, NOW, 10, share=SHARE)
    assert out["long"].start == NOW + 3600
    assert project(L8, running, [q("long", ram=3, mx=DAY)], {}, NOW, 10)["long"].start == NOW


def test_full_share_changes_nothing():
    running = [r("old", ram=4, mx=DAY, cpu_ids=(0,))]
    queued = [q("long", ram=4, mx=DAY)]
    assert started(share_run(L8, running, queued, share=Share(12 * 3600, 100))) == ["long"]


def test_a_job_at_the_threshold_is_not_long():
    running = [r("old", ram=4, mx=DAY, cpu_ids=(0,))]
    queued = [q("edge", ram=3, mx=12 * 3600)]
    assert started(share_run(L8, running, queued)) == ["edge"]


def test_cores_outside_lowered_limits_do_not_count_against_the_share():
    # old started on cores 4 to 7 before the limits were lowered to cores 0 to 3
    four = Limits(8 * G, (0, 1, 2, 3), ())
    running = [r("old", mx=DAY, cpu_ids=(4, 5, 6, 7))]
    queued = [q("long", cpus=2, mx=DAY)]
    assert started(share_run(four, running, queued, share=Share(12 * 3600, 100))) == ["long"]
    assert started(share_run(four, running, queued)) == ["long"]


def test_a_long_job_over_the_gpu_share_runs_alone():
    one = Limits(64 * G, tuple(range(8)), (Gpu(0, "u0", 24 * G),))
    # 20G of a 24G GPU is over the 18G share, so it runs only with no other long job
    assert started(share_run(one, [], [q("big", mx=DAY, gpus=1, vram=20)])) == ["big"]
    running = [r("big", mx=DAY, gpu_ids=(0,), vram=20)]
    # and while it runs, no other long job starts, even one without a GPU
    assert started(share_run(one, running, [q("long", mx=DAY)])) == []
    assert started(share_run(one, running, [q("short", mx=60)])) == ["short"]


# A job held too long


def hold_run(queued, free, hold_limit=1800):
    return plan(L8, [], queued, {}, NOW, 10, free=free, hold_limit=hold_limit)


def held_q(id, ram, t, since):
    return replace(q(id, ram=ram, t=t), held_since=since)


def test_later_job_may_pass_a_held_one_before_the_hold_limit():
    queued = [held_q("big", 6, 1, NOW - 1799), q("small", ram=1, t=2)]
    p = hold_run(queued, Free(3 * G))
    assert started(p) == ["small"]
    assert p.held == {"big": "ram"}


def test_no_later_job_may_pass_a_job_held_past_the_hold_limit():
    queued = [held_q("big", 6, 1, NOW - 1801), q("small", ram=1, t=2), q("tiny", ram=1, t=3)]
    p = hold_run(queued, Free(3 * G))
    assert started(p) == []
    assert p.held == {"big": "overdue-ram", "small": "behind", "tiny": "behind"}


def test_jobs_ahead_of_an_overdue_job_still_start():
    queued = [q("first", ram=1, t=0), held_q("big", 6, 1, NOW - 7200), q("small", ram=1, t=2)]
    p = hold_run(queued, Free(3 * G))
    assert started(p) == ["first"]
    assert p.held == {"big": "overdue-ram", "small": "behind"}


def test_a_hold_limit_of_zero_never_stops_later_jobs():
    queued = [held_q("big", 6, 1, NOW - 86_400), q("small", ram=1, t=2)]
    assert started(hold_run(queued, Free(3 * G), hold_limit=0)) == ["small"]


def test_overdue_job_starts_once_the_memory_is_free():
    queued = [held_q("big", 6, 1, NOW - 7200), q("small", ram=1, t=2)]
    p = hold_run(queued, Free(8 * G))
    assert started(p) == ["big", "small"]
    assert p.held == {}


def test_hold_time_matters_only_while_the_job_is_held():
    # without free memory nothing is held, so an old hold time stops nothing
    queued = [held_q("big", 6, 1, NOW - 7200), q("small", ram=1, t=2)]
    assert started(plan(L8, [], queued, {}, NOW, 10, hold_limit=1800)) == ["big", "small"]
