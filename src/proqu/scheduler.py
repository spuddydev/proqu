"""Pure scheduling and projection. No I/O, no clock reads, deterministic.

Contract with the caller:
- running includes starting jobs, since they hold resources.
- dep_states maps ids of dependencies that are neither running nor queued to their
  terminal state. A missing id counts as unknown.
- gpu_ids are Gpu.index values. Limits are already lowered by config.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass

from proqu.units import format_size


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    vram: int


@dataclass(frozen=True)
class Limits:
    ram: int
    cpu_ids: tuple[int, ...]
    gpus: tuple[Gpu, ...]


@dataclass(frozen=True)
class Request:
    ram: int
    cpus: int
    gpus: int = 0
    vram: int = 0


@dataclass(frozen=True)
class Running:
    id: str
    req: Request
    cpu_ids: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    started_at: float
    est: float
    max: float


@dataclass(frozen=True)
class Queued:
    id: str
    req: Request
    bid: int
    submitted_at: float
    est: float
    max: float
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Start:
    id: str
    cpu_ids: tuple[int, ...]
    gpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class Fail:
    id: str
    reason: str


@dataclass(frozen=True)
class Plan:
    start: list[Start]
    fail: list[Fail]


@dataclass(frozen=True)
class Projection:
    position: int
    start: float
    end: float


def order(queued: Iterable[Queued], aging_per_minute: float) -> list[Queued]:
    """Highest score first, then earliest submitted, then id.

    Score is bid - aging_per_minute * submitted_at / 60. It is scaled by 60 here so that
    equal scores stay exactly equal for whole number inputs.
    """
    return sorted(
        queued,
        key=lambda q: (aging_per_minute * q.submitted_at - q.bid * 60, q.submitted_at, q.id),
    )


_Held = tuple[Request, tuple[int, ...], tuple[int, ...]]


class _Pool:
    """Free resources: ram, cpu ids and vram per GPU index."""

    def __init__(self, limits: Limits, held: Iterable[_Held] = ()):
        self.allowed = frozenset(limits.cpu_ids)
        self.ram = limits.ram
        self.cpus = set(limits.cpu_ids)
        self.vram = {g.index: g.vram for g in limits.gpus}
        for h in held:
            self.take(*h)

    def copy(self) -> _Pool:
        new = copy.copy(self)
        new.cpus, new.vram = set(self.cpus), dict(self.vram)
        return new

    def take(self, req: Request, cpu_ids: tuple[int, ...], gpu_ids: tuple[int, ...]) -> None:
        # Ids outside the limits belong to jobs started before the limits were lowered.
        self.ram -= req.ram
        self.cpus -= set(cpu_ids)
        for g in gpu_ids:
            if g in self.vram:
                self.vram[g] -= req.vram

    def give(self, req: Request, cpu_ids: tuple[int, ...], gpu_ids: tuple[int, ...]) -> None:
        self.ram += req.ram
        self.cpus |= self.allowed.intersection(cpu_ids)
        for g in gpu_ids:
            if g in self.vram:
                self.vram[g] += req.vram

    def fit(self, req: Request) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """The cpu ids and GPU indices the request would get now, or None."""
        if req.ram > self.ram or req.cpus > len(self.cpus):
            return None
        # Best fit: the GPUs with the least free vram that still hold the request.
        free = sorted((v, g) for g, v in self.vram.items() if v >= req.vram)
        if len(free) < req.gpus:
            return None
        cpu_ids = tuple(sorted(self.cpus)[: req.cpus])
        gpu_ids = tuple(sorted(g for _, g in free[: req.gpus]))
        return cpu_ids, gpu_ids


def _over_limits(req: Request, limits: Limits) -> str | None:
    # only reachable when the limits were lowered after the job was submitted
    lowered = "the queue limits were lowered after it was submitted"
    if req.ram > limits.ram:
        return f"needs {format_size(req.ram)} RAM but {lowered} to {format_size(limits.ram)}"
    if req.cpus > len(limits.cpu_ids):
        return f"needs {req.cpus} CPUs but {lowered} to {len(limits.cpu_ids)}"
    if req.gpus > len(limits.gpus):
        return f"needs {req.gpus} GPUs but {lowered} to {len(limits.gpus)}"
    have = sum(g.vram >= req.vram for g in limits.gpus)
    if req.gpus > have:
        return (
            f"needs {req.gpus} GPUs with {format_size(req.vram)} each but {lowered}, "
            f"only {have} allow that much"
        )
    return None


def _doomed(
    limits: Limits, live: set[str], ordered: list[Queued], dep_states: dict[str, str]
) -> dict[str, str]:
    """Failure reasons for queued jobs that can never start, by id."""
    reasons = {}
    for q in ordered:
        for dep in q.after:
            state = dep_states.get(dep, "unknown")
            if dep not in live and state != "succeeded":
                ended = "does not exist" if state == "unknown" else f"ended {state}"
                reasons[q.id] = f"job {dep}, which it waits for, {ended}"
                break
        else:
            reason = _over_limits(q.req, limits)
            if reason:
                reasons[q.id] = reason
    # A job waiting on a job that fails in this pass fails too.
    changed = True
    while changed:
        changed = False
        for q in ordered:
            dep = next((d for d in q.after if d in reasons), None)
            if q.id not in reasons and dep:
                reasons[q.id] = f"job {dep}, which it waits for, can never start"
                changed = True
    return reasons


def _reserve(
    pool: _Pool, releases: list[tuple[float, _Held]], req: Request
) -> tuple[float, _Pool] | None:
    """The earliest time the request fits as jobs reach their hard end, and the pool then."""
    pool = pool.copy()
    releases = sorted(releases, key=lambda r: r[0])
    for i, (end, held) in enumerate(releases):
        pool.give(*held)
        # Release everything ending at the same moment before checking.
        if i + 1 < len(releases) and releases[i + 1][0] == end:
            continue
        if pool.fit(req):
            return end, pool
    return None


def plan(
    limits: Limits,
    running: Iterable[Running],
    queued: Iterable[Queued],
    dep_states: dict[str, str],
    now: float,
    aging_per_minute: float,
) -> Plan:
    """Which queued jobs start now and which fail. Fails are in score order."""
    running = list(running)
    ordered = order(queued, aging_per_minute)
    live = {r.id for r in running} | {q.id for q in ordered}
    doomed = _doomed(limits, live, ordered, dep_states)
    pool = _Pool(limits, [(r.req, r.cpu_ids, r.gpu_ids) for r in running])
    releases = [(r.started_at + r.max, (r.req, r.cpu_ids, r.gpu_ids)) for r in running]
    starts = []
    # The first blocked job: its reserved start, its request and the pool at that time.
    reserved: tuple[float, Request, _Pool] | None = None
    blocked = False
    for q in ordered:
        if q.id in doomed or any(d in live for d in q.after):
            continue
        fit = pool.fit(q.req)
        if fit is None:
            if not blocked:
                blocked = True
                # Never None for a job within the limits, as all releases restore them.
                found = _reserve(pool, releases, q.req)
                if found:
                    reserved = (found[0], q.req, found[1])
            continue
        if reserved and now + q.max > reserved[0]:
            # Runs past the reservation, so it must leave room for the reserved job.
            at, req, shadow = reserved
            shadow = shadow.copy()
            shadow.take(q.req, *fit)
            if shadow.fit(req) is None:
                continue
            reserved = (at, req, shadow)
        pool.take(q.req, *fit)
        releases.append((now + q.max, (q.req, *fit)))
        starts.append(Start(q.id, *fit))
    return Plan(starts, [Fail(q.id, doomed[q.id]) for q in ordered if q.id in doomed])


def project(
    limits: Limits,
    running: Iterable[Running],
    queued: Iterable[Queued],
    dep_states: dict[str, str],
    now: float,
    aging_per_minute: float,
) -> dict[str, Projection]:
    """Expected position, start and end of each queued job that is not about to fail.

    Replays the scheduler at each expected job end, assuming every job takes its estimate
    and every dependency succeeds. A job that can never start has no entry.
    """
    running = list(running)
    ordered = order(queued, aging_per_minute)
    position = {q.id: i for i, q in enumerate(ordered, 1)}
    live = {r.id for r in running} | set(position)
    doomed = _doomed(limits, live, ordered, dep_states)
    waiting = {q.id: q for q in ordered if q.id not in doomed}
    states = dict(dep_states)
    # A job past its estimate is expected to end now, and never after its hard end.
    ends = {r.id: max(now, min(r.started_at + r.max, r.started_at + r.est)) for r in running}
    out = {}
    t = now
    while waiting:
        for s in plan(limits, running, waiting.values(), states, t, aging_per_minute).start:
            q = waiting.pop(s.id)
            running.append(Running(q.id, q.req, s.cpu_ids, s.gpu_ids, t, q.est, q.max))
            ends[q.id] = t + q.est
            out[q.id] = Projection(position[q.id], t, t + q.est)
        if not waiting or not running:
            break
        t = min(ends[r.id] for r in running)
        for r in running:
            if ends[r.id] <= t:
                states[r.id] = "succeeded"
        running = [r for r in running if ends[r.id] > t]
    return out
