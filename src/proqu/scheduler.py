"""Pure scheduling and projection. No I/O, no clock reads, deterministic.

Contract with the caller:
- running includes starting jobs, since they hold resources.
- dep_states maps ids of dependencies that are neither running nor queued to their
  terminal state. A missing id counts as unknown.
- gpu_ids are Gpu.index values. Limits are already lowered by config.
- free, when given, is the memory free right now for new jobs. It only ever holds a
  job back. Reservations, backfill and projections stay on the limits alone.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

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
    # set by plan for a job whose max runtime makes it long, see Share
    long: bool = False


@dataclass(frozen=True)
class Share:
    """Long jobs, with a max runtime over after seconds, together hold at most percent
    of the RAM, the cores and each GPU's memory. A long job may still run alone when
    no other long job is running, however large it is, so none is left unrunnable.
    """

    after: float
    percent: int


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
class Free:
    """Memory free now for new jobs, net of what running jobs may still grow into.

    vram maps GPU index to free bytes. None means it could not be read, and then
    no job that needs a GPU starts.
    """

    ram: int
    vram: dict[int, int] | None = None


@dataclass(frozen=True)
class Plan:
    start: list[Start]
    fail: list[Fail]
    # jobs that fit the limits but not the memory free now, by id, with what they lack
    held: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Projection:
    position: int
    start: float
    end: float


def order(queued: Iterable[Queued], aging_per_minute: float, delay_cap: float = 0) -> list[Queued]:
    """Highest score first, then earliest submitted, then id.

    Score is bid - aging_per_minute * (submitted_at + delay) / 60, where delay is a tenth
    of the job's max runtime, at most delay_cap. A long job so counts as submitted later.
    It is scaled by 600 here so that equal scores stay exactly equal for whole number inputs.
    """
    return sorted(
        queued,
        key=lambda q: (
            aging_per_minute * (q.submitted_at * 10 + min(q.max, delay_cap * 10)) - q.bid * 600,
            q.submitted_at,
            q.id,
        ),
    )


_Held = tuple[Request, tuple[int, ...], tuple[int, ...]]


class _Pool:
    """Free resources: ram, cpu ids and vram per GPU index.

    It also counts what long jobs hold, against the share they may hold together.
    """

    def __init__(self, limits: Limits, held: Iterable[_Held] = (), percent: int = 100):
        self.allowed = frozenset(limits.cpu_ids)
        self.ram = limits.ram
        self.cpus = set(limits.cpu_ids)
        self.vram = {g.index: g.vram for g in limits.gpus}
        self.share_ram = limits.ram * percent // 100
        self.share_cpus = len(limits.cpu_ids) * percent // 100
        self.share_vram = {g.index: g.vram * percent // 100 for g in limits.gpus}
        self.long_jobs = 0
        self.long_ram = 0
        self.long_cpus = 0
        self.long_vram = {g.index: 0 for g in limits.gpus}
        for h in held:
            self.take(*h)

    def copy(self) -> _Pool:
        new = copy.copy(self)
        new.cpus, new.vram, new.long_vram = set(self.cpus), dict(self.vram), dict(self.long_vram)
        return new

    def take(self, req: Request, cpu_ids: tuple[int, ...], gpu_ids: tuple[int, ...]) -> None:
        # Ids outside the limits belong to jobs started before the limits were lowered.
        self.ram -= req.ram
        self.cpus -= set(cpu_ids)
        for g in gpu_ids:
            if g in self.vram:
                self.vram[g] -= req.vram
        if req.long:
            self._count_long(req, gpu_ids, 1)

    def give(self, req: Request, cpu_ids: tuple[int, ...], gpu_ids: tuple[int, ...]) -> None:
        self.ram += req.ram
        self.cpus |= self.allowed.intersection(cpu_ids)
        for g in gpu_ids:
            if g in self.vram:
                self.vram[g] += req.vram
        if req.long:
            self._count_long(req, gpu_ids, -1)

    def _count_long(self, req: Request, gpu_ids: tuple[int, ...], sign: int) -> None:
        self.long_jobs += sign
        self.long_ram += sign * req.ram
        self.long_cpus += sign * req.cpus
        for g in gpu_ids:
            if g in self.long_vram:
                self.long_vram[g] += sign * req.vram

    def _over_share(self, req: Request) -> bool:
        """Whether a long job would take long jobs past their share of RAM or cores."""
        if not req.long or not self.long_jobs:
            return False
        return self.long_ram + req.ram > self.share_ram or (
            self.long_cpus + req.cpus > self.share_cpus
        )

    def _gpu_in_share(self, req: Request, g: int) -> bool:
        if not req.long or not self.long_jobs:
            return True
        return self.long_vram[g] + req.vram <= self.share_vram[g]

    def fit(
        self, req: Request, vram_free: dict[int, int] | None = None
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """The cpu ids and GPU indices the request would get now, or None.

        With vram_free, only GPUs that also have that much memory free are used.
        """
        if req.ram > self.ram or req.cpus > len(self.cpus) or self._over_share(req):
            return None
        # Best fit: the GPUs with the least free vram that still hold the request.
        free = sorted(
            (v, g)
            for g, v in self.vram.items()
            if v >= req.vram
            and (vram_free is None or vram_free.get(g, 0) >= req.vram)
            and self._gpu_in_share(req, g)
        )
        if len(free) < req.gpus:
            return None
        cpu_ids = tuple(sorted(self.cpus)[: req.cpus])
        gpu_ids = tuple(sorted(g for _, g in free[: req.gpus]))
        return cpu_ids, gpu_ids


def over_limits(req: Request, limits: Limits) -> str | None:
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
            reason = over_limits(q.req, limits)
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


_Reserved = tuple[float, Request, _Pool]


def _after(
    reserved: _Reserved | None, now: float, q: Queued, fit: tuple[tuple[int, ...], tuple[int, ...]]
) -> tuple[bool, _Reserved | None]:
    """Whether q may start with fit, and the reservation once it has.

    A job that runs past the reserved start must leave room for the reserved job.
    """
    if not reserved or now + q.max <= reserved[0]:
        return True, reserved
    at, req, shadow = reserved
    shadow = shadow.copy()
    shadow.take(q.req, *fit)
    if shadow.fit(req) is None:
        return False, reserved
    return True, (at, req, shadow)


def _live_fit(
    pool: _Pool, req: Request, ram: int, vram: dict[int, int] | None
) -> tuple[str | None, tuple[tuple[int, ...], tuple[int, ...]] | None]:
    """What memory a job lacks now, "ram", "gpu" or "gpu-unread", else None and where it fits."""
    if req.ram > ram:
        return "ram", None
    if not req.gpus:
        return None, pool.fit(req)
    fit = pool.fit(req, vram) if vram is not None else None
    if fit is None:
        return ("gpu" if vram is not None else "gpu-unread"), None
    return None, fit


def _tag(req: Request, max_runtime: float, share: Share | None) -> Request:
    """The request marked long when the share applies to it."""
    return replace(req, long=bool(share) and max_runtime > share.after)


def plan(
    limits: Limits,
    running: Iterable[Running],
    queued: Iterable[Queued],
    dep_states: dict[str, str],
    now: float,
    aging_per_minute: float,
    delay_cap: float = 0,
    free: Free | None = None,
    share: Share | None = None,
) -> Plan:
    """Which queued jobs start now, which fail and which wait for free memory.

    Fails are in score order. A job held for free memory counts as blocked, so when
    it is the first blocked job its place is reserved from now and later jobs may
    only start if they leave room for it.
    """
    running = [replace(r, req=_tag(r.req, r.max, share)) for r in running]
    queued = [replace(q, req=_tag(q.req, q.max, share)) for q in queued]
    ordered = order(queued, aging_per_minute, delay_cap)
    live = {r.id for r in running} | {q.id for q in ordered}
    doomed = _doomed(limits, live, ordered, dep_states)
    percent = share.percent if share else 100
    pool = _Pool(limits, [(r.req, r.cpu_ids, r.gpu_ids) for r in running], percent)
    releases = [(r.started_at + r.max, (r.req, r.cpu_ids, r.gpu_ids)) for r in running]
    starts = []
    # The first blocked job: its reserved start, its request and the pool at that time.
    reserved: _Reserved | None = None
    blocked = False
    held: dict[str, str] = {}
    free_ram = free.ram if free else 0
    free_vram = dict(free.vram) if free and free.vram is not None else None
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
        may, after = _after(reserved, now, q, fit)
        if not may:
            continue
        if free is not None:
            lacking, live_fit = _live_fit(pool, q.req, free_ram, free_vram)
            if lacking or live_fit is None:
                held[q.id] = lacking or "gpu"
                if not blocked:
                    blocked = True
                    reserved = (now, q.req, pool.copy())
                continue
            # the free memory may pick other GPUs, which must leave room as well
            fit = live_fit
            may, after = _after(reserved, now, q, fit)
            if not may:
                continue
            free_ram -= q.req.ram
            for g in fit[1]:
                if free_vram is not None:
                    free_vram[g] = free_vram.get(g, 0) - q.req.vram
        reserved = after
        pool.take(q.req, *fit)
        releases.append((now + q.max, (q.req, *fit)))
        starts.append(Start(q.id, *fit))
    fails = [Fail(q.id, doomed[q.id]) for q in ordered if q.id in doomed]
    return Plan(starts, fails, held)


def project(
    limits: Limits,
    running: Iterable[Running],
    queued: Iterable[Queued],
    dep_states: dict[str, str],
    now: float,
    aging_per_minute: float,
    delay_cap: float = 0,
    share: Share | None = None,
) -> dict[str, Projection]:
    """Expected position, start and end of each queued job that is not about to fail.

    Replays the scheduler at each expected job end, assuming every job takes its estimate
    and every dependency succeeds. A job that can never start has no entry.
    """
    running = list(running)
    ordered = order(queued, aging_per_minute, delay_cap)
    position = {q.id: i for i, q in enumerate(ordered, 1)}
    live = {r.id for r in running} | set(position)
    doomed = _doomed(limits, live, ordered, dep_states)
    waiting = {q.id: q for q in ordered if q.id not in doomed}
    states = dict(dep_states)
    # A job ends at the earlier of its estimate and hard end, or now if that has passed.
    ends = {r.id: max(now, min(r.started_at + r.max, r.started_at + r.est)) for r in running}
    out = {}
    t = now
    while waiting:
        for s in plan(
            limits, running, waiting.values(), states, t, aging_per_minute, delay_cap, share=share
        ).start:
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
