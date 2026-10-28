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


def plan(
    limits: Limits,
    running: Iterable[Running],
    queued: Iterable[Queued],
    dep_states: dict[str, str],
    now: float,
    aging_per_minute: float,
) -> Plan:
    """Which queued jobs start now and which fail."""
    running = list(running)
    ordered = order(queued, aging_per_minute)
    pool = _Pool(limits, [(r.req, r.cpu_ids, r.gpu_ids) for r in running])
    starts = []
    for q in ordered:
        fit = pool.fit(q.req)
        if fit is None:
            continue
        pool.take(q.req, *fit)
        starts.append(Start(q.id, *fit))
    return Plan(starts, [])
