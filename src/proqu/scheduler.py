"""Pure scheduling and projection. No I/O, no clock reads, deterministic.

Contract with the caller:
- running includes starting jobs, since they hold resources.
- dep_states maps ids of dependencies that are neither running nor queued to their
  terminal state. A missing id counts as unknown.
- gpu_ids are Gpu.index values. Limits are already lowered by config.
"""

from __future__ import annotations

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
