"""What the agent is told about its jobs: queue outlook, outcomes and log tails."""

import os
from dataclasses import dataclass
from pathlib import Path

from proqu import scheduler
from proqu import tick as ticking
from proqu.store import Job, Store
from proqu.units import format_duration


@dataclass(frozen=True)
class Place:
    """Where a queued job stands: its position and, when known, its projected times."""

    position: int
    start: float | None = None
    end: float | None = None


def outlook(
    store: Store, session: str, limits: scheduler.Limits, aging: float, now: float
) -> dict[str, Place]:
    """The place of each of a session's queued jobs, from the inputs a scheduling pass uses.

    Positions count every queued job of every session, but only the session's own
    jobs are returned, so nothing else about other jobs can leak out.
    """
    active = store.active()
    running = [ticking.to_running(job, now) for job in active if job.state != "queued"]
    queued = [ticking.to_queued(job) for job in active if job.state == "queued"]
    states = ticking.dependency_states(store, active)
    projected = scheduler.project(limits, running, queued, states, now, aging)
    own = {job.id for job in active if job.session == session}
    places = {}
    for position, job in enumerate(scheduler.order(queued, aging), 1):
        if job.id not in own:
            continue
        seen = projected.get(job.id)
        places[job.id] = Place(position, seen.start, seen.end) if seen else Place(position)
    return places


def expected_end(job: Job, now: float) -> float:
    """When a running job is expected to end, by the rule the projection uses."""
    started = job.started_at or now
    return max(now, min(started + job.max, started + job.est))


def describe_place(place: Place, now: float) -> str:
    text = f"position {place.position} in the queue"
    if place.start is None or place.end is None:
        return f"{text}, no start time can be projected yet"
    return f"{text}, expected to start {_from_now(place.start, now)} and finish in " + (
        format_duration(place.end - now)
    )


def _from_now(at: float, now: float) -> str:
    # a job that fits now is started by the next pass
    return "shortly" if at - now < 1 else f"in {format_duration(at - now)}"


# how much of the end of a log is read for a tail
TAIL_BYTES = 256 * 1024


def tail_lines(path: Path, count: int) -> list[str] | None:
    """The last count lines of a log, or None when there is no log yet.

    Only the end of the file is read, so a huge log stays cheap. Progress bars
    redraw a line with carriage returns, so only the last drawing of each is kept.
    """
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - TAIL_BYTES)
            handle.seek(start)
            data = handle.read()
    except FileNotFoundError:
        return None
    lines = data.decode(errors="replace").split("\n")
    if start and len(lines) > 1:
        # the read began mid line
        lines = lines[1:]
    if lines[-1] == "":
        lines.pop()
    lines = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in lines]
    return lines[-count:]
