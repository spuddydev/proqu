"""What the agent is told about its jobs: queue outlook, outcomes and log tails."""

from dataclasses import dataclass

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
