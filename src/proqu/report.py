"""What the agent is told about its jobs: queue outlook, outcomes and log tails."""

import os
import signal
from dataclasses import dataclass
from pathlib import Path

from proqu import scheduler
from proqu import tick as ticking
from proqu.store import Job, Store
from proqu.units import format_duration, format_size


@dataclass(frozen=True)
class Place:
    """Where a queued job stands: its position and, when known, its projected times.

    held says what memory the job waits for when it fits the queue but other
    programs are using that memory now, as scheduler.Plan.held does. uncertain is
    set when any job is held, as projections cannot foresee other programs.
    """

    position: int
    start: float | None = None
    end: float | None = None
    held: str | None = None
    uncertain: bool = False


def outlook(
    store: Store,
    session: str,
    limits: scheduler.Limits,
    aging: float,
    delay_cap: float,
    now: float,
    held: dict[str, str] | None = None,
) -> dict[str, Place]:
    """The place of each of a session's queued jobs, from the inputs a scheduling pass uses.

    Positions count every queued job of every session, but only the session's own
    jobs are returned, so nothing else about other jobs can leak out. held is what
    the last pass held back for free memory. That any job was held says only that
    the machine is short of memory, so it is shared with every session.
    """
    held = held or {}
    active = store.active()
    running = [ticking.to_running(job, now) for job in active if job.state != "queued"]
    queued = [ticking.to_queued(job) for job in active if job.state == "queued"]
    states = ticking.dependency_states(store, active)
    projected = scheduler.project(limits, running, queued, states, now, aging, delay_cap)
    own = {job.id for job in active if job.session == session}
    places = {}
    for position, job in enumerate(scheduler.order(queued, aging, delay_cap), 1):
        if job.id not in own:
            continue
        seen = projected.get(job.id)
        start, end = (seen.start, seen.end) if seen else (None, None)
        places[job.id] = Place(position, start, end, held.get(job.id), bool(held))
    return places


def expected_end(job: Job, now: float) -> float:
    """When a running job is expected to end, by the rule the projection uses."""
    started = job.started_at or now
    return max(now, min(started + job.max, started + job.est))


WAITING = {
    "ram": "waiting for memory other programs are using",
    "gpu": "waiting for GPU memory other programs are using",
    "gpu-unread": "waiting until the free GPU memory can be read",
}


def describe_place(place: Place, now: float) -> str:
    text = f"position {place.position} in the queue"
    if place.held:
        waiting = WAITING.get(place.held, WAITING["ram"])
        return (
            f"{text}, {waiting}. It fits the queue and starts once that memory is free, "
            "so when it starts is uncertain"
        )
    if place.start is None or place.end is None:
        return f"{text}, no start time can be projected yet"
    later = ", or later while other programs use the memory" if place.uncertain else ""
    return f"{text}, expected to start {_from_now(place.start, now)}{later} and finish in " + (
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
            midline = False
            handle.seek(max(0, start - 1))
            if start:
                midline = handle.read(1) != b"\n"
            data = handle.read()
    except FileNotFoundError:
        return None
    lines = data.decode(errors="replace").split("\n")
    if lines[-1] == "":
        lines.pop()
    if midline and len(lines) > 1:
        # the first piece is the end of a line that began before the window, but a
        # lone piece is kept: it is all that can be shown of a very long last line
        lines = lines[1:]
    lines = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in lines]
    return lines[-count:]


def outcome(job: Job, log: list[str] | None) -> str:
    """What the agent reads when its job has ended: what happened and what to do next."""
    label = f" ({job.name})" if job.name else ""
    if job.started_at is None:
        when = " before it started"
    else:
        when = f" after {format_duration((job.ended_at or job.started_at) - job.started_at)}"
    reason = f": {job.reason}" if job.reason else ""
    lines = [f"Job {job.id}{label} {job.state}{when}{reason}"]
    facts = []
    if job.exit_code is not None:
        facts.append(exit_text(job.exit_code))
    if job.peak_ram is not None:
        facts.append(f"peak RAM {format_size(job.peak_ram)}")
    if facts:
        text = ", ".join(facts)
        lines.append(f"{text[:1].upper()}{text[1:]}.")
    lines.append(f"Next: {next_step(job)}")
    if log:
        lines.append(
            f"Last {len(log)} lines of the log (whole log: `proqu logs {job.id} --tail 0`):"
        )
        lines += log
    else:
        lines.append("The job wrote no output.")
    return "\n".join(lines)


def next_step(job: Job) -> str:
    if job.state == "succeeded":
        return "check its output and results before relying on them."
    if job.state == "killed":
        return (
            "the job broke its own limit, not the queue's. Find out why from the log before "
            "resubmitting, and raise its request only if it truly needs more."
        )
    if job.state == "cancelled":
        return "nothing, unless you still need its results, then submit it again."
    if job.reason == ticking.LOST:
        return "this was not the job's fault, resubmit it."
    if _dependency_failed(job):
        return (
            "fix and resubmit the job it waits for first, then resubmit this one "
            "with --after set to the new id."
        )
    return "read the log to find out why, fix the cause, then resubmit."


def _dependency_failed(job: Job) -> bool:
    reason = job.reason or ""
    return (
        job.started_at is None and reason.startswith("job ") and ", which it waits for, " in reason
    )


def exit_text(code: int) -> str:
    if code >= 0:
        return f"exit code {code}"
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = str(-code)
    return f"ended by signal {name}"
