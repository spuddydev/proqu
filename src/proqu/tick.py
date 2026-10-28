"""One scheduling pass: recover lost jobs, fail doomed ones, start what fits.

There is no resident server. A pass runs in every command that reads or
changes jobs, and whenever a supervisor finishes, so the queue moves whenever
anything happens. Recovery after a crash or restart therefore happens at the
next proqu command.
"""

import os
import signal
import time

from proqu import config, machine, paths, procs, runner, scheduler
from proqu.store import Job, Store
from proqu.units import format_size

LOST = "its supervisor was lost (machine restart or crash), resubmit it"


def tick(store: Store | None = None, cached_only: bool = False) -> None:
    """Run one pass. With cached_only the machine is never detected, and a
    pass with no cached snapshot is skipped, so the call stays quick."""
    store = store or Store()
    try:
        current = config.load(paths.config_path())
    except ValueError:
        # submit, status and limits report a broken config, the queue waits for a fix
        return
    seen = machine.cached() if cached_only else machine.snapshot()
    if seen is None:
        return
    limits = to_limits(config.effective_limits(seen, current))
    unreserved = to_limits(config.effective_limits(seen, current, reserved=False))
    settings = config.settings(current)
    # a supervisor started from inside a job would share that job's cgroup when
    # cgroups enforce limits, and die with it, so passes inside jobs leave
    # starting to the next pass outside one
    may_start = not os.environ.get("PROQU_JOB_ID")
    with store.locked():
        now = time.time()
        reap(store, now)
        active = store.active()
        plan = scheduler.plan(
            limits,
            [to_running(job, now) for job in active if job.state != "queued"],
            [to_queued(job) for job in active if job.state == "queued"],
            dependency_states(store, active),
            now,
            settings.aging_per_minute,
            settings.length_delay_cap,
        )
        for fail in plan.fail:
            job = next(j for j in active if j.id == fail.id)
            reason = reserve_reason(request(job), fail.reason, limits, unreserved)
            end_queued(store, fail.id, reason, now)
        if may_start:
            for start in plan.start:
                begin(store, start)


def reserve_reason(
    req: scheduler.Request, reason: str, limits: scheduler.Limits, unreserved: scheduler.Limits
) -> str:
    """Name the reserve when it alone makes a queued job too big, else keep the reason."""
    if scheduler.over_limits(req, limits) != reason or scheduler.over_limits(req, unreserved):
        return reason
    if req.ram > limits.ram:
        need, have, key = f"{format_size(req.ram)} RAM", format_size(limits.ram), "reserve_ram"
    else:
        need, have, key = f"{req.cpus} CPUs", str(len(limits.cpu_ids)), "reserve_cpus"
    return (
        f"needs {need} but the queue now keeps part of the machine in reserve for services "
        f"and interactive work, leaving {have}. Only the machine's owner should change that, "
        f"with `proqu config set {key}`"
    )


def to_limits(limits: machine.Machine) -> scheduler.Limits:
    gpus = tuple(scheduler.Gpu(gpu.index, gpu.uuid, gpu.vram) for gpu in limits.gpus)
    return scheduler.Limits(ram=limits.ram, cpu_ids=limits.cpu_ids, gpus=gpus)


def request(job: Job) -> scheduler.Request:
    return scheduler.Request(ram=job.ram, cpus=job.cpus, gpus=job.gpus, vram=job.vram)


def to_running(job: Job, now: float) -> scheduler.Running:
    return scheduler.Running(
        id=job.id,
        req=request(job),
        cpu_ids=tuple(job.cpu_ids),
        gpu_ids=tuple(job.gpu_ids),
        started_at=job.started_at or now,
        est=job.est,
        max=job.max,
    )


def to_queued(job: Job) -> scheduler.Queued:
    return scheduler.Queued(
        id=job.id,
        req=request(job),
        bid=job.bid,
        submitted_at=job.submitted_at,
        est=job.est,
        max=job.max,
        after=tuple(job.after),
    )


def dependency_states(store: Store, active: list[Job]) -> dict[str, str]:
    """Terminal states of every dependency that is no longer queued or running."""
    live = {job.id for job in active}
    wanted = sorted({dep for job in active for dep in job.after} - live)
    return store.states(wanted) if wanted else {}


def end_queued(store: Store, job_id: str, reason: str, now: float) -> None:
    """Fail a job that never started, giving its bid back."""
    job = store.get(job_id)
    if not job or not store.transition(job_id, ("queued",), "failed", reason=reason, ended_at=now):
        return
    if job.bid:
        store.add_bid_spent(job.session, -job.bid)


def begin(store: Store, start: scheduler.Start) -> None:
    job = store.get(start.id)
    moved = store.transition(
        start.id,
        ("queued",),
        "starting",
        cpu_ids=list(start.cpu_ids),
        gpu_ids=list(start.gpu_ids),
    )
    if not job or not moved:
        return
    try:
        pid, ticks = runner.spawn_supervisor(start.id)
    except (OSError, ValueError) as error:
        store.transition(
            start.id,
            ("starting",),
            "failed",
            reason=f"could not start its supervisor: {error}",
            ended_at=time.time(),
        )
        if job.bid:
            store.add_bid_spent(job.session, -job.bid)
        return
    store.update(start.id, supervisor_pid=pid, supervisor_start=ticks)


def reap(store: Store, now: float) -> list[str]:
    """End jobs whose supervisor is gone, returning their ids. Call inside locked().

    Leftover processes are killed before the job's end is recorded, so if this
    pass is interrupted the job stays active and the next pass tries again.
    """
    rebooted = _rebooted()
    lost = []
    for job in store.active():
        if job.state == "queued":
            continue
        if not rebooted and procs.is_alive(job.supervisor_pid, job.supervisor_start):
            continue
        if job.cancel_requested:
            state, reason = "cancelled", runner.CANCELLED
        else:
            state, reason = "failed", LOST
        if not rebooted:
            kill_leftovers([job.id])
        if store.transition(job.id, ("starting", "running"), state, reason=reason, ended_at=now):
            if job.state == "starting" and job.bid:
                store.add_bid_spent(job.session, -job.bid)
            lost.append(job.id)
    return lost


def _rebooted() -> bool:
    """Whether the machine restarted since the last pass. Call under the lock."""
    current = procs.boot_id()
    path = paths.state_dir() / "boot_id"
    try:
        previous = path.read_text().strip()
    except OSError:
        previous = ""
    if current and current != previous:
        path.write_text(current + "\n")
    return bool(previous and current and previous != current)


def kill_leftovers(job_ids: list[str]) -> None:
    """Best effort: kill processes of lost jobs, found by the id they carry."""
    for job_id in job_ids:
        for pid in procs.job_members(job_id, None):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
