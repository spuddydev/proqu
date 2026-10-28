"""The proqu command line. Every message is written for an agent to act on."""

import argparse
import contextlib
import os
import shlex
import shutil
import signal
import sqlite3
import sys
import time

from proqu import config, hooks, machine, paths, procs, report, watch
from proqu import limits as limits_module
from proqu import tick as ticking
from proqu.store import ACTIVE, TERMINAL, Job, LockTimeout, Store, StoreError
from proqu.units import format_duration, format_size, parse_duration, parse_size

VERSION = "1.0.0"
MAX_BID = 10000


class ProquError(Exception):
    """An error to show the agent. Exit code 2 means the request was invalid."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def invalid(message: str) -> ProquError:
    return ProquError(message, code=2)


def current_session() -> str:
    return os.environ.get("PROQU_SESSION") or "default"


def load_config() -> dict:
    try:
        return config.load(paths.config_path())
    except ValueError as error:
        raise ProquError(str(error)) from None


def own_job(store: Store, job_id: str) -> Job:
    job = store.get(job_id, session=current_session())
    if job is None:
        raise ProquError(f"no job {job_id} in this session, see `proqu status` for your jobs")
    return job


def queue_outlook(
    store: Store, now: float, held: dict[str, str] | None = None
) -> dict[str, report.Place]:
    """Positions and projections of queued jobs, from the limits a pass would use.

    held is what the pass just run held back for free memory.
    """
    current = load_config()
    limits = ticking.to_limits(config.effective_limits(machine.snapshot(), current))
    settings = config.settings(current)
    return report.outlook(
        store,
        current_session(),
        limits,
        settings.aging_per_minute,
        settings.length_delay_cap,
        now,
        held,
        ticking.share_of(settings),
    )


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def describe_request(ram: int, cpus: int, gpus: int, vram: int) -> str:
    parts = [f"{format_size(ram)} RAM", f"{cpus} CPU{'s' if cpus != 1 else ''}"]
    if gpus and not vram:
        parts.append(f"{gpus} GPU{'s' if gpus != 1 else ''} using the RAM")
    elif gpus:
        parts.append(f"{gpus} GPU{'s' if gpus != 1 else ''} with {format_size(vram)} each")
    return ", ".join(parts)


# submit


def cmd_submit(args, command: list[str]) -> int:
    if not command:
        raise invalid(
            "no command given: put it after --, for example: proqu submit ... -- make -j8"
        )
    ram = parse_flag("--ram", args.ram, parse_size)
    vram = parse_flag("--vram", args.vram, parse_size) if args.vram is not None else 0
    est = parse_flag("--est", args.est, parse_duration)
    max_time = parse_flag("--max", args.max, parse_duration)
    check_name(args.name)

    detected, current = machine.snapshot(), load_config()
    limits = config.effective_limits(detected, current)
    unreserved = config.effective_limits(detected, current, reserved=False)
    check_request(limits, unreserved, detected, ram, args.cpus, args.gpus, vram)
    if est > max_time:
        raise invalid(
            f"--est {format_duration(est)} is longer than --max {format_duration(max_time)}: "
            "max is the hard kill time and must be the larger, overestimated value"
        )
    if not 0 <= args.bid <= MAX_BID:
        raise invalid(f"--bid must be between 0 and {MAX_BID}")

    if args.cwd == "":
        raise invalid("--cwd is empty, give a directory")
    try:
        cwd = os.path.abspath(args.cwd if args.cwd is not None else os.getcwd())
    except FileNotFoundError:
        raise invalid("your current directory no longer exists, pass --cwd") from None
    if not os.path.exists(cwd):
        raise invalid(f"--cwd {cwd} does not exist")
    if not os.path.isdir(cwd):
        raise invalid(f"--cwd {cwd} is not a directory")
    env = dict(os.environ)
    check_executable(command[0], cwd, env.get("PATH", ""))

    session = current_session()
    store = Store()
    settings = config.settings(load_config())
    with store.locked():
        after = check_dependencies(store, session, args.after)
        remaining = max(0, settings.bid_budget - store.bid_spent(session))
        if args.bid > remaining:
            raise invalid(
                f"--bid {args.bid} is more than this session's remaining bid budget of {remaining}"
            )
        job = Job(
            id=store.new_id(),
            session=session,
            name=args.name,
            argv=command,
            cwd=cwd,
            env=env,
            ram=ram,
            cpus=args.cpus,
            gpus=args.gpus,
            vram=vram,
            est=est,
            max=max_time,
            bid=args.bid,
            after=after,
            submitted_at=time.time(),
        )
        store.insert(job)
        if job.bid:
            store.add_bid_spent(session, job.bid)
    held = run_pass(store)
    now = own_job(store, job.id)

    label = f" ({job.name})" if job.name else ""
    if now.state in TERMINAL:
        print(f"Job {job.id}{label} {now.state} at once: {now.reason or 'no reason recorded'}")
        return 1
    verb = "Queued" if now.state == "queued" else "Started"
    print(
        f"{verb} job {job.id}{label}: {describe_request(ram, args.cpus, args.gpus, vram)}, ", end=""
    )
    print(f"estimated {format_duration(est)}, killed at {format_duration(max_time)}.")
    if after:
        print(f"Waits for: {', '.join(after)}.")
    if now.state == "queued":
        delay = min(max_time / 10, settings.length_delay_cap) if settings.aging_per_minute else 0
        if delay > 0:
            print(f"Long max runtime: counts as submitted {format_duration(round(delay))} later.")
        if max_time > settings.long_job_after and settings.long_job_share < 100:
            print(
                f"Long job: long jobs together may hold at most {settings.long_job_share}% of "
                "RAM, cores and GPU memory, so it may wait for another long job to end."
            )
        # the job is queued already, so a failed estimate must not fail the submit
        with contextlib.suppress(ProquError, OSError, sqlite3.Error):
            when = time.time()
            place = queue_outlook(store, when, held).get(job.id)
            if place:
                print(f"{sentence(report.describe_place(place, when))}.")
    if watch.is_watched(session):
        print(
            "You will be woken when it ends, nothing more to run. "
            f"Check on it with `proqu status {job.id}`."
        )
    else:
        print(
            f"Next: run `proqu wait {job.id}` as a background command so you are woken when it "
            f"ends. Check on it with `proqu status {job.id}`."
        )
    return 0


MAX_NAME = 64


def parse_flag(flag: str, text: str, parse):
    try:
        return parse(text)
    except ValueError as error:
        raise invalid(f"{flag}: {error}") from None


def check_name(name: str | None) -> None:
    if name is None:
        return
    if not name or len(name) > MAX_NAME or not name.isprintable():
        raise invalid(f"--name must be 1 to {MAX_NAME} printable characters on one line")


def check_request(
    limits: machine.Machine,
    unreserved: machine.Machine,
    detected: machine.Machine,
    ram: int,
    cpus: int,
    gpus: int,
    vram: int,
) -> None:
    """Refuse a request over the queue limits.

    unreserved is the same without the reserve, detected is the machine itself.
    """
    if ram > limits.ram:
        message = (
            f"--ram {format_size(ram)} is more than the queue allows ({format_size(limits.ram)}): "
            "lower it or split the work"
        )
        if ram <= unreserved.ram:
            message += (
                f". The machine has {format_size(detected.ram)} but the rest is kept in "
                "reserve for services and interactive work. Only the machine's owner should "
                "change that, with `proqu config set reserve_ram`"
            )
        raise invalid(message)
    if not 1 <= cpus <= len(limits.cpu_ids):
        message = (
            f"--cpus {cpus} is outside what the queue allows: between 1 and {len(limits.cpu_ids)}"
        )
        if len(limits.cpu_ids) < cpus <= len(unreserved.cpu_ids):
            message += (
                f". The machine has {len(detected.cpu_ids)} but the rest are kept in reserve "
                "for services and interactive work. Only the machine's owner should change "
                "that, with `proqu config set reserve_cpus`"
            )
        raise invalid(message)
    if gpus < 0:
        raise invalid("--gpus cannot be negative")
    if any(gpu.vram == 0 for gpu in detected.gpus):
        check_unified(limits, gpus, vram)
        return
    if gpus and not vram:
        raise invalid("--gpus needs --vram, the GPU memory the job needs on each GPU")
    if vram and not gpus:
        raise invalid("--vram needs --gpus, the number of GPUs the job uses")
    if not gpus:
        return
    if not limits.gpus:
        raise invalid("this machine offers no GPUs to the queue, run without --gpus")
    if gpus > len(limits.gpus):
        raise invalid(f"--gpus {gpus} is more than the {len(limits.gpus)} GPUs the queue allows")
    big_enough = [gpu for gpu in limits.gpus if gpu.vram >= vram]
    if len(big_enough) < gpus:
        sizes = ", ".join(format_size(gpu.vram) for gpu in limits.gpus)
        raise invalid(
            f"--gpus {gpus} --vram {format_size(vram)} does not fit: only {len(big_enough)} "
            f"of the queue's GPUs allow that much (they allow {sizes})"
        )


def check_unified(limits: machine.Machine, gpus: int, vram: int) -> None:
    """Refuse a request for a GPU that uses the RAM, as on Apple silicon, that cannot be met."""
    if vram:
        raise invalid(
            "--vram does not apply on this machine: its GPU uses the RAM, so count the GPU "
            "memory the job needs in --ram and leave --vram out"
        )
    if gpus and not limits.gpus:
        raise invalid("this machine offers no GPUs to the queue, run without --gpus")
    if gpus > len(limits.gpus):
        raise invalid(
            f"--gpus {gpus} is more than the {len(limits.gpus)} GPU the queue allows, use --gpus 1"
        )


def check_executable(program: str, cwd: str, path: str) -> None:
    if "/" not in program:
        if not shutil.which(program, path=path):
            raise invalid(f"cannot find {program!r} on your PATH")
        return
    found = os.path.join(cwd, program)
    if not os.path.isfile(found):
        raise invalid(f"no file {found}")
    if not os.access(found, os.X_OK):
        raise invalid(
            f"{found} is not executable: run chmod +x on it, or run it via its interpreter"
        )


def check_dependencies(store: Store, session: str, after: list[str]) -> list[str]:
    unique = list(dict.fromkeys(after))
    for job_id in unique:
        job = store.get(job_id, session=session)
        if job is None:
            raise invalid(f"--after {job_id}: no job {job_id} in this session")
        if job.state in TERMINAL and job.state != "succeeded":
            raise invalid(
                f"--after {job_id}: that job already ended {job.state}, resubmit it first"
            )
    return unique


# status


RECENT_FINISHED = 10


def run_pass(store: Store) -> dict[str, str]:
    """Move the queue on, returning the jobs held back for free memory.

    A failed pass is only a warning, the command's own change stands.
    """
    try:
        return ticking.tick(store) or {}
    except (LockTimeout, StoreError, OSError, sqlite3.Error) as error:
        print(
            f"proqu: warning: the queue could not be moved on just now ({error})", file=sys.stderr
        )
        return {}


def cmd_status(args, command: list[str]) -> int:
    store = Store()
    held = run_pass(store)
    session = current_session()
    now = time.time()
    # built before printing, so a broken config fails before any output
    places = queue_outlook(store, now, held)
    budget = config.settings(load_config()).bid_budget
    if args.id:
        print("\n".join(job_details(own_job(store, args.id), now, places)))
    else:
        jobs = store.for_session(session)
        active = [job for job in jobs if job.state in ACTIVE]
        finished = sorted(
            (job for job in jobs if job.state in TERMINAL), key=lambda job: job.ended_at or 0
        )
        hidden = 0 if args.all else max(0, len(finished) - RECENT_FINISHED)
        shown = active + finished[hidden:]
        if not shown:
            print("No jobs in this session.")
        for job in shown:
            print(job_line(job, now, places))
        if hidden:
            print(f"({hidden} older finished jobs hidden, `proqu status --all` shows them)")
    remaining = max(0, budget - store.bid_spent(session))
    print(f"Bid budget: {remaining} of {budget} left for this session.")
    return 0


def job_line(job: Job, now: float, places: dict[str, report.Place]) -> str:
    label = f" {job.name}" if job.name else ""
    return f"{job.id}{label}: {job_state(job, now, places)}"


def job_state(job: Job, now: float, places: dict[str, report.Place]) -> str:
    if job.state == "queued":
        waited = format_duration(now - job.submitted_at)
        place = places.get(job.id)
        where = f", {report.describe_place(place, now)}" if place else ""
        return f"queued for {waited}{where}" + (f", bid {job.bid}" if job.bid else "")
    if job.cancel_requested and job.state in ("starting", "running"):
        return f"{job.state}, stopping because you cancelled it"
    if job.state == "starting":
        return "starting"
    if job.state == "running":
        ran = now - (job.started_at or now)
        if ran >= job.max:
            return f"running for {format_duration(ran)}, past its max runtime and being stopped"
        left = format_duration(job.max - ran)
        if ran < job.est:
            until = format_duration(report.expected_end(job, now) - now)
            expected = f"expected to finish in about {until}"
        else:
            expected = f"past its {format_duration(job.est)} estimate"
        return f"running for {format_duration(ran)}, {expected}, killed in {left} if still running"
    took = ""
    if job.started_at and job.ended_at:
        took = f" after {format_duration(job.ended_at - job.started_at)}"
    if job.state == "succeeded":
        return f"succeeded{took}"
    return f"{job.state}{took}: {job.reason or 'no reason recorded'}"


def job_details(job: Job, now: float, places: dict[str, report.Place]) -> list[str]:
    lines = [
        f"job {job.id}" + (f" ({job.name})" if job.name else ""),
        f"  state: {job_state(job, now, places)}",
        f"  command: {shlex.join(job.argv)}",
        f"  in: {job.cwd}",
        f"  request: {describe_request(job.ram, job.cpus, job.gpus, job.vram)}",
        f"  estimate {format_duration(job.est)}, killed at {format_duration(job.max)}",
        f"  bid: {job.bid}",
    ]
    if job.after:
        lines.append(f"  waits for: {', '.join(job.after)}")
    if job.exit_code is not None:
        lines.append(f"  exit code: {job.exit_code}")
    if job.peak_ram is not None:
        lines.append(f"  peak RAM: {format_size(job.peak_ram)}")
    return lines


# cancel


def cmd_cancel(args, command: list[str]) -> int:
    store = Store()
    session = current_session()
    with store.locked():
        job = own_job(store, args.id)
        if job.state in TERMINAL:
            raise ProquError(f"job {job.id} already {job.state}")
        if store.transition(
            job.id, ("queued",), "cancelled", reason="cancelled by you", ended_at=time.time()
        ):
            if job.bid:
                store.add_bid_spent(session, -job.bid)
            refund = f", bid of {job.bid} refunded" if job.bid else ""
            message = f"Cancelled job {job.id} before it started{refund}."
        else:
            store.update(job.id, cancel_requested=True)
            message = f"Cancel requested for job {job.id}, it will be stopped shortly."
    run_pass(store)
    print(message)
    return 0


# wait

# how often wait looks at its job, and how often it runs a scheduling pass
WAIT_POLL = 2.0
WAIT_TICK = 10.0
LOG_LINES = 20
TIMED_OUT = 3
CONFIG_BROKEN = 4


# signals that end a wait, each cleaning up first: SIGHUP is what a background command
# gets when its terminal or session goes away
STOPPING = (signal.SIGTERM, signal.SIGHUP)


class Terminated(Exception):
    """Raised on SIGTERM or SIGHUP, so a killed wait still cleans up after itself."""

    def __init__(self, signum: int = signal.SIGTERM):
        super().__init__(signum)
        self.signum = signum


def cmd_wait(args, command: list[str]) -> int:
    timeout = None
    if args.timeout is not None:
        timeout = parse_flag("--timeout", args.timeout, parse_duration)
    store = Store()
    job = own_job(store, args.id)
    pid, ticks = procs.current()
    previous = {signum: signal.getsignal(signum) for signum in STOPPING}

    def terminated(signum, frame):
        # one is enough, cleanup must not be interrupted by another
        for other in STOPPING:
            signal.signal(other, signal.SIG_IGN)
        raise Terminated(signum)

    for signum in STOPPING:
        signal.signal(signum, terminated)
    stopped = (
        f"Stopped waiting for job {job.id} before it ended. Run `proqu wait {job.id}` "
        "as a background command again to keep waiting."
    )
    try:
        with store.locked():
            store.add_waiter(job.id, pid, ticks)
        return wait_for(store, job.id, timeout)
    except KeyboardInterrupt:
        print(stopped, flush=True)
        return 130
    except Terminated as error:
        print(stopped, flush=True)
        return 128 + error.signum
    finally:
        drop_waiter(job.id, pid)
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def drop_waiter(job_id: str, pid: int) -> None:
    """Remove the waiter row even if a signal lands before the signals are ignored."""
    for _ in range(3):
        try:
            for signum in STOPPING:
                signal.signal(signum, signal.SIG_IGN)
            remove_waiter(job_id, pid)
            return
        except (Terminated, KeyboardInterrupt):
            continue


def wait_for(store: Store, job_id: str, timeout: float | None) -> int:
    began = time.monotonic()
    next_tick = began
    first = True
    held: dict[str, str] = {}
    while True:
        if time.monotonic() >= next_tick:
            held = quiet_tick(store)
            next_tick = time.monotonic() + WAIT_TICK
        job = own_job(store, job_id)
        if job.state in TERMINAL:
            return report_end(store, job)
        if job.state == "queued":
            try:
                load_config()
            except ProquError as error:
                print(
                    f"Stopped waiting: job {job.id} is still queued and nothing can start while "
                    f"the queue config is broken ({error}). Tell the user, then run "
                    f"`proqu wait {job.id}` again once it is fixed.",
                    flush=True,
                )
                return CONFIG_BROKEN
        if first and job.state == "queued":
            now = time.time()
            place = queue_outlook(store, now, held).get(job.id)
            if place:
                print(f"Job {job.id} is queued, {report.describe_place(place, now)}.", flush=True)
        first = False
        left = None if timeout is None else timeout - (time.monotonic() - began)
        if left is not None and left <= 0:
            print(
                f"Stopped waiting after {format_duration(timeout)}: job {job.id} is still "
                f"{job.state}. Run `proqu wait {job.id}` again to keep waiting.",
                flush=True,
            )
            return TIMED_OUT
        time.sleep(WAIT_POLL if left is None else min(WAIT_POLL, left))


def report_end(store: Store, job: Job) -> int:
    print(report.outcome(job, report.tail_lines(paths.log_path(job.id), LOG_LINES)), flush=True)
    # after printing, so a crash repeats the report rather than losing it
    # the report is out and the job's outcome stands, so a failed mark must not change the exit code
    try:
        with contextlib.suppress(LockTimeout, sqlite3.Error, OSError), store.locked():
            store.update(job.id, notified=True)
    except Terminated:
        # the job did end and was reported, so this is not a wait that stopped early
        pass
    return 0 if job.state == "succeeded" else 1


def quiet_tick(store: Store) -> dict[str, str]:
    """A scheduling pass that never ends the wait: a busy queue just means try later.

    Returns the jobs held back for free memory, empty when the pass failed.
    """
    with contextlib.suppress(LockTimeout, sqlite3.Error):
        return ticking.tick(store) or {}
    return {}


def remove_waiter(job_id: str, pid: int) -> None:
    # a fresh connection, as an interrupted one may be left mid transaction
    with contextlib.suppress(LockTimeout, StoreError, sqlite3.Error):
        store = Store(lock_timeout=10)
        try:
            with store.locked():
                store.remove_waiter(job_id, pid)
        finally:
            store.close()


# logs


def cmd_logs(args, command: list[str]) -> int:
    if args.tail < 0:
        raise invalid("--tail cannot be negative, use 0 for the whole log")
    job = own_job(Store(), args.id)
    path = paths.log_path(job.id)
    if args.tail:
        lines = report.tail_lines(path, args.tail) or []
        for line in lines:
            print(line)
        wrote = bool(lines)
    else:
        wrote = copy_log(path)
    if not wrote:
        print(f"No output from job {job.id} so far (state: {job.state}).")
    return 0


def copy_log(path) -> bool:
    """Write a whole log out, returning whether it held anything."""
    wrote = False
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as log:
            while chunk := log.read(1 << 16):
                sys.stdout.write(chunk)
                wrote = True
    except FileNotFoundError:
        pass
    return wrote


# limits and config


def cmd_limits(args, command: list[str]) -> int:
    detected = machine.snapshot(refresh=True)
    current = load_config()
    limits = config.effective_limits(detected, current)
    settings = config.settings(current)
    unreserved = config.effective_limits(detected, current, reserved=False)
    held_ram, held_cpus = config.reserve(detected, current)
    print("What the queue may use on this machine:")
    print(
        f"  RAM: {format_size(limits.ram)}"
        + _of(limits.ram, unreserved.ram, detected.ram, format_size)
    )
    print(
        f"  CPUs: {len(limits.cpu_ids)}"
        + _of(len(limits.cpu_ids), len(unreserved.cpu_ids), len(detected.cpu_ids), str)
    )
    if limits.gpus:
        for gpu in limits.gpus:
            if gpu.vram:
                print(f"  GPU {gpu.index}: {format_size(gpu.vram)} memory")
            else:
                print(
                    f"  GPU {gpu.index}: {gpu.uuid}, which uses the RAM, so one job at a time "
                    "and its GPU memory counts in --ram"
                )
    else:
        print("  GPUs: none")
    if sys.platform == "darwin":
        print(
            "RAM is enforced by watching each job four times a second, counting the memory "
            "footprint of all its processes, GPU memory included. macOS cannot pin a job to "
            "CPUs, so its --cpus is counted for scheduling but a job may use more: run it with "
            "that many threads. The Mac is kept awake while a job runs, as sleep would pause "
            "the job but not its max runtime clock. A job without --gpus is trusted not to use "
            "the GPU."
        )
    elif limits_module.cgroups_available():
        print(
            "RAM and CPU are enforced by the kernel: each job runs in its own cgroup. "
            "GPU use is watched every few seconds."
        )
    else:
        print(
            "RAM is enforced by watching each job about once a second, and CPUs by pinning. "
            "GPU use is watched every few seconds. For kernel enforcement of RAM and CPU the "
            "machine needs systemd for the user and linger (`loginctl enable-linger`)."
        )
    print(
        f"Kept in reserve for services and interactive work: {format_size(held_ram)} RAM and "
        f"{held_cpus} CPU {'core' if held_cpus == 1 else 'cores'}. Only the machine's owner "
        "should change it, with reserve_ram and reserve_cpus in `proqu config`."
    )
    print(f"Each session may bid up to {settings.bid_budget} points in total.")
    print(f"Every queued job gains {settings.aging_per_minute:g} priority points per minute.")
    # the delay is counted in aging, so without aging it changes nothing
    if settings.length_delay_cap and settings.aging_per_minute:
        print(
            "A long job counts as submitted later by a tenth of its --max, at most "
            f"{format_duration(settings.length_delay_cap)}, so shorter jobs tend to go first."
        )
    if settings.long_job_share < 100:
        print(
            f"Jobs with a --max over {format_duration(settings.long_job_after)} are long. Long "
            f"jobs together may hold at most {settings.long_job_share}% of RAM, cores and each "
            "GPU's memory. A larger long job runs only when no other long job is running."
        )
    if settings.hold_limit:
        print(
            "A job waiting for memory other programs are using may be passed by later jobs for "
            f"{format_duration(settings.hold_limit)}. After that no later job starts until it has."
        )
    return 0


def _of(limit, unreserved, total, show) -> str:
    """Why a limit is below the machine: the reserve only if it took off more than config did."""
    if limit >= total:
        return ""
    if limit < unreserved:
        return f" (of {show(total)}, after the reserve)"
    return f" (of {show(total)}, lowered in config)"


def cmd_config(args, command: list[str]) -> int:
    path = paths.config_path()
    if args.action == "show":
        if args.key:
            raise invalid("config show takes no key, it lists every setting")
        current = load_config()
        for key, description in config.DESCRIPTIONS.items():
            value = current.get(key, config.DEFAULTS.get(key))
            if value is None:
                shown = config.RESERVE_DEFAULTS.get(key, "machine default")
            else:
                shown = config.describe(key, value)
            print(f"{key} = {shown}    # {description}")
        return 0
    if not args.key:
        raise invalid(f"config {args.action} needs a key")
    if args.action == "set" and args.value is None:
        raise invalid("config set needs a value")
    if args.action == "unset" and args.value is not None:
        raise invalid("config unset takes only a key")
    detected = machine.snapshot()
    store = Store()
    with store.locked():
        # read without validation, so a bad value can still be set or unset
        try:
            current = config.load(path, validate=False)
        except ValueError as error:
            raise ProquError(str(error)) from None
        try:
            if args.action == "set":
                updated = config.set_value(current, args.key, args.value, detected)
            else:
                updated = config.unset(current, args.key)
        except ValueError as error:
            raise invalid(str(error)) from None
        config.save(path, updated)
    print("Saved. Run `proqu limits` to see what the queue may use now.")
    return 0


# entry point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proqu",
        description="A shared compute queue for coding agents.",
    )
    parser.add_argument("--version", action="version", version=f"proqu {VERSION}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    submit = commands.add_parser(
        "submit",
        help="queue a heavy job",
        description="Queue a job. Put the command after --.",
    )
    submit.add_argument("--ram", required=True, help="peak RAM the job needs, for example 6G")
    submit.add_argument("--cpus", required=True, type=int, help="CPU cores the job needs")
    submit.add_argument("--gpus", type=int, default=0, help="GPUs the job needs (default 0)")
    submit.add_argument("--vram", help="GPU memory needed on each GPU, for example 10G")
    submit.add_argument("--est", required=True, help="honest runtime estimate, for example 20m")
    submit.add_argument(
        "--max", required=True, help="generous runtime limit, the job is killed when it is reached"
    )
    submit.add_argument(
        "--after",
        action="append",
        default=[],
        metavar="ID",
        help="start only after this job succeeds",
    )
    submit.add_argument("--bid", type=int, default=0, help="priority bid, only when truly urgent")
    submit.add_argument("--name", help="a short label to recognise the job by")
    submit.add_argument("--cwd", help="working directory (default: the current one)")
    submit.set_defaults(handler=cmd_submit)

    status = commands.add_parser("status", help="show this session's jobs")
    status.add_argument("id", nargs="?", help="show one job in detail")
    status.add_argument("--all", action="store_true", help="include every finished job")
    status.set_defaults(handler=cmd_status)

    cancel = commands.add_parser("cancel", help="cancel one of this session's jobs")
    cancel.add_argument("id", help="the job id from submit or status")
    cancel.set_defaults(handler=cmd_cancel)

    wait = commands.add_parser(
        "wait",
        help="wait for one of this session's jobs to end and report how it went",
        description=(
            "Wait for a job to end, then report how it ended, what to do next and the end of "
            "its log. Run it as a background command so you are woken when the job ends. "
            "Exit codes: 0 the job succeeded, 1 it ended any other way, 3 --timeout ran out, "
            "4 the job is still queued but the queue config is broken, 129, 130 or 143 the wait "
            "was stopped by a signal, 141 its output was closed. For 3, 4, 129, 130, 141 and 143 "
            "the job may not have ended, so wait again."
        ),
    )
    wait.add_argument("id", help="the job id from submit or status")
    wait.add_argument("--timeout", help="stop waiting after this long, for example 30m")
    wait.set_defaults(handler=cmd_wait)

    logs = commands.add_parser("logs", help="show the output of one of this session's jobs")
    logs.add_argument("id", help="the job id from submit or status")
    logs.add_argument(
        "--tail",
        type=int,
        default=200,
        metavar="N",
        help=(
            "last N lines, taken from the last 256 KiB of the log so fewer come back when "
            "lines are long (default 200, 0 for the whole log)"
        ),
    )
    logs.set_defaults(handler=cmd_logs)

    limits = commands.add_parser("limits", help="show what the queue may use on this machine")
    limits.set_defaults(handler=cmd_limits)

    keys = "\n".join(f"  {key}: {text}" for key, text in config.DESCRIPTIONS.items())
    configure = commands.add_parser(
        "config",
        help="show or change the queue settings (for the machine's owner, not agents)",
        description=(
            "Show or change the queue settings. These belong to the person who owns the "
            "machine; agents should not change them. Resource limits can only be lowered "
            f"below what the machine has.\n\nkeys:\n{keys}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configure.add_argument(
        "action", nargs="?", default="show", choices=["show", "set", "unset"], help="default: show"
    )
    configure.add_argument("key", nargs="?", help="the setting to change")
    configure.add_argument("value", nargs="?", help="the new value, for set")
    configure.set_defaults(handler=cmd_config)

    # no help text, so it is left out of the command list the agent reads
    watching = commands.add_parser("watch")
    watching.set_defaults(handler=lambda args, command: watch.run())

    hook = commands.add_parser("hook")
    hook.add_argument("event", nargs="?", default="", help="session-start, prompt or stop")
    hook.set_defaults(handler=lambda args, command: hooks.run(args.event))
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command == "submit":
        parser.error(
            f"unexpected arguments {' '.join(extra)}: put the job's command after --, "
            "for example: proqu submit --ram 2G ... -- python train.py"
        )
    if extra or (command and args.command != "submit"):
        parser.error(f"unexpected arguments {' '.join(extra or command)}")
    try:
        return args.handler(args, command)
    except ProquError as error:
        print(f"proqu: {error}", file=sys.stderr)
        return error.code
    except (LockTimeout, paths.StateDirError, StoreError) as error:
        print(f"proqu: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # the reader went away, as in `proqu logs ID | head`: say nothing, as head would
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 128 + signal.SIGPIPE
    except (OSError, sqlite3.Error) as error:
        print(f"proqu: unexpected error: {error}", file=sys.stderr)
        return 1
