"""The proqu command line. Every message is written for an agent to act on."""

import argparse
import os
import shlex
import shutil
import sqlite3
import sys
import time

from proqu import config, machine, paths, report
from proqu import limits as limits_module
from proqu import tick as ticking
from proqu.store import ACTIVE, TERMINAL, Job, LockTimeout, Store, StoreError
from proqu.units import format_duration, format_size, parse_duration, parse_size

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


def queue_outlook(store: Store, now: float) -> dict[str, report.Place]:
    """Positions and projections of queued jobs, from the limits a pass would use."""
    current = load_config()
    limits = ticking.to_limits(config.effective_limits(machine.snapshot(), current))
    aging = config.settings(current).aging_per_minute
    return report.outlook(store, current_session(), limits, aging, now)


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def describe_request(ram: int, cpus: int, gpus: int, vram: int) -> str:
    parts = [f"{format_size(ram)} RAM", f"{cpus} CPU{'s' if cpus != 1 else ''}"]
    if gpus:
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

    limits = config.effective_limits(machine.snapshot(), load_config())
    check_request(limits, ram, args.cpus, args.gpus, vram)
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
    run_pass(store)
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
        when = time.time()
        place = queue_outlook(store, when).get(job.id)
        if place:
            print(f"{sentence(report.describe_place(place, when))}.")
    print(
        f"Next: run `proqu wait {job.id}` as a background command so you are woken when it ends. "
        f"Check on it with `proqu status {job.id}`."
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


def check_request(limits: machine.Machine, ram: int, cpus: int, gpus: int, vram: int) -> None:
    if ram > limits.ram:
        raise invalid(
            f"--ram {format_size(ram)} is more than the queue allows ({format_size(limits.ram)}): "
            "lower it or split the work"
        )
    if not 1 <= cpus <= len(limits.cpu_ids):
        raise invalid(
            f"--cpus {cpus} is outside what the queue allows: between 1 and {len(limits.cpu_ids)}"
        )
    if gpus < 0:
        raise invalid("--gpus cannot be negative")
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


def run_pass(store: Store) -> None:
    """Move the queue on. A failed pass is only a warning, the command's own change stands."""
    try:
        ticking.tick(store)
    except (LockTimeout, StoreError, OSError, sqlite3.Error) as error:
        print(
            f"proqu: warning: the queue could not be moved on just now ({error})", file=sys.stderr
        )


def cmd_status(args, command: list[str]) -> int:
    store = Store()
    run_pass(store)
    session = current_session()
    now = time.time()
    # built before printing, so a broken config fails before any output
    places = queue_outlook(store, now)
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
    print("What the queue may use on this machine:")
    print(f"  RAM: {format_size(limits.ram)}" + _of(limits.ram, detected.ram, format_size))
    print(f"  CPUs: {len(limits.cpu_ids)}" + _of(len(limits.cpu_ids), len(detected.cpu_ids), str))
    if limits.gpus:
        for gpu in limits.gpus:
            print(f"  GPU {gpu.index}: {format_size(gpu.vram)} memory")
    else:
        print("  GPUs: none")
    if limits_module.cgroups_available():
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
    print(f"Each session may bid up to {settings.bid_budget} points in total.")
    print(f"Every queued job gains {settings.aging_per_minute:g} priority points per minute.")
    return 0


def _of(limit, total, show) -> str:
    return f" (of {show(total)}, lowered in config)" if limit < total else ""


def cmd_config(args, command: list[str]) -> int:
    path = paths.config_path()
    if args.action == "show":
        if args.key:
            raise invalid("config show takes no key, it lists every setting")
        current = load_config()
        for key, description in config.DESCRIPTIONS.items():
            value = current.get(key, config.DEFAULTS.get(key))
            shown = "machine default" if value is None else config.describe(key, value)
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

    logs = commands.add_parser("logs", help="show the output of one of this session's jobs")
    logs.add_argument("id", help="the job id from submit or status")
    logs.add_argument(
        "--tail", type=int, default=200, metavar="N", help="last N lines (default 200, 0 for all)"
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
    except (OSError, sqlite3.Error) as error:
        print(f"proqu: unexpected error: {error}", file=sys.stderr)
        return 1
