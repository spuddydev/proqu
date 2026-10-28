"""The proqu command line. Every message is written for an agent to act on."""

import argparse
import os
import shutil
import sys
import time

from proqu import config, machine, paths
from proqu.store import TERMINAL, Job, LockTimeout, Store
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
    try:
        ram = parse_size(args.ram)
        vram = parse_size(args.vram) if args.vram else 0
        est = parse_duration(args.est)
        max_time = parse_duration(args.max)
    except ValueError as error:
        raise invalid(str(error)) from None

    limits = config.effective_limits(machine.snapshot(), load_config())
    check_request(limits, ram, args.cpus, args.gpus, vram)
    if est > max_time:
        raise invalid(
            f"--est {format_duration(est)} is longer than --max {format_duration(max_time)}: "
            "max is the hard kill time and must be the larger, overestimated value"
        )
    if not 0 <= args.bid <= MAX_BID:
        raise invalid(f"--bid must be between 0 and {MAX_BID}")

    cwd = os.path.abspath(args.cwd or os.getcwd())
    if not os.path.isdir(cwd):
        raise invalid(f"working directory {cwd} does not exist")
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

    label = f" ({job.name})" if job.name else ""
    print(
        f"Queued job {job.id}{label}: {describe_request(ram, args.cpus, args.gpus, vram)}, ", end=""
    )
    print(f"estimated {format_duration(est)}, killed at {format_duration(max_time)}.")
    if after:
        print(f"Waits for: {', '.join(after)}.")
    print(
        f"Next: run `proqu wait {job.id}` as a background command so you are woken when it ends. "
        f"Check on it with `proqu status {job.id}`."
    )
    return 0


def check_request(limits: machine.Machine, ram: int, cpus: int, gpus: int, vram: int) -> None:
    if ram > limits.ram:
        raise invalid(
            f"--ram {format_size(ram)} is more than the queue allows ({format_size(limits.ram)}): "
            "lower it or split the work"
        )
    if not 1 <= cpus <= len(limits.cpu_ids):
        raise invalid(f"--cpus must be between 1 and {len(limits.cpu_ids)} on this machine")
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
        largest = max(gpu.vram for gpu in limits.gpus)
        raise invalid(
            f"--vram {format_size(vram)} on {gpus} GPU(s) does not fit: "
            f"the largest GPU allows {format_size(largest)}"
        )


def check_executable(program: str, cwd: str, path: str) -> None:
    if "/" in program:
        found = os.path.join(cwd, program)
        if os.path.isfile(found) and os.access(found, os.X_OK):
            return
    elif shutil.which(program, path=path):
        return
    raise invalid(f"cannot find an executable {program!r} from {cwd} on your PATH")


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
        "--max", required=True, help="generous runtime limit; the job is killed at this point"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args, command)
    except ProquError as error:
        print(f"proqu: {error}", file=sys.stderr)
        return error.code
    except (LockTimeout, paths.StateDirError) as error:
        print(f"proqu: {error}", file=sys.stderr)
        return 1
