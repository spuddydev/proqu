"""Starting jobs and supervising them while they run.

Each running job has its own supervisor process, which exists only while the
job does. It launches the job, watches it, stops it when it must stop, records
how it ended and then lets the queue move on.
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from proqu import config, machine, paths, procs
from proqu.store import Job, Store
from proqu.units import format_duration

# how often the supervisor looks at its job
POLL = 1.0
# how long to keep killing a job's processes before giving up on stragglers
MEMBER_WAIT = 10.0

CANCELLED = "cancelled by you"

_SRC = str(Path(__file__).resolve().parent.parent)
_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from proqu.runner import supervise; supervise(sys.argv[2])"
)


def spawn_supervisor(job_id: str) -> tuple[int, int]:
    """Start a detached supervisor for a job and return its pid and start ticks.

    The supervisor is forked twice so it is adopted by init, and no long lived
    caller is left holding a dead child that still looks alive.
    """
    log = str(paths.supervisor_log_path(job_id))
    # a supervisor started from inside a job must not count as part of that job
    env = {key: value for key, value in os.environ.items() if key != "PROQU_JOB_ID"}
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read)
            os.setsid()
            grandchild = os.fork()
            if grandchild == 0:
                devnull = os.open(os.devnull, os.O_RDWR)
                out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                os.dup2(devnull, 0)
                os.dup2(out, 1)
                os.dup2(out, 2)
                os.execve(sys.executable, [sys.executable, "-c", _BOOT, _SRC, job_id], env)
            os.write(write, str(grandchild).encode())
        finally:
            os._exit(0)
    os.close(write)
    os.waitpid(child, 0)
    with os.fdopen(read) as pipe:
        pid = int(pipe.read())
    ticks = procs.start_ticks(pid)
    if ticks is None:
        raise OSError(f"supervisor {pid} exited at once, see {log}")
    return pid, ticks


def supervise(job_id: str) -> None:
    """Run one starting job to its end. This is the supervisor's whole life."""
    store = Store()
    settings = _settings()
    with store.locked():
        job = store.get(job_id)
        if job is None or job.state != "starting":
            return
        if job.cancel_requested:
            store.transition(
                job_id, ("starting",), "cancelled", reason=CANCELLED, ended_at=time.time()
            )
            return
    with open(paths.log_path(job_id), "ab") as log:
        try:
            proc = launch(job, log)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            _finish(store, job_id, "failed", f"could not start: {_describe(error)}", None)
            return
    with store.locked():
        running = store.transition(
            job_id, ("starting",), "running", job_pid=proc.pid, started_at=time.time()
        )
    outcome = watch(store, job, proc) if running else None
    stop_members(job_id, proc.pid, settings.kill_grace)
    code = proc.wait()
    if running:
        state, reason = outcome or classify(code)
        _finish(store, job_id, state, reason, code)


def launch(job: Job, log) -> subprocess.Popen:
    """Start the job's command in its own session, pinned to its CPUs."""
    env = dict(job.env)
    env["PROQU_JOB_ID"] = job.id
    uuids = {gpu.index: gpu.uuid for gpu in machine.snapshot().gpus}
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids.get(i, str(i)) for i in job.gpu_ids)
    cpu_ids = set(job.cpu_ids)

    def pin() -> None:
        if cpu_ids:
            os.sched_setaffinity(0, cpu_ids)

    return subprocess.Popen(
        job.argv,
        cwd=job.cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=pin,
    )


def watch(store: Store, job: Job, proc: subprocess.Popen) -> tuple[str, str] | None:
    """Wait for the job, returning why it had to be stopped, or None if it ended."""
    deadline = time.monotonic() + job.max
    while True:
        try:
            proc.wait(timeout=min(POLL, max(0.0, deadline - time.monotonic())))
            return None
        except subprocess.TimeoutExpired:
            pass
        current = store.get(job.id)
        if current is not None and current.cancel_requested:
            return "cancelled", CANCELLED
        if time.monotonic() >= deadline:
            return (
                "killed",
                f"still running at its max runtime of {format_duration(job.max)}: "
                "if it was making progress, resubmit with a larger --max, otherwise find the hang",
            )


def stop_members(job_id: str, session: int, grace: float) -> None:
    """Stop every process of the job, politely first, then by force."""
    members = procs.job_members(job_id, session)
    if not members:
        return
    _signal_all(members, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        time.sleep(0.1)
        if not procs.job_members(job_id, session):
            return
    deadline = time.monotonic() + MEMBER_WAIT
    while time.monotonic() < deadline:
        members = procs.job_members(job_id, session)
        if not members:
            return
        _signal_all(members, signal.SIGKILL)
        time.sleep(0.1)


def classify(code: int) -> tuple[str, str | None]:
    """The outcome of a job that ended by itself."""
    if code == 0:
        return "succeeded", None
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return "failed", f"terminated by signal {name}"
    return "failed", f"exited with code {code}"


def _finish(store: Store, job_id: str, state: str, reason: str | None, code: int | None) -> None:
    # the first writer wins, so a job already marked lost or cancelled keeps that
    with store.locked():
        store.transition(
            job_id,
            ("starting", "running"),
            state,
            reason=reason,
            exit_code=code,
            ended_at=time.time(),
        )


def _signal_all(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, sig)


def _settings() -> config.Settings:
    try:
        return config.settings(config.load(paths.config_path()))
    except ValueError:
        return config.settings({})


def _describe(error: Exception) -> str:
    if isinstance(error, OSError) and error.strerror:
        name = f" {error.filename}" if error.filename else ""
        return f"{error.strerror}{name}"
    return str(error)
