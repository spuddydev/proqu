"""Starting jobs and supervising them while they run.

Each running job has its own supervisor process, which exists only while the
job does. It launches the job, watches it, stops it when it must stop, records
how it ended and then lets the queue move on.
"""

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from proqu import config, limits, machine, paths, procs
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
                # a neutral start, whichever command happened to start the job
                os.chdir("/")
                os.umask(0o022)
                # isolated mode keeps modules in the caller's directory out of the supervisor
                argv = [sys.executable, "-I", "-c", _BOOT, _SRC, job_id]
                os.execve(sys.executable, argv, env)
            os.write(write, str(grandchild).encode())
        finally:
            os._exit(0)
    os.close(write)
    os.waitpid(child, 0)
    with os.fdopen(read) as pipe:
        text = pipe.read()
    if not text.isdigit():
        raise OSError(f"could not fork a supervisor, see {log}")
    pid = int(text)
    ticks = procs.start_ticks(pid)
    if ticks is None:
        raise OSError(f"supervisor {pid} exited at once, see {log}")
    return pid, ticks


def supervise(job_id: str) -> None:
    """Run one starting job to its end, then let the queue move on."""
    store = Store()
    try:
        _supervise(store, job_id)
    finally:
        from proqu.tick import tick

        tick(store)


def _supervise(store: Store, job_id: str) -> None:
    settings = _settings()
    with store.locked():
        job = store.get(job_id)
        if job is None or job.state != "starting":
            return
        if job.cancel_requested:
            _end_unstarted(store, job, "cancelled", CANCELLED)
            return
    with open(paths.log_path(job_id), "ab") as log:
        try:
            _check_cpus(job)
            proc, monitor = start(job, log)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            with store.locked():
                _end_unstarted(store, job, "failed", f"could not start: {_describe(error)}")
            return
    outcome = None
    running = False
    try:
        with store.locked():
            running = store.transition(
                job_id, ("starting",), "running", job_pid=proc.pid, started_at=time.time()
            )
        if running:
            outcome = watch(store, job, proc, monitor)
    finally:
        # whatever went wrong above, the job must not outlive its supervisor
        stop_members(job_id, proc.pid, settings.kill_grace)
        monitor.kill_all()
        code = _reap(proc)
        evidence = monitor.finish(code)
    if running:
        state, reason = outcome or evidence or classify(code)
        _finish(store, job_id, state, reason, code, monitor.peak)


def _end_unstarted(store: Store, job: Job, state: str, reason: str) -> None:
    """End a job that never ran, giving its bid back. Call inside locked()."""
    moved = store.transition(job.id, ("starting",), state, reason=reason, ended_at=time.time())
    if moved and job.bid:
        store.add_bid_spent(job.session, -job.bid)


def _check_cpus(job: Job) -> None:
    missing = set(job.cpu_ids) - os.sched_getaffinity(0)
    if missing:
        raise ValueError(f"its CPUs {sorted(missing)} are no longer available on this machine")


def _reap(proc: subprocess.Popen) -> int | None:
    """The job's exit code, or None if it cannot be made to exit at all."""
    try:
        return proc.wait(timeout=MEMBER_WAIT)
    except subprocess.TimeoutExpired:
        return None


def start(job: Job, log) -> tuple[subprocess.Popen, limits.Monitor]:
    """Launch the job with the strongest enforcement this machine offers."""
    _check_runnable(job)
    uuids = gpu_uuids(job)
    if limits.cgroups_available():
        marker = paths.state_dir() / "logs" / f"{job.id}.started"
        proc = launch(job, log, limits.scope_command(job, marker))
        started, scope = limits.await_scope(proc, marker)
        marker.unlink(missing_ok=True)
        if started:
            if scope is None:
                return proc, limits.Monitor(job, proc.pid, uuids)
            return proc, limits.CgroupMonitor(job, proc.pid, scope, uuids)
        # the scope was never made, so the job never ran and can run without it
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait()
        limits.mark_unavailable()
    proc = launch(job, log, job.argv)
    return proc, limits.Monitor(job, proc.pid, uuids)


def gpu_uuids(job: Job) -> tuple[str, ...]:
    """The uuids of the GPUs assigned to the job."""
    known = {gpu.index: gpu.uuid for gpu in machine.snapshot().gpus}
    return tuple(known.get(index, str(index)) for index in job.gpu_ids)


def _check_runnable(job: Job) -> None:
    """Fail early and plainly when the command cannot run at all.

    Inside a scope the shell step would only report exit code 127.
    """
    if not os.path.isdir(job.cwd):
        raise FileNotFoundError(2, "No such directory", job.cwd)
    program = job.argv[0]
    if "/" in program:
        if not os.access(os.path.join(job.cwd, program), os.X_OK):
            raise FileNotFoundError(2, "No such file or directory", program)
    elif not shutil.which(program, path=job.env.get("PATH", "")):
        raise FileNotFoundError(2, "No such file or directory", program)


def launch(job: Job, log, argv: list[str]) -> subprocess.Popen:
    """Start a command for the job in its own session, pinned to its CPUs."""
    env = dict(job.env)
    env["PROQU_JOB_ID"] = job.id
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids(job))
    cpu_ids = set(job.cpu_ids)

    def pin() -> None:
        if cpu_ids:
            os.sched_setaffinity(0, cpu_ids)

    if argv is not job.argv:
        env = limits.scope_env(env)
    return subprocess.Popen(
        argv,
        cwd=job.cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=pin,
    )


def watch(
    store: Store, job: Job, proc: subprocess.Popen, monitor: limits.Monitor
) -> tuple[str, str] | None:
    """Wait for the job, returning why it had to be stopped, or None if it ended."""
    deadline = time.monotonic() + job.max
    while True:
        try:
            proc.wait(timeout=min(POLL, max(0.0, deadline - time.monotonic())))
            ended = True
        except subprocess.TimeoutExpired:
            ended = False
        # a cancel counts even if the job ended by itself since the last look
        current = store.get(job.id)
        if current is not None and current.cancel_requested:
            return "cancelled", CANCELLED
        if ended:
            return None
        if time.monotonic() >= deadline:
            return (
                "killed",
                f"still running at its max runtime of {format_duration(job.max)}: "
                "if it was making progress, resubmit with a larger --max, otherwise find the hang",
            )
        broken = monitor.check()
        if broken:
            return broken


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


def classify(code: int | None) -> tuple[str, str | None]:
    """The outcome of a job that ended by itself."""
    if code is None:
        return "failed", "could not be stopped, its processes may be stuck in the kernel"
    if code == 0:
        return "succeeded", None
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return "failed", f"terminated by signal {name}"
    return "failed", f"exited with code {code}"


def _finish(
    store: Store,
    job_id: str,
    state: str,
    reason: str | None,
    code: int | None,
    peak: int | None = None,
) -> None:
    # the first writer wins, so a job already marked lost or cancelled keeps that
    with store.locked():
        store.transition(
            job_id,
            ("starting", "running"),
            state,
            reason=reason,
            exit_code=code,
            ended_at=time.time(),
            peak_ram=peak or None,
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
