"""Small helpers for checking on processes by pid without trusting reused pids."""

import os


def _stat_fields(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            stat = handle.read()
    except OSError:
        return None
    # the command name is in parentheses and may itself contain spaces or brackets
    return stat[stat.rindex(")") + 2 :].split()


def start_ticks(pid: int) -> int | None:
    """The process start time in clock ticks since boot, or None if it is gone.

    A pid can be reused once its process ends, so a pid is only trusted
    together with the start time recorded when it was first seen.
    """
    fields = _stat_fields(pid)
    return int(fields[19]) if fields else None


def is_alive(pid: int | None, ticks: int | None) -> bool:
    """Whether the process recorded as (pid, ticks) is still running.

    A zombie has exited even though its entry is still visible.
    """
    if pid is None or ticks is None:
        return False
    fields = _stat_fields(pid)
    return bool(fields) and fields[0] not in ("Z", "X") and int(fields[19]) == ticks


def current() -> tuple[int, int]:
    """This process as a (pid, ticks) pair."""
    pid = os.getpid()
    ticks = start_ticks(pid)
    assert ticks is not None
    return pid, ticks


def boot_id() -> str:
    """An id that changes on every boot, so pids from before a restart are never trusted."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def session_of(pid: int) -> int | None:
    fields = _stat_fields(pid)
    return int(fields[3]) if fields else None


def job_members(job_id: str, session: int | None) -> list[int]:
    """Every live process of a job: those in its session, or carrying its id.

    A process may leave the job's session with setsid, but it keeps the
    PROQU_JOB_ID it inherited, so it is still found.
    """
    return members_of({job_id: session})[job_id]


def members_of(jobs: dict[str, int | None]) -> dict[str, list[int]]:
    """The live processes of several jobs, given as job id to session, in one look at /proc."""
    markers = {f"PROQU_JOB_ID={job_id}".encode(): job_id for job_id in jobs}
    sessions = {session: job_id for job_id, session in jobs.items() if session is not None}
    found: dict[str, list[int]] = {job_id: [] for job_id in jobs}
    if not jobs:
        return found
    uid = os.getuid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            fields = _stat_fields(pid)
            if not fields or fields[0] in ("Z", "X"):
                continue
            owner = sessions.get(int(fields[3]))
            if owner is None:
                with open(f"/proc/{pid}/environ", "rb") as handle:
                    owner = next(
                        (markers[e] for e in handle.read().split(b"\0") if e in markers), None
                    )
            if owner is not None:
                found[owner].append(pid)
        except OSError:
            continue
    return found
