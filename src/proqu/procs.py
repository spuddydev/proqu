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
    marker = f"PROQU_JOB_ID={job_id}".encode()
    uid = os.getuid()
    members = []
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
            if session is not None and int(fields[3]) == session:
                members.append(pid)
                continue
            with open(f"/proc/{pid}/environ", "rb") as handle:
                if marker in handle.read().split(b"\0"):
                    members.append(pid)
        except OSError:
            continue
    return members
