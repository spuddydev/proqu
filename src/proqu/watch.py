"""The session's watch: a Claude Code monitor that tells the agent when its jobs end.

Claude Code runs `proqu watch` for the whole session and turns each line it prints into
a notification, which wakes the agent even when it is idle. The watch registers itself
so submit and the stop hook know no background wait is needed. It belongs to the session
it started in: once Claude Code moves on to another session, as after /clear, it stops,
so a new session never hears about an old one's jobs.
"""

import contextlib
import hashlib
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

from proqu import paths, procs, tick
from proqu.store import TERMINAL, LockTimeout, Store, StoreError

POLL = 2.0
TICK = 10.0


def _sessions_dir() -> Path:
    path = paths.state_dir() / "sessions"
    path.mkdir(exist_ok=True)
    return path


def _key(session: str) -> str:
    # session ids come from outside, so they are never used as file names directly
    return hashlib.sha256(session.encode()).hexdigest()[:32]


def _claude_ticks(claude_pid: str) -> int | None:
    return procs.start_ticks(int(claude_pid)) if claude_pid.isdigit() else None


def record_session(claude_pid: str, session: str) -> None:
    """Note the session a Claude Code process now runs, called at session start.

    The record names the process by its start time as well, so one left by an earlier
    process with the same pid is ignored. It is replaced whole, so a watch never reads
    half of it.
    """
    path = _sessions_dir() / f"claude-{_key(claude_pid)}"
    spare = path.with_suffix(f".{os.getpid()}")
    spare.write_text(f"{_claude_ticks(claude_pid)} {session}")
    os.replace(spare, path)


def _current_session(claude_pid: str) -> str | None:
    """The session last recorded for this Claude Code process, None if none was."""
    with contextlib.suppress(OSError, ValueError):
        ticks, session = (_sessions_dir() / f"claude-{_key(claude_pid)}").read_text().split(" ", 1)
        if session and int(ticks) == _claude_ticks(claude_pid):
            return session
    return None


def _watcher_file(session: str) -> Path:
    return _sessions_dir() / f"watch-{_key(session)}"


def is_watched(session: str) -> bool:
    """Whether a live watch reports this session's job ends."""
    with contextlib.suppress(OSError, ValueError):
        pid, ticks = map(int, _watcher_file(session).read_text().split())
        return procs.is_alive(pid, ticks)
    return False


def line(job) -> str:
    """One line for a job's end, which reaches the agent as one notification."""
    # imported here, as the hooks use this module
    from proqu import hooks

    return "proqu:" + hooks._describe(job, time.time()).removeprefix("-")


def report_ended(store: Store, session: str) -> None:
    """Print each ended job not yet reported, oldest first, then mark it reported.

    Printing first means a watch stopped in between repeats a report rather than losing it.
    """
    ended = [j for j in store.for_session(session) if j.state in TERMINAL and not j.notified]
    for job in sorted(ended, key=lambda j: j.ended_at or 0):
        print(line(job), flush=True)
        with store.locked():
            store.update(job.id, notified=True)


def run() -> int:
    session = os.environ.get("CLAUDE_CODE_SESSION_ID") or os.environ.get("PROQU_SESSION")
    claude_pid = os.environ.get("CLAUDE_PID", "")
    if not session:
        return 0
    me = procs.current()
    mine = _watcher_file(session)
    # Claude Code stops a monitor with SIGTERM, which must still clean up
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        mine.write_text(f"{me[0]} {me[1]}")
        store = Store(lock_timeout=10)
        next_tick = 0.0
        while True:
            if claude_pid and _current_session(claude_pid) not in (None, session):
                return 0
            if claude_pid.isdigit() and _claude_ticks(claude_pid) is None:
                return 0
            with contextlib.suppress(LockTimeout, StoreError, sqlite3.Error):
                if time.monotonic() >= next_tick:
                    tick.tick(store)
                    next_tick = time.monotonic() + TICK
                report_ended(store, session)
            time.sleep(POLL)
    except KeyboardInterrupt:
        return 130
    finally:
        with contextlib.suppress(OSError):
            if mine.read_text() == f"{me[0]} {me[1]}":
                mine.unlink()
