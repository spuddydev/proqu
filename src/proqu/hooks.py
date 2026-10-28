"""Claude Code hook entry points.

Each hook reads the hook input JSON on stdin. A hook must never break or stall
the agent's session, so every error becomes one line on stderr and exit 0.
"""

import contextlib
import json
import os
import shlex
import sys
import time

from proqu import paths, procs, tick
from proqu.store import ACTIVE, TERMINAL, Job, Store
from proqu.units import format_duration

# a busy queue must not hold up the agent's session for long
LOCK_TIMEOUT = 3.0
# jobs listed per message, so a long history cannot flood the agent's context
SHOWN = 20

WAIT_HINT = (
    "run `proqu wait <id>` as a background command (Bash with run_in_background) "
    "so you are woken when it ends"
)


def run(event: str, stdin=None, stdout=None) -> int:
    """Handle one hook event. Always returns 0."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        handler = HANDLERS.get(event)
        if handler is None:
            raise ValueError(f"unknown event, expected one of {', '.join(HANDLERS)}")
        try:
            data = json.loads(stdin.read())
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise ValueError("hook input on stdin is not a JSON object")
        session = data.get("session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("hook input has no session_id")
        output = handler(data, session)
        if output is not None:
            stdout.write(json.dumps(output) + "\n")
    except Exception as error:
        message = " ".join(str(error).split()) or type(error).__name__
        print(f"proqu hook {event}: {message}", file=sys.stderr)
    return 0


def session_start(data: dict, session: str) -> dict | None:
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if env_file:
        with open(env_file, "a") as handle:
            handle.write(f"export PROQU_SESSION={shlex.quote(session)}\n")
    store = _open_store()
    if store is None:
        return None
    try:
        active = [job for job in store.for_session(session) if job.state in ACTIVE]
        finished, more = _claim_finished(store, session)
    finally:
        store.close()
    lines = []
    if finished:
        lines += ["proqu jobs of this session finished without being reported to you:"]
        lines += _listed(finished, more)
        lines += ["Tell the user these outcomes if they matter to the current work."]
    if active:
        lines += ["proqu jobs of this session are still in the queue:"]
        lines += _listed(active, 0)
        lines += [f"Unless a wait is already running for one, {WAIT_HINT}."]
    return _context("SessionStart", lines)


def prompt(data: dict, session: str) -> dict | None:
    store = _open_store()
    if store is None:
        return None
    try:
        finished, more = _claim_finished(store, session)
    finally:
        store.close()
    if not finished:
        return None
    lines = ["proqu jobs of this session finished since you last heard:"]
    lines += _listed(finished, more)
    lines += ["Tell the user these outcomes if they matter to the current work."]
    return _context("UserPromptSubmit", lines)


def stop(data: dict, session: str) -> dict | None:
    if data.get("stop_hook_active") is True:
        return None
    store = _open_store()
    if store is None:
        return None
    try:
        jobs = store.for_session(session)
        unwatched = [job for job in jobs if job.state in ACTIVE and not _watched(store, job)]
        finished, more = _claim_finished(store, session)
    finally:
        store.close()
    if not unwatched and not finished:
        return None
    lines = []
    if unwatched:
        lines += ["proqu jobs of this session are not being waited on:"]
        lines += _listed(unwatched, 0)
        lines += [f"For each, {WAIT_HINT}, or run `proqu cancel <id>` if it is no longer needed."]
    if finished:
        lines += ["proqu jobs of this session finished and are not yet reported:"]
        lines += _listed(finished, more)
        lines += ["Report these outcomes to the user."]
    return {"decision": "block", "reason": "\n".join(lines)}


HANDLERS = {"session-start": session_start, "prompt": prompt, "stop": stop}


def _open_store() -> Store | None:
    """The job store, or None when no job was ever submitted on this machine."""
    if not paths.has_db():
        return None
    store = Store(lock_timeout=LOCK_TIMEOUT)
    store.conn.execute(f"PRAGMA busy_timeout = {int(LOCK_TIMEOUT * 1000)}")
    # with no daemon, each hook is also a chance to move the queue on
    with contextlib.suppress(Exception):
        tick.tick(store, cached_only=True)
    return store


def _claim_finished(store: Store, session: str) -> tuple[list[Job], int]:
    """Mark up to SHOWN unreported outcomes notified, returning them and how many remain.

    Reading and marking share one locked transaction, so two hooks racing on
    the same outcome cannot both report it. The lock is skipped when there is
    nothing to claim, so a quiet session does not contend with the queue.
    """
    if not _unreported(store.for_session(session)):
        return [], 0
    with store.locked():
        unreported = _unreported(store.for_session(session))
        unreported.sort(key=lambda job: job.ended_at or 0)
        claimed = unreported[:SHOWN]
        for job in claimed:
            store.update(job.id, notified=True)
    return claimed, len(unreported) - len(claimed)


def _unreported(jobs: list[Job]) -> list[Job]:
    return [job for job in jobs if job.state in TERMINAL and not job.notified]


def _watched(store: Store, job: Job) -> bool:
    return any(procs.is_alive(pid, ticks) for pid, ticks in store.waiters(job.id))


def _listed(jobs: list[Job], more: int) -> list[str]:
    now = time.time()
    lines = [_describe(job, now) for job in jobs[:SHOWN]]
    more += max(0, len(jobs) - SHOWN)
    if more:
        lines.append(f"and {more} more, see `proqu status`")
    return lines


def _describe(job: Job, now: float) -> str:
    text = f"- job {job.id}" + (f" ({job.name})" if job.name else "") + f": {job.state}"
    if job.state == "running" and job.started_at:
        return text + f" for {format_duration(now - job.started_at)}"
    if job.state in ACTIVE:
        return text
    if job.exit_code is not None and job.state != "succeeded":
        text += f", exit code {job.exit_code}"
    if job.reason:
        text += f": {job.reason}"
    return text + f". Log: `proqu logs {job.id} --tail 50`"


def _context(event: str, lines: list[str]) -> dict | None:
    if not lines:
        return None
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n".join(lines)}}
