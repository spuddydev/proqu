"""Claude Code hook entry points.

Each hook reads the hook input JSON on stdin. A hook must never break or stall
the agent's session, so every error becomes one line on stderr and exit 0.
"""

import contextlib
import json
import os
import shlex
import signal
import sys
import time

from proqu import paths, procs, report, tick, watch
from proqu.store import ACTIVE, TERMINAL, Job, Store
from proqu.units import format_duration

# a busy queue must not hold up the agent's session for long
LOCK_TIMEOUT = 3.0
# a hook ends by itself after this long, inside the timeout in hooks.json
BUDGET = 8
# jobs listed per message, so a long history cannot flood the agent's context
SHOWN = 20


def _commands(verb: str, jobs: list[Job]) -> str:
    return ", ".join(f"`proqu {verb} {job.id}`" for job in jobs[:SHOWN])


def _wait_hint(jobs: list[Job]) -> str:
    return (
        "these as background commands (Bash with run_in_background) so you are "
        f"woken when each ends: {_commands('wait', jobs)}"
    )


def run(event: str, stdin=None, stdout=None) -> int:
    """Handle one hook event. Always returns 0."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    previous = None
    try:
        previous = signal.signal(signal.SIGALRM, _out_of_time)
        signal.alarm(BUDGET)
    except ValueError:
        # not the main thread, so no alarm can be set
        pass
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
    finally:
        if previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return 0


def _out_of_time(signum, frame):
    raise TimeoutError(f"gave up after {BUDGET}s so the session is not held up")


def session_start(data: dict, session: str) -> dict | None:
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    exported = False
    if env_file:
        # the job report below does not depend on this write
        try:
            with open(env_file, "a") as handle:
                handle.write(f"export PROQU_SESSION={shlex.quote(session)}\n")
            exported = True
        except OSError as error:
            print(f"proqu hook session-start: cannot write {env_file}: {error}", file=sys.stderr)
    if os.environ.get("CLAUDE_PID") and paths.has_db():
        # a running watch stops once its Claude Code process has moved on to this session
        with contextlib.suppress(OSError, paths.StateDirError):
            watch.record_session(os.environ["CLAUDE_PID"], session)
    lines = []
    if not exported:
        lines += [
            f"proqu could not set PROQU_SESSION for your shell. Put `PROQU_SESSION={session}` "
            "in front of every proqu command, for example "
            f"`PROQU_SESSION={session} proqu status`. Without it proqu uses the shared "
            "`default` session, and you will not be woken or reminded about these jobs."
        ]
    store = _open_store()
    if store is None:
        return _context("SessionStart", lines)
    try:
        active = [job for job in store.for_session(session) if job.state in ACTIVE]
        finished, more = _claim_finished(store, session)
    finally:
        store.close()
    if finished:
        lines += ["proqu jobs of this session finished without being reported to you:"]
        lines += _listed(finished, more)
        lines += ["Tell the user these outcomes if they matter to the current work."]
    if active:
        lines += ["proqu jobs of this session are still in the queue:"]
        lines += _listed(active, 0)
        if watch.is_watched(session):
            lines += ["The proqu monitor tells you when each ends, nothing more to run."]
        else:
            lines += [f"Unless a wait is already running for one, run {_wait_hint(active)}."]
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
        lines += [
            f"Run {_wait_hint(unwatched)}. "
            f"Or cancel any no longer needed: {_commands('cancel', unwatched)}."
        ]
    if finished:
        lines += ["proqu jobs of this session finished and are not yet reported:"]
        lines += _listed(finished, more)
        lines += ["Report these outcomes to the user."]
    return {"decision": "block", "reason": "\n".join(lines)}


HANDLERS = {"session-start": session_start, "prompt": prompt, "stop": stop}


def _open_store() -> Store | None:
    """The job store, or None when this user has no queue database yet."""
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
    if watch.is_watched(job.session):
        return True
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
    # a plain exit code is already in the reason, a signal is not
    if job.exit_code is not None and job.exit_code < 0:
        text += f", {report.exit_text(job.exit_code)}"
    if job.reason:
        text += f": {job.reason}"
    return text + f". Next: {report.next_step(job)} Log: `proqu logs {job.id} --tail 50`"


def _context(event: str, lines: list[str]) -> dict | None:
    if not lines:
        return None
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n".join(lines)}}
