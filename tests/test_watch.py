import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from proqu import procs, watch
from proqu.cli import main
from proqu.store import Store
from tests.test_hooks import SESSION, add, hook, notified


def register(session=SESSION, pid_ticks=None):
    watch._watcher_file(session).write_text("{} {}".format(*(pid_ticks or procs.current())))


def test_a_live_watch_marks_the_session_watched(home):
    assert not watch.is_watched(SESSION)
    register()
    assert watch.is_watched(SESSION)
    assert not watch.is_watched("another")


def test_a_dead_watch_does_not_count(home):
    child = subprocess.Popen(["true"])
    ticks = procs.start_ticks(child.pid)
    child.wait()
    register(pid_ticks=(child.pid, ticks or 1))
    assert not watch.is_watched(SESSION)


def test_stop_allows_when_the_session_is_watched(home, capsys):
    add("running", started_at=time.time())
    register()
    assert hook("stop", {"session_id": SESSION}, capsys) == (0, None, "")


def test_submit_says_nothing_more_to_run_when_watched(home, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", SESSION)
    register()
    main(["submit", "--ram", "1G", "--cpus", "1", "--est", "1m", "--max", "2m", "--", "true"])
    out = capsys.readouterr().out
    assert "nothing more to run" in out
    assert "proqu wait" not in out


def test_ended_jobs_are_printed_oldest_first_once_each(home, capsys):
    later = add("failed", reason="exited with code 3")
    first = add("succeeded")
    store = Store()
    with store.locked():
        store.update(first.id, ended_at=time.time() - 60)
    add("running", started_at=time.time())
    watch.report_ended(store, SESSION)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"proqu: job {first.id}: succeeded")
    assert lines[1].startswith(f"proqu: job {later.id}: failed: exited with code 3")
    assert notified(first.id) and notified(later.id)
    watch.report_ended(store, SESSION)
    assert capsys.readouterr().out == ""


def start_watch(home, claude):
    env = {**os.environ, "PROQU_HOME": str(home / "state")}
    env |= {"CLAUDE_CODE_SESSION_ID": SESSION, "CLAUDE_PID": claude}
    code = "import sys; from proqu import watch; sys.exit(watch.run())"
    return subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True
    )


def read_line(proc, timeout=10):
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    assert ready, "the watch printed nothing"
    return proc.stdout.readline()


def until(check, timeout=10):
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end
        time.sleep(0.05)


@pytest.mark.parametrize("change", ["new session", "claude gone"])
def test_the_watch_stops_when_its_session_is_over(home, change):
    if change == "new session":
        claude = str(os.getpid())
        watch.record_session(claude, "the next one")
    else:
        child = subprocess.Popen(["true"])
        child.wait()
        claude = str(child.pid)
    proc = start_watch(home, claude)
    try:
        assert proc.wait(timeout=10) == 0
        assert proc.stdout.read() == ""
        assert not watch.is_watched(SESSION)
    finally:
        proc.kill()


def test_the_watch_registers_reports_and_stops_with_its_session(home):
    claude = str(os.getpid())
    proc = start_watch(home, claude)
    try:
        # no session recorded yet, as when the watch starts before the hook runs
        until(lambda: watch.is_watched(SESSION))
        job = add("succeeded")
        assert read_line(proc).startswith(f"proqu: job {job.id}: succeeded")
        assert proc.poll() is None
        watch.record_session(claude, "the next one")
        assert proc.wait(timeout=10) == 0
        assert not watch.is_watched(SESSION)
    finally:
        proc.kill()


def test_a_stopped_watch_leaves_no_registration(home):
    proc = start_watch(home, str(os.getpid()))
    try:
        until(lambda: watch.is_watched(SESSION))
        proc.terminate()
        proc.wait(timeout=10)
        assert not watch._watcher_file(SESSION).exists()
    finally:
        proc.kill()


def test_a_record_from_an_earlier_process_with_the_same_pid_is_ignored(home):
    claude = str(os.getpid())
    watch.record_session(claude, SESSION)
    path = next(watch._sessions_dir().glob("claude-*"))
    path.write_text(f"1 {SESSION}")
    assert watch._current_session(claude) is None


def test_session_start_says_nothing_to_run_when_watched(home, capsys):
    add("running", started_at=time.time())
    register()
    _, output, _ = hook("session-start", {"session_id": SESSION}, capsys)
    text = output["hookSpecificOutput"]["additionalContext"]
    assert "nothing more to run" in text
    assert "proqu wait" not in text


def test_a_session_start_records_the_session_for_its_claude(home, capsys, monkeypatch):
    add("running", started_at=time.time())
    monkeypatch.setenv("CLAUDE_PID", str(os.getpid()))
    hook("session-start", {"session_id": "fresh"}, capsys)
    assert watch._current_session(str(os.getpid())) == "fresh"


def test_watch_is_not_listed_in_the_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "watch" not in capsys.readouterr().out


def test_a_report_is_one_line_even_when_the_reason_is_not(home):
    job = add("failed", reason="could not start its supervisor: first\nsecond")
    assert "\n" not in watch.line(Store().get(job.id))


def test_the_plugin_runs_the_watch_as_a_monitor():
    root = Path(__file__).resolve().parent.parent
    (monitor,) = json.loads((root / "monitors" / "monitors.json").read_text())
    assert monitor["name"] and monitor["description"]
    assert monitor["command"] == '"${CLAUDE_PLUGIN_ROOT}/bin/proqu" watch'


def test_a_job_a_hook_reported_meanwhile_is_not_repeated(home, capsys, monkeypatch):
    job = add("succeeded")
    store = Store()
    stale = store.for_session(SESSION)
    with store.locked():
        store.update(job.id, notified=True)
    monkeypatch.setattr(store, "for_session", lambda session: stale)
    watch.report_ended(store, SESSION)
    assert capsys.readouterr().out == ""
