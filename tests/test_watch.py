import os
import subprocess
import sys
import time

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


def test_each_ended_job_is_one_line_and_reported_once(home):
    job = add("failed", reason="exited with code 3")
    add("running", started_at=time.time())
    store = Store()
    ended = watch._claim(store, SESSION)
    assert [j.id for j in ended] == [job.id]
    text = watch.line(ended[0])
    assert "\n" not in text
    assert text.startswith(f"proqu: job {job.id}: failed")
    assert notified(job.id)
    assert watch._claim(store, SESSION) == []


def run_watch(home, env, timeout=10):
    env = {**os.environ, "PROQU_HOME": str(home / "state"), **env}
    code = "import sys; from proqu import watch; sys.exit(watch.run())"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout
    )


@pytest.mark.parametrize("change", ["new session", "claude gone"])
def test_the_watch_stops_when_its_session_is_over(home, change):
    if change == "new session":
        claude = str(os.getpid())
        watch.record_session(claude, "the next one")
    else:
        child = subprocess.Popen(["true"])
        child.wait()
        claude = str(child.pid)
    result = run_watch(home, {"CLAUDE_CODE_SESSION_ID": SESSION, "CLAUDE_PID": claude})
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not watch.is_watched(SESSION)


def test_the_watch_prints_an_ended_job_then_stops_with_its_session(home):
    job = add("succeeded")
    claude = str(os.getpid())
    watch.record_session(claude, SESSION)
    env = {**os.environ, "PROQU_HOME": str(home / "state")}
    env |= {"CLAUDE_CODE_SESSION_ID": SESSION, "CLAUDE_PID": claude}
    code = "import sys; from proqu import watch; sys.exit(watch.run())"
    proc = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().startswith(f"proqu: job {job.id}: succeeded")
        watch.record_session(claude, "the next one")
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill()


def test_a_session_start_records_the_session_for_its_claude(home, capsys, monkeypatch):
    add("running", started_at=time.time())
    monkeypatch.setenv("CLAUDE_PID", "4242")
    hook("session-start", {"session_id": "fresh"}, capsys)
    assert watch._current_session("4242") == "fresh"


def test_watch_is_not_listed_in_the_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "watch" not in capsys.readouterr().out
