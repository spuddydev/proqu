import fcntl
import io
import json
import time

import pytest

from proqu import hooks, paths, procs
from proqu.cli import main
from proqu.store import Job, Store

SESSION = "sess-1"


def add(state="queued", session=SESSION, **fields) -> Job:
    store = Store()
    job = Job(
        id=store.new_id(),
        session=session,
        name=fields.pop("name", None),
        argv=["true"],
        cwd="/",
        env={},
        ram=1,
        cpus=1,
        gpus=0,
        vram=0,
        est=60,
        max=120,
        bid=0,
        after=[],
        submitted_at=time.time(),
        state=state,
        **fields,
    )
    with store.locked():
        store.insert(job)
        if state in ("succeeded", "failed", "killed", "cancelled"):
            store.update(job.id, ended_at=time.time())
    store.close()
    return job


def hook(event, payload, capsys):
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    stdout = io.StringIO()
    code = hooks.run(event, io.StringIO(stdin), stdout)
    text = stdout.getvalue()
    return code, json.loads(text) if text else None, capsys.readouterr().err


def context(output):
    return output["hookSpecificOutput"]["additionalContext"]


def notified(job_id):
    store = Store()
    try:
        return store.get(job_id).notified
    finally:
        store.close()


# session start


def test_session_start_exports_the_quoted_session_id(home, capsys, monkeypatch):
    env_file = home / "env.sh"
    monkeypatch.setenv("CLAUDE_ENV_FILE", str(env_file))
    code, output, err = hook("session-start", {"session_id": "it's mine"}, capsys)
    assert (code, output, err) == (0, None, "")
    assert env_file.read_text() == "export PROQU_SESSION='it'\"'\"'s mine'\n"


def test_session_start_without_an_env_file_is_quiet(home, capsys, monkeypatch):
    monkeypatch.delenv("CLAUDE_ENV_FILE", raising=False)
    assert hook("session-start", {"session_id": SESSION}, capsys) == (0, None, "")


def test_hooks_create_no_state_for_a_new_user(home, capsys):
    for event in ("session-start", "prompt", "stop"):
        assert hook(event, {"session_id": SESSION}, capsys) == (0, None, "")
    assert not paths.state_path().exists()


def test_session_start_lists_active_and_unreported_jobs(home, capsys, monkeypatch):
    monkeypatch.delenv("CLAUDE_ENV_FILE", raising=False)
    running = add("running", name="train", started_at=time.time() - 120)
    done = add("failed", exit_code=3, reason="exited with code 3")
    other = add("running", session="someone-else")
    _, output, _ = hook("session-start", {"session_id": SESSION, "source": "resume"}, capsys)
    text = context(output)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert f"- job {running.id} (train): running for 2m" in text
    assert f"- job {done.id}: failed: exited with code 3. Next: read the log" in text
    assert f"proqu logs {done.id} --tail 50" in text
    assert "proqu wait <id>" in text
    assert other.id not in text
    assert notified(done.id)


# prompt


def test_prompt_reports_an_outcome_once(home, capsys):
    job = add("killed", reason="still running at its max runtime of 2m")
    _, output, _ = hook("prompt", {"session_id": SESSION, "prompt": "hi"}, capsys)
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert f"- job {job.id}: killed: still running at its max" in context(output)
    assert "the job broke its own limit, not the queue's" in context(output)
    assert notified(job.id)
    assert hook("prompt", {"session_id": SESSION}, capsys) == (0, None, "")


def test_prompt_uses_the_input_session_not_the_environment(home, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "s1")
    mine = add("succeeded", session="s1")
    _, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert output is None
    assert not notified(mine.id)


def test_prompt_caps_the_list_and_keeps_the_rest(home, capsys):
    jobs = [add("succeeded") for _ in range(hooks.SHOWN + 10)]
    _, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert context(output).count("- job ") == hooks.SHOWN
    assert "and 10 more, see `proqu status`" in context(output)
    assert sum(notified(job.id) for job in jobs) == hooks.SHOWN
    _, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert context(output).count("- job ") == 10


# stop


def test_stop_blocks_while_a_job_is_unwatched(home, capsys):
    job = add("queued", name="sweep")
    code, output, err = hook("stop", {"session_id": SESSION, "stop_hook_active": False}, capsys)
    assert (code, err) == (0, "")
    assert output["decision"] == "block"
    assert f"- job {job.id} (sweep): queued" in output["reason"]
    assert "proqu wait <id>" in output["reason"]
    assert "proqu cancel <id>" in output["reason"]


def test_stop_allows_when_a_live_waiter_exists(home, capsys):
    job = add("running", started_at=time.time())
    store = Store()
    with store.locked():
        store.add_waiter(job.id, *procs.current())
    store.close()
    assert hook("stop", {"session_id": SESSION}, capsys) == (0, None, "")


@pytest.mark.parametrize("dead", ["gone", "reused"])
def test_stop_blocks_when_the_waiter_is_dead(home, capsys, dead):
    job = add("queued")
    pid, ticks = procs.current()
    waiter = (2**22 + 7, 1) if dead == "gone" else (pid, ticks + 1)
    store = Store()
    with store.locked():
        store.add_waiter(job.id, *waiter)
    store.close()
    _, output, _ = hook("stop", {"session_id": SESSION}, capsys)
    assert output["decision"] == "block"


def test_stop_allows_when_already_continuing(home, capsys):
    job = add("failed", exit_code=1)
    add("queued")
    assert hook("stop", {"session_id": SESSION, "stop_hook_active": True}, capsys) == (0, None, "")
    assert not notified(job.id)


def test_stop_reports_finished_outcomes_once(home, capsys):
    job = add("failed", exit_code=2, reason="exited with code 2")
    _, output, _ = hook("stop", {"session_id": SESSION}, capsys)
    assert output["decision"] == "block"
    assert f"- job {job.id}: failed: exited with code 2" in output["reason"]
    assert "exit code" not in output["reason"]
    assert "Report these outcomes to the user." in output["reason"]
    assert "proqu wait" not in output["reason"]
    assert notified(job.id)
    assert hook("stop", {"session_id": SESSION}, capsys) == (0, None, "")


def test_stop_for_unwatched_jobs_marks_nothing(home, capsys):
    add("queued")
    done = add("succeeded")
    store = Store()
    with store.locked():
        store.update(done.id, notified=True)
    store.close()
    hook("stop", {"session_id": SESSION}, capsys)
    assert notified(done.id)


# never break the session


@pytest.mark.parametrize(
    ("event", "stdin", "note"),
    [
        ("stop", "not json{", "not a JSON object"),
        ("stop", "", "not a JSON object"),
        ("prompt", "[1, 2]", "not a JSON object"),
        ("session-start", "{}", "no session_id"),
        ("finish", '{"session_id": "x"}', "unknown event"),
    ],
)
def test_bad_input_never_raises(home, capsys, event, stdin, note):
    code, output, err = hook(event, stdin, capsys)
    assert (code, output) == (0, None)
    assert err.count("\n") == 1
    assert err.startswith(f"proqu hook {event}: ")
    assert note in err


def test_a_held_lock_gives_up_quietly(home, capsys, monkeypatch):
    add("succeeded")
    monkeypatch.setattr(hooks, "LOCK_TIMEOUT", 0.2)
    with open(paths.lock_path(), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        started = time.monotonic()
        code, output, err = hook("prompt", {"session_id": SESSION}, capsys)
    assert time.monotonic() - started < 2
    assert (code, output) == (0, None)
    assert "queue lock" in err


def test_cli_runs_hooks_and_tolerates_a_missing_event(home, capsys, monkeypatch):
    job = add("succeeded")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": SESSION})))
    assert main(["hook", "prompt"]) == 0
    assert job.id in context(json.loads(capsys.readouterr().out))
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert main(["hook"]) == 0
    assert "unknown event" in capsys.readouterr().err


@pytest.mark.parametrize("event", ["session-start", "prompt", "stop"])
def test_hooks_move_the_queue_on(home, monkeypatch, capsys, event):
    from proqu import tick

    passes = []

    def record(store=None, cached_only=False):
        passes.append(cached_only)

    monkeypatch.setattr(tick, "tick", record)
    add()
    hook(event, {"session_id": SESSION}, capsys)
    assert passes == [True]


def test_a_failing_pass_does_not_break_the_hook(home, monkeypatch, capsys):
    from proqu import tick

    def broken(store=None, cached_only=False):
        raise RuntimeError("pass failed")

    monkeypatch.setattr(tick, "tick", broken)
    job = add(state="failed")
    code, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert code == 0
    assert job.id in context(output)


def test_a_slow_hook_gives_up_within_its_budget(home, capsys, monkeypatch):
    monkeypatch.setattr(hooks, "BUDGET", 1)
    monkeypatch.setitem(hooks.HANDLERS, "prompt", lambda data, session: time.sleep(10))
    started = time.monotonic()
    code, output, err = hook("prompt", {"session_id": SESSION}, capsys)
    assert time.monotonic() - started < 4
    assert (code, output) == (0, None)
    assert "gave up after 1s" in err


def test_a_signal_death_names_the_signal_not_a_number(home, capsys):
    job = add("killed", exit_code=-15, reason="still running at its max runtime of 3s")
    _, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert f"- job {job.id}: killed, ended by signal SIGTERM: still running" in context(output)
    assert "-15" not in context(output)


def test_a_lost_supervisor_says_it_was_not_the_jobs_fault(home, capsys):
    from proqu import tick

    add("failed", reason=tick.LOST)
    _, output, _ = hook("prompt", {"session_id": SESSION}, capsys)
    assert "not the job's fault, resubmit it" in context(output)


def test_session_start_still_reports_when_the_env_file_cannot_be_written(home, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDE_ENV_FILE", str(home / "missing" / "env.sh"))
    done = add("failed", reason="exited with code 3")
    _, output, err = hook("session-start", {"session_id": SESSION}, capsys)
    assert done.id in context(output)
    assert "cannot write" in err
