import pytest

from proqu.cli import main
from proqu.store import Store

BASE = ["submit", "--ram", "2G", "--cpus", "2", "--est", "1m", "--max", "5m"]


def submit(*extra, command=("true",)):
    return main([*BASE, *extra, "--", *command])


def only_job():
    jobs = Store().for_session("s1")
    assert len(jobs) == 1
    return jobs[0]


def test_valid_submit_queues_a_job(home, capsys, monkeypatch):
    monkeypatch.setenv("MARKER", "kept")
    assert submit("--name", "train", command=("echo", "a b")) == 0
    job = only_job()
    assert (job.state, job.name, job.argv, job.ram, job.cpus) == (
        "queued",
        "train",
        ["echo", "a b"],
        2 * 1024**3,
        2,
    )
    assert (job.est, job.max, job.env["MARKER"]) == (60, 300, "kept")
    out = capsys.readouterr().out
    assert f"Queued job {job.id} (train)" in out
    assert f"proqu wait {job.id}" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--ram", "20G"], "more than the queue allows"),
        (["--ram", "20"], "give a number and a unit"),
        (["--cpus", "5"], "between 1 and 4"),
        (["--cpus", "0"], "between 1 and 4"),
        (["--gpus", "1"], "needs --vram"),
        (["--vram", "1G"], "needs --gpus"),
        (["--gpus", "3", "--vram", "1G"], "more than the 2 GPUs"),
        (["--gpus", "2", "--vram", "20G"], "does not fit"),
        (["--est", "10m"], "longer than --max"),
        (["--bid", "10001"], "between 0 and 10000"),
        (["--bid", "-1"], "between 0 and 10000"),
        (["--after", "abcdef"], "no job abcdef in this session"),
        (["--cwd", "/no/such/dir"], "does not exist"),
    ],
)
def test_invalid_submit_is_refused(home, capsys, extra, message):
    assert submit(*extra) == 2
    assert message in capsys.readouterr().err
    assert Store().for_session("s1") == []


def test_missing_command(home, capsys):
    assert main(BASE) == 2
    assert "no command given" in capsys.readouterr().err


def test_unknown_program(home, capsys):
    assert submit(command=("no-such-program-xyz",)) == 2
    assert "cannot find an executable" in capsys.readouterr().err


def test_relative_program_resolves_from_cwd(home, tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    assert submit("--cwd", str(tmp_path), command=("./run.sh",)) == 0


def test_no_gpus_on_machine(home, capsys, monkeypatch):
    from proqu import config

    monkeypatch.setattr(config, "load", lambda path: {"gpus": []})
    assert submit("--gpus", "1", "--vram", "1G") == 2
    assert "offers no GPUs" in capsys.readouterr().err


def test_bid_is_charged_and_capped_by_budget(home, capsys):
    assert submit("--bid", "10000") == 0
    assert Store().bid_spent("s1") == 10000
    assert submit("--bid", "1001") == 2
    assert "remaining bid budget of 1000" in capsys.readouterr().err
    assert submit("--bid", "1000") == 0


def test_after_a_failed_job_is_refused(home, capsys):
    assert submit() == 0
    parent = only_job()
    store = Store()
    with store.locked():
        store.update(parent.id, state="killed")
    assert submit("--after", parent.id) == 2
    assert "already ended killed" in capsys.readouterr().err


def test_after_is_recorded_once(home):
    assert submit() == 0
    parent = only_job()
    assert submit("--after", parent.id, "--after", parent.id) == 0
    child = next(job for job in Store().for_session("s1") if job.id != parent.id)
    assert child.after == [parent.id]


def test_other_sessions_jobs_cannot_be_depended_on(home, monkeypatch, capsys):
    assert submit() == 0
    parent = only_job()
    monkeypatch.setenv("PROQU_SESSION", "s2")
    assert submit("--after", parent.id) == 2
    assert f"no job {parent.id} in this session" in capsys.readouterr().err
