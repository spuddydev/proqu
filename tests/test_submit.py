import pytest

from proqu import cli
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
        (["--cpus", "5"], "--cpus 5 is outside what the queue allows: between 1 and 4"),
        (["--cpus", "0"], "between 1 and 4"),
        (["--gpus", "1"], "needs --vram"),
        (["--vram", "1G"], "needs --gpus"),
        (["--gpus", "3", "--vram", "1G"], "more than the 2 GPUs"),
        (["--gpus", "2", "--vram", "20G"], "only 1 of the queue's GPUs allow that much"),
        (["--gpus", "-1"], "cannot be negative"),
        (["--vram", ""], "--vram: cannot read size"),
        (["--est", "5"], "--est: cannot read duration"),
        (["--cwd", ""], "--cwd is empty"),
        (["--cwd", "/etc/hostname"], "is not a directory"),
        (["--name", "two\nlines"], "printable characters"),
        (["--name", "x" * 65], "printable characters"),
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
    assert "cannot find 'no-such-program-xyz' on your PATH" in capsys.readouterr().err


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


def test_non_executable_script_is_refused(home, tmp_path, capsys):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o644)
    assert submit("--cwd", str(tmp_path), command=("./run.sh",)) == 2
    assert "run chmod +x" in capsys.readouterr().err


def test_missing_relative_program(home, tmp_path, capsys):
    assert submit("--cwd", str(tmp_path), command=("./nope.sh",)) == 2
    assert "no file" in capsys.readouterr().err


def test_relative_cwd_is_stored_absolute(home, tmp_path, monkeypatch):
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path)
    assert submit("--cwd", "work") == 0
    assert only_job().cwd == str(tmp_path / "work")


def test_deleted_current_directory(home, tmp_path, monkeypatch, capsys):
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert submit() == 2
    assert "pass --cwd" in capsys.readouterr().err


def test_forgotten_double_dash_gets_a_hint(home, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([*BASE, "python", "train.py"])
    assert exit_info.value.code == 2
    assert "put the job's command after --" in capsys.readouterr().err


def test_after_a_succeeded_job_is_allowed(home):
    assert submit() == 0
    parent = only_job()
    store = Store()
    with store.locked():
        store.update(parent.id, state="succeeded")
    assert submit("--after", parent.id) == 0


def test_whole_environment_is_kept(home, monkeypatch):
    monkeypatch.setenv("PROQU_EXTRA", "1")
    assert submit() == 0
    env = only_job().env
    assert env["PROQU_EXTRA"] == "1"
    assert env["PROQU_SESSION"] == "s1"


def test_empty_session_means_default(home, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "")
    assert submit() == 0
    assert len(Store().for_session("default")) == 1


def test_queued_submit_shows_its_place(home, capsys):
    assert submit("--cpus", "4") == 0
    capsys.readouterr()
    assert submit("--cpus", "4") == 0
    out = capsys.readouterr().out
    assert "Position 2 in the queue, expected to start in 1m and finish in 2m." in out


def test_a_long_max_says_it_counts_as_submitted_later(home, capsys):
    assert submit("--cpus", "4", "--max", "10h") == 0
    assert "Long max runtime: counts as submitted 1h later." in capsys.readouterr().out
    assert submit("--cpus", "4", "--max", "100h") == 0
    assert "counts as submitted 4h later." in capsys.readouterr().out


def test_a_short_max_or_no_delay_says_nothing_of_it(home, capsys):
    main(["config", "set", "length_delay_cap", "0"])
    assert submit("--cpus", "4", "--max", "10h") == 0
    assert "Long max runtime" not in capsys.readouterr().out
    main(["config", "unset", "length_delay_cap"])
    main(["config", "set", "aging_per_minute", "0"])
    assert submit("--cpus", "4", "--max", "10h") == 0
    assert "Long max runtime" not in capsys.readouterr().out


def test_place_never_shows_other_sessions(home, capsys, monkeypatch):
    monkeypatch.setenv("PROQU_SESSION", "s2")
    assert submit("--cpus", "4", "--name", "theirs", "--bid", "77") == 0
    theirs = Store().for_session("s2")[0]
    monkeypatch.setenv("PROQU_SESSION", "s1")
    capsys.readouterr()
    assert submit("--cpus", "4") == 0
    out = capsys.readouterr().out
    assert "Position 2 in the queue, expected to start in 1m" in out
    assert theirs.id not in out and "theirs" not in out and "77" not in out


def test_submit_succeeds_when_the_queue_outlook_fails(home, capsys, monkeypatch):
    assert submit("--cpus", "4") == 0
    monkeypatch.setattr(
        cli,
        "queue_outlook",
        lambda store, now, held=None: (_ for _ in ()).throw(cli.ProquError("broken")),
    )
    capsys.readouterr()
    assert submit("--cpus", "4") == 0
    out = capsys.readouterr().out
    assert out.startswith("Queued job ")
    assert "Next: run `proqu wait" in out
    assert len(Store().for_session("s1")) == 2


def reserve_on():
    main(["config", "unset", "reserve_ram"])
    main(["config", "unset", "reserve_cpus"])


def test_ram_over_the_reserve_names_the_reserve(home, capsys):
    reserve_on()
    capsys.readouterr()
    assert submit("--ram", "15G") == 2
    err = capsys.readouterr().err
    assert "more than the queue allows (14.4G)" in err
    assert "kept in reserve" in err
    assert "owner" in err
    assert "reserve_ram" in err


def test_cpus_over_the_reserve_name_the_reserve(home, capsys):
    reserve_on()
    capsys.readouterr()
    assert submit("--cpus", "4") == 2
    err = capsys.readouterr().err
    assert "between 1 and 3" in err
    assert "kept in reserve" in err
    assert "reserve_cpus" in err


def test_reserve_refusal_names_the_machine_not_a_lowered_figure(home, capsys):
    reserve_on()
    main(["config", "set", "ram", "15G"])
    main(["config", "set", "cpus", "3"])
    capsys.readouterr()
    assert submit("--ram", "14.7G") == 2
    assert "The machine has 16.0G" in capsys.readouterr().err
    main(["config", "set", "reserve_cpus", "2"])
    capsys.readouterr()
    assert submit("--cpus", "3") == 2
    assert "The machine has 4 but" in capsys.readouterr().err


def test_request_over_the_machine_does_not_blame_the_reserve(home, capsys):
    reserve_on()
    capsys.readouterr()
    assert submit("--ram", "20G") == 2
    assert submit("--cpus", "5") == 2
    assert "reserve" not in capsys.readouterr().err


def test_request_over_a_config_limit_does_not_blame_the_reserve(home, capsys):
    reserve_on()
    main(["config", "set", "ram", "4G"])
    capsys.readouterr()
    assert submit("--ram", "8G") == 2
    assert "reserve" not in capsys.readouterr().err


def test_request_inside_the_reserved_limits_is_accepted(home):
    reserve_on()
    assert submit("--ram", "14G", "--cpus", "3") == 0


def test_a_long_job_is_told_about_the_long_job_share(home, capsys):
    assert submit("--cpus", "4", "--max", "13h") == 0
    out = capsys.readouterr().out
    assert "Long job: long jobs together may hold at most 75% of RAM" in out
    assert submit("--cpus", "4", "--max", "12h") == 0
    assert "Long job:" not in capsys.readouterr().out
    main(["config", "set", "long_job_share", "100"])
    assert submit("--cpus", "4", "--max", "13h") == 0
    assert "Long job:" not in capsys.readouterr().out
