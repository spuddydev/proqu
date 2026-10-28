import os

import pytest

from proqu import paths
from proqu.cli import main
from proqu.store import Store

BASE = ["submit", "--ram", "2G", "--cpus", "2", "--est", "1m", "--max", "5m"]


def submit():
    assert main([*BASE, "--", "true"]) == 0
    return max(Store().for_session(os.environ["PROQU_SESSION"]), key=lambda job: job.submitted_at)


def test_logs_show_the_last_lines(home, capsys):
    job = submit()
    paths.log_path(job.id).write_text("".join(f"line {i}\n" for i in range(1, 301)))
    capsys.readouterr()
    assert main(["logs", job.id]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert (len(lines), lines[0], lines[-1]) == (200, "line 101", "line 300")
    assert main(["logs", job.id, "--tail", "2"]) == 0
    assert capsys.readouterr().out == "line 299\nline 300\n"
    assert main(["logs", job.id, "--tail", "0"]) == 0
    assert capsys.readouterr().out.count("\n") == 300


@pytest.mark.parametrize("tail", ["200", "0"])
def test_logs_before_any_output(home, capsys, tail):
    job = submit()
    capsys.readouterr()
    assert main(["logs", job.id, "--tail", tail]) == 0
    assert capsys.readouterr().out == f"No output from job {job.id} so far (state: queued).\n"
    paths.log_path(job.id).write_text("")
    assert main(["logs", job.id, "--tail", tail]) == 0
    assert "No output" in capsys.readouterr().out


def test_logs_of_another_session_are_refused(home, capsys, monkeypatch):
    job = submit()
    paths.log_path(job.id).write_text("secret\n")
    monkeypatch.setenv("PROQU_SESSION", "s2")
    capsys.readouterr()
    assert main(["logs", job.id]) == 1
    captured = capsys.readouterr()
    assert f"no job {job.id} in this session" in captured.err
    assert "secret" not in captured.out


def test_negative_tail_is_refused(home, capsys):
    job = submit()
    assert main(["logs", job.id, "--tail", "-1"]) == 2
    assert "cannot be negative" in capsys.readouterr().err
