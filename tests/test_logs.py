import os
import subprocess
import sys

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


def test_tail_help_says_the_window_can_cut_lines_short(capsys):
    with pytest.raises(SystemExit):
        main(["logs", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "256 KiB" in out and "0 for the whole log" in out


def test_whole_log_is_written_raw(home, capsys):
    job = submit()
    paths.log_path(job.id).write_bytes(b"10%\r50%\r100%\r\nok\xff\n")
    capsys.readouterr()
    assert main(["logs", job.id, "--tail", "0"]) == 0
    assert capsys.readouterr().out == "10%\r50%\r100%\r\nok�\n"


def test_a_closed_pipe_ends_quietly(home):
    job = submit()
    paths.log_path(job.id).write_text(("x" * 99 + "\n") * 100_000)
    proc = subprocess.Popen(
        [sys.executable, "-m", "proqu", "logs", job.id, "--tail", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdout.read(1)
    proc.stdout.close()
    err = proc.stderr.read()
    assert proc.wait(timeout=30) == 141
    assert err == b""
