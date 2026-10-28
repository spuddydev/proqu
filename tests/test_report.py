import pytest

from proqu import report, tick
from proqu.cli import main
from proqu.store import Job, Store

real_tick = tick.tick


def write(tmp_path, data: bytes):
    path = tmp_path / "job.log"
    path.write_bytes(data)
    return path


def test_tail_takes_the_last_lines(tmp_path):
    path = write(tmp_path, b"".join(b"%d\n" % i for i in range(1, 301)))
    assert report.tail_lines(path, 3) == ["298", "299", "300"]
    assert report.tail_lines(path, 500)[0] == "1"


def test_tail_edges(tmp_path):
    assert report.tail_lines(tmp_path / "missing.log", 5) is None
    assert report.tail_lines(write(tmp_path, b""), 5) == []
    assert report.tail_lines(write(tmp_path, b"a\n\nb"), 5) == ["a", "", "b"]
    assert report.tail_lines(write(tmp_path, b"\xffok\n"), 5) == ["�ok"]


def test_tail_keeps_the_last_drawing_of_a_progress_bar(tmp_path):
    path = write(tmp_path, b"10%\r50%\r100%\ndone\r\n")
    assert report.tail_lines(path, 5) == ["100%", "done"]


def test_tail_reads_only_the_end_of_a_huge_log(tmp_path):
    path = write(tmp_path, b"x" * (report.TAIL_BYTES * 2) + b"\nend\n")
    assert report.tail_lines(path, 5) == ["end"]
    one_line = write(tmp_path, b"y" * (report.TAIL_BYTES * 2))
    assert report.tail_lines(one_line, 5) == ["y" * report.TAIL_BYTES]


def ended(state, reason=None, started=True, **extra):
    job = Job(
        id="abc123",
        session="s1",
        name="train",
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
        submitted_at=1000.0,
        state=state,
        reason=reason,
        started_at=1010.0 if started else None,
        ended_at=1100.0,
    )
    for key, value in extra.items():
        setattr(job, key, value)
    return job


def test_outcome_of_a_succeeded_job():
    text = report.outcome(ended("succeeded", exit_code=0, peak_ram=3 * 1024**3), ["a", "b"])
    assert text.splitlines() == [
        "Job abc123 (train) succeeded after 1m30s",
        "Exit code 0, peak RAM 3.0G.",
        "Next: check its output and results before relying on them.",
        "Last 2 lines of the log (full log: `proqu logs abc123`):",
        "a",
        "b",
    ]


def test_outcome_of_a_failed_job_without_output():
    text = report.outcome(ended("failed", "exited with code 4", exit_code=4, name=None), None)
    assert text.splitlines() == [
        "Job abc123 failed after 1m30s: exited with code 4",
        "Exit code 4.",
        "Next: read the log to find out why, fix the cause, then resubmit.",
        "The job wrote no output.",
    ]


def test_outcome_of_a_job_ended_by_a_signal():
    text = report.outcome(ended("failed", "terminated by signal SIGKILL", exit_code=-9), [])
    assert "Ended by signal SIGKILL." in text


@pytest.mark.parametrize(
    ("job", "step"),
    [
        (ended("killed", "still running at its max runtime of 2m"), "its own limit"),
        (ended("cancelled", "cancelled by you", started=False), "nothing, unless"),
        (ended("failed", tick.LOST), "not the job's fault, resubmit"),
        (ended("failed", tick.LOST, started=False), "not the job's fault, resubmit"),
        (ended("failed", "could not start: No such file", started=False), "read the log"),
    ],
)
def test_next_step_depends_on_the_outcome(job, step):
    assert step in report.next_step(job)


def test_dependency_failure_reads_as_one(home):
    """Pinned to the scheduler's own wording, so a change to it shows up here."""
    base = ["submit", "--ram", "1G", "--cpus", "1", "--est", "1m", "--max", "2m"]
    assert main([*base, "--", "true"]) == 0
    parent = Store().for_session("s1")[0]
    assert main([*base, "--after", parent.id, "--", "true"]) == 0
    child = max(Store().for_session("s1"), key=lambda job: job.submitted_at)
    store = Store()
    with store.locked():
        store.update(parent.id, state="failed", ended_at=1.0)
    real_tick(store)
    failed = Store().get(child.id)
    assert failed.state == "failed"
    assert report.outcome(failed, None).startswith(f"Job {child.id} failed before it started: ")
    assert "resubmit the job it waits for first" in report.next_step(failed)


def test_tail_of_a_last_line_longer_than_the_window(tmp_path):
    path = write(tmp_path, b"y\n" + b"x" * (report.TAIL_BYTES * 2) + b"\n")
    assert report.tail_lines(path, 5) == ["x" * (report.TAIL_BYTES - 1)]


def test_tail_keeps_a_line_that_starts_exactly_at_the_window(tmp_path):
    first = b"zz\n"
    tail = b"yyyyyyyyyy\n"
    long_line = b"q" * (report.TAIL_BYTES - len(tail) - 1) + b"\n"
    path = write(tmp_path, first + long_line + tail)
    lines = report.tail_lines(path, 5)
    assert lines == ["q" * (len(long_line) - 1), "yyyyyyyyyy"]
