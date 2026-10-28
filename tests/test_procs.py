import os
import subprocess
import sys
import time

from proqu import procs


def test_current_process_is_alive():
    assert procs.is_alive(*procs.current())


def test_wrong_start_ticks_mean_a_reused_pid():
    pid, ticks = procs.current()
    assert not procs.is_alive(pid, ticks + 1)


def test_missing_values_are_dead():
    assert not procs.is_alive(None, 5)
    assert not procs.is_alive(os.getpid(), None)


def test_exited_process_is_dead_even_as_a_zombie():
    child = subprocess.Popen(["true"])
    ticks = procs.start_ticks(child.pid)
    time.sleep(0.3)
    try:
        assert ticks is not None
        assert not procs.is_alive(child.pid, ticks)
    finally:
        child.wait()


def test_name_with_brackets_and_spaces(tmp_path):
    odd = tmp_path / "a) (b c"
    odd.symlink_to(sys.executable)
    child = subprocess.Popen([str(odd), "-c", "import time; time.sleep(5)"])
    try:
        time.sleep(0.2)
        assert procs.is_alive(child.pid, procs.start_ticks(child.pid))
        assert procs.session_of(child.pid) == os.getsid(0)
    finally:
        child.kill()
        child.wait()


def test_boot_id_is_stable():
    assert procs.boot_id() == procs.boot_id() != ""


def test_job_members_by_session_and_by_marker():
    env = dict(os.environ, PROQU_JOB_ID="job123")
    leader = subprocess.Popen(["sleep", "5"], start_new_session=True)
    marked = subprocess.Popen(["sleep", "5"], env=env, start_new_session=True)
    try:
        time.sleep(0.2)
        assert set(procs.job_members("job123", leader.pid)) == {leader.pid, marked.pid}
        assert procs.job_members("other", None) == []
    finally:
        for child in (leader, marked):
            child.kill()
            child.wait()
