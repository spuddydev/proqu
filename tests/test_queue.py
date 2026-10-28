"""The queue end to end: real passes, real supervisors, real processes."""

import json
import os
import subprocess
import sys
import time

import pytest

from proqu import cli, limits, machine, paths, procs, scheduler, tick
from proqu.cli import main
from proqu.store import Store

G = 1024**3
CPUS = tuple(sorted(os.sched_getaffinity(0)))[:2]


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PROQU_SESSION", "s1")
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    small = machine.Machine(ram=4 * G, cpu_ids=CPUS, gpus=())
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: small)
    paths.config_path().write_text(json.dumps({"reserve_ram": 0, "reserve_cpus": 0}))
    yield tmp_path
    for job in Store().active():
        tick.kill_leftovers([job.id])


def submit(*extra, command=("sleep", "1")):
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "2s", "--max", "30s"]
    assert main([*request, *extra, "--", *command]) == 0
    return max(Store().for_session("s1"), key=lambda job: job.submitted_at)


def settle(job_ids, timeout=30):
    store = Store()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = [store.get(job_id) for job_id in job_ids]
        if all(job.state not in ("queued", "starting", "running") for job in jobs):
            return jobs
        time.sleep(0.1)
    raise AssertionError(f"jobs still active: {[(j.id, j.state) for j in jobs]}")


@pytest.mark.skipif(len(CPUS) < 2, reason="needs two CPUs")
def test_jobs_start_when_they_fit_and_the_rest_follow(queue):
    first = submit("--ram", "3G")
    second = submit("--ram", "2G")
    third = submit("--ram", "1G")
    store = Store()
    assert store.get(first.id).state in ("starting", "running")
    assert store.get(second.id).state == "queued"
    assert store.get(third.id).state in ("starting", "running")
    done = settle([first.id, second.id, third.id])
    assert [job.state for job in done] == ["succeeded"] * 3
    assert done[1].started_at >= min(done[0].ended_at, done[2].ended_at)


def test_dependant_of_a_failed_job_fails_and_is_refunded(queue):
    parent = submit(command=("sh", "-c", "sleep 1; exit 4"))
    child = submit("--after", parent.id, "--bid", "100")
    assert Store().bid_spent("s1") == 100
    parent_done, child_done = settle([parent.id, child.id])
    assert parent_done.reason == "exited with code 4"
    assert child_done.state == "failed"
    assert child_done.reason == f"job {parent.id}, which it waits for, ended failed"
    assert Store().bid_spent("s1") == 0


def test_dependant_waits_even_when_it_fits(queue):
    parent = submit()
    child = submit("--after", parent.id, command=("true",))
    assert Store().get(child.id).state == "queued"
    parent_done, child_done = settle([parent.id, child.id])
    assert child_done.state == "succeeded"
    assert child_done.started_at >= parent_done.ended_at


def test_lost_supervisor_is_noticed_by_the_next_command(queue, capsys):
    job = submit(command=("sleep", "30"))
    store = Store()
    deadline = time.monotonic() + 10
    while store.get(job.id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    running = store.get(job.id)
    os.kill(running.supervisor_pid, 9)
    time.sleep(0.3)
    assert main(["status"]) == 0
    lost = store.get(job.id)
    assert lost.state == "failed"
    assert lost.reason == tick.LOST
    time.sleep(0.2)
    assert not procs.job_members(job.id, None)


def test_restart_loses_running_jobs(queue):
    job = submit(command=("sleep", "30"))
    store = Store()
    (queue / "state" / "boot_id").write_text("an-earlier-boot\n")
    tick.tick(store)
    assert store.get(job.id).reason == tick.LOST


def test_cancelled_job_that_lost_its_supervisor_reads_cancelled(queue):
    job = submit(command=("sleep", "30"))
    store = Store()
    with store.locked():
        store.update(job.id, cancel_requested=True, supervisor_pid=1, supervisor_start=0)
    tick.tick(store)
    assert store.get(job.id).state == "cancelled"


def test_passes_inside_a_job_do_not_start_jobs(queue, monkeypatch):
    monkeypatch.setenv("PROQU_JOB_ID", "outer1")
    job = submit(command=("true",))
    assert Store().get(job.id).state == "queued"
    monkeypatch.delenv("PROQU_JOB_ID")
    tick.tick()
    assert settle([job.id])[0].state == "succeeded"


def test_first_pass_is_not_taken_for_a_restart(queue):
    job = submit(command=("sleep", "2"))
    (queue / "state" / "boot_id").unlink()
    tick.tick()
    assert Store().get(job.id).state in ("starting", "running")


def test_lost_starting_job_gets_its_bid_back(queue):
    store = Store()
    job = submit("--bid", "200", command=("sleep", "30"))
    tick.kill_leftovers([job.id])
    with store.locked():
        store.update(job.id, state="starting", supervisor_pid=1, supervisor_start=0)
    tick.tick(store)
    assert store.get(job.id).reason == tick.LOST
    assert store.bid_spent("s1") == 0


def test_cancelling_a_parent_fails_its_dependant_at_once(queue, capsys):
    blocker = submit("--ram", "4G", command=("sleep", "30"))
    parent = submit("--ram", "4G", command=("true",))
    child = submit("--after", parent.id, command=("true",))
    assert main(["cancel", parent.id]) == 0
    assert Store().get(child.id).state == "failed"
    main(["cancel", blocker.id])


def test_submit_says_whether_the_job_started(queue, capsys):
    submit(command=("true",))
    assert "Started job" in capsys.readouterr().out


def test_submit_reports_a_job_failed_at_once(queue, capsys):
    parent = submit(command=("sh", "-c", "exit 1"))
    settle([parent.id])
    capsys.readouterr()
    store = Store()
    with store.locked():
        store.update(parent.id, state="running", supervisor_pid=1, supervisor_start=0)
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "2s", "--max", "30s"]
    assert main([*request, "--after", parent.id, "--", "true"]) == 1
    assert "failed at once" in capsys.readouterr().out


def test_failed_pass_does_not_fail_the_submit(queue, capsys, monkeypatch):
    from proqu.store import LockTimeout

    def busy(store=None):
        raise LockTimeout("busy")

    monkeypatch.setattr(tick, "tick", busy)
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "2s", "--max", "30s"]
    assert main([*request, "--", "true"]) == 0
    assert "could not be moved on just now" in capsys.readouterr().err


# waiting and logs


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(cli, "WAIT_POLL", 0.2)


def waiter_for(job_id, timeout=30):
    """Start `proqu wait` in its own process and return it once it is registered."""
    machine._save(paths.machine_path(), machine.snapshot())
    proc = subprocess.Popen(
        [sys.executable, "-m", "proqu", "wait", job_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if [pid for pid, _ in Store().waiters(job_id)] == [proc.pid]:
            return proc
        assert proc.poll() is None, proc.communicate()
        time.sleep(0.1)
    proc.kill()
    raise AssertionError("wait never registered itself")


def test_wait_reports_a_succeeded_job(queue, quick, capsys):
    job = submit("--name", "hi", command=("sh", "-c", "echo hello; sleep 0.5"))
    capsys.readouterr()
    assert main(["wait", job.id]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Job {job.id} (hi) succeeded after ")
    assert "Exit code 0" in out
    assert out.endswith(f"(whole log: `proqu logs {job.id} --tail 0`):\nhello\n")
    assert Store().get(job.id).notified


def test_wait_reports_a_failed_job(queue, quick, capsys):
    job = submit(command=("sh", "-c", "echo oops; exit 4"))
    capsys.readouterr()
    assert main(["wait", job.id]) == 1
    out = capsys.readouterr().out
    assert ": exited with code 4\n" in out
    assert "Next: read the log" in out
    assert out.endswith("oops\n")


def test_wait_on_a_queued_job_that_then_runs(queue, tmp_path):
    release = tmp_path / "release"
    hold = f"while [ ! -e {release} ]; do sleep 0.1; done"
    submit("--ram", "3G", command=("sh", "-c", hold))
    second = submit("--ram", "2G", command=("echo", "done"))
    assert Store().get(second.id).state == "queued"
    proc = waiter_for(second.id)
    # the first job is held until the wait is running, so no timing is assumed
    release.touch()
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert out.startswith(f"Job {second.id} is queued, position 1 in the queue, expected to start")
    assert f"Job {second.id} succeeded after " in out
    assert out.endswith("done\n")


def test_waiter_is_registered_while_waiting_and_removed_after(queue):
    job = submit(command=("sleep", "3"))
    proc = waiter_for(job.id)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert f"Job {job.id} succeeded after " in out
    assert Store().waiters(job.id) == []


def test_terminated_wait_removes_its_waiter(queue):
    job = submit(command=("sleep", "30"))
    proc = waiter_for(job.id)
    proc.terminate()
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143
    assert f"Stopped waiting for job {job.id} before it ended." in out
    assert Store().waiters(job.id) == []
    assert Store().get(job.id).state in ("starting", "running")


def test_wait_times_out_while_the_job_runs(queue, quick, capsys):
    job = submit(command=("sleep", "30"))
    capsys.readouterr()
    assert main(["wait", job.id, "--timeout", "1s"]) == 3
    out = capsys.readouterr().out
    assert f"job {job.id} is still " in out
    assert Store().waiters(job.id) == []
    assert not Store().get(job.id).notified


def test_another_session_cannot_wait_or_read_logs(queue, capsys, monkeypatch):
    job = submit(command=("sh", "-c", "echo private"))
    settle([job.id])
    monkeypatch.setenv("PROQU_SESSION", "s2")
    capsys.readouterr()
    assert main(["wait", job.id]) == 1
    assert main(["logs", job.id]) == 1
    captured = capsys.readouterr()
    assert captured.err.count(f"no job {job.id} in this session") == 2
    assert "private" not in captured.out
    assert not Store().get(job.id).notified


def test_logs_of_a_finished_job(queue, capsys):
    job = submit(command=("seq", "1", "300"))
    settle([job.id])
    capsys.readouterr()
    assert main(["logs", job.id, "--tail", "3"]) == 0
    assert capsys.readouterr().out == "298\n299\n300\n"
    assert main(["logs", job.id]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert (len(lines), lines[0]) == (200, "101")


def test_wait_reports_a_job_killed_at_its_max_runtime(queue, quick, capsys):
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "1s", "--max", "1s"]
    assert main([*request, "--", "sh", "-c", "echo started; sleep 30"]) == 0
    job = max(Store().for_session("s1"), key=lambda job: job.submitted_at)
    capsys.readouterr()
    assert main(["wait", job.id]) == 1
    out = capsys.readouterr().out
    assert f"Job {job.id} killed after " in out
    assert "still running at its max runtime of 1s" in out
    assert "the job broke its own limit, not the queue's" in out
    assert out.endswith("started\n")


def test_wait_reports_a_job_failed_by_its_parent(queue, quick, capsys):
    parent = submit(command=("sh", "-c", "exit 3"))
    child = submit("--after", parent.id)
    settle([parent.id, child.id])
    capsys.readouterr()
    assert main(["wait", child.id]) == 1
    out = capsys.readouterr().out
    assert f"Job {child.id} failed before it started: job {parent.id}, which it waits for" in out
    assert "resubmit the job it waits for first" in out


def test_wait_reports_a_job_whose_supervisor_was_lost(queue, quick, capsys):
    job = submit(command=("sleep", "30"))
    proc_pid = Store().get(job.id).supervisor_pid
    os.kill(proc_pid, 9)
    deadline = time.monotonic() + 10
    while procs.is_alive(proc_pid, Store().get(job.id).supervisor_start):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    capsys.readouterr()
    assert main(["wait", job.id]) == 1
    out = capsys.readouterr().out
    assert f"Job {job.id} failed" in out
    assert "its supervisor was lost" in out
    assert "this was not the job's fault, resubmit it" in out


def test_a_cached_only_pass_never_detects_the_machine(queue, monkeypatch):
    monkeypatch.setenv("PROQU_JOB_ID", "outer1")
    job = submit(command=("true",))
    monkeypatch.delenv("PROQU_JOB_ID")

    def detect(refresh=False):
        raise AssertionError("detected the machine")

    monkeypatch.setattr(machine, "snapshot", detect)
    monkeypatch.setattr(machine, "cached", lambda: None)
    tick.tick(cached_only=True)
    assert Store().get(job.id).state == "queued"
    monkeypatch.setattr(machine, "cached", lambda: machine.Machine(4 * G, CPUS, ()))
    tick.tick(cached_only=True)
    assert settle([job.id])[0].state == "succeeded"


def test_a_pass_hands_the_length_delay_cap_to_the_scheduler(queue, monkeypatch):
    seen = []

    def spy(*args, **kwargs):
        seen.append(args[5:])
        return scheduler.Plan([], [])

    monkeypatch.setattr(scheduler, "plan", spy)
    tick.tick()
    main(["config", "set", "length_delay_cap", "2h"])
    tick.tick()
    assert seen == [(10.0, 4 * 3600), (10.0, 2 * 3600)]


def test_a_queued_job_failed_by_the_reserve_names_it(queue, monkeypatch):
    real = tick.tick
    monkeypatch.setattr(tick, "tick", lambda store=None: None)
    job = submit("--ram", "4G")
    monkeypatch.setattr(tick, "tick", real)
    main(["config", "set", "reserve_ram", "1G"])
    real()
    failed = Store().get(job.id)
    assert failed.state == "failed"
    assert "reserve" in failed.reason
    assert "reserve_ram" in failed.reason


def test_a_queued_job_failed_by_a_config_limit_keeps_the_old_wording(queue, monkeypatch):
    real = tick.tick
    monkeypatch.setattr(tick, "tick", lambda store=None: None)
    job = submit("--ram", "4G")
    monkeypatch.setattr(tick, "tick", real)
    main(["config", "set", "ram", "2G"])
    real()
    assert "lowered after it was submitted" in Store().get(job.id).reason


# Free memory


def test_a_pass_hands_the_long_job_share_to_the_scheduler(queue, monkeypatch):
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("share"))
        return scheduler.Plan([], [])

    monkeypatch.setattr(scheduler, "plan", spy)
    main(["config", "set", "long_job_share", "50"])
    main(["config", "set", "long_job_after", "2h"])
    tick.tick()
    assert seen == [scheduler.Share(2 * 3600, 50)]


def test_long_jobs_wait_for_their_share(queue, capsys):
    # two cores, so long jobs together may hold one of them
    first = submit("--max", "13h", command=("sleep", "30"))
    second = submit("--max", "13h", command=("true",))
    assert Store().get(second.id).state == "queued"
    capsys.readouterr()
    main(["status", second.id])
    assert "expected to start in" in capsys.readouterr().out
    main(["config", "set", "long_job_share", "100"])
    tick.tick()
    assert settle([second.id])[0].state == "succeeded"
    main(["cancel", first.id])
    settle([first.id])


def test_job_waits_while_other_programs_use_the_memory(queue, monkeypatch):
    monkeypatch.setattr(machine, "free_ram", lambda: G // 2)
    job = submit(command=("true",))
    assert Store().get(job.id).state == "queued"
    assert tick.tick() == {job.id: "ram"}
    monkeypatch.setattr(machine, "free_ram", lambda: 3 * G)
    assert tick.tick() == {}
    assert settle([job.id])[0].state == "succeeded"


def test_status_and_submit_say_a_job_waits_for_memory(queue, monkeypatch, capsys):
    monkeypatch.setattr(machine, "free_ram", lambda: G // 2)
    job = submit(command=("true",))
    assert "waiting for memory other programs are using" in capsys.readouterr().out
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert f"{job.id}: queued" in out
    assert "waiting for memory other programs are using" in out
    assert "uncertain" in out


def test_only_memory_other_programs_use_makes_projections_uncertain(queue, monkeypatch):
    monkeypatch.setattr(tick, "tick", lambda store=None, cached_only=False: {})
    first, second = submit(), submit()
    store, now = Store(), time.time()
    places = cli.queue_outlook(store, now, {first.id: "gpu-unread"})
    assert not places[second.id].uncertain
    assert cli.queue_outlook(store, now, {first.id: "ram"})[second.id].uncertain


def test_running_jobs_keep_the_memory_they_may_still_use(queue, monkeypatch):
    # the sleeping job uses almost none of its 2G, but may grow into all of it
    first = submit("--ram", "2G", command=("sleep", "30"))
    monkeypatch.setattr(machine, "free_ram", lambda: 2 * G + G // 2)
    second = submit(command=("true",))
    assert Store().get(second.id).state == "queued"
    assert tick.tick() == {second.id: "ram"}
    main(["cancel", first.id])
    settle([first.id])


def test_passes_inside_a_job_never_read_free_memory(queue, monkeypatch):
    def unread():
        raise AssertionError("free memory read inside a job")

    monkeypatch.setattr(machine, "free_ram", unread)
    monkeypatch.setenv("PROQU_JOB_ID", "outer1")
    submit(command=("true",))


def test_running_gpu_jobs_keep_the_gpu_memory_they_may_still_use(queue, monkeypatch):
    gpu = machine.Machine(ram=4 * G, cpu_ids=CPUS, gpus=(machine.GpuInfo(0, "GPU-a", 8 * G),))
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: gpu)
    monkeypatch.setattr(limits, "gpu_apps", lambda: [])
    monkeypatch.setattr(machine, "free_vram", lambda: {0: 8 * G})
    first = submit("--gpus", "1", "--vram", "4G", command=("sleep", "30"))
    assert Store().get(first.id).state in ("starting", "running")
    # the first job uses none of its 4G yet, but may grow into all of it
    monkeypatch.setattr(machine, "free_vram", lambda: {0: 6 * G})
    second = submit("--gpus", "1", "--vram", "3G", command=("true",))
    assert tick.tick() == {second.id: "gpu"}
    main(["cancel", first.id])
    settle([first.id])


def test_wait_says_a_job_waits_for_memory(queue, quick, monkeypatch, capsys):
    monkeypatch.setattr(machine, "free_ram", lambda: G // 2)
    job = submit(command=("true",))
    capsys.readouterr()
    assert main(["wait", job.id, "--timeout", "1s"]) == 3
    assert "waiting for memory other programs are using" in capsys.readouterr().out


def test_gpu_memory_is_checked_when_free_ram_cannot_be_read(queue, monkeypatch):
    gpu = machine.Machine(ram=4 * G, cpu_ids=CPUS, gpus=(machine.GpuInfo(0, "GPU-a", 8 * G),))
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: gpu)
    monkeypatch.setattr(machine, "free_ram", lambda: None)
    monkeypatch.setattr(machine, "free_vram", lambda: {0: 0})
    job = submit("--gpus", "1", "--vram", "1G", command=("true",))
    assert tick.tick() == {job.id: "gpu"}
    assert Store().get(job.id).state == "queued"


def test_hook_passes_never_start_gpu_jobs(queue, monkeypatch):
    gpu = machine.Machine(ram=4 * G, cpu_ids=CPUS, gpus=(machine.GpuInfo(0, "GPU-a", 8 * G),))
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: gpu)
    monkeypatch.setattr(machine, "cached", lambda: gpu)
    monkeypatch.setattr(machine, "free_vram", lambda: {0: 8 * G})
    monkeypatch.setenv("PROQU_JOB_ID", "outer1")
    job = submit("--gpus", "1", "--vram", "1G", command=("true",))
    monkeypatch.delenv("PROQU_JOB_ID")
    assert tick.tick(cached_only=True) == {job.id: "gpu-unread"}
    assert Store().get(job.id).state == "queued"
