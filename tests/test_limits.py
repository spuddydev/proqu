import contextlib
import itertools
import os
import signal
import subprocess
import sys
import time

import pytest

from proqu import limits, procs, runner
from proqu.store import Store

from .test_runner import cancel_after, ended, log_of, starting_job

M = 1024**2

GROW = (
    "import time\n"
    "blocks = []\n"
    "for _ in range(80):\n"
    "    blocks.append(bytearray(10 * 1024 * 1024))\n"
    "    time.sleep(0.05)\n"
    "time.sleep(30)\n"
)


def test_memory_of_this_process_is_counted():
    assert limits.memory_of(os.getpid()) > M
    assert limits.memory_of(2**22 + 12345) == 0


def test_growing_job_is_killed_at_its_ram_limit(home):
    job = starting_job([sys.executable, "-c", GROW], ram=200 * M)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used more than its RAM limit of 200.0M")
    assert "this was the job's own limit" in done.reason
    assert done.peak_ram > 200 * M
    assert time.monotonic() - started < 20


def test_peak_ram_is_recorded_for_a_job_within_limits(home):
    code = "import time\nx = bytearray(60 * 1024 * 1024)\ntime.sleep(2.5)\n"
    job = starting_job([sys.executable, "-c", code], ram=500 * M)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "succeeded"
    assert 60 * M < done.peak_ram < 500 * M


def test_children_count_towards_the_limit(home):
    child = "import time\nx = bytearray(150 * 1024 * 1024)\ntime.sleep(30)\n"
    script = f"{sys.executable} -c '{child}' & {sys.executable} -c '{child}' & wait"
    job = starting_job(["sh", "-c", script], ram=250 * M)
    runner.supervise(job.id)
    assert ended(job).state == "killed"
    assert Store().get(job.id).peak_ram > 250 * M


REAL_PROBE = limits.cgroups_available
# the cgroup probe and its cache exist only on Linux, macOS has no cgroups
PROBES = pytest.mark.skipif(sys.platform == "darwin", reason="no cgroups on macOS")


@pytest.fixture
def cgroups(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    if not REAL_PROBE():
        pytest.skip("this machine cannot give jobs their own cgroup")


@pytest.mark.parametrize("attempt", range(5))
def test_kernel_kills_a_job_over_its_ram_in_a_scope(cgroups, attempt):
    job = starting_job([sys.executable, "-c", GROW], ram=200 * M)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used more than its RAM limit of 200.0M")


def test_scope_carries_the_cpu_quota_and_peak(cgroups):
    script = 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/cpu.max"; sleep 1.5'
    job = starting_job(["sh", "-c", script], cpus=2, cpu_ids=[0, 1])
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "succeeded"
    assert log_of(job).split()[:2] == ["200000", "100000"]
    assert done.peak_ram


def test_fast_job_in_a_scope_runs_once(cgroups, tmp_path):
    counter = tmp_path / "count"
    job = starting_job(["sh", "-c", f"echo x >> {counter}"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert counter.read_text() == "x\n"


def test_failing_systemd_run_falls_back_to_watching(home, monkeypatch, tmp_path, capsys):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemd-run").write_text("#!/bin/sh\necho no bus >&2\nexit 1\n")
    (fake / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setattr(limits, "cgroups_available", lambda: True)
    forgotten = []
    monkeypatch.setattr(limits, "mark_unavailable", lambda: forgotten.append(1))
    counter = tmp_path / "count"
    job = starting_job(["sh", "-c", f"echo x >> {counter}"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert counter.read_text() == "x\n"
    assert forgotten == [1]
    assert "systemd-run did not start the job in a cgroup" in capsys.readouterr().out


@PROBES
def test_probe_result_is_cached(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    calls = []
    monkeypatch.setattr(limits, "_probe", lambda: calls.append(1) or True)
    assert limits.cgroups_available() and limits.cgroups_available()
    assert calls == [1]


@PROBES
def test_a_failed_launch_makes_the_next_job_probe_again(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    calls = []
    monkeypatch.setattr(limits, "_probe", lambda: calls.append(1) or True)
    assert limits.cgroups_available()
    limits.mark_unavailable()
    assert limits.cgroups_available()
    assert calls == [1, 1]


def fake_nvidia_smi(tmp_path, monkeypatch, line):
    """A nvidia-smi that reports the job's shell, whose pid it reads from a file."""
    bin_dir = tmp_path / "gpu-bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "pid"
    script = bin_dir / "nvidia-smi"
    script.write_text(f'#!/bin/sh\n[ -f {pid_file} ] && echo "$(cat {pid_file}), {line}"\nexit 0\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(limits, "GPU_EVERY", 0.2)
    return ["sh", "-c", f"echo $$ > {pid_file}; sleep 30"]


def test_gpu_use_without_a_request_is_killed(home, tmp_path, monkeypatch):
    command = fake_nvidia_smi(tmp_path, monkeypatch, "GPU-a, 500")
    job = starting_job(command)
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used a GPU without asking for one")


def test_use_of_a_gpu_not_given_is_killed(home, tmp_path, monkeypatch):
    command = fake_nvidia_smi(tmp_path, monkeypatch, "GPU-b, 500")
    job = starting_job(command, gpus=1, vram=1024**3, gpu_ids=[0])
    runner.supervise(job.id)
    assert ended(job).reason.startswith("used GPU GPU-b, which was not given to it")


def test_gpu_memory_over_the_request_is_killed(home, tmp_path, monkeypatch):
    command = fake_nvidia_smi(tmp_path, monkeypatch, "GPU-a, 2048")
    job = starting_job(command, gpus=1, vram=1024**3, gpu_ids=[0])
    runner.supervise(job.id)
    assert ended(job).reason.startswith("used 2.0G of GPU memory, more than its --vram of 1.0G")


def test_gpu_use_within_the_request_is_fine(home, tmp_path, monkeypatch):
    fake_nvidia_smi(tmp_path, monkeypatch, "GPU-a, 512")
    pid_file = tmp_path / "pid"
    job = starting_job(
        ["sh", "-c", f"echo $$ > {pid_file}; sleep 1.5"], gpus=1, vram=1024**3, gpu_ids=[0]
    )
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"


def test_gpu_apps_parsing(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text("#!/bin/sh\necho '12, GPU-a, 100'\necho '13, GPU-b, [N/A]'\necho junk\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert limits.gpu_apps() == [(12, "GPU-a", 100 * M), (13, "GPU-b", None)]


def test_failing_gpu_query_is_ignored(home, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(limits, "GPU_EVERY", 0.2)
    job = starting_job(["sleep", "3"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    assert capsys.readouterr().out.count("could not query GPU use") == 1


def test_scope_limit_rounded_up_to_a_page_counts_as_applied(home, tmp_path):
    page = os.sysconf("SC_PAGE_SIZE")
    (tmp_path / "memory.max").write_text(str(100 * M + page))
    (tmp_path / "memory.events").write_text("oom_kill 0\n")
    job = starting_job(["true"], ram=100 * M + 1)
    monitor = limits.CgroupMonitor(job, 1, tmp_path)
    monitor.finish()
    assert monitor.enforced


def test_peak_of_a_job_shorter_than_one_poll_is_recorded(home):
    code = "x = bytearray(80 * 1024 * 1024)\nx[::4096] = b'a' * len(x[::4096])\n"
    job = starting_job([sys.executable, "-c", code], ram=500 * M)
    # a fresh process, so the peak of earlier test jobs is not inherited
    script = (
        "from proqu import limits, procs, runner\n"
        "limits.cgroups_available = lambda: False\n"
        f"runner.supervise('{job.id}')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    done = ended(job)
    assert done.state == "succeeded"
    assert 80 * M < done.peak_ram < 500 * M


def test_a_process_hiding_its_job_id_does_not_outlive_its_scope(cgroups, tmp_path):
    pid_file = tmp_path / "pid"
    script = f"env -i setsid /bin/sleep 300 & echo $! > {pid_file}; sleep 0.5"
    job = starting_job(["sh", "-c", script])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"
    pid = int(pid_file.read_text())
    try:
        assert not procs.is_alive(pid, procs.start_ticks(pid))
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_scope_command_names_systemd_run_by_full_path(home, monkeypatch, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemd-run").write_text("#!/bin/sh\n")
    (fake / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake))
    job = starting_job(["true"], env={"PATH": "/nowhere"})
    assert limits.scope_command(job, tmp_path / "m")[0] == str(fake / "systemd-run")


def test_failing_gpu_query_is_retried_less_and_less_often(home, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(limits.time, "monotonic", lambda: now[0])
    calls = []
    monkeypatch.setattr(limits, "gpu_apps", lambda: calls.append(now[0]))
    monitor = limits.Monitor(starting_job(["true"]), 1)
    monitor.watch_gpus = True
    for _ in range(300):
        monitor.check_gpus()
        now[0] += 1
    gaps = [later - earlier for earlier, later in itertools.pairwise(calls)]
    assert gaps == sorted(gaps)
    assert gaps[-1] == limits.GPU_BACKOFF_MAX
    assert len(calls) < 15


def test_job_that_starts_just_after_the_scope_wait_is_not_run_twice(home, monkeypatch, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemd-run").write_text(
        '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n'
    )
    (fake / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setattr(limits, "cgroups_available", lambda: True)

    def late(proc, marker):
        time.sleep(0.5)
        return False, None

    monkeypatch.setattr(limits, "await_scope", late)
    counter = tmp_path / "count"
    job = starting_job(["sh", "-c", f"echo x >> {counter}; sleep 30"])
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "failed"
    assert done.reason.startswith("could not start: its cgroup was too slow")
    assert counter.read_text() == "x\n"


def test_kernel_alone_kills_a_job_over_its_ram(cgroups, monkeypatch):
    # with the supervisor's own RAM check off, only the cgroup can stop the job
    monkeypatch.setattr(limits.Monitor, "check_ram", lambda self: None)
    monkeypatch.setattr(limits.CgroupMonitor, "check_ram", lambda self: None)
    job = starting_job([sys.executable, "-c", GROW], ram=200 * M)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("used more than its RAM limit of 200.0M")
    assert time.monotonic() - started < 20


def test_max_runtime_stops_a_job_in_a_scope(cgroups):
    job = starting_job(["sleep", "30"], max=1)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("still running at its max runtime of 1s")
    assert time.monotonic() - started < 10


def test_cancel_stops_a_job_in_a_scope(cgroups):
    job = starting_job(["sleep", "30"])
    cancel_after(job, 0.3)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert (done.state, done.reason) == ("cancelled", "cancelled by you")
    assert time.monotonic() - started < 10


@PROBES
def test_probe_result_expires(home, monkeypatch):
    monkeypatch.setattr(limits, "cgroups_available", REAL_PROBE)
    calls = []
    monkeypatch.setattr(limits, "_probe", lambda: calls.append(1) or True)
    now = time.time()
    monkeypatch.setattr(limits.time, "time", lambda: now)
    assert limits.cgroups_available()
    monkeypatch.setattr(limits.time, "time", lambda: now + limits.PROBE_AGE - 1)
    assert limits.cgroups_available()
    assert calls == [1]
    monkeypatch.setattr(limits.time, "time", lambda: now + limits.PROBE_AGE + 1)
    assert limits.cgroups_available()
    assert calls == [1, 1]


def test_vram_is_summed_per_gpu_and_not_across_gpus(home):
    job = starting_job(["true"], gpus=2, vram=1024**3, gpu_ids=[0, 1])
    both = ("GPU-a", "GPU-b")
    same_gpu = [(1, "GPU-a", 600 * M), (2, "GPU-a", 600 * M)]
    assert limits.gpu_breach(job, both, same_gpu)[1].startswith("used 1.1G of GPU memory")
    two_gpus = [(1, "GPU-a", 600 * M), (2, "GPU-b", 600 * M)]
    assert limits.gpu_breach(job, both, two_gpus) is None


def test_gpu_use_by_other_processes_is_ignored(home, tmp_path, monkeypatch):
    # nvidia-smi reports this test process, which is not part of the job
    fake_nvidia_smi(tmp_path, monkeypatch, "GPU-a, 500")
    (tmp_path / "pid").write_text(str(os.getpid()))
    job = starting_job(["sleep", "1.5"])
    runner.supervise(job.id)
    assert ended(job).state == "succeeded"


def test_ram_reason_gives_a_peak_only_when_it_is_above_the_limit():
    over = limits.ram_reason(200 * M, 250 * M)
    assert over.startswith("used more than its RAM limit of 200.0M (peak 250.0M): ")
    lagging = limits.ram_reason(200 * M, 150 * M)
    assert lagging.startswith("used more than its RAM limit of 200.0M (stopped by the kernel): ")
    assert "peak" not in lagging


def test_a_changed_scope_limit_is_noticed(home, tmp_path):
    page = os.sysconf("SC_PAGE_SIZE")
    (tmp_path / "memory.max").write_text(str(100 * M))
    (tmp_path / "memory.events").write_text("oom_kill 0\n")
    monitor = limits.CgroupMonitor(starting_job(["true"], ram=100 * M), 1, tmp_path)
    assert monitor.enforced and monitor.applied % page == 0
    assert monitor.check_ram() is None
    (tmp_path / "memory.max").write_text("max\n")
    monitor.sample()
    state, reason = monitor.check_ram()
    assert state == "killed"
    assert reason.startswith("changed the memory limit of its own cgroup")
    monitor.finish()


def test_job_lifting_its_own_scope_limit_is_killed(cgroups):
    # the pause lets the supervisor see the applied limit first
    script = (
        'sleep 1; echo max > "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.max"; '
        "sleep 30"
    )
    job = starting_job(["sh", "-c", script], ram=200 * M)
    started = time.monotonic()
    runner.supervise(job.id)
    done = ended(job)
    assert done.state == "killed"
    assert done.reason.startswith("changed the memory limit of its own cgroup")
    assert time.monotonic() - started < 10


def run_scope_command(job, monkeypatch, tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemd-run").write_text(
        '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n'
    )
    (fake / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    command = limits.scope_command(job, tmp_path / "marker")
    run = subprocess.run(
        command, env=limits.scope_env(dict(job.env)), capture_output=True, text=True, check=True
    )
    return run.stdout.split()


SHOW = 'echo "${XDG_RUNTIME_DIR-unset}" "${DBUS_SESSION_BUS_ADDRESS-unset}"'


def test_variables_only_systemd_run_needed_are_not_left_in_the_job(home, monkeypatch, tmp_path):
    env = {"PATH": os.environ["PATH"]}
    job = starting_job(["sh", "-c", SHOW], env=env)
    assert run_scope_command(job, monkeypatch, tmp_path) == ["unset", "unset"]


def test_variables_the_job_was_submitted_with_are_kept(home, monkeypatch, tmp_path):
    env = {"PATH": os.environ["PATH"], "XDG_RUNTIME_DIR": "/run/mine"}
    job = starting_job(["sh", "-c", SHOW], env=env)
    assert run_scope_command(job, monkeypatch, tmp_path) == ["/run/mine", "unset"]
