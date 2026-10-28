import pytest

from proqu import limits, machine, tick

G = 1024**3

FAKE_MACHINE = machine.Machine(
    ram=16 * G,
    cpu_ids=(0, 1, 2, 3),
    gpus=(machine.GpuInfo(0, "GPU-a", 24 * G), machine.GpuInfo(1, "GPU-b", 12 * G)),
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private state directory, a fixed session and a fake machine."""
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PROQU_SESSION", "s1")
    monkeypatch.delenv("PROQU_JOB_ID", raising=False)
    monkeypatch.setattr(machine, "snapshot", lambda refresh=False: FAKE_MACHINE)
    # command tests check the commands alone, real runs are in the queue tests
    monkeypatch.setattr(tick, "tick", lambda store=None: None)
    # supervisor tests use the watch enforcement unless they ask for cgroups
    monkeypatch.setattr(limits, "cgroups_available", lambda: False)
    return tmp_path
