import pytest

from proqu import config, paths
from proqu.machine import GpuInfo, Machine

G = 1024**3
MACHINE = Machine(
    ram=16 * G,
    cpu_ids=(0, 1, 2, 3),
    gpus=(GpuInfo(0, "GPU-a", 24 * G), GpuInfo(1, "GPU-b", 12 * G)),
)


def test_defaults():
    assert config.settings({}) == config.Settings(
        bid_budget=11000, aging_per_minute=10.0, kill_grace=10
    )


@pytest.mark.parametrize(
    ("key", "text", "stored"),
    [
        ("ram", "8G", 8 * G),
        ("cpus", "2", 2),
        ("gpus", "1", [1]),
        ("gpus", "none", []),
        ("vram", "10G", 10 * G),
        ("bid_budget", "500", 500),
        ("aging_per_minute", "2.5", 2.5),
        ("kill_grace", "30s", 30),
    ],
)
def test_set_value(key, text, stored):
    assert config.set_value({}, key, text, MACHINE) == {key: stored}


@pytest.mark.parametrize(
    ("key", "text", "message"),
    [
        ("ram", "32G", "can only be lowered"),
        ("cpus", "5", "between 1 and 4"),
        ("cpus", "0", "between 1 and 4"),
        ("gpus", "2", "no GPU with index 2"),
        ("vram", "25G", "can only be lowered"),
        ("bid_budget", "-1", "negative"),
        ("aging_per_minute", "fast", "number"),
        ("colour", "red", "unknown config key"),
    ],
)
def test_set_value_refuses(key, text, message):
    with pytest.raises(ValueError, match=message):
        config.set_value({}, key, text, MACHINE)


def test_unset():
    assert config.unset({"ram": 1, "cpus": 2}, "ram") == {"cpus": 2}


def test_effective_limits_lower_the_machine():
    limits = config.effective_limits(
        MACHINE, {"ram": 8 * G, "cpus": 2, "gpus": [1], "vram": 10 * G}
    )
    assert limits == Machine(ram=8 * G, cpu_ids=(0, 1), gpus=(GpuInfo(1, "GPU-b", 10 * G),))


def test_effective_limits_follow_a_smaller_machine():
    small = Machine(ram=4 * G, cpu_ids=(0,), gpus=())
    assert config.effective_limits(small, {"ram": 8 * G, "cpus": 2}) == small


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    path = paths.config_path()
    assert config.load(path) == {}
    config.save(path, {"ram": 5})
    assert config.load(path) == {"ram": 5}


def test_state_dir_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    assert paths.state_dir().stat().st_mode & 0o777 == 0o700


def test_state_dir_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("PROQU_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert paths.state_dir() == tmp_path / "proqu"


def test_corrupt_config_names_the_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=r"config\.json is not valid JSON"):
        config.load(path)


def test_existing_open_state_dir_is_made_private(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(mode=0o755)
    state.chmod(0o755)
    monkeypatch.setenv("PROQU_HOME", str(state))
    paths.state_dir()
    assert state.stat().st_mode & 0o777 == 0o700


def test_relative_xdg_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("PROQU_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", "relative/dir")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.state_dir() == tmp_path / ".local" / "state" / "proqu"
