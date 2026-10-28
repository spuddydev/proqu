import json

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
        ("aging_per_minute", "nan", "number"),
        ("aging_per_minute", "inf", "number"),
        ("cpus", "\u0662", "whole number"),
        ("cpus", "1_0", "whole number"),
        ("cpus", "+2", "whole number"),
        ("bid_budget", "1" * 13, "whole number"),
        ("aging_per_minute", "\u0661", "number"),
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
    assert config.effective_limits(small, {"ram": 8 * G, "cpus": 2, "reserve_ram": 0}) == small


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


@pytest.mark.parametrize(
    "bad",
    [
        {"ram": "12G"},
        {"cpus": 0},
        {"cpus": True},
        {"gpus": "0"},
        {"bid_budget": -5},
        {"aging_per_minute": "fast"},
        {"kill_grace": 0.5},
        {"aging_per_minute": True},
        {"aging_per_minute": 10**400},
        {"aging_per_minute": -1},
        {"gpus": [True]},
        {"vram": 0},
        {"bid_budget": -1},
    ],
)
def test_bad_values_in_the_file_are_refused(tmp_path, bad):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match=f"bad value for {next(iter(bad))}"):
        config.load(path)


def test_file_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="not valid JSON"):
        config.load(path)


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"later": "x"}')
    assert config.load(path) == {"later": "x"}


def test_settings_read_the_config():
    assert config.settings({"bid_budget": 5, "aging_per_minute": 1.5, "kill_grace": 3}) == (
        config.Settings(bid_budget=5, aging_per_minute=1.5, kill_grace=3)
    )


def test_effective_limits_with_no_gpus_allowed():
    assert config.effective_limits(MACHINE, {"gpus": []}).gpus == ()


def test_vram_limit_never_raises_a_smaller_gpu():
    gpus = config.effective_limits(MACHINE, {"vram": 20 * G}).gpus
    assert [gpu.vram for gpu in gpus] == [20 * G, 12 * G]


@pytest.mark.parametrize(
    ("key", "value", "shown"),
    [
        ("ram", 8 * G, "8.0G"),
        ("kill_grace", 90, "1m30s"),
        ("gpus", [], "none"),
        ("gpus", [0, 1], "0,1"),
    ],
)
def test_describe(key, value, shown):
    assert config.describe(key, value) == shown


def test_state_dir_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(paths.os, "geteuid", lambda: 12345)
    with pytest.raises(paths.StateDirError, match="belongs to another user"):
        paths.state_dir()


def test_state_dir_leaves_the_umask_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("PROQU_HOME", str(tmp_path / "state"))
    import os

    before = os.umask(0o022)
    try:
        paths.state_dir()
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(before)


def test_relative_proqu_home_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROQU_HOME", "rel")
    assert paths.state_dir() == tmp_path / "rel"


def test_no_home_directory_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("PROQU_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    def no_home():
        raise RuntimeError("no home")

    monkeypatch.setattr(paths.Path, "home", no_home)
    with pytest.raises(paths.StateDirError, match="set PROQU_HOME"):
        paths.state_dir()


def test_unusable_state_dir_gives_a_clear_error(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("PROQU_HOME", str(blocker / "state"))
    with pytest.raises(paths.StateDirError, match="cannot use state directory"):
        paths.state_dir()


def test_bid_budget_of_zero_is_allowed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"bid_budget": 0}')
    assert config.load(path) == {"bid_budget": 0}


def test_bad_values_load_without_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"ram": -5}')
    assert config.load(path, validate=False) == {"ram": -5}


def test_save_leaves_no_temp_files(tmp_path):
    config.save(tmp_path / "config.json", {"ram": 5})
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]


# The reserve

FOUR = Machine(ram=16 * G, cpu_ids=(0, 1, 2, 3), gpus=())
TWO = Machine(ram=16 * G, cpu_ids=(0, 1), gpus=())


def test_default_reserve_on_a_four_cpu_machine():
    limits = config.effective_limits(FOUR, {})
    assert limits.ram == 16 * G - 16 * G // 10
    assert limits.cpu_ids == (0, 1, 2)


def test_default_reserve_on_a_two_cpu_machine_keeps_every_cpu():
    limits = config.effective_limits(TWO, {})
    assert limits.cpu_ids == (0, 1)
    assert limits.ram == 16 * G - 16 * G // 10


def test_reserve_of_zero_gives_the_whole_machine():
    limits = config.effective_limits(FOUR, {"reserve_ram": 0, "reserve_cpus": 0})
    assert (limits.ram, limits.cpu_ids) == (16 * G, (0, 1, 2, 3))


def test_reserve_can_be_set_higher():
    limits = config.effective_limits(FOUR, {"reserve_ram": 4 * G, "reserve_cpus": 2})
    assert (limits.ram, limits.cpu_ids) == (12 * G, (0, 1))


def test_explicit_limits_only_lower_the_reserved_limits():
    lower = config.effective_limits(FOUR, {"ram": 8 * G, "cpus": 2})
    assert (lower.ram, lower.cpu_ids) == (8 * G, (0, 1))
    higher = config.effective_limits(FOUR, {"ram": 16 * G, "cpus": 4})
    assert (higher.ram, higher.cpu_ids) == (16 * G - 16 * G // 10, (0, 1, 2))


def test_effective_limits_can_leave_the_reserve_out():
    limits = config.effective_limits(FOUR, {"ram": 8 * G}, reserved=False)
    assert (limits.ram, limits.cpu_ids) == (8 * G, (0, 1, 2, 3))


@pytest.mark.parametrize("ram", [1, 5, 10, 10**6])
def test_reserve_never_leaves_no_ram(ram):
    tiny = Machine(ram=ram, cpu_ids=(0,), gpus=())
    assert config.effective_limits(tiny, {}).ram >= 1
    assert config.effective_limits(tiny, {"reserve_ram": 10**9}).ram >= 1


def test_reserve_never_leaves_no_cpus():
    assert config.effective_limits(TWO, {"reserve_cpus": 9}).cpu_ids == (0,)


@pytest.mark.parametrize(
    ("key", "text", "stored"),
    [("reserve_ram", "2G", 2 * G), ("reserve_ram", "0", 0), ("reserve_cpus", "0", 0)],
)
def test_set_reserve(key, text, stored):
    assert config.set_value({}, key, text, MACHINE) == {key: stored}


@pytest.mark.parametrize(
    ("key", "text", "message"),
    [
        ("reserve_ram", "16G", "less than"),
        ("reserve_ram", "lots", "cannot read size"),
        ("reserve_cpus", "4", "between 0 and 3"),
        ("reserve_cpus", "-1", "between 0 and 3"),
    ],
)
def test_set_reserve_refuses(key, text, message):
    with pytest.raises(ValueError, match=message):
        config.set_value({}, key, text, MACHINE)


def test_zero_reserve_in_the_file_is_valid(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"reserve_ram": 0, "reserve_cpus": 0}')
    assert config.load(path) == {"reserve_ram": 0, "reserve_cpus": 0}
    path.write_text('{"reserve_cpus": -1}')
    with pytest.raises(ValueError, match="reserve_cpus"):
        config.load(path)
