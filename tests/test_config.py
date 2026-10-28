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
        bid_budget=11000,
        aging_per_minute=10.0,
        kill_grace=10,
        length_delay_cap=4 * 3600,
        long_job_after=12 * 3600,
        long_job_share=75,
        hold_limit=30 * 60,
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
    given = {
        "bid_budget": 5,
        "aging_per_minute": 1.5,
        "kill_grace": 3,
        "length_delay_cap": 60,
        "long_job_after": 7200,
        "long_job_share": 50,
        "hold_limit": 600,
    }
    assert config.settings(given) == config.Settings(**given)


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


# The length delay cap


@pytest.mark.parametrize(("text", "stored"), [("2h", 7200), ("90m", 5400), ("0", 0)])
def test_set_length_delay_cap(text, stored):
    assert config.set_value({}, "length_delay_cap", text, MACHINE) == {"length_delay_cap": stored}


@pytest.mark.parametrize("text", ["0G", "0B", "0.0M"])
def test_set_reserve_ram_accepts_zero_with_a_unit(text):
    assert config.set_value({}, "reserve_ram", text, MACHINE) == {"reserve_ram": 0}


@pytest.mark.parametrize("text", ["0s", "0h", "0m"])
def test_set_length_delay_cap_accepts_zero_with_a_unit(text):
    assert config.set_value({}, "length_delay_cap", text, MACHINE) == {"length_delay_cap": 0}


@pytest.mark.parametrize("text", ["0", "0s"])
def test_set_kill_grace_refuses_zero(text):
    with pytest.raises(ValueError, match="more than zero"):
        config.set_value({}, "kill_grace", text, MACHINE)


def test_set_length_delay_cap_refuses_nonsense():
    with pytest.raises(ValueError, match="cannot read duration"):
        config.set_value({}, "length_delay_cap", "long", MACHINE)


def test_length_delay_cap_zero_is_valid_in_the_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"length_delay_cap": 0}')
    assert config.load(path) == {"length_delay_cap": 0}
    assert config.settings({"length_delay_cap": 0}).length_delay_cap == 0


def test_describe_length_delay_cap():
    assert config.describe("length_delay_cap", 4 * 3600) == "4h"


@pytest.mark.parametrize(("text", "stored"), [("50", 50), ("50%", 50), ("100", 100), (" 1 ", 1)])
def test_set_long_job_share(text, stored):
    assert config.set_value({}, "long_job_share", text, MACHINE) == {"long_job_share": stored}


@pytest.mark.parametrize("text", ["0", "101", "half", "0.5", "-5"])
def test_set_long_job_share_refuses_what_is_not_a_percent(text):
    with pytest.raises(ValueError):
        config.set_value({}, "long_job_share", text, MACHINE)


@pytest.mark.parametrize("value", [0, 101, 0.5, True, "75"])
def test_long_job_share_out_of_range_in_the_file_is_refused(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"long_job_share": value}))
    with pytest.raises(ValueError, match="long_job_share"):
        config.load(path)


def test_set_long_job_after():
    assert config.set_value({}, "long_job_after", "6h", MACHINE) == {"long_job_after": 6 * 3600}
    with pytest.raises(ValueError):
        config.set_value({}, "long_job_after", "0", MACHINE)


def test_long_job_settings_read_as_a_person_would_write_them():
    assert config.describe("long_job_after", 12 * 3600) == "12h"
    assert config.describe("long_job_share", 75) == "75%"


def test_zero_long_job_after_in_the_file_is_refused(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"long_job_after": 0}')
    with pytest.raises(ValueError, match="long_job_after"):
        config.load(path)
    path.write_text('{"long_job_after": 3600}')
    assert config.load(path) == {"long_job_after": 3600}


@pytest.mark.parametrize(("text", "stored"), [("45m", 2700), ("2h", 7200), ("0", 0), ("0s", 0)])
def test_set_hold_limit(text, stored):
    assert config.set_value({}, "hold_limit", text, MACHINE) == {"hold_limit": stored}


@pytest.mark.parametrize("text", ["off", "30", "-5m"])
def test_set_hold_limit_refuses_nonsense(text):
    with pytest.raises(ValueError):
        config.set_value({}, "hold_limit", text, MACHINE)


def test_hold_limit_in_the_file_is_checked(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"hold_limit": 0}')
    assert config.settings(config.load(path)).hold_limit == 0
    path.write_text('{"hold_limit": -1}')
    with pytest.raises(ValueError, match="hold_limit"):
        config.load(path)


def test_describe_hold_limit():
    assert config.describe("hold_limit", 1800) == "30m"
