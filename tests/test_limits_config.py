from proqu.cli import main


def test_limits_show_the_machine(home, capsys):
    assert main(["limits"]) == 0
    out = capsys.readouterr().out
    assert "RAM: 16.0G" in out
    assert "CPUs: 4" in out
    assert "GPU 0: 24.0G memory" in out
    assert "bid up to 11000" in out


def test_config_set_lowers_limits(home, capsys):
    assert main(["config", "set", "ram", "8G"]) == 0
    assert main(["config", "set", "gpus", "1"]) == 0
    capsys.readouterr()
    main(["limits"])
    out = capsys.readouterr().out
    assert "RAM: 8.0G (of 16.0G, lowered in config)" in out
    assert "GPU 0" not in out
    assert "GPU 1: 12.0G memory" in out


def test_config_set_refuses_raising(home, capsys):
    assert main(["config", "set", "ram", "64G"]) == 2
    assert "can only be lowered" in capsys.readouterr().err


def test_config_set_needs_a_value(home, capsys):
    assert main(["config", "set", "ram"]) == 2
    assert "needs a value" in capsys.readouterr().err


def test_config_unset_restores_default(home, capsys):
    main(["config", "set", "bid_budget", "50"])
    main(["config", "unset", "bid_budget"])
    capsys.readouterr()
    main(["config"])
    assert "bid_budget = 11000" in capsys.readouterr().out


def test_lowered_limit_applies_to_submit(home, capsys):
    main(["config", "set", "ram", "1G"])
    code = main(
        ["submit", "--ram", "2G", "--cpus", "1", "--est", "1m", "--max", "2m", "--", "true"]
    )
    assert code == 2
    assert "more than the queue allows (1.0G)" in capsys.readouterr().err


def test_lowered_budget_applies_to_bids(home, capsys):
    main(["config", "set", "bid_budget", "100"])
    request = ["submit", "--ram", "1G", "--cpus", "1", "--est", "1m", "--max", "2m"]
    code = main([*request, "--bid", "101", "--", "true"])
    assert code == 2
    assert "remaining bid budget of 100" in capsys.readouterr().err


def test_bad_config_value_can_be_unset(home, capsys):
    from proqu import paths

    paths.config_path().write_text('{"ram": -5}')
    assert main(["status"]) == 1
    assert "proqu config unset ram" in capsys.readouterr().err
    assert main(["config", "unset", "ram"]) == 0
    assert main(["status"]) == 0


def test_config_refuses_stray_arguments(home, capsys):
    assert main(["config", "unset", "ram", "8G"]) == 2
    assert main(["config", "show", "ram"]) == 2
    assert main(["config", "set"]) == 2


def test_limits_without_gpus_and_with_ageing(home, capsys):
    main(["config", "set", "gpus", "none"])
    main(["config", "set", "aging_per_minute", "2.5"])
    capsys.readouterr()
    main(["limits"])
    out = capsys.readouterr().out
    assert "GPUs: none" in out
    assert "gains 2.5 priority points per minute" in out


def test_limits_refresh_the_snapshot(home, monkeypatch):
    from proqu import machine

    from .conftest import FAKE_MACHINE

    calls = []
    monkeypatch.setattr(
        machine, "snapshot", lambda refresh=False: calls.append(refresh) or FAKE_MACHINE
    )
    main(["limits"])
    assert calls == [True]


def test_unusable_state_directory_is_a_message(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("PROQU_HOME", str(blocker / "state"))
    assert main(["status"]) == 1
    assert "cannot use state directory" in capsys.readouterr().err


def test_damaged_database_is_a_message(home, capsys):
    from proqu import paths

    paths.db_path().write_bytes(b"not a database" * 200)
    assert main(["status"]) == 1
    assert "damaged" in capsys.readouterr().err


def test_limits_name_the_enforcement(home, capsys, monkeypatch):
    from proqu import limits

    main(["limits"])
    watched = capsys.readouterr().out
    assert "enable-linger" in watched
    assert "GPU use is watched" in watched
    monkeypatch.setattr(limits, "cgroups_available", lambda: True)
    main(["limits"])
    kernel = capsys.readouterr().out
    assert "RAM and CPU are enforced by the kernel" in kernel
    assert "GPU use is watched" in kernel


def test_limits_show_the_default_reserve(home, capsys):
    main(["config", "unset", "reserve_ram"])
    main(["config", "unset", "reserve_cpus"])
    capsys.readouterr()
    main(["limits"])
    out = capsys.readouterr().out
    assert "RAM: 14.4G (of 16.0G, after the reserve)" in out
    assert "CPUs: 3 (of 4, after the reserve)" in out
    assert "Kept in reserve for services and interactive work: 1.5G RAM and 1 CPU cores" in out


def test_limits_show_a_zero_reserve(home, capsys):
    main(["limits"])
    out = capsys.readouterr().out
    assert "RAM: 16.0G\n" in out
    assert "work: 0B RAM and 0 CPU cores" in out


def test_config_show_names_the_reserve_defaults(home, capsys):
    main(["config", "unset", "reserve_ram"])
    main(["config", "unset", "reserve_cpus"])
    capsys.readouterr()
    main(["config", "show"])
    out = capsys.readouterr().out
    assert "reserve_ram = 10% of the machine" in out
    assert "reserve_cpus = 1 on machines with 4 or more CPUs, else 0" in out


def test_config_show_a_set_reserve(home, capsys):
    main(["config", "set", "reserve_ram", "2G"])
    capsys.readouterr()
    main(["config", "show"])
    assert "reserve_ram = 2.0G" in capsys.readouterr().out


def test_limits_describe_the_length_delay(home, capsys):
    main(["limits"])
    assert "a tenth of its --max, at most 4h" in capsys.readouterr().out


def test_limits_leave_out_the_length_delay_when_off(home, capsys):
    main(["config", "set", "length_delay_cap", "0"])
    capsys.readouterr()
    main(["limits"])
    assert "tenth of its --max" not in capsys.readouterr().out


def test_config_show_the_length_delay(home, capsys):
    main(["config", "show"])
    assert "length_delay_cap = 4h" in capsys.readouterr().out
