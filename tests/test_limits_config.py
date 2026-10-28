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
