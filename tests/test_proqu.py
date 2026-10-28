import proqu


def test_package_imports():
    assert callable(proqu.main)


def test_module_entry_point_exits_with_the_command_status(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ, PROQU_HOME=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "proqu", "cancel", "ffffff"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "no job ffffff" in result.stderr
