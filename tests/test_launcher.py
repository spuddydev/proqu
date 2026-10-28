import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "bin" / "proqu"


def outside_env(tmp_path) -> dict:
    """An environment where only the launcher can make proqu importable."""
    venv_bin = os.path.join(sys.prefix, "bin")
    path = os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if os.path.realpath(entry) != os.path.realpath(venv_bin)
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PROQU_") and key not in ("PYTHONPATH", "VIRTUAL_ENV")
    }
    env.update(PATH=path, PROQU_HOME=str(tmp_path / "state"), PYTHONNOUSERSITE="1")
    python = shutil.which("python3", path=path)
    if python is None:
        pytest.skip("no python3 outside the virtualenv")
    version = subprocess.run(
        [python, "-c", "import sys; print(sys.version_info >= (3, 10))"],
        capture_output=True,
        text=True,
        env=env,
    )
    if version.stdout.strip() != "True":
        pytest.skip("python3 outside the virtualenv is older than 3.10")
    return env


@pytest.mark.parametrize("via", ["direct", "symlink"])
def test_launcher_runs_from_another_directory(tmp_path, via):
    env = outside_env(tmp_path)
    program = LAUNCHER
    if via == "symlink":
        program = tmp_path / "proqu"
        program.symlink_to(LAUNCHER)
    result = subprocess.run(
        [str(program), "limits"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "What the queue may use on this machine:" in result.stdout


def test_plugin_version_matches_the_package():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{manifest["version"]}"' in re.findall(r'^version = ".*"$', pyproject, re.M)


def test_launcher_refuses_an_old_python():
    code = (
        "import sys; sys.version_info = (3, 9, 0); "
        f"exec(compile(open({str(LAUNCHER)!r}).read(), 'proqu', 'exec'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "proqu needs Python 3.10 or newer" in result.stderr
