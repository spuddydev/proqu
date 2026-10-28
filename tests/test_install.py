import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"


def archive(tmp_path):
    path = tmp_path / "proqu.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in ("bin", "src", "skills", "hooks", ".claude-plugin"):
            tar.add(ROOT / name, f"proqu/{name}")
    return path


def fake_claude(tmp_path, listed):
    folder = tmp_path / "fakebin"
    folder.mkdir(exist_ok=True)
    claude = folder / "claude"
    claude.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {tmp_path / "calls"}\n'
        f'[ "$3" = list ] && echo "{listed}"\n'
        "exit 0\n"
    )
    claude.chmod(0o755)
    return folder


def install(tmp_path, path_dirs, **extra):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"]),
        "PROQU_ARCHIVE": str(archive(tmp_path)),
        **extra,
    }
    return subprocess.run(["sh", str(SCRIPT)], env=env, capture_output=True, text=True)


def test_install_links_the_command_into_a_fixed_place(tmp_path):
    result = install(tmp_path, [], PROQU_NO_PLUGIN="1")
    assert result.returncode == 0, result.stderr
    link = tmp_path / "home" / ".local" / "bin" / "proqu"
    assert link.resolve() == tmp_path / "home" / ".local" / "share" / "proqu" / "bin" / "proqu"
    version = subprocess.run([str(link), "--version"], capture_output=True, text=True)
    assert version.stdout.startswith("proqu ")


def test_install_registers_the_plugin_from_the_same_folder(tmp_path):
    result = install(tmp_path, [fake_claude(tmp_path, "")])
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    home = tmp_path / "home" / ".local" / "share" / "proqu"
    assert f"plugin marketplace add {home}" in calls
    assert "plugin install proqu@proqu" in calls


def test_installing_again_updates_in_place(tmp_path):
    assert install(tmp_path, [], PROQU_NO_PLUGIN="1").returncode == 0
    stale = tmp_path / "home" / ".local" / "share" / "proqu" / "stale"
    stale.write_text("left from the last version")
    result = install(tmp_path, [fake_claude(tmp_path, "proqu")])
    assert result.returncode == 0, result.stderr
    assert not stale.exists()
    calls = (tmp_path / "calls").read_text().splitlines()
    assert "plugin marketplace update proqu" in calls
    assert "plugin update proqu@proqu" in calls
    assert not any(call.startswith("plugin marketplace add") for call in calls)


def test_install_without_claude_says_what_to_run_later(tmp_path):
    result = install(tmp_path, [])
    assert result.returncode == 0, result.stderr
    assert "claude plugin install proqu@proqu" in result.stdout
