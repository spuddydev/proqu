import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from proqu.cli import VERSION

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"
PLUGIN_FILES = ["bin/proqu", "src/proqu/cli.py", "skills/proqu/SKILL.md", "hooks/hooks.json"]


def archive(tmp_path):
    path = tmp_path / "proqu.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in ("bin", "src", "skills", "hooks", ".claude-plugin"):
            tar.add(ROOT / name, f"proqu/{name}")
    return path


def fake_claude(tmp_path, marketplace=None, update_fails=False):
    """A claude that logs its calls and lists a proqu marketplace at the given path."""
    folder = tmp_path / "fakebin"
    folder.mkdir(exist_ok=True)
    listing = [{"name": "proqu", "source": "directory", "path": str(marketplace)}]
    claude = folder / "claude"
    claude.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {tmp_path / "calls"}\n'
        f"[ \"$3\" = list ] && echo '{json.dumps(listing if marketplace else [])}'\n"
        f'[ "$2" = update ] && exit {1 if update_fails else 0}\n'
        "exit 0\n"
    )
    claude.chmod(0o755)
    return folder


def install(tmp_path, path_dirs=(), **extra):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"]),
        "PROQU_ARCHIVE": str(archive(tmp_path)),
        **extra,
    }
    return subprocess.run(["sh", str(SCRIPT)], env=env, capture_output=True, text=True)


def calls(tmp_path):
    return (tmp_path / "calls").read_text().splitlines()


def default_home(tmp_path):
    return tmp_path / "home" / ".local" / "share" / "proqu"


def test_install_links_the_command_into_a_fixed_place(tmp_path):
    result = install(tmp_path, PROQU_NO_PLUGIN="1")
    assert result.returncode == 0, result.stderr
    link = tmp_path / "home" / ".local" / "bin" / "proqu"
    assert link.resolve() == default_home(tmp_path) / "bin" / "proqu"
    version = subprocess.run([str(link), "--version"], capture_output=True, text=True)
    assert version.stdout == f"proqu {VERSION}\n"


def test_install_folders_can_be_chosen(tmp_path):
    files, links = tmp_path / "files", tmp_path / "links"
    result = install(tmp_path, PROQU_NO_PLUGIN="1", PROQU_DIR=str(files), PROQU_BIN_DIR=str(links))
    assert result.returncode == 0, result.stderr
    assert (links / "proqu").resolve() == files / "bin" / "proqu"
    assert not default_home(tmp_path).exists()


def test_install_registers_the_plugin_from_the_same_folder(tmp_path):
    result = install(tmp_path, [fake_claude(tmp_path)])
    assert result.returncode == 0, result.stderr
    assert f"plugin marketplace add {default_home(tmp_path)}" in calls(tmp_path)
    assert "plugin install proqu@proqu" in calls(tmp_path)


def test_installing_again_updates_in_place(tmp_path):
    assert install(tmp_path, PROQU_NO_PLUGIN="1").returncode == 0
    stale = default_home(tmp_path) / "stale"
    stale.write_text("left from the last version")
    result = install(tmp_path, [fake_claude(tmp_path, default_home(tmp_path))])
    assert result.returncode == 0, result.stderr
    assert not stale.exists()
    assert "plugin marketplace update proqu" in calls(tmp_path)
    assert "plugin update proqu@proqu" in calls(tmp_path)
    assert not any(call.startswith("plugin marketplace add") for call in calls(tmp_path))


def test_a_failed_plugin_update_falls_back_to_installing_it(tmp_path):
    assert install(tmp_path, PROQU_NO_PLUGIN="1").returncode == 0
    claude = fake_claude(tmp_path, default_home(tmp_path), update_fails=True)
    result = install(tmp_path, [claude])
    assert result.returncode == 0, result.stderr
    assert "plugin install proqu@proqu" in calls(tmp_path)


def test_a_marketplace_from_elsewhere_is_left_alone(tmp_path):
    clone = tmp_path / "clone"
    result = install(tmp_path, [fake_claude(tmp_path, clone)])
    assert result.returncode == 0, result.stderr
    assert calls(tmp_path) == ["plugin marketplace list --json"]
    assert f"comes from {clone}" in result.stdout


def test_a_folder_the_installer_did_not_make_is_never_replaced(tmp_path):
    mine = default_home(tmp_path)
    mine.mkdir(parents=True)
    (mine / "work").write_text("not proqu")
    result = install(tmp_path, PROQU_NO_PLUGIN="1")
    assert result.returncode == 1
    assert "was not made by this installer" in result.stderr
    assert (mine / "work").read_text() == "not proqu"


def test_install_without_claude_says_what_to_run_later(tmp_path):
    result = install(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "claude plugin install proqu@proqu" in result.stdout


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git repository")
def test_release_archive_holds_the_plugin_and_no_dev_files():
    data = subprocess.run(
        ["git", "archive", "--format=tar.gz", "--prefix=proqu/", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = set(tar.getnames())
        launcher = tar.getmember("proqu/bin/proqu")
    for name in [*PLUGIN_FILES, ".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"]:
        assert f"proqu/{name}" in names
    assert launcher.mode & 0o111
    assert not any(name.startswith(("proqu/tests", "proqu/.claude/")) for name in names)
    assert "proqu/CLAUDE.md" not in names
