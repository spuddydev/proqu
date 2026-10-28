import json
import re
from pathlib import Path

from proqu import cli, hooks

ROOT = Path(__file__).resolve().parent.parent
SKILL = (ROOT / "skills" / "proqu" / "SKILL.md").read_text()


def subparsers() -> dict:
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.choices and "submit" in a.choices)
    return action.choices


def flags_of(command: str) -> set[str]:
    parser = subparsers()[command]
    return {opt for action in parser._actions for opt in action.option_strings}


def test_hooks_json_runs_each_real_event_inside_its_budget():
    wiring = json.loads((ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    commands = {}
    for event, groups in wiring.items():
        for group in groups:
            for entry in group["hooks"]:
                assert entry["type"] == "command"
                assert entry["timeout"] > hooks.BUDGET
                pattern = r'"\$\{CLAUDE_PLUGIN_ROOT\}/bin/proqu" hook (\S+)'
                match = re.fullmatch(pattern, entry["command"])
                assert match, entry["command"]
                commands[event] = match.group(1)
    assert commands == {
        "SessionStart": "session-start",
        "UserPromptSubmit": "prompt",
        "Stop": "stop",
    }
    assert set(commands.values()) == set(hooks.HANDLERS)


def test_marketplace_lists_the_plugin_from_this_directory():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert [(p["name"], p["source"]) for p in market["plugins"]] == [(plugin["name"], "./")]


def test_every_command_the_skill_shows_exists_with_its_flags():
    commands = subparsers()
    found = 0
    for text in re.findall(r"`(proqu [^`]+)`", SKILL) + re.findall(r"^\s+(proqu .+)$", SKILL, re.M):
        words = text.split("--")[0].split()
        assert words[1] in commands, text
        for flag in re.findall(r"--[a-z]+", text):
            assert flag in flags_of(words[1]), (text, flag)
            found += 1
    assert found >= 10


def test_every_option_in_the_skill_is_real():
    every = set().union(*(flags_of(command) for command in subparsers()))
    for flag in set(re.findall(r"`(--[a-z]+)", SKILL)):
        assert flag in every, flag


def test_the_skill_names_every_agent_option():
    for command in ("submit", "wait", "status", "logs"):
        for flag in flags_of(command) - {"--help", "-h"}:
            assert flag in SKILL, (command, flag)
