"""User configuration. Resource limits may only lower what the machine has."""

import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from proqu.machine import Machine
from proqu.units import format_duration, format_size, parse_duration, parse_size

DEFAULTS = {"bid_budget": 11000, "aging_per_minute": 10.0, "kill_grace": 10}

DESCRIPTIONS = {
    "ram": "RAM jobs may use in total, for example 12G",
    "cpus": "CPU cores jobs may use in total",
    "gpus": "GPU indices jobs may use, for example 0,1, or none",
    "vram": "GPU memory jobs may use on each GPU, for example 20G",
    "bid_budget": "bid points each session may spend",
    "aging_per_minute": "priority points every queued job gains per minute",
    "kill_grace": "time between the polite and the forced kill, for example 10s",
}


@dataclass(frozen=True)
class Settings:
    bid_budget: int
    aging_per_minute: float
    kill_grace: int


def load(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError:
        config = None
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} is not valid JSON, fix or delete it")
    for key, value in config.items():
        if key in DESCRIPTIONS and not _valid(key, value):
            raise ValueError(
                f"config file {path} has a bad value for {key}, run `proqu config unset {key}`"
            )
    return config


def _valid(key: str, value) -> bool:
    """Whether a value read back from the config file has the right shape."""
    whole = isinstance(value, int) and not isinstance(value, bool)
    if key in ("ram", "vram", "cpus", "kill_grace"):
        return whole and value >= 1
    if key == "bid_budget":
        return whole and value >= 0
    if key == "gpus":
        return isinstance(value, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in value
        )
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def save(path: Path, config: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def settings(config: dict) -> Settings:
    return Settings(**{key: config.get(key, default) for key, default in DEFAULTS.items()})


def set_value(config: dict, key: str, text: str, machine: Machine) -> dict:
    """Return a copy of config with key set, refusing anything above the machine."""
    if key not in DESCRIPTIONS:
        raise ValueError(f"unknown config key {key!r}, known keys: {', '.join(DESCRIPTIONS)}")
    value = _parse(key, text, machine)
    return {**config, key: value}


def unset(config: dict, key: str) -> dict:
    if key not in DESCRIPTIONS:
        raise ValueError(f"unknown config key {key!r}, known keys: {', '.join(DESCRIPTIONS)}")
    return {k: v for k, v in config.items() if k != key}


def _parse(key: str, text: str, machine: Machine):
    if key == "ram":
        value = parse_size(text)
        if value > machine.ram:
            raise ValueError(
                f"ram can only be lowered: this machine has {format_size(machine.ram)}"
            )
        return value
    if key == "cpus":
        value = _int(key, text)
        if not 1 <= value <= len(machine.cpu_ids):
            raise ValueError(f"cpus must be between 1 and {len(machine.cpu_ids)} on this machine")
        return value
    if key == "gpus":
        if text.strip().lower() == "none":
            return []
        indices = sorted({_int(key, part) for part in text.split(",")})
        known = {gpu.index for gpu in machine.gpus}
        unknown = [i for i in indices if i not in known]
        if unknown:
            raise ValueError(
                f"no GPU with index {unknown[0]} on this machine, known: {sorted(known)}"
            )
        return indices
    if key == "vram":
        value = parse_size(text)
        largest = max((gpu.vram for gpu in machine.gpus), default=0)
        if value > largest:
            raise ValueError(
                f"vram can only be lowered: the largest GPU has {format_size(largest)}"
            )
        return value
    if key == "bid_budget":
        value = _int(key, text)
        if value < 0:
            raise ValueError("bid_budget cannot be negative")
        return value
    if key == "aging_per_minute":
        try:
            value = float(text) if text.strip().isascii() else math.nan
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise ValueError("aging_per_minute must be a number")
        if value < 0:
            raise ValueError("aging_per_minute cannot be negative")
        return value
    return parse_duration(text)


def _int(key: str, text: str) -> int:
    if not re.fullmatch(r"-?[0-9]{1,12}", text.strip()):
        raise ValueError(f"{key} must be a whole number")
    return int(text.strip())


def effective_limits(machine: Machine, config: dict) -> Machine:
    """The machine as jobs see it after configured limits are applied."""
    ram = min(machine.ram, config.get("ram", machine.ram))
    cpu_ids = machine.cpu_ids[: config.get("cpus", len(machine.cpu_ids))]
    gpus = machine.gpus
    if "gpus" in config:
        gpus = tuple(gpu for gpu in gpus if gpu.index in config["gpus"])
    if "vram" in config:
        gpus = tuple(replace(gpu, vram=min(gpu.vram, config["vram"])) for gpu in gpus)
    return Machine(ram=ram, cpu_ids=cpu_ids, gpus=gpus)


def describe(key: str, value) -> str:
    """A config value as a person would write it."""
    if key in ("ram", "vram"):
        return format_size(value)
    if key == "kill_grace":
        return format_duration(value)
    if key == "gpus":
        return ",".join(map(str, value)) or "none"
    return str(value)
