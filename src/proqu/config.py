"""User configuration. Resource limits may only lower what the machine has."""

import contextlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from proqu.machine import Machine
from proqu.units import format_duration, format_size, parse_duration, parse_size

DEFAULTS = {
    "bid_budget": 11000,
    "aging_per_minute": 10.0,
    "kill_grace": 10,
    "length_delay_cap": 4 * 3600,
    "long_job_after": 12 * 3600,
    "long_job_share": 75,
    "hold_limit": 30 * 60,
}

# What an unset reserve means, as shown by `proqu config show`.
RESERVE_DEFAULTS = {
    "reserve_ram": "10% of the machine",
    "reserve_cpus": "1 on machines with 4 or more CPUs, else 0",
}

DESCRIPTIONS = {
    "ram": "RAM jobs may use in total, for example 12G",
    "cpus": "CPU cores jobs may use in total",
    "reserve_ram": "RAM kept back from jobs for services and interactive work, 0 for none",
    "reserve_cpus": "CPU cores kept back from jobs for services and interactive work, 0 for none",
    "gpus": "GPU indices jobs may use, for example 0,1, or none",
    "vram": "GPU memory jobs may use on each GPU, for example 20G",
    "bid_budget": "bid points each session may spend",
    "aging_per_minute": "priority points every queued job gains per minute",
    "length_delay_cap": "most a long job is delayed in the queue order, 0 for no delay",
    "long_job_after": "max runtime past which a job counts as long for long_job_share",
    "long_job_share": "percent of RAM, cores and GPU memory long jobs may hold together",
    "hold_limit": (
        "how long a job waiting for memory other programs use may be passed by later jobs, "
        "0 to let them always pass"
    ),
    "kill_grace": "time between the polite and the forced kill, for example 10s",
}


@dataclass(frozen=True)
class Settings:
    bid_budget: int
    aging_per_minute: float
    kill_grace: int
    length_delay_cap: float
    long_job_after: float
    long_job_share: int
    hold_limit: float


def load(path: Path, validate: bool = True) -> dict:
    """Read the config. Without validation a bad value can still be unset."""
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError:
        config = None
    except OSError as error:
        raise ValueError(f"cannot read config file {path}: {error.strerror}") from None
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} is not valid JSON, fix or delete it")
    for key, value in config.items():
        if validate and key in DESCRIPTIONS and not _valid(key, value):
            raise ValueError(
                f"config file {path} has a bad value for {key}, run `proqu config unset {key}`"
            )
    return config


def _valid(key: str, value) -> bool:
    """Whether a value read back from the config file has the right shape."""
    whole = isinstance(value, int) and not isinstance(value, bool)
    if key in ("ram", "vram", "cpus", "kill_grace"):
        return whole and value >= 1
    if key in ("bid_budget", "reserve_ram", "reserve_cpus"):
        return whole and value >= 0
    if key == "long_job_share":
        return whole and 1 <= value <= 100
    if key == "gpus":
        return isinstance(value, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in value
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        # set refuses a zero long_job_after, as it would make every job long
        least = value > 0 if key == "long_job_after" else value >= 0
        return math.isfinite(value) and least
    except OverflowError:
        return False


def save(path: Path, config: dict) -> None:
    """Replace the config file atomically."""
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix="config.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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
    if key == "reserve_ram":
        value = 0 if _is_zero(text, "KMGTB") else parse_size(text)
        if value >= machine.ram:
            raise ValueError(
                f"reserve_ram must be less than the {format_size(machine.ram)} this machine has"
            )
        return value
    if key == "reserve_cpus":
        value = _int(key, text)
        if not 0 <= value < len(machine.cpu_ids):
            raise ValueError(
                f"reserve_cpus must be between 0 and {len(machine.cpu_ids) - 1} on this machine"
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
    if key in ("length_delay_cap", "hold_limit"):
        return 0 if _is_zero(text, "smhd") else parse_duration(text)
    if key == "long_job_share":
        value = _int(key, text.strip().removesuffix("%"))
        if not 1 <= value <= 100:
            raise ValueError("long_job_share must be a percent from 1 to 100")
        return value
    if text.strip() == "0":
        raise ValueError(f"{key} must be more than zero")
    return parse_duration(text)


def _is_zero(text: str, units: str) -> bool:
    """Whether text is zero, bare or with a unit such as 0G or 0s."""
    return re.fullmatch(rf"0+(?:\.0+)?\s*(?:[{units}](?:i?b)?)?", text.strip(), re.I) is not None


def _int(key: str, text: str) -> int:
    if not re.fullmatch(r"-?[0-9]{1,12}", text.strip()):
        raise ValueError(f"{key} must be a whole number")
    return int(text.strip())


def reserve(machine: Machine, config: dict) -> tuple[int, int]:
    """RAM bytes and CPU cores kept back from jobs, never enough to leave nothing."""
    cpus = len(machine.cpu_ids)
    ram = config.get("reserve_ram", machine.ram // 10)
    held = config.get("reserve_cpus", 1 if cpus >= 4 else 0)
    return min(ram, machine.ram - 1), min(held, cpus - 1)


def effective_limits(machine: Machine, config: dict, reserved: bool = True) -> Machine:
    """The machine as jobs see it: minus the reserve, then lowered by configured limits.

    With reserved False the reserve is left out, which shows what a request may be
    compared with to tell a reserve refusal from a config one.
    """
    held_ram, held_cpus = reserve(machine, config) if reserved else (0, 0)
    ram = min(machine.ram - held_ram, config.get("ram", machine.ram))
    cpu_ids = machine.cpu_ids[
        : min(len(machine.cpu_ids) - held_cpus, config.get("cpus", len(machine.cpu_ids)))
    ]
    gpus = machine.gpus
    if "gpus" in config:
        gpus = tuple(gpu for gpu in gpus if gpu.index in config["gpus"])
    if "vram" in config:
        gpus = tuple(replace(gpu, vram=min(gpu.vram, config["vram"])) for gpu in gpus)
    return Machine(ram=ram, cpu_ids=cpu_ids, gpus=gpus)


def describe(key: str, value) -> str:
    """A config value as a person would write it."""
    if key in ("ram", "vram", "reserve_ram"):
        return format_size(value)
    if key in ("kill_grace", "length_delay_cap", "long_job_after", "hold_limit"):
        return format_duration(value)
    if key == "long_job_share":
        return f"{value}%"
    if key == "gpus":
        return ",".join(map(str, value)) or "none"
    return str(value)
