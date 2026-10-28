"""Where proqu keeps its state."""

import os
from pathlib import Path


class StateDirError(Exception):
    pass


def state_dir() -> Path:
    """The per user state directory, created private on first use."""
    xdg = os.environ.get("XDG_STATE_HOME", "")
    if os.environ.get("PROQU_HOME"):
        path = Path(os.environ["PROQU_HOME"])
    elif xdg and os.path.isabs(xdg):
        path = Path(xdg) / "proqu"
    else:
        path = Path.home() / ".local" / "state" / "proqu"
    # the directory holds environments captured from agents, so keep it private
    os.umask(0o077)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.geteuid():
        raise StateDirError(f"state directory {path} belongs to another user, set PROQU_HOME")
    if info.st_mode & 0o077:
        path.chmod(0o700)
    (path / "logs").mkdir(mode=0o700, exist_ok=True)
    return path


def db_path() -> Path:
    return state_dir() / "proqu.db"


def lock_path() -> Path:
    return state_dir() / "lock"


def machine_path() -> Path:
    return state_dir() / "machine.json"


def config_path() -> Path:
    return state_dir() / "config.json"


def log_path(job_id: str) -> Path:
    return state_dir() / "logs" / f"{job_id}.log"
