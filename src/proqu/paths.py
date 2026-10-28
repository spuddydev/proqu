"""Where proqu keeps its state."""

import os
from pathlib import Path

DB_NAME = "proqu.db"


class StateDirError(Exception):
    pass


def state_path() -> Path:
    """Where the state directory is, without creating it."""
    xdg = os.environ.get("XDG_STATE_HOME", "")
    try:
        if os.environ.get("PROQU_HOME"):
            return Path(os.path.abspath(os.environ["PROQU_HOME"]))
        if xdg and os.path.isabs(xdg):
            return Path(xdg) / "proqu"
        return Path.home() / ".local" / "state" / "proqu"
    except RuntimeError:
        raise StateDirError("cannot find a home directory, set PROQU_HOME") from None


def state_dir() -> Path:
    """The per user state directory, created private on first use.

    The directory itself is 0700, which keeps every file in it private without
    changing the process umask, so jobs keep the umask the agent gave them.
    """
    path = state_path()
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.stat()
        if info.st_uid != os.geteuid():
            raise StateDirError(f"state directory {path} belongs to another user, set PROQU_HOME")
        if info.st_mode & 0o077:
            path.chmod(0o700)
        (path / "logs").mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise StateDirError(f"cannot use state directory {path}: {error.strerror}") from None
    return path


def db_path() -> Path:
    return state_dir() / DB_NAME


def has_db() -> bool:
    """Whether this user's queue database exists, checked without creating state.

    Any command that opens the queue creates it, so this does not mean a job was submitted.
    """
    return (state_path() / DB_NAME).exists()


def lock_path() -> Path:
    return state_dir() / "lock"


def machine_path() -> Path:
    return state_dir() / "machine.json"


def config_path() -> Path:
    return state_dir() / "config.json"


def log_path(job_id: str) -> Path:
    return state_dir() / "logs" / f"{job_id}.log"


def supervisor_log_path(job_id: str) -> Path:
    return state_dir() / "logs" / f"{job_id}.proqu.log"
