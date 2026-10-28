"""The SQLite job store and the lock that serialises changes to it."""

import fcntl
import json
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from proqu import paths

ACTIVE = ("queued", "starting", "running")
TERMINAL = ("succeeded", "failed", "killed", "cancelled")

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    name TEXT,
    argv TEXT NOT NULL,
    cwd TEXT NOT NULL,
    env TEXT NOT NULL,
    ram INTEGER NOT NULL,
    cpus INTEGER NOT NULL,
    gpus INTEGER NOT NULL,
    vram INTEGER NOT NULL,
    est REAL NOT NULL,
    max REAL NOT NULL,
    bid INTEGER NOT NULL,
    after TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    state TEXT NOT NULL,
    reason TEXT,
    exit_code INTEGER,
    peak_ram INTEGER,
    cpu_ids TEXT NOT NULL,
    gpu_ids TEXT NOT NULL,
    supervisor_pid INTEGER,
    supervisor_start INTEGER,
    job_pid INTEGER,
    cancel_requested INTEGER NOT NULL,
    notified INTEGER NOT NULL
);
CREATE INDEX jobs_session ON jobs (session);
CREATE INDEX jobs_state ON jobs (state);
CREATE TABLE sessions (
    session TEXT PRIMARY KEY,
    bid_spent INTEGER NOT NULL
);
CREATE TABLE waiters (
    job_id TEXT NOT NULL,
    pid INTEGER NOT NULL,
    pid_start INTEGER NOT NULL,
    PRIMARY KEY (job_id, pid)
);
CREATE TABLE held (
    job_id TEXT PRIMARY KEY,
    since REAL NOT NULL
);
"""

# each step takes the schema from the version before it to the next, by index
# held jobs live in their own table, so a supervisor still running from an older
# release keeps reading the jobs table it knows
MIGRATIONS = (
    "",
    "CREATE TABLE held (job_id TEXT PRIMARY KEY, since REAL NOT NULL)",
)

# how long a command waits for another to release the queue lock
LOCK_TIMEOUT = 60.0


class LockTimeout(Exception):
    pass


class StoreError(Exception):
    pass


# columns held as JSON text in the database
_JSON = ("argv", "env", "after", "cpu_ids", "gpu_ids")


@dataclass
class Job:
    id: str
    session: str
    name: str | None
    argv: list[str]
    cwd: str
    env: dict[str, str]
    ram: int
    cpus: int
    gpus: int
    vram: int
    est: float
    max: float
    bid: int
    after: list[str]
    submitted_at: float
    state: str = "queued"
    started_at: float | None = None
    ended_at: float | None = None
    reason: str | None = None
    exit_code: int | None = None
    peak_ram: int | None = None
    cpu_ids: list[int] = field(default_factory=list)
    gpu_ids: list[int] = field(default_factory=list)
    supervisor_pid: int | None = None
    supervisor_start: int | None = None
    job_pid: int | None = None
    cancel_requested: bool = False
    notified: bool = False


_COLUMNS = [f.name for f in fields(Job)]


def _to_row(job: Job) -> dict:
    row = asdict(job)
    for name in _JSON:
        row[name] = json.dumps(row[name])
    return row


def _from_row(row: sqlite3.Row) -> Job:
    values = dict(row)
    for name in _JSON:
        values[name] = json.loads(values[name])
    values["cancel_requested"] = bool(values["cancel_requested"])
    values["notified"] = bool(values["notified"])
    return Job(**values)


class Store:
    def __init__(
        self, db: Path | None = None, lock: Path | None = None, lock_timeout: float = LOCK_TIMEOUT
    ):
        self.lock_timeout = lock_timeout
        self.db = db or paths.db_path()
        self.lock_file = lock or paths.lock_path()
        try:
            self.conn = sqlite3.connect(self.db, timeout=30, isolation_level=None)
        except sqlite3.Error as error:
            raise StoreError(f"cannot open queue database {self.db}: {error}") from None
        self.conn.row_factory = sqlite3.Row
        try:
            if self._version() == 0:
                with self._flock():
                    if self._version() == 0:
                        self._create()
            if 0 < self._version() < SCHEMA_VERSION:
                with self._flock():
                    self._migrate()
            version = self._version()
        except sqlite3.DatabaseError as error:
            raise StoreError(
                f"queue database {self.db} is damaged ({error}), move it aside to start afresh"
            ) from None
        if version != SCHEMA_VERSION:
            raise StoreError(
                f"queue database {self.db} has version {version}, this proqu understands "
                f"{SCHEMA_VERSION}: update proqu"
            )

    def _version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _create(self) -> None:
        """Create the schema in one transaction, so a crash leaves nothing half made."""
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    self.conn.execute(statement)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def _migrate(self) -> None:
        """Bring an older schema up to date, one version at a time, each in one transaction."""
        while 0 < (version := self._version()) < SCHEMA_VERSION:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(MIGRATIONS[version])
                self.conn.execute(f"PRAGMA user_version = {version + 1}")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the proqu lock and one transaction for a read, decide, write sequence."""
        with self._flock():
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    @contextmanager
    def _flock(self) -> Iterator[None]:
        with open(self.lock_file, "a") as handle:
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise LockTimeout(
                            "another proqu command is holding the queue lock, retry shortly"
                        ) from None
                    time.sleep(0.05)
            yield

    def new_id(self) -> str:
        while True:
            job_id = secrets.token_hex(3)
            if not self.conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone():
                return job_id

    def insert(self, job: Job) -> None:
        row = _to_row(job)
        names = ", ".join(_COLUMNS)
        marks = ", ".join(f":{name}" for name in _COLUMNS)
        self.conn.execute(f"INSERT INTO jobs ({names}) VALUES ({marks})", row)

    def update(self, job_id: str, **changes) -> None:
        if not changes:
            return
        unknown = set(changes) - set(_COLUMNS)
        if unknown:
            raise KeyError(f"unknown job fields {sorted(unknown)}")
        for name in _JSON:
            if name in changes:
                changes[name] = json.dumps(changes[name])
        assignments = ", ".join(f"{name} = :{name}" for name in changes)
        self.conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = :job_id", {**changes, "job_id": job_id}
        )

    def transition(
        self, job_id: str, from_states: tuple[str, ...], to_state: str, **changes
    ) -> bool:
        """Move a job to a new state only if it is in one of from_states.

        Returns whether the move happened, so two processes racing to change
        the same job cannot both act on it. Call it inside locked().
        """
        marks = ", ".join("?" for _ in from_states)
        row = self.conn.execute(
            f"SELECT 1 FROM jobs WHERE id = ? AND state IN ({marks})", (job_id, *from_states)
        ).fetchone()
        if not row:
            return False
        self.update(job_id, state=to_state, **changes)
        return True

    def get(self, job_id: str, session: str | None = None) -> Job | None:
        """One job, or None. With a session, jobs owned by others are invisible."""
        query = "SELECT * FROM jobs WHERE id = ?"
        params: tuple = (job_id,)
        if session is not None:
            query += " AND session = ?"
            params += (session,)
        row = self.conn.execute(query, params).fetchone()
        return _from_row(row) if row else None

    def for_session(self, session: str) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE session = ? ORDER BY submitted_at, id", (session,)
        )
        return [_from_row(row) for row in rows]

    def active(self) -> list[Job]:
        """Every queued, starting and running job across all sessions."""
        marks = ", ".join("?" for _ in ACTIVE)
        rows = self.conn.execute(
            f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY submitted_at, id", ACTIVE
        )
        return [_from_row(row) for row in rows]

    def states(self, job_ids: list[str]) -> dict[str, str]:
        marks = ", ".join("?" for _ in job_ids)
        rows = self.conn.execute(f"SELECT id, state FROM jobs WHERE id IN ({marks})", job_ids)
        return {row["id"]: row["state"] for row in rows}

    def bid_spent(self, session: str) -> int:
        row = self.conn.execute(
            "SELECT bid_spent FROM sessions WHERE session = ?", (session,)
        ).fetchone()
        return row["bid_spent"] if row else 0

    def add_bid_spent(self, session: str, amount: int) -> None:
        self.conn.execute(
            "INSERT INTO sessions (session, bid_spent) VALUES (?, ?) "
            "ON CONFLICT (session) DO UPDATE SET bid_spent = bid_spent + excluded.bid_spent",
            (session, amount),
        )

    def held_since(self) -> dict[str, float]:
        """When each queued job was first held back for free memory, by id."""
        rows = self.conn.execute("SELECT job_id, since FROM held")
        return {row["job_id"]: row["since"] for row in rows}

    def mark_held(self, job_id: str, since: float) -> None:
        """Record that a job was held back, keeping the first time it was."""
        self.conn.execute(
            "INSERT OR IGNORE INTO held (job_id, since) VALUES (?, ?)", (job_id, since)
        )

    def forget_held(self) -> None:
        """Drop the held time of every job that is no longer queued."""
        self.conn.execute(
            "DELETE FROM held WHERE job_id NOT IN (SELECT id FROM jobs WHERE state = 'queued')"
        )

    def add_waiter(self, job_id: str, pid: int, pid_start: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO waiters (job_id, pid, pid_start) VALUES (?, ?, ?)",
            (job_id, pid, pid_start),
        )

    def remove_waiter(self, job_id: str, pid: int) -> None:
        self.conn.execute("DELETE FROM waiters WHERE job_id = ? AND pid = ?", (job_id, pid))

    def waiters(self, job_id: str) -> list[tuple[int, int]]:
        rows = self.conn.execute("SELECT pid, pid_start FROM waiters WHERE job_id = ?", (job_id,))
        return [(row["pid"], row["pid_start"]) for row in rows]
