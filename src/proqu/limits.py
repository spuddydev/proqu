"""Hard limits on a running job.

The supervisor asks its monitor about the job once a poll. A monitor answers
with the reason the job must be stopped, or None while it is within limits.
"""

from proqu import procs
from proqu.store import Job
from proqu.units import format_size

OWN_LIMIT = "this was the job's own limit"


def ram_reason(limit: int, peak: int) -> str:
    return (
        f"used more than its RAM limit of {format_size(limit)} (peak {format_size(peak)}): "
        f"{OWN_LIMIT}, reduce its memory use or resubmit with a larger --ram if it truly needs it"
    )


class Monitor:
    """Watches RAM by summing the memory of every process in the job."""

    def __init__(self, job: Job, session: int):
        self.job = job
        self.session = session
        self.peak = 0

    def check(self) -> tuple[str, str] | None:
        used = sum(memory_of(pid) for pid in procs.job_members(self.job.id, self.session))
        self.peak = max(self.peak, used)
        if used > self.job.ram:
            return "killed", ram_reason(self.job.ram, used)
        return None

    def finish(self) -> tuple[str, str] | None:
        """Evidence found only once the job has ended."""
        return None


def memory_of(pid: int) -> int:
    """Proportional memory of a process, so shared pages are not counted twice."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0
