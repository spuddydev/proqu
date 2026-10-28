# proqu design notes

How proqu works, why, and what it does not promise.

## Shape

proqu is a command line tool that uses only the Python standard library. It needs no install step beyond the plugin. It has no server, no daemon and no network.

## No resident server

Nothing runs between commands. The queue moves on in a scheduling pass, and a pass runs:

- in every command that reads or changes jobs
- in every hook
- when a supervisor ends, which is when a job ends
- every 10 seconds inside `proqu wait`

Each pass:

1. Finds jobs whose supervisor has gone and ends them.
2. Fails queued jobs that can never start.
3. Starts the jobs that fit.

Trade-offs:

- Nothing to install, supervise or restart.
- A crash or reboot is only noticed at the next proqu command. Until then the queue is stale. The first command afterwards ends the lost jobs as "supervisor lost" and carries on. A reboot is spotted from the boot id. A crash is spotted because the supervisor's pid and start time no longer match a live process.
- If nobody runs a command and no job ends, nothing starts. In practice a waiting agent keeps passes running.

## One supervisor per running job

Each running job has a short lived supervisor process. It:

- launches the job in its own session, pinned to its cores
- enforces the max runtime, and watches RAM and GPU use
- stops the job's processes when it must, politely and then by force
- records how the job ended
- runs a scheduling pass last, so the next job starts at once

The supervisor is detached from the command that started it. It lives only as long as its job.

Trade-offs:

- Cheap when the queue is idle, and a stuck supervisor affects only its own job.
- Many running jobs means many small processes.
- Supervisors run outside the job's scope, not inside it. If they ran inside, the job's cap and its out of memory kill would take the supervisor too. A scheduling pass started from inside a job never starts other jobs, for the same reason.
- If a supervisor is killed, its job loses its watcher. The next pass finds the supervisor gone, kills any leftover job processes and marks the job lost.

## SQLite with a file lock

State is one SQLite database in the state directory, with a file lock around every read, decide and write sequence. A command that cannot get the lock for 60 seconds gives up and says to retry. Hooks wait only 3 seconds so a busy queue does not stall a session.

Trade-offs:

- Simple and inspectable, and survives crashes.
- One writer at a time. That is plenty for one queue per user per machine.
- A lock held by a stuck command blocks others until it ends.
- The queue is per user. proqu does not share a queue between users, so it cannot protect one user's jobs from another's.

## Limits stop mistakes, not evasion

The limits are there to catch an honest agent that got an estimate wrong. They are not a sandbox. A job that wants to escape can:

- leave the job's session with `setsid` and clear its environment. The supervisor finds a job's processes by session and by the job id in their environment, so a process that has done both is no longer seen
- call `systemd-run` itself and start work outside the job's scope
- in watch mode, hold memory in tmpfs or shared memory, which is not counted against the process

Do not run proqu as a security boundary against code you do not trust.

## Watch mode samples once a second

In watch mode RAM is read once a second. A spike that rises and falls between two samples is not seen. A fast allocation can also overshoot the limit before the next sample ends it. Cgroup mode does not have this problem, because the kernel enforces the cap. Cgroup mode is preferred, and `proqu limits` says which one is in use.

Memory is summed as proportional set size, so a page shared between processes counts once.

## GPU checks are best effort

GPU use is checked by running `nvidia-smi` every 5 seconds, because it is too slow to run every second. A job can use a GPU for less than 5 seconds between checks without being seen. In a container with its own pid namespace, the pids that `nvidia-smi` reports may not match the pids proqu sees. Then GPU limits cannot be enforced reliably. If `nvidia-smi` fails, GPU limits are not checked, and the supervisor log says so. There is no kernel enforcement of GPU memory.

## Logs and the database are never pruned

A job's output goes to a log file with no size cap. A job that prints without end fills the disk. The database keeps every job forever, and so do the logs. `proqu status` hides all but the last 10 finished jobs, but nothing is deleted. The owner can remove old files by hand from the state directory.

## Sessions

A session is the unit of visibility, bidding and notification.

- Under Claude Code the SessionStart hook sets `PROQU_SESSION` to the Claude Code session id.
- Outside Claude Code every caller shares the session `default`, so they all see and can cancel each other's jobs.
- `/clear` or forking a session gives a new session id. The new session cannot see the earlier one's jobs, bids or budget. Jobs from the earlier session keep running and finish. Nobody is woken for them.
- A subagent shares its parent's session and can see the same jobs. A background `proqu wait` started by a subagent ends when the subagent does. The parent must run its own wait for any job it needs to hear about. The Stop hook then reminds it if a job is unwatched.

## Notifications

The agent hears about a finished job through a background `proqu wait`. The hooks are a safety net. Prompt and Stop report any outcome the wait did not, and each outcome is reported once. The Stop hook blocks the agent from finishing while a job is unwatched, so a job is not left to run with nobody told.

## Not supported yet

- macOS. Linux only, because it uses `/proc`, cgroups and systemd.
- AMD GPUs. NVIDIA only, through `nvidia-smi`.
