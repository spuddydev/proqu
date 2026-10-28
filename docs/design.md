# proqu design notes

How proqu works, why, and what it does not promise.

## Shape

proqu is a command line tool that uses only the Python standard library. It installs with one script, or as a plain Claude Code plugin, and needs nothing else. It has no server, no daemon and no network.

## No resident server

Nothing runs between commands. The queue moves on in a scheduling pass, and a pass runs:

- in `submit`, `status`, `cancel` and `wait`. `logs`, `limits` and `config` run none
- in every hook once the database exists, using only the cached machine snapshot and skipped if there is none, so a hook stays quick
- when a supervisor ends, which is when a job ends
- every 10 seconds inside `proqu wait`

Each pass:

1. Finds jobs whose supervisor has gone and ends them.
2. Fails queued jobs that can never start.
3. Starts the jobs that fit the limits and the memory free right now. See [Free memory at start](#free-memory-at-start).

Trade-offs:

- No server to install, supervise or restart.
- A crash or reboot is only noticed at the next proqu command. Until then the queue is stale. The first command afterwards ends the lost jobs as "supervisor lost" and carries on. A reboot is spotted from the boot id. A crash is spotted because the supervisor's pid and start time no longer match a live process.
- If nobody runs a command and no job ends, nothing starts. In practice a waiting agent keeps passes running.
- A broken config file stalls scheduling until it is fixed. No queued job starts and lost jobs are not ended. `status` and `submit` report it, and a `wait` on a queued job exits 4.

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
- The queue is per user. proqu does not share a queue between users, so it cannot protect one user's jobs from another's. The free memory check at start partly helps, because it sees memory other users hold. It does not see their CPU use, and it does not help once a job has started.

## A reserve below the machine limits

The queue may use the machine minus a reserve. By default that is 10% of RAM and one core on a machine with 4 or more cores. The owner can change it with `reserve_ram` and `reserve_cpus`.

Trade-offs:

- Services and interactive work keep some room, so a full queue does not freeze the machine or the owner's own session.
- Jobs cannot use the whole machine, so a request near the machine's size is refused. The refusal names the reserve and says only the owner should change it.
- The default is a guess. It is too much on a dedicated box, which is why it is config. On a busy machine the free memory check now covers memory others use, so the reserve matters less there.
- The reserve is never allowed to take everything. Jobs always keep some RAM and a core.

## Long jobs count as submitted later

Priority is the bid plus waiting time. A job also counts as submitted later by a tenth of its `--max`, at most 4 hours by default, set by `length_delay_cap`. So with no bids a job with a short `--max` tends to go ahead of a long one submitted a little earlier.

Why reorder rather than refuse long jobs:

- A multi-day training run is a normal and honest request. Refusing it would turn away real work, and any limit for "too long" is arbitrary.
- The delay is capped, so a long job that has waited long enough still goes first. It is slowed, not starved.
- Short jobs finish and free the machine sooner, so more jobs get through.
- A job that understates its `--max` to gain order is killed at it.

Trade-offs:

- The delay only changes who starts first. Once a long job runs it holds its share until it ends or reaches its `--max`.
- A job's `--max` is the agent's own figure. An agent that overstates it only delays itself.
- A tenth is a rule of thumb, not a measured fit. It is not tuned to any one workload.
- It is counted in aging, so with `aging_per_minute` at 0 it does nothing.

### Not done yet: a cap on the share long jobs may hold

A few long jobs can fill the machine for days, and short jobs then wait the whole time whatever their priority. The delay does not help once a long job is running. A cap would limit how much of RAM, cores and GPUs jobs with a long `--max` may hold together.

It is deferred. What counts as long, how big the share is and whether it is a config key are not decided. It is tracked in issue #7.

## Free memory at start

The limits say what the queue may use. They do not say what is free. When a pass is about to start a job, it also reads the memory free now and starts the job only if its request fits. For RAM that is `MemAvailable` less what running jobs may still grow into, which is each job's `--ram` less its use now. A starting job counts its whole request. For GPUs it is the free memory per GPU from `nvidia-smi`, less the unused part of running jobs' `--vram`, and the job is placed on a GPU with room. CPU is not checked, because more work than cores only slows things down.

A job held back for this stays queued. It never fails for it. Projections, reservations and backfill use the limits alone, so the check only ever delays a start. A held job counts as the first blocked job.

Trade-offs:

- Free memory is read afresh at each pass that is about to start a job, and never cached. It costs a read of `/proc/meminfo` and one look through `/proc` for the processes of running jobs. For GPU jobs it also runs `nvidia-smi` for the free memory and, when jobs already hold GPUs, again for their use. It happens with the queue locked, so the free memory read gives up after 5 seconds.
- Hooks never read GPU memory, because `nvidia-smi` is too slow. A GPU job starts only in a pass from a command, a supervisor ending, or the 10 second pass in `proqu wait`. If the free GPU memory cannot be read, GPU jobs wait. If only the use of running jobs cannot be read, each counts its whole `--vram`. If `MemAvailable` cannot be read, the limits alone decide on RAM.
- It is a check at start time and it is racy. Another program can take the memory a moment later, and nothing stops it after the job has started.
- Later jobs can go ahead of a held job if they leave room for it within the limits and fit the free memory. Stopping the whole queue behind it could idle the machine for as long as another user holds the memory. The agent can see the wait in status and cancel.
- No reserve or margin is taken off the free figure. The reserve applies to the limits only. Taking it off twice would leave a job sized to the limit unable to start even on an idle machine.
- A job's use is its proportional set size. File backed pages counted in it make the figure slightly wrong.
- In a container, free memory is the host's `MemAvailable`. The limits already respect a cgroup memory limit.

## Limits stop mistakes, not evasion

The limits are there to catch an honest agent that got an estimate wrong. They are not a sandbox. A job that wants to escape can:

- leave the job's session with `setsid` and clear its environment. In watch mode the supervisor finds a job's processes by session and by the job id in their environment, so a process that has done both is no longer seen. In cgroup mode it stays in the scope, so it still counts against the cap and is killed when the job ends. GPU checks use the same lookup, so it escapes them in both modes
- call `systemd-run` itself and start work outside the job's scope
- in watch mode, hold memory in tmpfs or shared memory, which is not counted against the process

Do not run proqu as a security boundary against code you do not trust.

## Watch mode samples once a second

In watch mode RAM is read once a second. A spike that rises and falls between two samples is not seen. A fast allocation can also overshoot the limit before the next sample ends it. Cgroup mode does not have this problem, because the kernel enforces the cap.

CPU is not capped in watch mode. Each job is only pinned to its cores, and a job can move itself onto cores given to others. Cgroup mode adds a CPU quota. Cgroup mode is preferred, and `proqu limits` says which one is in use.

Memory is summed as proportional set size, so a page shared between processes counts once.

## GPU checks are best effort

GPU use is checked by running `nvidia-smi` every 5 seconds, because it is too slow to run every second. A job can use a GPU for less than 5 seconds between checks without being seen. In a container with its own pid namespace, the pids that `nvidia-smi` reports may not match the pids proqu sees. Then GPU limits cannot be enforced reliably. If `nvidia-smi` fails, GPU limits are not checked, and the supervisor log says so. There is no kernel enforcement of GPU memory.

## Logs and the database are never pruned

A job's output goes to a log file with no size cap. A job that prints without end fills the disk. The database keeps every job forever, and so do the logs. `proqu status` hides all but the last 10 finished jobs, but nothing is deleted. The owner can remove old files by hand from the state directory.

Each job's whole environment, captured at submit, is stored in the database too. So secrets in the environment stay on disk for as long as the database does. The state directory is private, mode 0700, which limits who can read them.

## Sessions

A session is the unit of visibility, bidding and notification.

- Under Claude Code the SessionStart hook sets `PROQU_SESSION` to the Claude Code session id.
- Outside Claude Code every caller shares the session `default`, so they all see and can cancel each other's jobs.
- The owner has no view across sessions. `status --all` covers only the caller's session. To see or stop an agent's jobs the owner reads the database for the session id and runs the command with `PROQU_SESSION` set to it. The guide shows how.
- `/clear` or forking a session gives a new session id. The new session cannot see the earlier one's jobs, bids or budget. Jobs from the earlier session keep running and finish. Nobody is woken for them.
- A subagent shares its parent's session and can see the same jobs. A background `proqu wait` started by a subagent ends when the subagent does. The parent must run its own wait for any job it needs to hear about. The Stop hook then reminds it if a job is unwatched.

## Notifications

The agent hears about a finished job through a background `proqu wait`. The hooks are a safety net. SessionStart, Prompt and Stop report any outcome the wait did not, and each outcome is reported once. Each says what happened, what to do next and the command for the log. A `proqu wait` on a job that already ended prints the outcome again. The Stop hook blocks the agent from finishing while a job is unwatched, and lists the exact wait and cancel commands for each such job. It blocks once. Claude Code sets `stop_hook_active` on the retry, and the hook then lets the agent stop. So a job can still be left unwatched if the agent ignores the reminder.

A hook ends itself after 8 seconds, inside the 10 second timeout the plugin sets. If the environment file is missing, SessionStart cannot export the session id. It shows the id instead, and says to put it in front of every command.

## Not supported yet

- macOS. Linux only, because it uses `/proc`, cgroups and systemd.
- AMD GPUs. NVIDIA only, through `nvidia-smi`.
