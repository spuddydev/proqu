# proqu user guide

proqu is a queue for heavy jobs. Agents submit a job with what it needs. proqu starts it when it fits, holds it to its limits and reports how it ended.

## Contents

- [Quick start](#quick-start)
- [Commands](#commands)
- [What not to queue](#what-not-to-queue)
- [Scheduling](#scheduling)
- [Limits](#limits)
- [Kill reasons](#kill-reasons)
- [Hooks](#hooks)
- [Config](#config)
- [For the owner](#for-the-owner)
- [Outside Claude Code](#outside-claude-code)
- [Where things live](#where-things-live)

## Quick start

```sh
proqu limits
proqu submit --ram 6G --cpus 4 --est 20m --max 45m -- python train.py
proqu wait <id>
```

Run `proqu wait` as a background command so the agent is woken when the job ends. The skill that comes with the plugin tells the agent this. Sizes and times always need a unit.

- Sizes: `512M`, `6G`, `1.5T`. A bare number is refused.
- Durations: `90s`, `20m`, `2h`, `1h30m`. Units are lowercase.

## Commands

Every command acts on the current session only. Exit code 2 means the command line was invalid. An unknown job id gives 1 for `status`, `cancel` and `logs`, and so does cancelling a finished job. `wait` has its own table. A job from another session reads as "no job in this session". See [Hooks](#hooks) for how the session is chosen.

### submit

```sh
proqu submit --ram SIZE --cpus N --est TIME --max TIME [options] -- COMMAND ARGS
```

The command goes after `--` and is run directly, with no shell. For pipes, redirects or `&&` use `-- bash -c '...'`. The working directory and environment are captured at submit, so activate virtualenvs and export variables first. The whole environment is stored with the job. See [Where things live](#where-things-live).

| Flag | Meaning |
| --- | --- |
| `--ram SIZE` | Peak RAM of the whole job, children included. Required. |
| `--cpus N` | CPU cores the job keeps busy. Required. |
| `--gpus N` | Number of GPUs. Default 0. |
| `--vram SIZE` | GPU memory needed on each GPU. Required with `--gpus`, and only allowed with it. |
| `--est TIME` | Honest runtime estimate. Used to plan and to project start times. Required. |
| `--max TIME` | Hard runtime limit. The job is killed when it is reached. Required, and not less than `--est`. |
| `--after ID` | Start only after this job succeeds. Repeatable. |
| `--bid N` | Priority bid from 0 to 10000. Default 0. |
| `--name TEXT` | A label of 1 to 64 printable characters. |
| `--cwd DIR` | Working directory. Default is the current one. |

A request is refused at once, with exit code 2, if:

- it asks for more than `proqu limits` shows. If the reserve alone makes it too big, the refusal says so and that only the machine's owner should change it
- `--est` is longer than `--max`
- `--bid` is more than the session's remaining budget
- the program cannot be found, or the directory does not exist
- an `--after` job does not exist in this session or has already ended without succeeding

Example:

```sh
proqu submit --ram 12G --cpus 8 --gpus 1 --vram 10G --est 40m --max 90m \
  --name finetune -- python finetune.py --epochs 3
```

On success it prints whether the job was queued or started, and for a queued job its position and projected start. If the job fails before it starts, it prints the reason and exits 1.

The job runs with:

- standard input closed
- standard output and error written to its log
- `PROQU_JOB_ID` set to its id
- `CUDA_VISIBLE_DEVICES` set to the UUIDs of the GPUs it was given, and empty when it was given none
- its CPUs pinned to the cores it was given

### status

```sh
proqu status [ID] [--all]
```

With no id it lists active jobs, then the ten most recent finished ones. `--all` lists every finished job. With an id it shows one job in full: command, directory, request, bid, dependencies, exit code and peak RAM. Both end with the session's remaining bid budget.

A queued job shows its position, projected start and projected finish. If it is held back only because other programs are using memory, it says "position N in the queue, waiting for memory other programs are using" (or "GPU memory"). It fits the queue and starts once that memory is free, so when it starts is uncertain. Other queued jobs then show "expected to start in X, or later while other programs use the memory". A running job shows how long it has run, when it is expected to end and how long until it is killed.

### wait

```sh
proqu wait ID [--timeout TIME]
```

Waits for the job to end, then prints how it ended, what to do next and the last 20 lines of the log, with the command for the whole log. If the job is queued it first prints its position. Run it in the background. It checks the job every 2 seconds and runs a scheduling pass every 10 seconds.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | The job succeeded. |
| 1 | The job ended any other way (failed, killed or cancelled), there is no such job, or the queue's lock could not be taken. |
| 2 | The command line was invalid. |
| 3 | `--timeout` ran out and the job is still going. |
| 4 | The job is still queued but the queue config is broken, so nothing can start. Tell the owner. |
| 129 | The wait was stopped with SIGHUP, as when its terminal or session went away. |
| 130 | The wait was interrupted with Ctrl-C. |
| 141 | The reader of its output went away (SIGPIPE). |
| 143 | The wait was stopped with SIGTERM. |

After 3, 4, 129, 130, 141 or 143 the job has not ended and the wait should be run again. A wait that stops early prints how to start it again. The job is not affected. A wait on a job that has already ended prints its outcome again, so running it twice is safe.

### logs

```sh
proqu logs ID [--tail N]
```

Prints the job's output. The default is the last 200 lines. `--tail 0` prints the whole log. A tail is taken from the last 256 KiB of the log, so fewer lines come back when lines are long. A progress bar that redraws with carriage returns shows only its last drawing in a tail.

### cancel

```sh
proqu cancel ID
```

A queued job is cancelled at once and its bid is refunded. A starting or running job is stopped within a few seconds. A job that never got as far as running has its bid refunded. One that was running does not. A finished job cannot be cancelled.

### limits

```sh
proqu limits
```

Shows what the queue may use: RAM, CPUs and GPUs, noting when a figure is below the machine's because of the reserve. It also shows which enforcement mode applies, how much is kept in reserve, the bid budget, the aging rate and the length delay. Run it before submitting. It detects the machine afresh.

### config

```sh
proqu config [show]
proqu config set KEY VALUE
proqu config unset KEY
```

For the machine's owner. See [Config](#config).

### hook

`proqu hook session-start|prompt|stop` is the entry point for the Claude Code hooks. It reads JSON on standard input. Nobody runs it by hand, so it is left out of `proqu --help`.

## What not to queue

The skill tells agents never to queue:

- services, such as dev servers, databases, file watchers or bots. They would hold a slot as long as they live and be killed at their max runtime
- multi-week jobs

Quick or light work does not belong in the queue either. A multi-day training run does.

## Scheduling

### Priority

Every queued job has a priority:

```
priority = bid + 10 points for each minute it has waited
```

A long job counts as submitted later than it was, so it has waited less:

```
delay = one tenth of --max, at most 4 hours
```

For example a job with `--max 1h` counts as submitted 6 minutes later, and one with `--max 3d` as 4 hours later. No bid counts as 0, so with no bids and equal `--max` this is plain first in, first out. With no bids and different `--max`, a shorter job tends to go ahead of a longer one submitted a little earlier.

The 10 is the default and the owner can change it with `aging_per_minute`. The 4 hours is the default and the owner can change it with `length_delay_cap`, where `0` turns the delay off. The delay is counted in aging, so it has no effect if `aging_per_minute` is 0. `proqu limits` shows both. Ties go to the job submitted first.

Waiting never stops adding points. A bid is only a head start over jobs that were already waiting, and a long job that has waited long enough still goes first. A queued long job is told at submit how much later it counts as submitted.

### Bid budget

Each session has a budget of 11000 points, set by `bid_budget`. A bid is spent when the job is submitted, up to 10000 on one job. It is refunded if the job is cancelled while queued, or fails before it starts. It is not refunded once the job has run, even if it was lost with its supervisor.

A session sees its own bids and remaining budget. It never sees another session's jobs or bids. Queue positions count every session's jobs, but show nothing about them.

### Starting jobs

Jobs are considered in priority order, with the delay for long jobs included. A job starts when the limits have enough RAM, cores and GPU memory for its whole request. A job never starts with less than it asked for.

It must also fit the memory that is free right now. When a pass is about to start a job, it reads the free RAM from `MemAvailable` in `/proc/meminfo`. It subtracts what running jobs may still grow into, which is each job's `--ram` less what it uses now (a job that is still starting counts its whole request). The job starts only if its `--ram` fits what is left.

GPU memory works the same way. proqu reads the free memory of each GPU from `nvidia-smi`, subtracts the unused part of running jobs' `--vram` on that GPU, and places the job on a GPU that has enough free. Hooks never read GPU memory, because `nvidia-smi` is too slow for them. So a GPU job starts in a pass from a command, from a job ending, or from the 10 second pass in `proqu wait`. If `nvidia-smi` cannot be read, GPU jobs wait, and status says "waiting until the free GPU memory can be read". If `MemAvailable` cannot be read, the limits alone decide on RAM.

Free memory is read afresh at each pass that is about to start a job. Nothing is kept back from it, because the reserve already applies to the limits. CPU is not checked this way, since more work than cores only slows things down.

A job held back by this stays queued. It never fails for it. Other programs using the memory is not the job's fault, and the memory may come free.

### Backfill and the reservation

A job still waiting for a parent is skipped. It neither starts nor holds a reservation, however high its priority. The first other job in order that does not fit is blocked. proqu works out when it will fit, assuming every running job lasts until its `--max`, and reserves that time for it.

A later job may start ahead of it only if:

- it fits now, and
- it will be killed by its `--max` before the reserved time, or it leaves enough room that the blocked job still fits at the reserved time

So small jobs fill the gaps while a large job waits, and the large job is not starved. This is why an honest `--max` matters. A very large `--max` stops a job being used as a filler.

Projected start and finish times assume every job takes its `--est`.

The reservation, backfill and projections use the limits alone. They do not foresee other programs. The free memory check only ever delays a start. A job held for free memory counts as the first blocked job. A later job may go ahead of it only if it leaves room for it within the limits and fits the free memory. So later jobs can still pass a job that waits on other programs. Stopping the whole queue behind it could leave the machine idle for as long as another user holds the memory. Watch the status, and cancel a job that waits too long.

### Dependencies

`--after ID` makes a job wait for another job of the same session. It can be repeated. A job with a dependency:

- never starts before every parent has succeeded
- fails without running if any parent fails, is killed or is cancelled
- fails if a parent of its parent fails, all the way down

A failed dependency is reported with the parent's id. The bid of a job that never ran is refunded. To recover, fix and resubmit the parent, then resubmit the child with the new id.

## Limits

### What the limits are

The queue may use what the machine has, as detected at run time, minus a reserve. The detection includes any memory limit or CPU quota on the cgroup that proqu itself runs in. The usable cores are the ones proqu itself may run on, trimmed to any quota, so their numbers may not start at 0 or be in a row. If an `nvidia-smi` query fails, proqu keeps the GPU list it had before. Anything over the limit is refused at submit. These limits are the ceiling. The memory free at start time is a separate check, described under [Starting jobs](#starting-jobs), and it can only delay a job that fits the limits.

The reserve keeps some of the machine free for services and interactive work:

- RAM: 10% of the machine by default.
- CPUs: 1 core on a machine with 4 or more, else none.
- It never takes everything. Some RAM and at least 1 core are always left for jobs.

The owner can change the reserve with `reserve_ram` and `reserve_cpus`, and can lower the limits further. Lowering `ram` is not needed to protect memory that other users hold, because a job waits for free memory anyway. It still caps what the queue may use in total. See [Config](#config). Agents should not change either.

CPU cores are handed out by number. Each job is pinned to its own cores, so jobs do not share them.

### Two enforcement modes

proqu enforces RAM and CPU in one of two modes. `proqu limits` shows which one applies.

Cgroup mode, enforced by the kernel. Each job runs in its own systemd user scope with:

- a memory cap equal to `--ram`, and no swap
- a CPU quota of 100% per core requested
- out of memory kills inside the scope

It needs all of these:

- `systemd-run` for the user
- a delegated memory controller for user scopes
- linger enabled, with `loginctl enable-linger` once, so the user's scopes outlive their last logout

Watch mode is the fallback. A supervisor adds up the memory of every process in the job once a second and kills the job when the total is over `--ram`. Shared pages are counted once. CPU is limited only by pinning to the assigned cores.

proqu probes for cgroup support and remembers the answer for a day. If a scope fails to start, proqu runs that job in watch mode and forgets the answer, so the next job probes again. If the memory cap was not applied, RAM is watched as well.

### GPU rules

GPU use is checked by watching in both modes, because cgroups cannot do it.

- `--gpus` and `--vram` go together. A job with `--gpus` and no `--vram` is refused.
- `--vram` is per GPU. The job gets that many GPUs, each with at least that much memory.
- The scheduler gives a job the GPUs with the least free memory that still fit, to keep large GPUs free.
- `CUDA_VISIBLE_DEVICES` holds only the GPUs the job was given.
- Every 5 seconds proqu asks `nvidia-smi` which processes use which GPU.
- A job is killed if it uses a GPU while having asked for none, uses a GPU it was not given, or uses more GPU memory than `--vram` on one GPU.
- GPUs in MIG mode are not offered to the queue.
- If `nvidia-smi` cannot be queried, GPU limits are not checked. A line is written to the job's supervisor log.

### Max runtime

A job is killed when it has run for `--max`. Set it to about 1.5 to 3 times `--est`. Too tight loses a slow run. Too loose blocks planning for other jobs.

### How a job is stopped

The whole job, including children that left its session, gets SIGTERM. After `kill_grace` (10 seconds by default) anything left gets SIGKILL.

## Kill reasons

Every kill is reported with the limit and what was measured. A kill is always the job's own fault, never the queue's.

| Reason | What to do |
| --- | --- |
| Used more than its RAM limit | Measure the real peak, or cut batch size, workers or parallelism. Raise `--ram` only if it truly needs more. |
| Still running at its max runtime | Look in the log for a hang first. Raise `--max` only if it was making progress. |
| Used a GPU without asking for one | Resubmit with `--gpus` and `--vram`. |
| Used a GPU that was not given to it | Use only the GPUs in `CUDA_VISIBLE_DEVICES`. |
| Used more GPU memory than its `--vram` | Measure with `nvidia-smi` and resubmit with the real figure. |
| Changed the memory limit of its own cgroup | A job must not change its own limits. Remove that from the job. |

In cgroup mode a job that dies of SIGKILL while its peak RAM was at least 90% of its limit is also reported as a RAM kill, whoever sent the signal.

Jobs can also end without being killed:

| State and reason | Meaning |
| --- | --- |
| failed, exited with code N | The command failed. Read the log. |
| failed, terminated by signal | Something sent the job a signal. |
| failed, could not start | The program or directory was not found, or its CPUs were no longer available on the machine. |
| failed, could not start its supervisor | proqu could not launch the process that watches the job. Resubmit. |
| failed, its cgroup was too slow to start | The job was stopped because it might have half started. Resubmit. |
| failed, could not be stopped | Its processes may be stuck in the kernel. Look at the machine, then resubmit. |
| failed, supervisor lost | The machine restarted or proqu crashed. Not the job's fault. Resubmit. A job that was running keeps its bid spent. |
| failed, a job it waits for ended | A dependency did not succeed. |
| failed, limits were lowered | The owner lowered the limits below its request while it queued. |
| cancelled | You or the user cancelled it. |

## Hooks

The plugin installs three Claude Code hooks. Each one runs a scheduling pass first, but only once the queue's database exists. Before the first `proqu` command creates it, a hook does nothing except SessionStart, which still exports the session id. A hook never breaks the session. Any error becomes one line on standard error and the hook exits 0.

To stay quick:

- A hook pass uses only the cached machine snapshot, whatever its age, and never detects the machine. With no snapshot cached the pass is skipped.
- A hook waits at most 3 seconds for the queue lock.
- A hook gives up after 8 seconds, inside the 10 second timeout in the plugin.
- At most 20 jobs are listed in one message, then a line says how many more.

A finished job is listed on one line: its state and reason, then "Next:" with what to do, then the command for the last 50 lines of its log. A job still active shows only its id and state.

### SessionStart

- Exports `PROQU_SESSION` with the Claude Code session id, so every `proqu` command in the session uses it.
- If the environment file is missing or cannot be written, it shows the session id and says to put `PROQU_SESSION=ID` in front of every `proqu` command. Without that the commands use the shared `default` session.
- Reports jobs of the session that finished without being reported.
- Lists jobs still in the queue and gives the exact `proqu wait ID` command for each.

### UserPromptSubmit

Reports jobs of the session that finished since the agent last heard. Each outcome is reported once.

### Stop

Blocks the agent from stopping while the session has an active job that nobody is waiting on, or an outcome not yet reported. The message lists the exact `proqu wait ID` command for each such job, to run in the background, and the exact `proqu cancel ID` command for each. It blocks once. Claude Code sets `stop_hook_active` on the retry and the hook then lets the agent stop, even if a job is still unwatched. It is a reminder, not a guarantee.

### Sessions

A job belongs to the session in `PROQU_SESSION`. Outside Claude Code the variable is unset and every caller shares the session `default`. `/clear` and forking a session start a new session, which cannot see the old one's jobs.

## Config

Config belongs to the person who owns the machine. Agents should not change it. The resource limits (`ram`, `cpus`, `gpus`, `vram`) can only be lowered below what the machine has. The reserve can be set to any amount that leaves something for jobs.

```sh
proqu config
proqu config set ram 12G
proqu config set gpus 0,1
proqu config set reserve_ram 4G
proqu config set reserve_cpus 0
proqu config unset ram
```

| Key | Meaning | Default |
| --- | --- | --- |
| `ram` | RAM jobs may use in total, for example `12G`. | the machine |
| `cpus` | CPU cores jobs may use in total. | the machine |
| `reserve_ram` | RAM kept back from jobs, for example `4G`. `0` for none. | 10% of the machine |
| `reserve_cpus` | CPU cores kept back from jobs. `0` for none. | 1 on 4 or more CPUs, else 0 |
| `gpus` | GPU indices jobs may use, for example `0,1`, or `none`. | the machine |
| `vram` | GPU memory jobs may use on each GPU. | the machine |
| `bid_budget` | Bid points each session may spend. | 11000 |
| `aging_per_minute` | Priority points a queued job gains per minute. | 10 |
| `length_delay_cap` | Most a long job is delayed in the queue order. `0` for no delay. See [Priority](#priority). | 4h |
| `kill_grace` | Time between the polite and the forced kill. Must be more than zero. | 10s |

A `ram`, `cpus`, `gpus` or `vram` above what the machine has is refused, and so is a reserve that would leave nothing for jobs. Jobs already running keep what they were given. A queued job that no longer fits fails with a reason that says the limits were lowered. If the reserve alone is what stops it fitting, the reason names the reserve. A job held back only for free memory does not fail. It waits.

If the config file is broken, commands report it and the queue waits for a fix. `proqu config unset KEY` removes a bad value.

## For the owner

The owner is the person who owns the machine. The install script links `proqu` into `~/.local/bin`, so the owner can run it too.

Each session sees only its own jobs, and `proqu status --all` covers only the caller's session. There is no command to list sessions or see everything. The owner can still look and act:

- List every job and its session by reading `proqu.db` in the state directory, for example with Python's `sqlite3` module. The `jobs` table has `id`, `session` and `state`.
- Then run any command as that session, for example `PROQU_SESSION=SESSION proqu cancel ID`.
- Or end the processes directly. The supervisor notices and the next command tidies the queue.

An owner running `proqu` without `PROQU_SESSION` uses the shared `default` session and sees none of this.

## Outside Claude Code

There is no skill, so there are no hooks and no reminders. A person or another agent should read `skills/proqu/SKILL.md` in the repository. The essential rules are:

- Queue only heavy work, such as training, large builds and long simulations. Never queue services or quick tasks.
- Measure RAM and cores honestly before submitting, for example with `/usr/bin/time -v`. Use `nvidia-smi` for GPU memory.
- Set `--max` to 1.5 to 3 times `--est`.
- Run `proqu wait ID` so you hear when the job ends. It must stay running, for example in a background process. It exits 0 only if the job succeeded.
- Set `PROQU_SESSION` to a name of your own, or everyone shares `default`.
- Read the outcome. A kill is the job's fault and says what to change.
- Do not change the config.

State is per user. The directory is, in order:

1. `$PROQU_HOME`
2. `$XDG_STATE_HOME/proqu`
3. `~/.local/state/proqu`

It is created private (mode 0700).

| Path | Holds |
| --- | --- |
| `proqu.db` | Jobs, bid budgets and waiters, in SQLite. |
| `lock` | The lock that serialises changes. |
| `logs/ID.log` | The job's output. |
| `logs/ID.proqu.log` | The supervisor's own messages. Look here if a job fails oddly. |
| `config.json` | Settings from `proqu config`. |
| `machine.json` | The detected machine, cached for an hour. |
| `cgroups.json` | The cgroup probe result, cached for a day. |
| `boot_id` | Used to notice a restart. |

Logs and the database are never pruned. The database holds each job's whole environment from submit, so any secret in it stays on disk. Keep tokens out of the environment where you can. The directory is private, mode 0700. See the [design notes](design.md).
