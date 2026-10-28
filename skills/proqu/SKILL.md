---
name: proqu
description: Use before running any compute heavy job on this machine, such as training or fine tuning a model, model inference over a large dataset, a large compile (kernel, LLVM, a full release build), long simulations or benchmarks, big data processing or hyperparameter sweeps, and anything that needs a GPU or will take more than a few minutes or many GB of RAM. proqu is the shared compute queue that runs it fairly beside other agents. Covers estimating resources, submitting, waiting, reading outcomes and bids.
---

# proqu: the shared compute queue

This machine is shared by several agents. proqu is one queue per user that all of them use, so heavy jobs run one after another instead of fighting over memory, cores and GPUs. Every job states what it needs, and the limits are hard: a job that goes over its own request is killed.

Never run heavy work outside the queue on a shared machine. It starves everyone else's jobs and can bring the machine down.

## When to use it

Use proqu for work that is heavy or long:

- training or fine tuning a model
- model inference over a large dataset
- a large compile such as a kernel, LLVM or a full release build
- long simulations or benchmarks
- big data processing
- hyperparameter sweeps (one job per run, or one job running the sweep)

Do not use it for quick or light work. Run these directly:

- quick scripts, tests that take seconds, linting, formatting
- git, package installs, small builds
- anything under about a minute that uses little memory

Use judgement between the two. A five minute build on a few cores is fine either way. Anything that needs a GPU goes through the queue.

Never put these through proqu:

- services: anything that stays up serving or watching at low or medium load, such as dev servers, databases, file watchers or bots. A queue slot would be held for as long as the service lives, and the job would be killed at its max runtime.
- multi-week jobs. Nothing here should hold the machine that long.

A multi-day training run is a normal queue job. Give it an honest `--max`, and expect it to start after shorter jobs, as described under Bids.

## Estimate honestly

Every request has to be right, because the queue plans around it and enforces it.

- `--ram`: peak resident memory of the whole job, children included. Measure it on a small run with `/usr/bin/time -v CMD` (Maximum resident set size) or by watching it, then scale up. A little over is safer than being killed, but large overestimates keep others waiting.
- `--cpus`: cores the job will really keep busy, for example the `-j` of a build or the worker count of a data loader. Not the core count of the machine.
- `--gpus` and `--vram`: number of GPUs and memory needed on each, as `nvidia-smi` reports it during a short run. Give both together, the CLI rejects one without the other. A job that asked for no GPU is killed if it touches one, so set these whenever the code may use CUDA.
- `--est`: your honest expected runtime. The queue uses it to plan and to show projected start times.
- `--max`: a generous overestimate, about 1.5 to 3 times `--est`. The job is killed when it reaches max, so a hang cannot hold resources forever. Too tight and a slow run is lost, too loose and it blocks planning.

Check what the queue allows with `proqu limits`. A request over the limit is refused, so split the work or lower it. The queue keeps a share of RAM and a core back for services and interactive work, so it allows a little less than the machine has. Do not change that reserve yourself, it is the machine owner's choice.

## The flow

1. Submit:

   ```
   proqu submit --ram 6G --cpus 4 [--gpus 1 --vram 10G] --est 20m --max 45m [--name NAME] [--after ID] [--bid N] [--cwd DIR] -- CMD ARGS
   ```

   The command after `--` is run directly, without a shell. For pipes, redirects or `&&`, use `-- bash -c '...'`. The working directory (`--cwd DIR` to choose another) and environment are captured at submit, so activate the virtualenv or export variables first. Sizes take `512M`, `6G`, durations take `90s`, `20m`, `2h`, `1h30m`.

2. Straight away, run `proqu wait <id>` with the Bash tool's `run_in_background` set. You are woken when the job ends, with its outcome and the end of its log. Do not poll `status` in a loop or sleep, the background wait does that for you.

   `proqu wait` exits 0 when the job succeeded and 1 for any other outcome, so a failed or killed job shows up as a failed background command. That is normal, read its output. `proqu wait ID --timeout 30m` stops waiting after that long and exits 3 with the job still going. Exit 4 means the job is still queued but the queue config is broken, so tell the user. Exit 130 or 143 means the wait itself was stopped. After 3, 4, 130 or 143 the job has not ended, so run the wait again.

3. Carry on with other work, or end your turn. If you end a turn with a job nobody waits on, you are asked to start a wait or cancel it.

Other commands:

- `proqu status [ID] [--all]`: your jobs with queue position, projected start and finish, time left, outcomes and your bid budget. `--all` includes every finished job.
- `proqu logs ID [--tail N]`: the job's output so far.
- `proqu cancel ID`: stop a job you no longer need. Do this rather than leaving it queued.
- `proqu limits`: what the queue may use on this machine.

You only ever see your own session's jobs. Another session's job ids read as "no job in this session".

## Handling outcomes

- **succeeded**: exit code 0. Check the output is what you expected before reporting.
- **failed** with an exit code or signal: the command itself failed. Read `proqu logs ID --tail 50`, fix the cause and resubmit. `could not start` means the program or working directory was not found.
- **killed** names the limit the job broke and what it measured. It is always the job's own fault, not the queue's:
  - RAM limit: the job used more than `--ram`. Measure the real peak, or reduce batch size, workers or parallelism.
  - max runtime: the job was still running at `--max`. Check the log for a hang or a stuck download first. Raise `--max` only if it was making real progress.
  - GPU used without a request, a GPU it was not given, or over `--vram`: request the GPU, or measure real GPU memory and request that.
- **cancelled**: you or the user cancelled it.
- **dependency failed**: a job it waited on with `--after` did not succeed. Fix and resubmit the parent first.
- **supervisor lost**: the machine restarted or proqu crashed. Resubmit.

Do not blindly resubmit with bigger numbers. Diagnose first from the log and the reason. Tell the user every outcome that matters to them.

## Dependencies

`--after ID` (repeatable) makes a job wait for another of your jobs. The waiting job never starts first, and fails without running if its parent does not succeed. Use it for pipelines such as preprocess, then train, then evaluate, and wait on the last job (or each, if you want to hear about every step).

## Bids

The default is no bid, and that is almost always right. With no bids the queue is first come first served, except that a long job counts as submitted later, by a tenth of its `--max` and at most 4 hours (by default). So short jobs tend to go first, and a job with a modest `--max` does not wait behind long ones. Small jobs also fill the gaps around large ones.

Each session has a small bid budget (11000 by default) that is spent at submit. Every queued job gains 10 priority points per minute (by default), so a bid only buys a head start over jobs that were waiting before it.

- Bid with `--bid N` on submit, and only when a person is actively blocked waiting on this result right now.
- Never bid to jump the queue for convenience or speed.
- Bid an honest number up to 10000 in proportion to the real urgency.
- You cannot see other sessions' bids. Do not try to infer or game them.

## Subagents

A background command ends with the agent that started it. If a subagent submits a job and then finishes, its wait stops and nobody is woken. The agent that will act on the result should submit and wait itself. A subagent may still submit for its parent, but it must hand back the job id so the parent runs `proqu wait`.
