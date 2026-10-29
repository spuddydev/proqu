# proqu

A shared compute queue that lets coding agents submit heavy jobs and share a machine fairly.

## Specification

### Purpose

- Coding agents (Claude Code first) queue medium and heavy compute jobs, such as slow test suites, full builds, data processing, model training and long simulations. Many agents share one machine without overloading it.
- The only real user is an agent. Every interface, message and error is written for an agent to act on.
- General use and open source. Nothing is tied to one machine, user or path. Limits are detected at run time.

### Shape

- Shipped as a Claude Code plugin: a skill that explains when and how to use it, hooks for session identity and notifications, a monitor that wakes the agent when its jobs end, and the `proqu` CLI on the agent's PATH.
- The CLI is Python standard library only. One install script fetches a release, links the command and installs the plugin.
- No resident server. State is one SQLite file under a file lock. Every command runs a scheduling pass, and so does each session's monitor every 10 seconds. Each running job has a short lived supervisor that enforces limits and runs a scheduling pass when the job ends.
- One queue per user per machine.

### Jobs

- A submit gives the command, RAM, CPU cores, an estimated runtime and a maximum runtime. GPU is optional.
- Anything over the machine limit is refused. Limits default to the machine minus a small reserve for services and interactive work. Config can change the reserve or only lower the limits further.
- Limits are hard. RAM and CPU use cgroups when available, with a watch and kill fallback. macOS has no cgroups and always watches. A job that asked for no GPU may not use one.
- A job is killed when it hits its maximum runtime.
- Every kill is reported to the agent with a reason that makes clear it was the job's fault.
- A job may depend on another of the same session's jobs. It never starts first, and fails if its parent fails.

### Scheduling

- Priority is the bid plus 10 points per minute waited, for every job. No bid counts as 0. A job counts as submitted later by a tenth of its maximum runtime, capped at 4 hours by default. With no bids and equal maximum runtimes this is plain first in first out.
- Each session has a bid budget, 11000 by default and configurable.
- Jobs start in priority order. A later job may start ahead only if it fits and its maximum runtime ends before the reserved start of the first blocked job (backfill), so large jobs are not starved.
- Jobs with a maximum runtime over 12 hours together hold at most 75% of RAM, cores and each GPU's memory. Both are configurable. A long job larger than that runs only when no other long job is running.
- A job also starts only when it fits the memory free right now, less what running jobs may still grow into. This covers RAM and GPU memory used by other programs. It only ever delays a start, and status says when a job waits for it.
- Later jobs may pass a job held for that memory, but only for `hold_limit`, 30 minutes by default. After that no later job starts until it has. The clock starts at the first hold and is kept until the job leaves the queue.

### Visibility

- A session sees only its own jobs: state, queue position, its own bids and budget, and projected start and finish from the estimates.
- A session never sees other sessions' jobs or bids.
- The agent is woken when its job ends, whether by success, failure, kill or a failed dependency. A plugin monitor does this for the session. Without it, a background `proqu wait` does.

### Scope

- Linux and macOS. macOS limits are watched, not enforced by the kernel.
- NVIDIA and AMD GPUs on Linux, and the Apple silicon GPU as one slot. AMD is untested on hardware.
- Not for menial tasks such as small helpers or quick checks. Anything over about two minutes may be queued. On a quiet machine it does not matter, on a busy one the queue is respected. The skill says so with examples.
- Services and multi-week jobs never go through proqu.

## Layout

- `src/proqu/` the CLI and scheduler.
- `tests/` pytest suite.
- `docs/` the user guide and design notes.
- `.claude-plugin/` plugin and marketplace manifests.
- `bin/proqu` launcher that runs the CLI from the plugin's files.
- `hooks/hooks.json` hook wiring for the plugin.
- `monitors/monitors.json` the session monitor that wakes the agent.
- `skills/proqu/` the skill agents read.
- `.github/` CI, PR checks and templates.
- `.claude/` project skills and workflows for working on this repo.

## Commands

```
uv sync                                  set up
uv run pytest                            test
uv run ruff check --fix && uv run ruff format   lint and format
uvx pre-commit install --install-hooks   hooks, once per clone
```

## Releasing

- Set the new version in `src/proqu/cli.py`, `pyproject.toml` and `.claude-plugin/plugin.json`, then run `uv lock`.
- Once that is on main, tag main with `vX.Y.Z` and push the tag. The release workflow tests, checks the tag against the version and publishes the archive the installer downloads.

## Branches and merging

- main is the release branch. dev is where work lands. Branch from dev for every change.
- Pull requests into dev are squash only. Pull requests from dev into main are squash or rebase.
- CI runs on pull requests into main and on pushes to main, never on dev. Run the tests and hooks locally before opening a pull request into dev.
- Every commit header and pull request title is conventional, `type: subject`, at most 50 characters, in plain words. A pull request into main fails its checks on any commit that is not.
- Pull requests follow the template. The What and Checks sections must be filled in.
- Issues are filed, claimed and closed through the file-issue skill.

## Prose

British English, plain words, no em dashes and no semicolons. This covers comments, docs, commits, issues and pull requests.
