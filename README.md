# proqu

A shared compute queue for coding agents.

Coding agents such as Claude Code use proqu to run heavy jobs (model training, large builds, long simulations) on a machine they share. The queue is per Unix user, so it does not arbitrate between users. Each job states the RAM, CPU cores and GPUs it needs and how long it may run. proqu starts jobs when they fit, enforces those limits, and wakes the agent when its job ends. It keeps a small reserve of RAM and one core back for services and interactive work.

The user of proqu is the agent. The person who owns the machine installs it, and can set limits for it.

## Install

Requires Linux and Python 3.10 or newer. Install it as a Claude Code plugin:

```sh
claude plugin marketplace add spuddydev/proqu
claude plugin install proqu@proqu
```

The plugin gives the agent a skill that explains when and how to use the queue, hooks that tell it about finished jobs, and the `proqu` command. The command uses only the Python standard library, so there is nothing else to install.

The plugin puts `proqu` on the agent's PATH, not on yours. The owner needs it to run `proqu config`, so link it into your own PATH. Claude Code keeps installed plugins under `~/.claude/plugins`. Find the command and link it:

```sh
find ~/.claude/plugins -path '*proqu/bin/proqu'
mkdir -p ~/.local/bin
ln -s PATH_FROM_FIND ~/.local/bin/proqu
```

The path usually holds the plugin version, so link again after an update. Or clone the repository as below.

### Without the plugin

Clone the repository and put `bin/proqu` on your PATH:

```sh
git clone https://github.com/spuddydev/proqu
mkdir -p ~/.local/bin
ln -s "$PWD/proqu/bin/proqu" ~/.local/bin/proqu
proqu limits
```

`~/.local/bin` must be on your PATH. The link can be followed from any directory.

Without the hooks, every caller shares one session called `default`. Set `PROQU_SESSION` to give a caller its own.

There is no skill outside Claude Code. An agent or person using it this way should read [the skill](skills/proqu/SKILL.md) first. The guide has a [short version of the rules](docs/guide.md#outside-claude-code).

## Example

The agent submits a job, then waits for it in the background:

```sh
proqu submit --ram 6G --cpus 4 --est 20m --max 45m -- python train.py
# Started job c8d90a: 6.0G RAM, 4 CPUs, estimated 20m, killed at 45m.
# Next: run `proqu wait c8d90a` as a background command so you are woken when it ends. Check on it with `proqu status c8d90a`.

proqu wait c8d90a   # run as a background command
```

On a busy machine the job is queued instead, and submit prints its position and projected start. When the job ends, the wait prints the outcome and the end of the log:

```
Job c8d90a succeeded after 2s
Exit code 0, peak RAM 3.1M.
Next: check its output and results before relying on them.
Last 1 lines of the log (whole log: `proqu logs c8d90a --tail 0`):
epoch 3 done
```

A job that breaks its own limit is killed, and the outcome says which limit and why.

## Documentation

- [User guide](docs/guide.md): commands, scheduling, limits, hooks and config.
- [Design notes](docs/design.md): how it works and what it does not promise.

## Status

The first version, v1. Linux only, NVIDIA GPUs only. macOS and AMD are not supported yet.

## Development

```sh
uv sync
uv run pytest
uv run ruff check --fix && uv run ruff format
uvx pre-commit install --install-hooks
```
