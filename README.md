# proqu

A shared compute queue for coding agents.

Coding agents such as Claude Code use proqu to run heavy jobs (model training, large builds, long simulations) on a machine they share. Each job states the RAM, CPU cores and GPUs it needs and how long it may run. proqu starts jobs when they fit, enforces those limits, and wakes the agent when its job ends.

The user of proqu is the agent. The person who owns the machine installs it, and can set limits for it.

## Install

Requires Linux and Python 3.10 or newer. Install it as a Claude Code plugin:

```sh
claude plugin marketplace add spuddydev/proqu
claude plugin install proqu@proqu
```

The plugin gives the agent a skill that explains when and how to use the queue, hooks that tell it about finished jobs, and the `proqu` command. The command uses only the Python standard library, so there is nothing else to install.

### Outside Claude Code

Clone the repository and put `bin/proqu` on your PATH:

```sh
git clone https://github.com/spuddydev/proqu
ln -s "$PWD/proqu/bin/proqu" ~/.local/bin/proqu
```

Without the hooks, every caller shares one session called `default`. Set `PROQU_SESSION` to give a caller its own.

## Example

The agent submits a job, then waits for it in the background:

```sh
proqu submit --ram 6G --cpus 4 --est 20m --max 45m -- python train.py
# Queued job 3f9a: 6.0G RAM, 4 CPUs, estimated 20m, killed at 45m.

proqu wait 3f9a   # run as a background command
```

When the job ends, the wait prints the outcome and the end of the log:

```
Job 3f9a succeeded after 18m
Exit code 0, peak RAM 4.8G.
Next: check its output and results before relying on them.
```

A job that breaks its own limit is killed, and the outcome says which limit and why.

## Documentation

- [User guide](docs/guide.md): commands, scheduling, limits, hooks and config.
- [Design notes](docs/design.md): how it works and what it does not promise.

## Status

Version 1. Linux only, NVIDIA GPUs only. macOS and AMD are not supported yet.

## Development

```sh
uv sync
uv run pytest
uv run ruff check --fix && uv run ruff format
uvx pre-commit install --install-hooks
```
