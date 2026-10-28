# proqu

A shared compute queue for coding agents.

proqu lets coding agents such as Claude Code queue heavy jobs (model training, large builds, long simulations) on a shared machine. Each job states the RAM, CPU and GPU it needs and how long it may run. proqu packs jobs onto the machine as tightly as it can, enforces those limits hard and wakes the agent when its job ends.

It is in early development.

## Setup

Requires Linux and Python 3.10 or newer. Install as a Claude Code plugin:

```sh
claude plugin marketplace add spuddydev/proqu
claude plugin install proqu@proqu
```

## Usage

Agents use proqu through the bundled skill. They submit a job with its limits, then wait on it in the background:

```sh
proqu submit --ram 6G --cpu 4 --est 20m --max 45m -- python train.py
proqu wait <job id>
```

## Development

```sh
uv sync
uv run pytest
uv run ruff check --fix && uv run ruff format
uvx pre-commit install --install-hooks
```
