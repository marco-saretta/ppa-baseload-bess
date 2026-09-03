# ppa-baseload-bess

Optimization course project: firming a baseload power purchase agreement (PPA)
with a battery energy storage system (BESS).

## Structure

```txt
├── configs/            # Hydra configuration
│   └── config.yaml
├── docs/               # mkdocs documentation
├── notebooks/          # Jupyter notebooks
├── src/
│   └── ppa_baseload_bess/
│       └── __init__.py
├── main.py             # entrypoint (test run)
└── pyproject.toml
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run python main.py
uv run python main.py message="hi" seed=7   # override config from the CLI
```
