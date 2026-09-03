# ppa-baseload-bess

Optimization course project: firming a baseload power purchase agreement (PPA)
with a battery energy storage system (BESS).

The goal is to size and schedule a battery so that a variable renewable
generation profile can be delivered as a flat, baseload contract. The
optimization model is built with `gurobipy`, configured with Hydra, and run
through a single entrypoint (`main.py`).

## Structure

```txt
├── configs/            # Hydra configuration
│   └── config.yaml
├── data/               # input data (folder tracked, contents ignored)
├── docs/               # mkdocs documentation
├── notebooks/          # Jupyter notebooks
├── src/
│   └── ppa_baseload_bess/
│       ├── __init__.py
│       ├── model.py    # optimization model construction
│       └── paths.py    # project filesystem paths
├── main.py             # entrypoint: build the model from the config
└── pyproject.toml
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.13
(Hydra 1.3 is not yet compatible with 3.14).

```bash
uv sync
```

## Run

```bash
uv run python main.py
uv run python main.py seed=7   # override config from the CLI
```

Each run writes logs and the resolved config to `outputs/<date>/<time>/`.
