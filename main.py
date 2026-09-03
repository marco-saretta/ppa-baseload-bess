"""Entrypoint: build the optimization model from the Hydra config.

Usage::

    uv run python main.py
    uv run python main.py seed=7   # override config from the CLI
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ppa_baseload_bess import init_model
from ppa_baseload_bess.paths import DATA_DIR

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve the config and construct the model as a smoke test."""
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg).rstrip())
    logger.info("Data directory: %s", DATA_DIR)

    model = init_model(cfg)
    logger.info("Model %r is ready (%d variables)", model.ModelName, model.NumVars)


if __name__ == "__main__":
    main()
