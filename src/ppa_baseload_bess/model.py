"""Construction of the optimization model."""

import logging

import gurobipy as gp
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def init_model(cfg: DictConfig) -> gp.Model:
    """Create and return the (still empty) optimization model.

    Decision variables and constraints are added by later build steps.
    """
    model = gp.Model("ppa_baseload_bess")
    model.setParam("Seed", cfg.seed)

    logger.info("Initialized Gurobi model %r", model.ModelName)
    return model
