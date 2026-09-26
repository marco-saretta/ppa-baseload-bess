"""Construction of the optimization model."""

import logging

import gurobipy as gp
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, cfg: DictConfig, data) -> None:
        """Create the (still empty) optimization model.

        Decision variables and constraints are added by later build steps.
        """
        self.cfg = cfg
        self.data = data
        self.model = gp.Model("ppa_baseload_bess")

        logger.info("Initialized Gurobi model %r", self.model.ModelName)

    def solve(self):
        self.model.optimize()
