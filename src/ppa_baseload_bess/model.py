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
        self.m = gp.Model("ppa_baseload_bess")
        logger.info("Initialized Gurobi model %r", self.model.ModelName)

    def build_model(self):
        self.add_parameters()
        self.add_variables()
        self.add_constraints()
        self.add_objective()

    def add_parameters(self):
        logger.info("Start adding parameters")
        self.d_G = None
        self.d_L = None
        #self.spot_prices = self.data.

        logger.info("End adding parameters")

    def add_variables(self):
        logger.info("Start adding variables")
        # self.u_G = self.m.addMVar(shape=(self.data.scenarios))
        # self.u_L = self.m.addMVar(shape=(self.data.scenarios))
        # self.w_L = self.m.addMVar(shape=(self.data.scenarios))
        # self.w_G = self.m.addMVar(shape=(self.data.scenarios))
        # self.S = self.m.addVar(name="S")
        self.M = self.m.addVar(name="M")
        logger.info("End adding variables")

    def add_constraints(self):
        logger.info("Start adding contraints")

        logger.info("End adding contraints")

    def add_objective(self):
        logger.info("Start adding objective")

        logger.info("End adding objective")

    def solve(self):
        self.model.optimize()
