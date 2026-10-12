import logging
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from omegaconf import DictConfig

from .utils.risk import risk_adjusted  # noqa: F401, used in TODO 5

logger = logging.getLogger(__name__)

# Stochastic model with CVaR: each side's utility is a mix of its expected profit and the
# expected profit in its worst scenarios (CVaR):
#   u = (1 - beta) * E[profit] + beta * CVaR_alpha[profit]
# alpha and beta come from config/simulations/<label>.yaml, block `cvar`, one pair per side.
#
# The task: add the CVaR. Follow the TODOs in order (1 to 6):
#   TODO 1-2  buyer: utility with CVaR and the constraint that defines VaR_aux_L
#   TODO 3-4  generator: the same
#   TODO 5    disagreement points measured with the same utility
#   TODO 6    check the result
# No for loops: use the MVars, @ and numpy operations, as in the profit constraints.
# Run `uv run python main.py` after every TODO and look at the status in the log and at
# results/<label>/scenarios.csv. Until TODO 1 and 3 are done u_L and u_G have no constraint,
# so the objective can grow without limit and Gurobi stops without an optimum (status 13).
#
# TODO 6 - Check the result, with the default config the model is now OPTIMAL:
#   - Set beta = 0 for both sides: you are back to the stochastic model. Which status, and why?
#   - With beta > 0, compute w_G + w_L from variables.csv. Is it still zero? Where does the
#     gain come from? Compare scenarios.csv with and without the contract.
#   - Change tau. How do w_G and w_L split? Does S or M change, and why that one?
#   - Is M unique? Fix M to a few values around the optimum (M_lower = M_upper in the
#     config) and compare w_G + w_L.


class Model:
    def __init__(self, cfg: DictConfig, data) -> None:
        """Create the (still empty) optimization model.

        Decision variables and constraints are added by later build steps.
        """
        self.cfg = cfg  # Get config dictionary
        self.data = data  # Get data object

        # Intialize gurobipy model
        self.setup_gurobi_model()

    def build_model(self):
        self.add_parameters()
        self.add_variables()
        self.add_constraints()
        self.add_objective()

    def setup_gurobi_model(self):
        """Initialise the Gurobi model with solver parameters and the variable namespace."""
        self.m = gp.Model("ppa_baseload_bess")
        self.m.Params.NonConvex = 2  # Allow bilinear terms (S×M, gamma×S)
        self.m.Params.FeasibilityTol = 1e-6  # Constraint violation tolerance
        self.m.Params.OutputFlag = 0  # Suppress Gurobi console output
        self.m.Params.TimeLimit = 420  # Hard stop at 7 minutes
        self.m.Params.DualReductions = 0  # Report INFEASIBLE or UNBOUNDED, not the ambiguous INF_OR_UNBD

        logger.info("Initialized Gurobi model %r", self.m.ModelName)

    def add_parameters(self):
        logger.info("Start adding parameters")
        sc = self.data.sc  # scenario data, every series has shape (T, S)
        self.p = sc.probabilities  # probability of each scenario, shape (S,)
        self.T = self.data.n_timesteps
        self.S = self.data.n_scenarios
        self.dt = self.data.dt  # hours per time step, MW * dt = MWh
        self.spot = sc.spot_prices  # EUR/MWh
        # Sum of the spot price over the horizon in each scenario, shape (S,): the spot value of
        # 1 MW of baseload is dt * spot_sum[s]
        self.spot_sum = self.spot.sum(axis=0)

        # --- Generator ---
        # Solar only. The series is the total for all of Denmark: clip the negative measurements
        # to zero, scale to a 0..1 profile and multiply by the installed capacity of the plant.
        solar = sc.solar.clip(min=0)
        self.P_G = self.data.solar_mw * solar / solar.max()  # MW
        # Disagreement point: revenue from selling all production at spot, without the contract
        self.rev_G = self.dt * (self.spot * self.P_G).sum(axis=0)  # EUR per scenario, shape (S,)
        self.d_G = float(self.p @ self.rev_G)  # expected, EUR

        # --- Buyer ---
        # The series is the whole DK1 business consumption: keep its shape and scale it so
        # that the buyer's average load is the one in the config.
        consumption = sc.consumption
        self.P_L = self.data.load_mw * consumption / consumption.mean()  # MW
        # Disagreement point: cost of buying all the load at spot, without the contract.
        # A cost, so the utility is negative.
        self.cost_L = -self.dt * (self.spot * self.P_L).sum(axis=0)  # EUR per scenario, shape (S,)
        self.d_L = float(self.p @ self.cost_L)  # expected, EUR

        # --- Risk aversion ---
        self.alpha_G, self.beta_G = self.data.alpha_G, self.data.beta_G  # generator
        self.alpha_L, self.beta_L = self.data.alpha_L, self.data.beta_L  # buyer

        # TODO 5 - Disagreement points with CVaR. Do it after TODO 1-4, and run the model before:
        #   which status do you get, and why?
        #   d_G is the utility without the contract, so it must be measured like u_G:
        #     d_G = (1 - beta_G) * E[rev_G] + beta_G * CVaR_alpha_G[rev_G]
        #   rev_G is data, not a variable, so this is a numpy computation. Write it in
        #   utils/risk.py (TODO 5a and 5b), test it on the small example there, then use it
        #   here for both sides, replacing the expected values above:
        # self.d_G = risk_adjusted(self.rev_G, self.p, self.alpha_G, self.beta_G)
        # self.d_L = ...

        logger.info("End adding parameters")

    def add_variables(self):
        logger.info("Start adding variables")
        self.v = SimpleNamespace()  # Namespace for all Gurobi decision variables

        # Contract
        self.v.S = self.m.addMVar(
            shape=(), lb=self.data.strike_lower, ub=self.data.strike_upper, name="S"
        )  # strike price, EUR/MWh
        self.v.M = self.m.addMVar(shape=(), lb=self.data.M_lower, ub=self.data.M_upper, name="M")  # baseload volume, MW

        # Generator
        self.v.profit_G = self.m.addMVar(self.S, lb=-GRB.INFINITY, name="profit_G")  # per scenario, EUR
        self.v.u_G = self.m.addVar(lb=-GRB.INFINITY, name="u_G")  # utility with the contract, EUR

        # Buyer
        self.v.profit_L = self.m.addMVar(self.S, lb=-GRB.INFINITY, name="profit_L")
        self.v.u_L = self.m.addVar(lb=-GRB.INFINITY, name="u_L")

        # Nash bargaining
        self.v.w_L = self.m.addVar(lb=1e-3, name="w_L")
        self.v.w_G = self.m.addVar(lb=1e-3, name="w_G")  # gain over no contract, EUR
        self.v.log_w_G = self.m.addVar(lb=-GRB.INFINITY, name="log_w_G")  # log of the generator's gain
        self.v.log_w_L = self.m.addVar(lb=-GRB.INFINITY, name="log_w_L")

        # CVaR terms
        # VaR: the profit threshold of the worst (1 - alpha) share of scenarios. Free: the buyer's
        # profits are all negative, so its VaR is too.
        # VaR_aux: per scenario, how far the profit falls below VaR (0 if above), EUR.
        self.v.VaR_G = self.m.addVar(lb=-GRB.INFINITY, name="VaR_G")  # EUR
        self.v.VaR_aux_G = self.m.addMVar(self.S, lb=0, name="VaR_aux_G")  # EUR, shape (S,)
        self.v.VaR_L = self.m.addVar(lb=-GRB.INFINITY, name="VaR_L")
        self.v.VaR_aux_L = self.m.addMVar(self.S, lb=0, name="VaR_aux_L")
        logger.info("End adding variables")

    def add_constraints(self):
        logger.info("Start adding contraints")
        v = self.v
        p = self.p
        dt = self.dt
        T = self.T

        # --- Generator ---
        # In scenario s the generator is paid S for the baseload volume M, sells its production
        # at spot and pays the spot price for M:
        #   profit_G[s] = sum_t dt * (S * M + spot[t, s] * (P_G[t, s] - M))
        # The sums over t are already taken (rev_G, spot_sum), so this is one matrix
        # constraint of shape (S,): one row per scenario.
        self.m.addConstr(v.profit_G == dt * (T * v.S * v.M - self.spot_sum * v.M) + self.rev_G, name="profit_G")
        # TODO 3 - Utility of the generator with CVaR, the mirror of TODO 1
        # self.m.addConstr(v.u_G == ..., name="utility_G")
        # TODO 4 - Shortfall of the generator below VaR_G, the mirror of TODO 2
        # self.m.addConstr(v.VaR_aux_G >= ..., name="cvar_aux_G")
        self.m.addConstr(v.w_G == v.u_G - self.d_G, name="gain_G")

        # --- Buyer ---
        # The mirror image: the buyer pays S for M, receives the spot price for M and buys its
        # whole load at spot:
        #   profit_L[s] = sum_t dt * (-S * M + spot[t, s] * (M - P_L[t, s]))
        self.m.addConstr(v.profit_L == dt * (-T * v.S * v.M + self.spot_sum * v.M) + self.cost_L, name="profit_L")
        # TODO 1 - Complete the utility constraint with CVaR:
        #   u_L = (1 - beta_L) * E[profit_L] + beta_L * CVaR_L
        #   CVaR_L = VaR_L - 1 / (1 - alpha_L) * sum_s p[s] * VaR_aux_L[s]
        #   Mind the brackets: in Python 1/1-alpha is 1 - alpha.
        # self.m.addConstr(v.u_L == (1 - self.beta_L) * p @ v.profit_L + self.beta_L * (...), name="utility_L")
        # TODO 2 - Add the aux CVaR constraint. VaR_aux_L[s] is how far profit_L[s] falls below VaR_L:
        #   VaR_aux_L[s] >= VaR_L - profit_L[s]   and   VaR_aux_L[s] >= 0 (already its lower bound)
        #   One matrix constraint of shape (S,), no for loop.
        #   Question: the definition is max(VaR_L - profit_L[s], 0), so why is ">=" enough?
        # self.m.addConstr(v.VaR_aux_L >= v.VaR_L - ..., name="cvar_aux_L")
        self.m.addConstr(v.w_L == v.u_L - self.d_L, name="gain_L")

        # --- Nash bargaining ---
        self.m.addGenConstrLog(v.w_G, v.log_w_G, name="log_gain_G")  # log_w_G = log(w_G)
        self.m.addGenConstrLog(v.w_L, v.log_w_L, name="log_gain_L")

        logger.info("End adding contraints")

    def add_objective(self):
        logger.info("Start adding objective")
        # Nash bargaining: each side's log gain, weighted by its bargaining power
        # bargaining power of the generator, buyer has 1 - tau
        self.obj_expression = self.data.tau * self.v.log_w_G + (1 - self.data.tau) * self.v.log_w_L

        self.m.setObjective(self.obj_expression, GRB.MAXIMIZE)
        logger.info("End adding objective")

    def solve(self):
        logger.info("Solving the model...")
        self.m.optimize()

        if self.m.Status == GRB.INFEASIBLE:
            logger.warning("Model is infeasible, computing the IIS")
            self._compute_iis()
            return

        if self.m.Status != GRB.OPTIMAL:
            logger.warning("Model did not reach optimality, Gurobi status %d", self.m.Status)
            return

        logger.info("Optimal, objective %.6g", self.m.ObjVal)

    def _compute_iis(self):
        """Log a minimal set of constraints and bounds that cannot all hold (the IIS).

        The exporter writes it to results/<label>/model.ilp.
        """
        m = self.m
        m.computeIIS()

        def in_iis(objects, name_attr, iis_attr):
            """Names of the objects that are part of the IIS."""
            names = np.array(m.getAttr(name_attr, objects), dtype=object)
            return names[np.array(m.getAttr(iis_attr, objects), dtype=bool)]

        # Linear, quadratic (here: those with S * M) and general (log) constraints are separate lists in Gurobi
        constrs = np.concatenate(
            [
                in_iis(m.getConstrs(), "ConstrName", "IISConstr"),
                in_iis(m.getQConstrs(), "QCName", "IISQConstr"),
                in_iis(m.getGenConstrs(), "GenConstrName", "IISGenConstr"),
            ]
        )
        bounds = np.concatenate(
            [in_iis(m.getVars(), "VarName", "IISLB") + " lb", in_iis(m.getVars(), "VarName", "IISUB") + " ub"]
        )
        logger.warning("IIS constraints: %s", ", ".join(constrs))
        logger.warning("IIS bounds: %s", ", ".join(bounds))
