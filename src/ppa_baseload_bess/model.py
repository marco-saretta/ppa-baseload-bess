import logging
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Deterministic model: the scenarios are averaged into one, so every time series is a vector
# of length T and every utility is a scalar.
#
# Expected outcome: INFEASIBLE. Without risk aversion the contract is zero sum, w_G + w_L = 0
# for every S and M, so the two gains cannot both be positive. The solver reports the
# conflicting constraints (IIS) in the log and in results/<label>/model.ilp.


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
        sc = self.data.sc  # scenario data

        # Deterministic model: one scenario only, the mean over all scenarios
        self.T = self.data.n_timesteps
        self.dt = self.data.dt  # hours per time step, MW * dt = MWh
        self.spot = sc.spot_prices.mean(axis=1)  # EUR/MWh

        # --- Generator ---
        # Solar only. The series is the total for all of Denmark: clip the negative measurements
        # to zero, scale to a 0..1 profile and multiply by the installed capacity of the plant.
        solar = sc.solar.mean(axis=1).clip(min=0)
        self.P_G = self.data.solar_mw * solar / solar.max()  # MW
        # Disagreement point: revenue from selling all production at spot, without the contract
        self.d_G = float(self.dt * (self.spot * self.P_G).sum())  # EUR

        # --- Buyer ---
        # The series is the whole DK1 business consumption: keep its shape and scale it so
        # that the buyer's average load is the one in the config.
        consumption = sc.consumption.mean(axis=1)
        self.P_L = self.data.load_mw * consumption / consumption.mean()  # MW
        # Disagreement point: cost of buying all the load at spot, without the contract.
        # A cost, so the utility is negative.
        self.d_L = -float(self.dt * (self.spot * self.P_L).sum())  # EUR

        logger.info("End adding parameters")

    def add_variables(self):
        logger.info("Start adding variables")
        self.v = SimpleNamespace()  # Namespace for all Gurobi decision variables

        # --- Contract ---
        # The solver needs finite bounds for the bilinear term S * M
        self.v.S = self.m.addVar(
            lb=self.data.strike_lower, ub=self.data.strike_upper, name="S"
        )  # strike price, EUR/MWh
        self.v.M = self.m.addVar(lb=self.data.M_lower, ub=self.data.M_upper, name="M")  # baseload volume, MW

        # --- Generator ---
        self.v.u_G = self.m.addVar(lb=-GRB.INFINITY, name="u_G")  # utility with the contract, EUR
        # Small positive lower bound: the log of the gain is undefined at 0
        self.v.w_G = self.m.addVar(lb=1e-3, name="w_G")  # gain over no contract, EUR

        # --- Buyer ---
        self.v.u_L = self.m.addVar(lb=-GRB.INFINITY, name="u_L")
        self.v.w_L = self.m.addVar(lb=1e-3, name="w_L")
        # By default, lb=0 in Gurobi. The buyer's utility is mainly negative.

        # --- Nash bargaining ---
        self.v.log_w_G = self.m.addVar(lb=-GRB.INFINITY, name="log_w_G")  # log of the generator's gain
        self.v.log_w_L = self.m.addVar(lb=-GRB.INFINITY, name="log_w_L")

        logger.info("End adding variables")

    def add_constraints(self):
        logger.info("Start adding contraints")
        v = self.v

        # --- Generator ---
        # Every time step the generator is paid S for the baseload volume M and settles the
        # difference between its production and M at the spot price:
        #   u_G = sum_t dt * (S * M + spot_t * (P_G_t - M))
        # written with the sums over t already taken, since S and M do not depend on t.
        self.m.addConstr(
            v.u_G == self.dt * (self.T * v.S * v.M + (self.spot * self.P_G).sum() - self.spot.sum() * v.M),
            name="utility_G",
        )
        self.m.addConstr(v.w_G == v.u_G - self.d_G, name="gain_G")

        # --- Buyer ---
        self.m.addConstr(
            v.u_L == self.dt * (-self.T * v.S * v.M + self.spot.sum() * v.M - (self.spot * self.P_L).sum()),
            name="utility_L",
        )
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
        """Log a minimal set of constraints and bounds that cannot all hold (the IIS) and write it to model.ilp."""
        self.m.computeIIS()
        constrs = [c.ConstrName for c in self.m.getConstrs() if c.IISConstr]
        constrs += [q.QCName for q in self.m.getQConstrs() if q.IISQConstr]
        constrs += [g.GenConstrName for g in self.m.getGenConstrs() if g.IISGenConstr]
        bounds = [f"{x.VarName} lb" for x in self.m.getVars() if x.IISLB]
        bounds += [f"{x.VarName} ub" for x in self.m.getVars() if x.IISUB]
        logger.warning("IIS constraints: %s", ", ".join(constrs))
        logger.warning("IIS bounds: %s", ", ".join(bounds))

        out_dir = Path(self.cfg.paths.results) / self.cfg.simulations.label
        out_dir.mkdir(parents=True, exist_ok=True)
        self.m.write(str(out_dir / "model.ilp"))
        logger.warning("IIS written to %s", out_dir / "model.ilp")
