import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB
from omegaconf import OmegaConf

from ..utils.logger import get_logger
from ..utils.risk import cvar, risk_adjusted

log = get_logger(__name__)

# Gurobi status code -> name, for the statuses this model can end with
STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


class DataExporter:
    """Write the results of a model run to results/<label>/:
    config.yaml       the settings the run used
    results.csv       solver status, solve time and objective
    variables.csv     value of every scalar decision variable            (only if solved)
    scenarios.csv     one row per scenario: probability, profit of each side without the
                      contract, and every variable indexed by scenario with
                      the gain over no contract                          (variables only if solved)
    spot_prices.csv   model inputs as matrices, index t and one column per scenario s0, s1, ...,
    P_G.csv           the same layout as the input files in data/scenarios/<label>/
    P_L.csv
    model.ilp         the conflicting constraints (IIS)                  (only if infeasible)

    The folder is emptied first, so it always describes a single run.
    """

    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.model = model
        self.has_solution = model.m.SolCount > 0

    def run(self) -> None:
        self.set_paths()
        self.read_variables()
        self.export_config()
        self.export_results()
        self.export_variables()
        self.export_scenarios()
        self.export_timeseries()
        self.export_iis()
        self.postprocess()
        self.sweep_M()
        self.sweep_row()
        log.info(f"Exported results to {self.out_dir}")

    def set_paths(self) -> None:
        self.out_dir = Path(self.cfg.paths.results) / self.cfg.simulations.label
        shutil.rmtree(self.out_dir, ignore_errors=True)  # remove the files of an earlier run
        self.out_dir.mkdir(parents=True)

    def read_variables(self) -> None:
        """Split the solution by variable name: scalars ("S", "u_G") and per scenario ("profit_G[3]")."""
        if not self.has_solution:
            return
        m = self.model.m
        all_vars = m.getVars()
        values = pd.Series(m.getAttr("X", all_vars), index=m.getAttr("VarName", all_vars), name="value")

        parts = values.index.str.extract(r"^(?P<variable>\w+)(?:\[(?P<scenario>\d+)\])?$")
        is_scalar = parts["scenario"].isna().to_numpy()

        self.scalars = values[is_scalar]
        self.scalars.index.name = "variable"

        # One column per variable, one row per scenario
        per_scenario = parts[~is_scalar].assign(value=values.to_numpy()[~is_scalar])
        self.per_scenario = per_scenario.pivot(index="scenario", columns="variable", values="value")
        self.per_scenario.index = self.scenario_names(self.per_scenario.index.astype(int))

    def export_config(self) -> None:
        (self.out_dir / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))

    def export_results(self) -> None:
        m = self.model.m
        results = pd.Series(
            {
                "status": m.Status,
                "status_name": STATUS_NAMES.get(m.Status, "OTHER"),
                "solve_time_s": m.Runtime,
                "objective": m.ObjVal if self.has_solution else np.nan,
            },
            name="value",
        )
        results.index.name = "key"
        results.to_csv(self.out_dir / "results.csv")

        if not self.has_solution:
            log.warning(f"No solution to export (Gurobi status {results['status_name']})")

    def export_variables(self) -> None:
        if self.has_solution:
            self.scalars.to_csv(self.out_dir / "variables.csv")

    def export_scenarios(self) -> None:
        """Per-scenario profits in EUR. Without the contract they are data, so always available;
        with the contract they are variables and need the solution."""
        model = self.model
        df = pd.DataFrame(
            {
                "probability": model.p,
                "profit_G_no_contract": model.rev_G,
                "profit_L_no_contract": model.cost_L,
            },
            index=self.scenario_names(np.arange(model.S)),
        )
        if self.has_solution:
            df = df.join(self.per_scenario)
            df["gain_G"] = df["profit_G"] - df["profit_G_no_contract"]
            df["gain_L"] = df["profit_L"] - df["profit_L_no_contract"]
        df.index.name = "scenario"
        df.to_csv(self.out_dir / "scenarios.csv")

    def export_timeseries(self) -> None:
        self.write_matrix("spot_prices", self.model.spot)
        self.write_matrix("P_G", self.model.P_G)
        self.write_matrix("P_L", self.model.P_L)

    def write_matrix(self, name: str, values) -> None:
        """One (T, S) matrix per file. A deterministic model's (T,) series counts as one scenario."""
        matrix = np.asarray(values).reshape(self.model.T, -1)
        df = pd.DataFrame(matrix, columns=self.scenario_names(np.arange(matrix.shape[1])))
        df.index.name = "t"
        df.to_csv(self.out_dir / f"{name}.csv")

    def export_iis(self) -> None:
        # The model computes the IIS when it finds the problem infeasible
        if self.model.m.Status == GRB.INFEASIBLE:
            self.model.m.write(str(self.out_dir / "model.ilp"))
            log.info(f"IIS written to {self.out_dir / 'model.ilp'}")

    def postprocess(self) -> None:
        """ Prepare all the values the plots need in data_visualizer.py 

        earnings_G.csv
        earnings_G_per_scenario.csv
        """
        if not self.has_solution:
            return #There is not solution for S and M

        model   = self.model
        strike  = float(model.v.S.X) #S*, EuR/MWh
        volume  = float(model.v.M.X) #M*, MW

        #Earnings per timestep per scenario (Same cash flows as the profit G constraint)
        market   = model.spot * model.P_G # EUR/h, (T, S)
        contract = (strike - model.spot) * volume #EUR/h, (T, S)
        earnings = model.dt * (market + contract) # EUR per time step, (T, S)

        # Time series: x axis in hours
        per_step = pd.DataFrame(earnings, columns=self.scenario_names(np.arange(model.S))) #(model.T, model.S)
        per_step.index = np.arange(model.T) * model.dt
        per_step.index.name = "hour"
        per_step.to_csv(self.out_dir / "earnings_G.csv")

        #Total earnings per scenario in MEUR IN ORDER from lowest to highest
        totals = pd.Series(earnings.sum(axis=0)/1e6, index=self.scenario_names(np.arange(model.S)), name="earnings_G_MEUR") #1D array (model.S,)
        totals.index.name = "scenario"
        totals.sort_values().to_csv(self.out_dir / "earnings_G_per_scenario.csv")

        #Expected Earnings & Title names
        summary = pd.Series({"S":strike, "M":volume, "Expected Earnings":float(model.p @ totals.to_numpy())}, name="value")
        summary.index.name = "key"
        summary.to_csv(self.out_dir / "plot_summary.csv")

    def sweep_M(self) -> None:
        """Gains and risk removed as a function of the baseload volume M, for surplus_vs_M.png.

        surplus_vs_M.csv     index M (MW); W, w_G, w_L, risk_G, risk_L in kEUR
        surplus_summary.csv  maximum and flat top of W, the peaks of each curve, the solver's W
        """
        if not self.has_solution:
            return  # S* comes from the solution

        model = self.model
        p = model.p
        strike = float(model.v.S.X)  # S*, EUR/MWh: panel B keeps the price fixed here
        H = model.dt * model.T  # hours in the horizon
        M_grid = np.linspace(0, 1.6 * model.data.load_mw, 321)  # volumes to try, MW

        def profits(M, S):
            """Profit per scenario of each side for a contract (M, S): the model's profit constraints."""
            profit_G = model.rev_G + M * (H * S - model.dt * model.spot_sum)  # (S,)
            profit_L = model.cost_L + M * (model.dt * model.spot_sum - H * S)  # (S,)
            return profit_G, profit_L

        def risk(x, alpha):
            """R = CVaR - E: how much worse the tail is than the average (negative), EUR."""
            return cvar(x, p, alpha) - float(p @ x)

        # Risk of each side without the contract (M = 0)
        R_G0 = risk(model.rev_G, model.alpha_G)
        R_L0 = risk(model.cost_L, model.alpha_L)

        rows = []
        for M in M_grid:
            profit_G, profit_L = profits(M, strike)
            # Gain over no contract, measured with the model's utility (1 - beta) E + beta CVaR
            w_G = risk_adjusted(profit_G, p, model.alpha_G, model.beta_G) - model.d_G
            w_L = risk_adjusted(profit_L, p, model.alpha_L, model.beta_L) - model.d_L
            rows.append(
                {
                    "M": M,
                    "W": w_G + w_L,  # total gain: the strike cancels in the sum
                    "w_G": w_G,  # generator's gain with S fixed at S*
                    "w_L": w_L,  # buyer's gain with S fixed at S*
                    "risk_G": model.beta_G * (risk(profit_G, model.alpha_G) - R_G0),  # risk removed
                    "risk_L": model.beta_L * (risk(profit_L, model.alpha_L) - R_L0),
                }
            )
        sweep = pd.DataFrame(rows).set_index("M") / 1e3  # EUR -> kEUR (the index M is not divided)
        sweep.to_csv(self.out_dir / "surplus_vs_M.csv")

        # Flat top: every M where W reaches its maximum
        is_top = np.isclose(sweep["W"], sweep["W"].max(), rtol=1e-6)
        summary = pd.Series(
            {
                "W_max": sweep["W"].max(),  # kEUR
                "flat_M_low": sweep.index[is_top].min(),  # MW
                "flat_M_high": sweep.index[is_top].max(),  # MW
                "W_solver": (model.v.w_G.X + model.v.w_L.X) / 1e3,  # the solver's total gain, kEUR
                "M_peak_G": sweep["w_G"].idxmax(),  # panel B peaks, MW
                "M_peak_L": sweep["w_L"].idxmax(),
                "M_pref_G": sweep["risk_G"].idxmax(),  # panel C preferred sizes, MW
                "M_pref_L": sweep["risk_L"].idxmax(),
            },
            name="value",
        )
        summary.index.name = "key"
        summary.to_csv(self.out_dir / "surplus_summary.csv")

    def sweep_row(self) -> None:
        """Add this run's key results as one row to results/sweep_<name>.csv (only for sweep runs).

        The file lives in results/ itself, not in the run's folder, so it collects all runs of the sweep.
        Hydra runs the jobs one after the other, so appending a row is safe."""
        
        if "sweep_name" not in self.cfg:
            return  # a normal run, not part of a sweep

        m, v, sim = self.model.m, self.model.v, self.cfg.simulations
        solved = self.has_solution
        row = {
            # the settings of this run
            "beta_G": sim.cvar.generator.beta,
            "beta_L": sim.cvar.buyer.beta,
            "alpha_G": sim.cvar.generator.alpha,
            "alpha_L": sim.cvar.buyer.alpha,
            "tau": sim.contract.tau,
            # the outcome (NaN when there is no solution -> grey square)
            "status": STATUS_NAMES.get(m.Status, "OTHER"),
            "S": float(v.S.X) if solved else np.nan,  # EUR/MWh
            "M": float(v.M.X) if solved else np.nan,  # MW
            "w_G": v.w_G.X if solved else np.nan,  # EUR
            "w_L": v.w_L.X if solved else np.nan,  # EUR
            "W": v.w_G.X + v.w_L.X if solved else np.nan,  # EUR
        }

        path = Path(self.cfg.paths.results) / f"sweep_{self.cfg.sweep_name}.csv"
        # mode="a" appends; the header is written only when the file is created by the first run
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    @staticmethod
    def scenario_names(indices) -> pd.Index:
        """s0, s1, ...: the column names of the input scenario files."""
        return "s" + pd.Index(indices).astype(str)
