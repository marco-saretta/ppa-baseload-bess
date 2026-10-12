import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB
from omegaconf import OmegaConf

from ..utils import get_logger

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

    @staticmethod
    def scenario_names(indices) -> pd.Index:
        """s0, s1, ...: the column names of the input scenario files."""
        return "s" + pd.Index(indices).astype(str)
