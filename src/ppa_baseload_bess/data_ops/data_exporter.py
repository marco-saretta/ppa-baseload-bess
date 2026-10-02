import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from omegaconf import OmegaConf

from ..utils import get_logger

log = get_logger(__name__)


class DataExporter:
    """Write the results of a solved model to results/<label>/:
    config.yaml       the settings the run used
    results.json      solver status, solve time and objective
    variables.csv     value of every scalar decision variable
    timeseries.csv    the model inputs per time step (prices, production, load)
    """

    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.model = model

    def run(self) -> None:
        self.set_paths()
        self.export_config()
        self.export_results()
        self.export_timeseries()
        log.info(f"Exported results to {self.out_dir}")

    def set_paths(self) -> None:
        self.out_dir = Path(self.cfg.paths.results) / self.cfg.simulations.label
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export_config(self) -> None:
        (self.out_dir / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))

    def export_results(self) -> None:
        m = self.model.m
        results = {"status": m.Status, "solve_time_s": m.Runtime}

        # Variable values only exist if the solver found a solution
        if m.SolCount > 0:
            results["objective"] = m.ObjVal
            # Every scalar variable in the model's namespace, whatever it is called
            variables = pd.Series(
                {name: var.X for name, var in vars(self.model.v).items() if isinstance(var, gp.Var)}, name="value"
            )
            variables.index.name = "variable"
            variables.to_csv(self.out_dir / "variables.csv")
            # TODO: from the stochastic model on, variables indexed by scenario (MVar) need
            #   their own file, one row per scenario.
        else:
            log.warning(f"No solution to export (Gurobi status {m.Status})")

        (self.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    def export_timeseries(self) -> None:
        model = self.model
        df = pd.DataFrame({"spot_price": model.spot, "P_G": model.P_G, "P_L": model.P_L})
        df.index.name = "t"
        df.to_csv(self.out_dir / "timeseries.csv")
