import json
from pathlib import Path
from ..utils import get_logger

import numpy as np
import pandas as pd

log = get_logger(__name__)


class DataLoader:
    """Load the scenarios written by DataPreprocessor and hand them to the model.

    Input: data/scenarios/<label>/, one CSV per series:
        spot_prices.csv          EUR/MWh
        solar_data.csv           MW (average over each time step)
        onshore_wind_data.csv    MW
        offshore_wind_data.csv   MW
        consumption.csv          MWh per time step
        drawn_days.csv           historical date behind each scenario day (not model input)

    Every series CSV has the same layout:
        - index column "t": time step 0 .. T-1 (T = horizon_days * steps per day)
        - one column per scenario: s0, s1, ..., s{S-1}
    Scenario sK is the SAME historical days in every file, so the files belong together.

    Goal: one numpy array `self.data` of shape (n_series, T, S), so that
        self.data[i, t, s]
    is the value of series i at time step t in scenario s.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self) -> None:
        self.set_paths()
        self.read_scenarios()
        self.check_scenarios()
        self.build_tensor()
        self.set_probabilities()

        n_series, T, S = self.scenarios.shape
        log.info(f"Loaded {n_series} series, {T} time steps, {S} scenarios from {self.scenario_dir}")

    def set_paths(self) -> None:
        """Store where the scenarios are and which series to load."""

        self.data_dir = Path(self.cfg.paths.data)
        self.scenario_dir = self.data_dir / "scenarios" / self.cfg.simulations.label
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

        self.series_names = ["spot_prices", "solar_data", "onshore_wind_data", "offshore_wind_data", "consumption"]

    def read_scenarios(self) -> None:
        """Read every series CSV into a DataFrame."""

        self.frames = {
            name: pd.read_csv(self.scenario_dir / f"{name}.csv", index_col="t") for name in self.series_names
        }
        log.info(f"Read {len(self.frames)} series from {self.scenario_dir}")

    def check_scenarios(self) -> None:
        """Make sure the files fit together before stacking them."""

        ref_name = self.series_names[0]
        ref = self.frames[ref_name]

        for name, frame in self.frames.items():
            if frame.shape != ref.shape:
                raise ValueError(f"{name}.csv has {frame.shape}, when the expected is {ref.shape}, like {ref_name}.csv")
            if not frame.columns.equals(ref.columns):
                raise ValueError(f"{name}.csv has different scenario columns or order than {ref_name}.csv")
            if not frame.index.equals(ref.index):
                raise ValueError(f"{name}.csv has different index than {ref_name}.csv")
            if frame.isna().to_numpy().any():
                raise ValueError(f"{name}.csv contains missing values")

        T, S = ref.shape
        scen_cfg = self.cfg.simulations.scenarios
        steps_per_day = pd.Timedelta("1D") // pd.Timedelta(self.cfg.data.preprocessor.timestep)
        expected = (scen_cfg.horizon_days * steps_per_day, scen_cfg.n_scenarios)
        if (T, S) != expected:
            raise ValueError(
                f"Scenarios have (T, S)={(T, S)}, but the config expects {expected}."
                f"Rerun the preprocessing to regenerate {self.scenario_dir}"
            )
        if not ref.index.equals(pd.RangeIndex(T)):
            raise ValueError(f"The time index must run 0..{T - 1}")
        log.info(f"Scenario files are consistent: T={T} time steps, S={S} scenarios")

    def build_tensor(self) -> None:
        """Stack all series into one array of shape (n_series, T, S)."""

        arrays = [self.frames[name].to_numpy() for name in self.series_names]

        self.scenarios = np.stack(arrays, axis=0)

        # store T and S for the model
        self.n_features, self.n_timesteps, self.n_scenarios = self.scenarios.shape

        self.series_index = {name: i for i, name in enumerate(self.series_names)}

    def set_probabilities(self) -> None:
        """Probability of each scenario."""

        self.probabilities = np.ones(self.n_scenarios) / self.n_scenarios
