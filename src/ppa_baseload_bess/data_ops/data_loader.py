from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..utils import get_logger

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

    Output: everything that changes with the scenario lives in the namespace `self.sc`.
    One numpy array of shape (T, S) per series, so that e.g.
        self.sc.spot_prices[t, s]
    is the spot price at time step t in scenario s:
        sc.spot_prices      EUR/MWh
        sc.solar            MW
        sc.onshore_wind     MW
        sc.offshore_wind    MW
        sc.consumption      MWh per time step
        sc.probabilities    probability of each scenario, length S
    Directly on the loader: n_timesteps (T), n_scenarios (S), dt (hours per time step).

    Also directly on the loader, the model settings from the simulation config:
        solar_mw        installed capacity of the generator's solar plant, MW
        load_mw         average load of the buyer, MW
        strike_lower    bounds of the strike price S, EUR/MWh
        strike_upper
        M_lower         bounds of the baseload volume M, MW
        M_upper
        tau             bargaining power of the generator, 0..1
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self) -> None:
        self.set_paths()
        self.read_scenarios()
        self.check_scenarios()
        self.build_arrays()
        self.set_probabilities()
        self.set_parameters()

        log.info(
            f"Loaded {len(self.series_names)} series, {self.n_timesteps} time steps, "
            f"{self.n_scenarios} scenarios from {self.scenario_dir}"
        )

    def set_paths(self) -> None:
        """Store where the scenarios are and which series to load."""

        self.data_dir = Path(self.cfg.paths.data)
        self.scenario_dir = self.data_dir / "scenarios" / self.cfg.simulations.label
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

        # CSV name -> attribute name on the loader
        self.series_names = {
            "spot_prices": "spot_prices",
            "solar_data": "solar",
            "onshore_wind_data": "onshore_wind",
            "offshore_wind_data": "offshore_wind",
            "consumption": "consumption",
        }

    def read_scenarios(self) -> None:
        """Read every series CSV into a DataFrame."""

        self.frames = {
            name: pd.read_csv(self.scenario_dir / f"{name}.csv", index_col="t") for name in self.series_names
        }
        log.info(f"Read {len(self.frames)} series from {self.scenario_dir}")

    def check_scenarios(self) -> None:
        """Make sure the files fit together before stacking them."""

        ref_name = next(iter(self.series_names))
        ref = self.frames[ref_name]

        for name, frame in self.frames.items():
            if frame.shape != ref.shape:
                raise ValueError(f"{name}.csv has {frame.shape}, when the expected is {ref.shape}, like {ref_name}.csv")
            if not frame.columns.equals(ref.columns):
                raise ValueError(f"{name}.csv has different scenario columns or order than {ref_name}.csv")
            if not frame.index.equals(ref.index):
                raise ValueError(f"{name}.csv has different index than {ref_name}.csv")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
                raise ValueError(f"{name}.csv contains non-numeric values")
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"{name}.csv contains missing or infinite values")

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

    def build_arrays(self) -> None:
        """Store each series as an array of shape (T, S) in self.sc, under its attribute name."""

        self.sc = SimpleNamespace()  # Namespace for all scenario-dependent data
        for name, attribute in self.series_names.items():
            setattr(self.sc, attribute, self.frames[name].to_numpy(dtype=float))

        # store T, S and the time step length for the model
        self.n_timesteps, self.n_scenarios = self.sc.spot_prices.shape
        self.dt = pd.Timedelta(self.cfg.data.preprocessor.timestep) / pd.Timedelta("1h")

    def set_probabilities(self) -> None:
        """Probability of each scenario."""

        self.sc.probabilities = np.ones(self.n_scenarios) / self.n_scenarios

    def set_parameters(self) -> None:
        """Collect the model settings from the simulation config."""

        sim_cfg = self.cfg.simulations
        self.solar_mw = sim_cfg.generator.solar_mw
        self.load_mw = sim_cfg.buyer.load_mw
        self.strike_lower = sim_cfg.contract.strike_lower
        self.strike_upper = sim_cfg.contract.strike_upper
        self.M_lower = sim_cfg.contract.M_lower
        self.M_upper = sim_cfg.contract.M_upper
        self.tau = sim_cfg.contract.tau
