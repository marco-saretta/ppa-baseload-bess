import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..utils import get_logger


log = get_logger(__name__)


class DataPreprocessor:
    """Download Energinet datasets to data/raw/, clean them into data/preprocessed/
    and bootstrap scenarios into data/scenarios/<label>/."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data.preprocessor
        self.config_dirs()

    def config_dirs(self):
        self.data_dir = Path(self.cfg.paths.data)
        self.raw_dir = self.data_dir / "raw"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        # One folder per simulation, since the scenarios depend on its config
        self.scenario_dir = self.data_dir / "scenarios" / self.cfg.simulations.label
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        if self.is_preprocessed():
            log.info(f"Vasiliki, preprocessed data already in {self.preprocessed_dir}, skipping download and preprocessing")
        else:
            self.prepare_spot_prices()
            self.prepare_power_system()
            self.prepare_consumption()
        # Always rebuilt: cheap, and they depend on the simulation config (seed, horizon, ...)
        self.prepare_scenarios()

    def is_preprocessed(self) -> bool:
        """True if every preprocessed file exists (delete data/preprocessed/ to redo it,
        e.g. after changing timestep, start or end)."""
        names = ["spot_prices", "power_system", "solar_data", "onshore_wind_data", "offshore_wind_data", "consumption"]
        return all((self.preprocessed_dir / f"{name}.csv").exists() for name in names)

    # Wrapper methods --> download + preprocess one dataset, then build scenarios from all of them

    def prepare_spot_prices(self) -> None:
        self.download_spot_prices()
        self.preprocess_spot_prices()

    def prepare_power_system(self) -> None:
        self.download_power_system()
        self.preprocess_power_system()

    def prepare_consumption(self) -> None:
        self.download_consumption()
        self.preprocess_consumption()

    def prepare_scenarios(self) -> None:
        """Build equally likely scenarios by bootstrapping whole days of history.

        Every series takes the same drawn days, so the link between prices,
        renewables and consumption is kept. Writes one CSV per series (rows are
        time steps, columns are scenarios) and drawn_days.csv with the source dates.
        """
        scen_cfg = self.cfg.simulations.scenarios
        log.info(f"Start generating {scen_cfg.n_scenarios} scenarios of {scen_cfg.horizon_days} days")

        # Preprocessed file name -> column to use
        series = {
            "spot_prices": "DayAheadPriceEUR",
            "solar_data": "SolarPower",
            "onshore_wind_data": "OnshoreWindPower",
            "offshore_wind_data": "OffshoreWindPower",
            "consumption": "ConsumptionMWh",
        }
        # join="inner" keeps the time steps present in every series, dropna the ones with a missing value
        df = pd.concat(
            [
                pd.read_csv(self.preprocessed_dir / f"{name}.csv", index_col=0, parse_dates=True)[column].rename(name)
                for name, column in series.items()
            ],
            axis=1,
            join="inner",
        ).dropna()

        # Keep only complete days (UTC), so every day has the same number of steps
        steps_per_day = pd.Timedelta("1D") // pd.Timedelta(self.data_cfg.timestep)
        day_of_step = df.index.normalize()
        steps_in_day = df.groupby(day_of_step).size()
        days = steps_in_day.index[steps_in_day == steps_per_day]
        df = df[day_of_step.isin(days)]
        log.info(f"Drawing from {len(days)} days between {days[0].date()} and {days[-1].date()}")

        # drawn[s, d] is the index in `days` of the d-th day of scenario s
        rng = np.random.default_rng(scen_cfg.seed)
        drawn = rng.integers(0, len(days), size=(scen_cfg.n_scenarios, scen_cfg.horizon_days))
        scenario_names = [f"s{s}" for s in range(scen_cfg.n_scenarios)]

        for name in series:
            # One row per historical day, one column per time step of that day
            daily = df[name].to_numpy().reshape(len(days), steps_per_day)
            # Stack the drawn days of each scenario end to end: shape (time steps, scenarios)
            values = daily[drawn].reshape(scen_cfg.n_scenarios, -1).T

            out = pd.DataFrame(values, columns=scenario_names)
            out.index.name = "t"
            out.to_csv(self.scenario_dir / f"{name}.csv")

        # Source date of every scenario day, to trace results back to history
        drawn_days = pd.DataFrame(days.strftime("%Y-%m-%d").to_numpy()[drawn].T, columns=scenario_names)
        drawn_days.index.name = "day"
        drawn_days.to_csv(self.scenario_dir / "drawn_days.csv")

        log.info(f"Saved scenarios to {self.scenario_dir}")

    def _fetch_energinet_dataset(self, dataset: str, **params) -> pd.DataFrame:
        """Fetch a whole dataset from the Energinet API as a DataFrame.

        `params` are query params (start, end, filter, sort, ...), see
        https://www.energidataservice.dk/guides/api-guides. A dict `filter` is
        JSON-encoded, since the API expects a JSON string.
        """
        if isinstance(params.get("filter"), dict):
            params["filter"] = json.dumps(params["filter"])

        # limit=0 returns every row; the API default is only 100
        response = requests.get(f"{self.data_cfg.energinet_api_url}/{dataset}", params={"limit": 0, **params})
        response.raise_for_status()
        return pd.DataFrame(response.json()["records"])

    # Download (skipped if the raw file exists: delete it to re-download after changing start/end)

    def download_spot_prices(self) -> None:
        out_path = self.raw_dir / "raw_spot_prices.csv"
        if out_path.exists():
            log.info(f"Spot prices already downloaded at {out_path}, skipping")
            return

        log.info("Start downloading spot prices")
        df = self._fetch_energinet_dataset(
            "DayAheadPrices",
            start=self.data_cfg.start,
            end=self.data_cfg.end,
            filter={"PriceArea": "DK1"},
            sort="TimeUTC ASC",
        )
        df.to_csv(out_path, index=False)
        log.info(f"Saved spot prices to {out_path}")

    def download_power_system(self) -> None:
        out_path = self.raw_dir / "raw_power_system.csv"
        if out_path.exists():
            log.info(f"Power system data already downloaded at {out_path}, skipping")
            return

        log.info("Start downloading power system data")
        df = self._fetch_energinet_dataset(
            "PowerSystemRightNow", start=self.data_cfg.start, end=self.data_cfg.end, sort="Minutes1UTC ASC"
        )
        df.to_csv(out_path, index=False)
        log.info(f"Saved power system data to {out_path}")

    def download_consumption(self) -> None:
        out_path = self.raw_dir / "raw_consumption.csv"
        if out_path.exists():
            log.info(f"Consumption data already downloaded at {out_path}, skipping")
            return

        log.info("Start downloading consumption data")
        df = self._fetch_energinet_dataset(
            "ConsumptionConsumerCategoryHour", start=self.data_cfg.start, end=self.data_cfg.end, sort="TimeUTC ASC"
        )
        df.to_csv(out_path, index=False)
        log.info(f"Saved consumption data to {out_path}")

    # Preprocess

    def preprocess_spot_prices(self) -> None:
        """DK1 day-ahead price in EUR/MWh."""
        log.info("Start preprocessing spot prices")
        raw_df = pd.read_csv(self.raw_dir / "raw_spot_prices.csv", parse_dates=["TimeUTC"], index_col="TimeUTC")

        # Prices are 15-minute from 2025-10-01; resample puts them on a gap-free `timestep` grid
        df = raw_df[["DayAheadPriceEUR"]].sort_index().resample(self.data_cfg.timestep).mean()

        out_path = self.preprocessed_dir / "spot_prices.csv"
        df.to_csv(out_path)
        log.info(f"Saved preprocessed spot prices to {out_path}")

    def preprocess_power_system(self) -> None:
        """All numeric columns in MW, averaged from 1-minute data. Gaps are interpolated."""
        log.info("Start preprocessing power system data")
        raw_df = pd.read_csv(
            self.raw_dir / "raw_power_system.csv", parse_dates=["Minutes1UTC"], index_col="Minutes1UTC"
        )

        # select_dtypes drops the Minutes1DK string column
        df = raw_df.select_dtypes("number").sort_index().resample(self.data_cfg.timestep).mean()
        # Fill steps with no source data (e.g. the Sept 2025 outage) linearly in time
        df = df.interpolate(method="time")

        df.to_csv(self.preprocessed_dir / "power_system.csv")
        log.info(f"Saved preprocessed power system data to {self.preprocessed_dir / 'power_system.csv'}")

        # Renewable production series used by the model, one file each
        for column, filename in {
            "SolarPower": "solar_data.csv",
            "OffshoreWindPower": "offshore_wind_data.csv",
            "OnshoreWindPower": "onshore_wind_data.csv",
        }.items():
            df[[column]].to_csv(self.preprocessed_dir / filename)
            log.info(f"Saved {column} to {self.preprocessed_dir / filename}")

    def preprocess_consumption(self) -> None:
        """DK1 business ("Erhverv") consumption in MWh per time step.

        The source is hourly kWh per region and category; each hour is split evenly
        over its time steps.
        """
        log.info("Start preprocessing consumption data")
        raw_df = pd.read_csv(self.raw_dir / "raw_consumption.csv", parse_dates=["TimeUTC"], index_col="TimeUTC")

        # Keep business consumption in the DK1 regions only
        df = raw_df[
            raw_df["RegionName"].isin(list(self.data_cfg.dk1_regions)) & (raw_df["ConsumerCategory3"] == "Erhverv")
        ]
        # Sum over regions, then kWh -> MWh
        hourly_mwh = df.groupby(level="TimeUTC")["ConsumptionkWh"].sum().sort_index() / 1000
        # Repeat each hour's value on its sub-hourly steps, then split the energy over them
        steps_per_hour = pd.Timedelta("1h") / pd.Timedelta(self.data_cfg.timestep)
        df = (hourly_mwh.resample(self.data_cfg.timestep).ffill() / steps_per_hour).to_frame("ConsumptionMWh")

        out_path = self.preprocessed_dir / "consumption.csv"
        df.to_csv(out_path)
        log.info(f"Saved preprocessed consumption data to {out_path}")
