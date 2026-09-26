import json
from pathlib import Path

import pandas as pd
import requests

from ..utils import get_logger


log = get_logger(__name__)

ENERGINET_API_URL = "https://api.energidataservice.dk/dataset"


class DataPreprocessor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.config_directories()
        self.get_spot_prices_data()
        self.get_power_system_data()

    def config_directories(self):
        self.data_dir = Path(self.cfg.paths.data)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_energinet_dataset(
        self,
        dataset: str,
        start: str | None = None,
        end: str | None = None,
        **params,
    ) -> pd.DataFrame:
        """Fetch a dataset from the Energinet API and return it as a DataFrame.

        `start`/`end` are the query's date range (e.g. "2025-01-01T00:00").
        `params` are any other query params (e.g. filter, columns, limit, sort)
        - see https://www.energidataservice.dk/guides/api-guides.
        A `filter` value given as a dict is JSON-encoded, since the API expects it
        as a JSON string (e.g. {"PriceArea": "DK1"}).
        """
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        if isinstance(params.get("filter"), dict):
            params["filter"] = json.dumps(params["filter"])

        url = f"{ENERGINET_API_URL}/{dataset}"
        response = requests.get(url=url, params=params)
        response.raise_for_status()

        return pd.DataFrame(data=response.json()["records"])

    def get_spot_prices_data(self, start: str | None = None, end: str | None = None):
        out_path = self.data_dir / "spot_prices.csv"
        if out_path.exists():
            log.info(f"Spot prices already downloaded at {out_path}, skipping")
            return

        log.info("Start fetching electricity prices data")
        df = self._fetch_energinet_dataset(
            "DayAheadPrices",
            start=start,
            end=end,
            filter={"PriceArea": "DK1"},
            sort="TimeUTC DESC",
            limit=0,
        )
        df.to_csv(path_or_buf=out_path, index=False)

        log.info(f"Saved spot prices to {out_path}")

    def get_power_system_data(self, start: str = "2025-01-01T00:00", end: str = "2026-09-17T00:00"):
        out_path = self.data_dir / "power_system.csv"
        if out_path.exists():
            log.info(f"Power system data already downloaded at {out_path}, skipping")
            return

        log.info("Start fetching power system data")
        df = self._fetch_energinet_dataset(
            "PowerSystemRightNow",
            start=start,
            end=end,
            sort="Minutes1UTC ASC",
        )
        df.to_csv(path_or_buf=out_path, index=False)

        log.info(f"Saved power system data to {out_path}")
