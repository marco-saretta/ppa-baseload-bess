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

    Goal: one numpy array `self.data` of shape (n_series, T, S), so that
        self.data[i, t, s]
    is the value of series i at time step t in scenario s.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # TODO: once the methods below are done, call them here in order:
        #   1. set_paths()
        #   2. read_scenarios()
        #   3. check_scenarios()
        #   4. build_tensor()
        #   5. set_probabilities()
        # Hint: log a short summary at the end (n_series, T, S) with log.info(...)

    def set_paths(self) -> None:
        """Store where the scenarios are and which series to load."""
        # TODO: build self.scenario_dir from the config.
        #   It must match DataPreprocessor.config_dirs(): data dir / "scenarios" / simulation label.
        #   Look at config/paths/default.yaml and config/simulations/default.yaml for the keys.
        #
        # TODO: define self.series_names, a list with the file names without ".csv".
        #   The ORDER of this list decides the first axis of the tensor, so keep it fixed
        #   and use it everywhere (e.g. index 0 is always spot prices).

    def read_scenarios(self) -> None:
        """Read every series CSV into a DataFrame."""
        # TODO: for each name in self.series_names, read <scenario_dir>/<name>.csv with pandas.
        #   Use "t" as the index column, so only the scenario columns are left.
        #
        # TODO: store the result in a dict self.frames: {series name: DataFrame}.
        #   A dict keeps the name next to its data, which helps when debugging.
        #
        # Question to think about: what should happen if a file is missing?
        # Which error would you like to see, and where?

    def check_scenarios(self) -> None:
        """Make sure the files fit together before stacking them."""
        # TODO: check that every DataFrame has
        #   - the same shape (T, S)
        #   - the same scenario columns, in the same order (s0, s1, ...)
        #   - the same index (t = 0 .. T-1)
        #   - no missing values
        # Raise a ValueError with a clear message if not (say WHICH file is wrong).
        #
        # TODO (optional): compare S with cfg.simulations.scenarios.n_scenarios and
        #   T with horizon_days * steps per day. Where can you find the steps per day?
        #
        # Why bother: np.stack does not know about column names. If two files had their
        # scenarios in a different order, the tensor would silently mix scenarios.

    def build_tensor(self) -> None:
        """Stack all series into one array of shape (n_series, T, S)."""
        # TODO: turn each DataFrame into a numpy array (shape (T, S)).
        # TODO: stack them along a NEW first axis, following the order of self.series_names.
        #   Hint: look at np.stack and its `axis` argument.
        # TODO: store it as self.data and also store T and S as attributes
        #   (self.n_timesteps, self.n_scenarios) so the model can build its index sets.
        #
        # Check yourself: pick one series, one t and one s, and compare self.data[i, t, s]
        # with the value you see when you open the CSV.
        #
        # Optional: a dict self.series_index = {name: i} makes the model code more readable,
        # e.g. self.data[self.series_index["spot_prices"]].

    def set_probabilities(self) -> None:
        """Probability of each scenario."""
        # TODO: the scenarios are drawn by bootstrap, so they are equally likely.
        #   Store a numpy array self.probabilities of length S that sums to 1.
        #   The model will need it for the expected value and for CVaR.
