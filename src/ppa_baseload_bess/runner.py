from omegaconf import DictConfig

from . import utils
from .data_ops import DataPreprocessor, DataLoader, DataExporter, DataVisualizer
from .model import Model

log = utils.get_logger(__name__)


class Runner:
    def __init__(self, cfg: DictConfig) -> None:
        utils.setup_logging(log_dir=cfg.paths.log)

        self.cfg = cfg
        self.data = None
        self.model = None

    def run(self):
        self.preprocess()
        self.load_data()
        self.build_model()
        self.solve_model()
        self.export_data()
        self.visualize_data()

    def preprocess(self):
        with utils.stage("preprocess", log):
            DataPreprocessor(self.cfg)

    def load_data(self):
        with utils.stage("load", log):
            self.data = DataLoader(self.cfg)

    def build_model(self):
        if self.data is None:
            raise RuntimeError("build_model() needs data: call load_data() first")
        with utils.stage("build", log):
            self.model = Model(self.cfg, self.data)

    def solve_model(self):
        with utils.stage("solve", log):
            self.model.solve()

    def export_data(self):
        with utils.stage("export", log):
            self.exporter = DataExporter(self.cfg, self.model)

    def visualize_data(self):
        with utils.stage("visualize", log):
            self.visualizer = DataVisualizer(self.cfg, self.model)
