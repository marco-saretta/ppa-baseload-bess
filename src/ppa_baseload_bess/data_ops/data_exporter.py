from ..utils import get_logger

log = get_logger(__name__)


class DataExporter:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.model = model
