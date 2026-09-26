from ..utils import get_logger

log = get_logger(__name__)


class DataLoader:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
