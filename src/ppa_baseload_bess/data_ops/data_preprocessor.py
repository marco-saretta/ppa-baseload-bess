from ..utils import get_logger

log = get_logger(__name__)


class DataPreprocessor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        log.info("I am alive!")
