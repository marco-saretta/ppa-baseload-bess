import logging
import time
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager

_STAGE: ContextVar[str] = ContextVar("stage", default="run")

# Third-party loggers that log outside the "ppa_baseload_bess" hierarchy, so our
# formatting and filters never apply to them. Quiet them instead of reformatting them.
_QUIET_LOGGERS = ("gurobipy",)


class _StageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.stage = f"[{_STAGE.get()}]"
        return True


def setup_logging(log_dir: str = "logs", log_file: str = "ppa_baseload_bess.log", level: int = logging.INFO) -> None:
    """
    Log one line per event to the console and to <log_dir>/<log_file>:

        10:50:31 INFO    [solve]      optimal, objective -5.12e+06

    The stage tag comes from the enclosing `stage(...)` block.
    """
    logger = logging.getLogger("ppa_baseload_bess")
    if logger.handlers:
        return  # already configured — avoid duplicate handlers on a second call
    logger.setLevel(level)
    logger.propagate = False  # Hydra's root handler would print every line twice

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(stage)-12s %(message)s", datefmt="%H:%M:%S"))
    file = logging.FileHandler(Path(log_dir) / log_file, mode="a", encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(stage)-12s %(name)s: %(message)s"))

    for handler in (console, file):
        handler.addFilter(_StageFilter())
        logger.addHandler(handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger; propagates to the configured ppa_baseload_bess root logger."""
    return logging.getLogger(name)


@contextmanager
def stage(name: str, logger: logging.Logger):
    """
    Tag every log line inside the block with [name]; on success, log how long
    the block took.
    """
    token = _STAGE.set(name)
    start = time.perf_counter()
    logger.info(f"start {name}")
    try:
        yield
        logger.info("done in %.1f s", time.perf_counter() - start)
    finally:
        _STAGE.reset(token)
