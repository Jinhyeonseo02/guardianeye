import logging
import sys
from guardianeye.utils.config import LOG_LEVEL, LOG_FILE


def _setup_root_logger() -> None:
    level   = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt     = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger("guardianeye")
    if root.handlers:
        return

    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("guardianeye"):
        name = f"guardianeye.{name}"
    return logging.getLogger(name)

