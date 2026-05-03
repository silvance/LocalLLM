import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import get_settings


# Project root: app/utils/logger.py -> app/utils/ -> app/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_logger(name: str = "localllm") -> logging.Logger:
    settings = get_settings()

    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 5 MB per file, keep 5 rotations -> ~25 MB max on disk per logger.
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
