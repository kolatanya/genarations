"""File logging. The terminal belongs to the rich display, so logs go to disk."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(log_dir: Path = config.LOG_DIR, level: int = logging.INFO) -> Path | None:
    """Attach a rotating file handler to the root logger. Returns the log path (None if unwritable)."""
    root = logging.getLogger()
    root.setLevel(level)
    log_path = log_dir / "evobot.log"
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return log_path
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        root.addHandler(logging.NullHandler())
        return None
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    # Third-party chatter stays out of our log unless it's serious.
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path
