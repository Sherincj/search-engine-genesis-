"""
Structured logging configuration.

Provides a consistent logging setup across the application.
Log level is controlled via the DEBUG setting.
"""

import logging
import sys


def setup_logging(*, debug: bool = False) -> None:
    """
    Configure the root logger with a structured format.

    Parameters
    ----------
    debug:
        If ``True``, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. tests).
    if not root.handlers:
        root.addHandler(handler)

    # Silence noisy third-party loggers — they flood the console
    # with low-level HTTP/DB chatter when the app runs in DEBUG mode.
    _NOISY_LOGGERS = (
        "httpcore",
        "httpx",
        "aiosqlite",
        "sqlalchemy.engine.Engine",
        "hpack",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
