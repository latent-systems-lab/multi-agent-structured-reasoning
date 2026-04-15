"""Logging helpers."""

from __future__ import annotations

import logging
import os

import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a JSON logger with minimal configuration.

    When the ``BACKTEST_DEBUG`` environment variable is set to ``"1"``, the
    logger is configured to emit debug-level messages. Otherwise, only info
    level and above are recorded.
    """

    level = logging.DEBUG if os.getenv("BACKTEST_DEBUG") == "1" else logging.INFO
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    return structlog.get_logger(name)
