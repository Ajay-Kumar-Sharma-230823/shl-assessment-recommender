"""
Logging Configuration
======================
Structured logging with rich output for development
and JSON logging for production.
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    log_level: str = "INFO",
    env: str = "development",
) -> None:
    """
    Configure structured logging.
    
    - Development: human-readable colored output via rich
    - Production: JSON structured logging
    """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    if env == "development":
        # Use structlog with console renderer
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
    else:
        # Production: JSON
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )

    # Also configure standard logging
    logging.basicConfig(
        level=log_level_int,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Suppress noisy loggers
    for noisy in ["httpcore", "httpx", "urllib3", "charset_normalizer"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
