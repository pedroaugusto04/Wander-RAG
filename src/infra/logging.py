"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=True)
    level_styles["debug"] = "\x1b[34m"     # Blue
    level_styles["info"] = "\x1b[32m"      # Green
    level_styles["warning"] = "\x1b[33m"   # Yellow
    level_styles["warn"] = "\x1b[33m"      # Yellow
    level_styles["error"] = "\x1b[31m"     # Red
    level_styles["exception"] = "\x1b[31m" # Red
    level_styles["critical"] = "\x1b[31;1m" # Bold Red

    # Decide if we are rendering for terminal/dev or JSON
    # Typically you check if sys.stderr.isatty() or an env var. 
    # The original author had `if True` here, so we keep that logic but fix the color palette.
    is_tty = True  # or sys.stderr.isatty() if you prefer later
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=is_tty, level_styles=level_styles) if is_tty else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
