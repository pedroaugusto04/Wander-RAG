"""Structured logging configuration using structlog."""

from __future__ import annotations

import dataclasses
import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application and intercept stdlib logs."""

    _original_get_column_styles = structlog.dev.ConsoleRenderer.get_default_column_styles
    def custom_column_styles(colors: bool = True, force_colors: bool = False):
        styles = _original_get_column_styles(colors, force_colors)
        if colors or force_colors:
            return dataclasses.replace(styles, logger_name="\x1b[36m")
        return styles
    structlog.dev.ConsoleRenderer.get_default_column_styles = staticmethod(custom_column_styles)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=True)
    level_styles["debug"] = "\x1b[34m"     # Blue
    level_styles["info"] = "\x1b[32m"      # Green
    level_styles["warning"] = "\x1b[33m"   # Yellow
    level_styles["warn"] = "\x1b[33m"      # Yellow
    level_styles["error"] = "\x1b[31m"     # Red
    level_styles["exception"] = "\x1b[31m" # Red
    level_styles["critical"] = "\x1b[31;1m" # Bold Red

    is_tty = True

    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=is_tty, force_colors=is_tty, level_styles=level_styles) if is_tty else structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())
