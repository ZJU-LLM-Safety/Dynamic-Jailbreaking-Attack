"""Logging utilities for the framework"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class Logger:
    """Custom logger wrapper using loguru"""

    def __init__(
        self,
        name: str = "AAF",
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
    ):
        """
        Initialize logger

        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.name = name
        self.log_level = log_level

        # Remove default logger
        logger.remove()

        # Console logging
        if log_to_console:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
                level=log_level,
                colorize=True,
            )

        # File logging
        if log_to_file and log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Regular log file
            logger.add(
                log_path / f"{name}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
                level=log_level,
                rotation="500 MB",
                retention="10 days",
                compression="zip",
            )

            # Error log file
            logger.add(
                log_path / f"{name}_error.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation="100 MB",
                retention="30 days",
                compression="zip",
            )

        self.logger = logger

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)


# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger(
    name: str = "AAF",
    log_level: str = "INFO",
    log_dir: Optional[str] = "logs",
    reset: bool = False,
) -> Logger:
    """
    Get or create global logger instance

    Args:
        name: Logger name
        log_level: Logging level
        log_dir: Log directory
        reset: Whether to reset existing logger

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None or reset:
        _global_logger = Logger(
            name=name,
            log_level=log_level,
            log_dir=log_dir,
        )

    return _global_logger


def setup_logging(config):
    """Setup logging from config"""
    return get_logger(
        name="AAF",
        log_level=config.log_level if hasattr(config, 'log_level') else "INFO",
        log_dir=config.log_dir if hasattr(config, 'log_dir') else "logs",
        reset=True,
    )
