"""Logging configuration for Greek Forge."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Singleton logger for Greek Forge application.

    This ensures consistent logging configuration across all modules.
    """

    _instance: Optional["Logger"] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger only once."""
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(
        self,
        name: str = "greek_forge",
        level: Optional[int] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        """
        Setup the singleton logger.

        Args:
            name: Logger name (default: 'greek_forge')
            level: Logging level. If None, uses LOG_LEVEL env var or defaults to INFO
            log_file: Optional path to log file. If None, uses LOG_FILE env var or console only
        """
        # Determine log level from env var or default to INFO
        if level is None:
            level_str = os.getenv("LOG_LEVEL", "INFO").upper()
            level = getattr(logging, level_str, logging.INFO)

        # Determine log file from env var
        if log_file is None and os.getenv("LOG_FILE"):
            log_file = Path(os.getenv("LOG_FILE"))

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()

        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Format: timestamp - name - level - message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the singleton logger instance.

        Returns:
            Logger instance
        """
        return self._logger

    def reconfigure(
        self,
        level: Optional[int] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        """
        Reconfigure the logger (useful for testing or runtime changes).

        Args:
            level: New logging level
            log_file: New log file path
        """
        self._setup_logger(level=level, log_file=log_file)


_logger_instance = Logger()


def get_logger() -> logging.Logger:
    """
    Get the singleton logger instance.

    This is the main function to use throughout the application.
    All modules will share the same logger configuration.

    Returns:
        Logger instance

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("This is a log message")
    """
    return _logger_instance.get_logger()


def setup_logger(
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup or reconfigure the singleton logger.

    Args:
        level: Logging level (default: INFO or from LOG_LEVEL env var)
        log_file: Optional path to log file (or from LOG_FILE env var)

    Returns:
        Configured logger instance

    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger(level=logging.DEBUG, log_file=Path("logs/app.log"))
    """
    _logger_instance.reconfigure(level=level, log_file=log_file)
    return _logger_instance.get_logger()


if __name__ == "__main__":
    # NOTE: Example usage
    logger = setup_logger(level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    log_file = Path("logs/test.log")
    logger = setup_logger(level=logging.INFO, log_file=log_file)
    logger.info("This message goes to both console and file")

    logger1 = get_logger()
    logger2 = get_logger()
    logger1.info("From module 1")
    logger2.info("From module 2")
    # logger1 is logger2 → True (same instance!)

    # Example: Using environment variables
    # Set LOG_LEVEL=DEBUG and LOG_FILE=logs/app.log in .env
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FILE"] = "logs/env_test.log"
    logger = setup_logger()
    logger.debug("This uses env var configuration")
