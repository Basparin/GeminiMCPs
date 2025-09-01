"""
CES Logging Configuration

Provides centralized logging configuration for the Cognitive Enhancement System.
Supports structured logging, multiple log levels, and configurable outputs.
"""

import logging
import logging.config
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(level: str = "INFO",
                 log_file: Optional[str] = None,
                 json_format: bool = False,
                 console_output: bool = True) -> None:
    """
    Setup logging configuration for CES

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Whether to use JSON format for logs
        console_output: Whether to output to console
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {},
        'root': {
            'handlers': [],
            'level': level
        }
    }

    # Add console handler
    if console_output:
        if json_format:
            config['formatters']['json'] = {
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
            }
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'formatter': 'json',
                'stream': sys.stdout
            }
        else:
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'formatter': 'detailed' if level == 'DEBUG' else 'standard',
                'stream': sys.stdout
            }
        config['root']['handlers'].append('console')

    # Add file handler
    if log_file:
        if json_format:
            config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'formatter': 'json'
            }
        else:
            config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'formatter': 'detailed'
            }
        config['root']['handlers'].append('file')

    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class CESLogger:
    """CES-specific logger with additional context"""

    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.context = {}

    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self.context.clear()

    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log('debug', message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log('warning', message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log('error', message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log('critical', message, **kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method"""
        # Combine context with additional kwargs
        extra = {**self.context, **kwargs}

        # Log the message
        log_method = getattr(self.logger, level)
        log_method(message, extra=extra if extra else None)


# Global logger instance
_ces_logger = None


def get_ces_logger(name: str = "ces") -> CESLogger:
    """Get CES logger instance"""
    global _ces_logger
    if _ces_logger is None:
        _ces_logger = CESLogger(name)
    return _ces_logger


# Initialize default logging on import
setup_logging()