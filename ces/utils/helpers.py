"""
CES Utility Functions

Common utility functions and helpers used throughout the CES system.
"""

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import json


def setup_logging(level: str = "INFO", debug_mode: bool = False) -> logging.Logger:
    """
    Set up logging configuration for CES

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        debug_mode: Enable debug mode with more detailed logging

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('ces')
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create formatter
    if debug_mode:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate CES configuration

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validation result with any issues found
    """
    issues = []

    # Required fields
    required_fields = ['debug_mode', 'log_level', 'max_memory_mb']
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")

    # Validate memory settings
    if 'max_memory_mb' in config:
        if not isinstance(config['max_memory_mb'], int) or config['max_memory_mb'] < 64:
            issues.append("max_memory_mb must be an integer >= 64")

    # Validate log level
    if 'log_level' in config:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config['log_level'].upper() not in valid_levels:
            issues.append(f"log_level must be one of: {', '.join(valid_levels)}")

    # Validate boolean fields
    boolean_fields = ['debug_mode', 'ethical_checks_enabled', 'cache_enabled']
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            issues.append(f"{field} must be a boolean")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse a JSON file

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data or None if loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_file(file_path: Path, data: Dict[str, Any], indent: int = 2):
    """
    Save data to a JSON file

    Args:
        file_path: Path where to save the JSON file
        data: Data to save
        indent: JSON indentation level
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logging.info(f"Data saved to {file_path}")
    except IOError as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}")
        raise


def format_task_summary(task_description: str, result: Dict[str, Any]) -> str:
    """
    Format a task execution summary

    Args:
        task_description: Original task description
        result: Task execution result

    Returns:
        Formatted summary string
    """
    status = result.get('status', 'unknown')
    assistant = result.get('assistant_used', 'unknown')
    timestamp = result.get('timestamp', 'unknown')

    summary = f"""
Task Summary:
-------------
Description: {task_description[:100]}{'...' if len(task_description) > 100 else ''}
Status: {status.upper()}
Assistant: {assistant}
Timestamp: {timestamp}

Result: {result.get('result', 'No result available')[:200]}{'...' if len(str(result.get('result', ''))) > 200 else ''}
"""

    if 'error' in result:
        summary += f"\nError: {result['error']}"

    return summary.strip()


def calculate_task_metrics(tasks: list) -> Dict[str, Any]:
    """
    Calculate metrics from a list of tasks

    Args:
        tasks: List of task dictionaries

    Returns:
        Dictionary with calculated metrics
    """
    if not tasks:
        return {"total_tasks": 0, "success_rate": 0, "average_complexity": 0}

    total_tasks = len(tasks)
    successful_tasks = sum(1 for task in tasks if task.get('status') == 'completed')

    complexities = [task.get('analysis', {}).get('complexity_score', 0) for task in tasks]
    average_complexity = sum(complexities) / len(complexities) if complexities else 0

    return {
        "total_tasks": total_tasks,
        "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
        "average_complexity": average_complexity,
        "successful_tasks": successful_tasks
    }


def safe_get_nested_value(data: Dict[str, Any], keys: list, default=None):
    """
    Safely get a nested value from a dictionary

    Args:
        data: Dictionary to search
        keys: List of keys to traverse
        default: Default value if path doesn't exist

    Returns:
        Value at the nested path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes

    Args:
        file_path: Path to the file

    Returns:
        File size in MB
    """
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except (OSError, FileNotFoundError):
        return 0.0