"""Configuration Module for CodeSage MCP Server.

This module manages the configuration for the CodeSage MCP Server, including API keys
for various services like Groq, OpenRouter, and Google AI.

It uses `python-dotenv` to automatically load environment variables from a `.env` file
located in the project root directory. This allows for secure configuration without
hardcoding sensitive information in the source code.

Environment Variables:
    GROQ_API_KEY (str): API key for Groq.
    OPENROUTER_API_KEY (str): API key for OpenRouter.
    GOOGLE_API_KEY (str): API key for Google AI.

Example .env file:
    ```env
    GROQ_API_KEY="gsk_..."
    OPENROUTER_API_KEY="sk-or-..."
    GOOGLE_API_KEY="AIza..."
    ```
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_required_env_var(var_name: str) -> str:
    """Get a required environment variable with validation.

    Args:
        var_name: Name of the environment variable

    Returns:
        The environment variable value

    Raises:
        ValueError: If the environment variable is not set or empty
    """
    value = os.getenv(var_name)
    if not value or value.strip() == "":
        raise ValueError(
            f"Required environment variable '{var_name}' is not set. "
            f"Please set it in your .env file or environment."
        )
    return value.strip()


def get_optional_env_var(var_name: str) -> Optional[str]:
    """Get an optional environment variable.

    Args:
        var_name: Name of the environment variable

    Returns:
        The environment variable value or None if not set
    """
    value = os.getenv(var_name)
    return value.strip() if value else None


# Load API keys from environment variables
try:
    GROQ_API_KEY = get_required_env_var("GROQ_API_KEY")
except ValueError:
    GROQ_API_KEY = None

try:
    OPENROUTER_API_KEY = get_required_env_var("OPENROUTER_API_KEY")
except ValueError:
    OPENROUTER_API_KEY = None

try:
    GOOGLE_API_KEY = get_required_env_var("GOOGLE_API_KEY")
except ValueError:
    GOOGLE_API_KEY = None


def validate_configuration() -> list[str]:
    """Validate the current configuration and return any issues found.

    Returns:
        List of configuration issues (empty if all valid)
    """
    issues = []

    if not GROQ_API_KEY:
        issues.append("GROQ_API_KEY is not configured")

    if not OPENROUTER_API_KEY:
        issues.append("OPENROUTER_API_KEY is not configured")

    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY is not configured")

    return issues


def get_configuration_status() -> dict:
    """Get the current configuration status.

    Returns:
        Dictionary with configuration status information
    """
    issues = validate_configuration()

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "configured_providers": {
            "groq": GROQ_API_KEY is not None,
            "openrouter": OPENROUTER_API_KEY is not None,
            "google": GOOGLE_API_KEY is not None,
        }
    }
