"""Configuration Tools Module for CodeSage MCP Server."""

import os
from codesage_mcp.utils import create_error_response, tool_error_handler
from codesage_mcp.config import ENABLE_CACHING
from codesage_mcp.cache import get_cache_instance


def configure_api_key_tool(llm_provider: str, api_key: str) -> dict:
    """Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI).

    This tool sets environment variables for API keys. For persistent configuration,
    add the keys to your .env file in the project root.
    """
    # Map provider to the environment variable name
    env_var_map = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var_name = env_var_map.get(llm_provider.lower())
    if not env_var_name:
        return {
            "error": {
                "code": "INVALID_PROVIDER",
                "message": (
                    f"Unsupported LLM provider: {llm_provider}. "
                    f"Supported providers are: groq, openrouter, google."
                ),
            }
        }

    # Validate the API key format (basic validation)
    if not api_key or len(api_key.strip()) == 0:
        return {
            "error": {
                "code": "INVALID_API_KEY",
                "message": "API key cannot be empty.",
            }
        }

    # Set the environment variable
    os.environ[env_var_name] = api_key.strip()

    # Update the global variables in config module
    import codesage_mcp.config as config_module

    if env_var_name == "GROQ_API_KEY":
        config_module.GROQ_API_KEY = api_key.strip()
    elif env_var_name == "OPENROUTER_API_KEY":
        config_module.OPENROUTER_API_KEY = api_key.strip()
    elif env_var_name == "GOOGLE_API_KEY":
        config_module.GOOGLE_API_KEY = api_key.strip()

    return {
        "message": (
            f"API key for {llm_provider} updated successfully. "
            "Note: This change is temporary and will be lost on server restart. "
            "For permanent configuration, add the key to your .env file."
        )
    }


@tool_error_handler
def get_configuration_tool() -> dict:
    """Returns the current configuration, with API keys masked for security."""
    from codesage_mcp.config import (
        GROQ_API_KEY,
        OPENROUTER_API_KEY,
        GOOGLE_API_KEY,
        get_configuration_status,
    )

    def mask_api_key(key: str) -> str:
        """Mask an API key, showing only the first and last few characters."""
        if not key:
            return "Not set"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"

    config_status = get_configuration_status()

    return {
        "message": "Current configuration retrieved successfully.",
        "configuration": {
            "groq_api_key": mask_api_key(GROQ_API_KEY),
            "openrouter_api_key": mask_api_key(OPENROUTER_API_KEY),
            "google_api_key": mask_api_key(GOOGLE_API_KEY),
        },
        "status": config_status,
    }


@tool_error_handler
def get_cache_statistics_tool() -> dict:
    """Returns comprehensive statistics about the intelligent caching system."""
    if not ENABLE_CACHING:
        return {
            "message": "Caching is disabled in configuration.",
            "caching_enabled": False,
        }

    try:
        cache = get_cache_instance()
        stats = cache.get_comprehensive_stats()

        return {
            "message": "Cache statistics retrieved successfully.",
            "caching_enabled": True,
            "statistics": stats,
        }

    except Exception as e:
        return create_error_response(
            "CACHE_ERROR", f"Failed to retrieve cache statistics: {e}"
        )
