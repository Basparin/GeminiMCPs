"""Configuration Tools Module for CodeSage MCP Server."""

import os
from codesage_mcp.codebase_manager import codebase_manager
from codesage_mcp.utils import create_error_response, tool_error_handler, safe_read_file


def configure_api_key_tool(llm_provider: str, api_key: str) -> dict:
    """Configures API keys for LLMs (e.g., Groq, OpenRouter, Google AI)."""
    config_file_path = "/home/basparin/Escritorio/GeminiMCPs/codesage_mcp/config.py"

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

    # Read the config file content
    lines = safe_read_file(config_file_path, as_lines=True)

    updated_lines = []
    key_updated = False
    for line in lines:
        if line.strip().startswith(f"{env_var_name} ="):
            updated_lines.append(f'{env_var_name} = "{api_key}"\n')
            key_updated = True
        else:
            updated_lines.append(line)

    if not key_updated:
        # If the key was not found, append it to the end of the file
        updated_lines.append(f'{env_var_name} = "{api_key}"\n')

    # Write the updated content back to the file
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    return {
        "message": (
            f"API key for {llm_provider} updated successfully. "
            "A server restart may be required for changes to take full effect."
        )
    }


@tool_error_handler
def get_configuration_tool() -> dict:
    """Returns the current configuration, with API keys masked for security."""
    from codesage_mcp.config import GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY

    def mask_api_key(key: str) -> str:
        """Mask an API key, showing only the first and last few characters."""
        if not key:
            return "Not set"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"

    return {
        "message": "Current configuration retrieved successfully.",
        "configuration": {
            "groq_api_key": mask_api_key(GROQ_API_KEY),
            "openrouter_api_key": mask_api_key(OPENROUTER_API_KEY),
            "google_api_key": mask_api_key(GOOGLE_API_KEY),
        },
    }