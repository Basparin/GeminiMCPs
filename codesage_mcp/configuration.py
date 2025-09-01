"""
Configuration Module for CodeSage MCP Server.

This module provides classes for managing configuration settings, environment
variables, and API key validation. It supports loading configuration from
environment variables, files, and provides validation and management features.

Classes:
    ConfigManager: Manages configuration loading and validation
    EnvVarHandler: Handles environment variable operations
    APIKeyValidator: Validates API keys and manages their lifecycle
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from .core.exceptions import ConfigurationError

# Set up logger
import logging
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading, validation, and management.

    This class handles loading configuration from various sources (environment
    variables, files, defaults), validates configuration values, and provides
    methods for saving and merging configurations.

    Attributes:
        config: Current configuration dictionary
        defaults: Default configuration values
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.config = {}
        self.defaults = {
            "api_key": None,
            "debug": False,
            "max_memory_mb": 2048,
            "cache_enabled": True,
            "log_level": "INFO"
        }

    def load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Dictionary containing loaded configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        try:
            config = {}

            # Load API key
            api_key = os.getenv("API_KEY")
            if api_key:
                config["api_key"] = api_key

            # Load debug setting
            debug_str = os.getenv("DEBUG", "false").lower()
            config["debug"] = debug_str in ("true", "1", "yes", "on")

            # Load memory limit
            max_memory = os.getenv("MAX_MEMORY_MB", "2048")
            try:
                config["max_memory_mb"] = int(max_memory)
            except ValueError:
                logger.warning(f"Invalid MAX_MEMORY_MB value '{max_memory}', using default")
                config["max_memory_mb"] = self.defaults["max_memory_mb"]

            # Load cache setting
            cache_enabled = os.getenv("CACHE_ENABLED", "true").lower()
            config["cache_enabled"] = cache_enabled in ("true", "1", "yes", "on")

            # Load log level
            config["log_level"] = os.getenv("LOG_LEVEL", self.defaults["log_level"])

            self.config.update(config)
            return config

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from environment: {e}",
                config_key="environment"
            )

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration values.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Validate API key if present
            if "api_key" in config:
                api_key = config["api_key"]
                if api_key and not isinstance(api_key, str):
                    raise ConfigurationError("API key must be a string", config_key="api_key")

            # Validate memory limit
            if "max_memory_mb" in config:
                memory = config["max_memory_mb"]
                if not isinstance(memory, int) or memory <= 0:
                    raise ConfigurationError("Memory limit must be a positive integer", config_key="max_memory_mb")

            # Validate debug setting
            if "debug" in config:
                debug = config["debug"]
                if not isinstance(debug, bool):
                    raise ConfigurationError("Debug must be a boolean", config_key="debug")

            # Validate log level
            if "log_level" in config:
                log_level = config["log_level"]
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                if log_level not in valid_levels:
                    raise ConfigurationError(f"Log level must be one of {valid_levels}", config_key="log_level")

            return True

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def save_to_file(self, config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration dictionary to save
            filepath: Path to save the configuration

        Raises:
            ConfigurationError: If save operation fails
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {filepath}: {e}",
                config_key="file_save"
            )

    def load_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            filepath: Path to load configuration from

        Returns:
            Loaded configuration dictionary

        Raises:
            ConfigurationError: If load operation fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Validate loaded configuration
            self.validate(config)

            self.config.update(config)
            return config

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {filepath}", config_key="file_load")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}", config_key="file_load")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {filepath}: {e}", config_key="file_load")

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration to override with

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Deep merge for nested dictionaries
                merged[key] = self.merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.config.copy()

    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)


class EnvVarHandler:
    """
    Handles environment variable operations.

    This class provides methods for getting, setting, and managing environment
    variables with support for defaults, type conversion, and validation.

    Attributes:
        prefix: Optional prefix for environment variables
    """

    def __init__(self, prefix: Optional[str] = None):
        """
        Initialize the environment variable handler.

        Args:
            prefix: Optional prefix for environment variables
        """
        self.prefix = prefix

    def get(self, key: str, default: Any = None, var_type: type = str) -> Any:
        """
        Get an environment variable with type conversion.

        Args:
            key: Environment variable name
            default: Default value if variable not set
            var_type: Type to convert the value to

        Returns:
            Environment variable value or default
        """
        full_key = f"{self.prefix}_{key}" if self.prefix else key

        value = os.getenv(full_key)
        if value is None:
            return default

        try:
            if var_type is bool:
                return value.lower() in ("true", "1", "yes", "on")
            elif var_type is int:
                return int(value)
            elif var_type is float:
                return float(value)
            else:
                return value
        except (ValueError, TypeError):
            logger.exception(f"Failed to convert environment variable {full_key} to {var_type.__name__}, using default")
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set an environment variable.

        Args:
            key: Environment variable name
            value: Value to set
        """
        full_key = f"{self.prefix}_{key}" if self.prefix else key
        os.environ[full_key] = str(value)

    def list_with_prefix(self, prefix: str) -> Dict[str, str]:
        """
        List all environment variables with a given prefix.

        Args:
            prefix: Prefix to filter environment variables

        Returns:
            Dictionary of matching environment variables
        """
        result = {}
        prefix_upper = prefix.upper()

        for key, value in os.environ.items():
            if key.upper().startswith(prefix_upper):
                result[key] = value

        return result

    def exists(self, key: str) -> bool:
        """
        Check if an environment variable exists.

        Args:
            key: Environment variable name

        Returns:
            True if the variable exists
        """
        full_key = f"{self.prefix}_{key}" if self.prefix else key
        return full_key in os.environ

    def delete(self, key: str) -> None:
        """
        Delete an environment variable.

        Args:
            key: Environment variable name
        """
        full_key = f"{self.prefix}_{key}" if self.prefix else key
        if full_key in os.environ:
            del os.environ[full_key]


class APIKeyValidator:
    """
    Validates API keys and manages their lifecycle.

    This class provides methods for validating API key format, checking
    expiry, and managing API key operations.

    Attributes:
        key_patterns: Dictionary of valid key patterns by provider
    """

    def __init__(self):
        """Initialize the API key validator."""
        self.key_patterns = {
            "openai": r"sk-\w{48}",
            "anthropic": r"sk-ant-\w{95}",
            "groq": r"gsk_\w{50}",
            "openrouter": r"sk-or-v1-\w{64}",
            "google": r"AIza\w{35}"
        }

    def validate(self, api_key: str, provider: Optional[str] = None) -> bool:
        """
        Validate an API key.

        Args:
            api_key: API key to validate
            provider: Optional provider name for specific validation

        Returns:
            True if the API key is valid

        Raises:
            ConfigurationError: If API key format is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ConfigurationError("API key must be a non-empty string", config_key="api_key")

        if len(api_key.strip()) == 0:
            raise ConfigurationError("API key cannot be empty", config_key="api_key")

        # Check minimum length
        if len(api_key) < 10:
            raise ConfigurationError("API key is too short", config_key="api_key")

        # Provider-specific validation
        if provider and provider in self.key_patterns:
            import re
            if not re.match(self.key_patterns[provider], api_key):
                raise ConfigurationError(f"Invalid API key format for {provider}", config_key="api_key")

        return True

    def format(self, api_key: str) -> str:
        """
        Format and normalize an API key.

        Args:
            api_key: API key to format

        Returns:
            Formatted API key
        """
        return api_key.strip()

    def is_expired(self, api_key: str) -> bool:
        """
        Check if an API key is expired.

        Args:
            api_key: API key to check

        Returns:
            True if the key is expired (simplified implementation)
        """
        # This is a simplified implementation
        # In a real system, you'd check against a key registry or external service
        return False

    def mask_key(self, api_key: str) -> str:
        """
        Mask an API key for logging/display.

        Args:
            api_key: API key to mask

        Returns:
            Masked API key
        """
        if not api_key or len(api_key) < 8:
            return "****"

        return f"{api_key[:4]}****{api_key[-4:]}"

    def get_provider_from_key(self, api_key: str) -> Optional[str]:
        """
        Attempt to identify the provider from API key format.

        Args:
            api_key: API key to analyze

        Returns:
            Provider name if identifiable, None otherwise
        """
        import re

        for provider, pattern in self.key_patterns.items():
            if re.match(pattern, api_key):
                return provider

        return None