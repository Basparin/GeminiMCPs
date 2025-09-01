"""
CES Configuration Management

Handles loading, validation, and management of CES configuration settings
from environment variables, config files, and runtime parameters.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import json


@dataclass
class CESConfig:
    """
    Configuration settings for the Cognitive Enhancement System.

    Manages all configurable aspects of CES including AI assistants,
    memory settings, ethical guidelines, and performance parameters.
    """

    # Core settings
    debug_mode: bool = False
    log_level: str = "INFO"
    max_memory_mb: int = 256

    # AI Assistant settings
    grok_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None

    # Memory settings
    memory_db_path: str = "ces_memory.db"
    max_context_age_days: int = 90
    semantic_memory_size_mb: int = 500

    # Ethical settings
    ethical_checks_enabled: bool = True
    bias_detection_enabled: bool = True

    # Performance settings
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 300
    cache_enabled: bool = True

    # Development settings
    development_mode: bool = True
    enable_metrics: bool = True
    metrics_port: int = 9090

    def __post_init__(self):
        """Load configuration from environment and config files"""
        self._load_from_environment()
        self._load_from_config_file()
        self._validate_configuration()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # API Keys
        self.grok_api_key = os.getenv('GROK_API_KEY', self.grok_api_key)
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', self.gemini_api_key)
        self.qwen_api_key = os.getenv('QWEN_API_KEY', self.qwen_api_key)

        # Core settings
        self.debug_mode = os.getenv('CES_DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('CES_LOG_LEVEL', self.log_level)
        self.max_memory_mb = int(os.getenv('CES_MAX_MEMORY_MB', self.max_memory_mb))

        # Memory settings
        self.memory_db_path = os.getenv('CES_MEMORY_DB_PATH', self.memory_db_path)
        self.max_context_age_days = int(os.getenv('CES_MAX_CONTEXT_AGE_DAYS', self.max_context_age_days))
        self.semantic_memory_size_mb = int(os.getenv('CES_SEMANTIC_MEMORY_SIZE_MB', self.semantic_memory_size_mb))

        # Ethical settings
        self.ethical_checks_enabled = os.getenv('CES_ETHICAL_CHECKS_ENABLED', 'true').lower() == 'true'
        self.bias_detection_enabled = os.getenv('CES_BIAS_DETECTION_ENABLED', 'true').lower() == 'true'

        # Performance settings
        self.max_concurrent_tasks = int(os.getenv('CES_MAX_CONCURRENT_TASKS', self.max_concurrent_tasks))
        self.task_timeout_seconds = int(os.getenv('CES_TASK_TIMEOUT_SECONDS', self.task_timeout_seconds))
        self.cache_enabled = os.getenv('CES_CACHE_ENABLED', 'true').lower() == 'true'

        # Development settings
        self.development_mode = os.getenv('CES_DEVELOPMENT_MODE', 'true').lower() == 'true'
        self.enable_metrics = os.getenv('CES_ENABLE_METRICS', 'true').lower() == 'true'
        self.metrics_port = int(os.getenv('CES_METRICS_PORT', self.metrics_port))

    def _load_from_config_file(self):
        """Load configuration from JSON config file"""
        config_paths = [
            Path.home() / '.ces' / 'config.json',
            Path.cwd() / 'ces_config.json',
            Path.cwd() / 'config' / 'ces_config.json'
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)

                    # Update instance attributes from config file
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)

                    logging.info(f"Loaded configuration from {config_path}")
                    break
                except (json.JSONDecodeError, IOError) as e:
                    logging.warning(f"Failed to load config from {config_path}: {e}")

    def _validate_configuration(self):
        """Validate configuration settings"""
        issues = []

        # Validate memory settings
        if self.max_memory_mb < 64:
            issues.append("max_memory_mb must be at least 64MB")

        if self.semantic_memory_size_mb < 100:
            issues.append("semantic_memory_size_mb must be at least 100MB")

        # Validate performance settings
        if self.max_concurrent_tasks < 1:
            issues.append("max_concurrent_tasks must be at least 1")

        if self.task_timeout_seconds < 30:
            issues.append("task_timeout_seconds must be at least 30 seconds")

        # Validate ports
        if not (1024 <= self.metrics_port <= 65535):
            issues.append("metrics_port must be between 1024 and 65535")

        if issues:
            raise ValueError(f"Configuration validation failed: {', '.join(issues)}")

    def save_to_file(self, config_path: Optional[Path] = None):
        """Save current configuration to file"""
        if config_path is None:
            config_dir = Path.home() / '.ces'
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / 'config.json'

        config_data = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
            if not key.startswith('_')
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logging.info(f"Configuration saved to {config_path}")

    def get_ai_assistant_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for AI assistants"""
        return {
            'grok': {
                'api_key': self.grok_api_key,
                'enabled': self.grok_api_key is not None
            },
            'gemini': {
                'api_key': self.gemini_api_key,
                'enabled': self.gemini_api_key is not None
            },
            'qwen': {
                'api_key': self.qwen_api_key,
                'enabled': self.qwen_api_key is not None
            }
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory-related configuration"""
        return {
            'db_path': self.memory_db_path,
            'max_context_age_days': self.max_context_age_days,
            'semantic_memory_size_mb': self.semantic_memory_size_mb,
            'cache_enabled': self.cache_enabled
        }

    def get_ethical_config(self) -> Dict[str, Any]:
        """Get ethical configuration"""
        return {
            'checks_enabled': self.ethical_checks_enabled,
            'bias_detection_enabled': self.bias_detection_enabled
        }

    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.development_mode

    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': self.log_level,
            'debug_mode': self.debug_mode
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
            if not key.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CESConfig':
        """Create configuration from dictionary"""
        # Filter out invalid keys
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)