"""CES Configuration Manager.

Manages configuration settings for the Cognitive Enhancement System,
including environment variables, web interface settings, and runtime configuration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class WebConfig:
    """Web interface configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    cors_origins: list = None
    session_timeout_minutes: int = 60
    max_upload_size_mb: int = 10

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8001"]

@dataclass
class AIConfig:
    """AI assistant configuration."""
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    default_assistant: str = "grok"
    max_tokens_per_request: int = 4000
    request_timeout_seconds: int = 30

@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "ces_memory.db"
    backup_interval_hours: int = 24
    max_connections: int = 10
    connection_timeout_seconds: int = 30

@dataclass
class MonitoringConfig:
    """Monitoring and analytics configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    alert_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "response_time_ms": 2000,
                "error_rate_percent": 5.0,
                "memory_usage_percent": 80.0
            }

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_authentication: bool = False
    jwt_secret_key: Optional[str] = None
    session_secret: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

class ConfigManager:
    """Manages all CES configuration settings."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "ces_config.json"
        self._config = {}

        # Initialize default configurations
        self.web = WebConfig()
        self.ai = AIConfig()
        self.database = DatabaseConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if exists
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._apply_config(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")

        # Override with environment variables
        self._load_from_env()

        # Generate secrets if not set
        self._ensure_secrets()

    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration dictionary to config objects."""
        if "web" in config:
            web_config = config["web"]
            for key, value in web_config.items():
                if hasattr(self.web, key):
                    setattr(self.web, key, value)

        if "ai" in config:
            ai_config = config["ai"]
            for key, value in ai_config.items():
                if hasattr(self.ai, key):
                    setattr(self.ai, key, value)

        if "database" in config:
            db_config = config["database"]
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)

        if "monitoring" in config:
            mon_config = config["monitoring"]
            for key, value in mon_config.items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)

        if "security" in config:
            sec_config = config["security"]
            for key, value in sec_config.items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Web configuration
        self.web.host = os.getenv("CES_WEB_HOST", self.web.host)
        self.web.port = int(os.getenv("CES_WEB_PORT", self.web.port))
        self.web.debug = os.getenv("CES_DEBUG", "false").lower() == "true"

        # AI configuration
        self.ai.groq_api_key = os.getenv("GROQ_API_KEY", self.ai.groq_api_key)
        self.ai.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.ai.openrouter_api_key)
        self.ai.google_api_key = os.getenv("GOOGLE_API_KEY", self.ai.google_api_key)
        self.ai.default_assistant = os.getenv("CES_DEFAULT_ASSISTANT", self.ai.default_assistant)

        # Database configuration
        self.database.path = os.getenv("CES_DATABASE_PATH", self.database.path)

        # Monitoring configuration
        self.monitoring.enable_metrics = os.getenv("CES_ENABLE_METRICS", "true").lower() == "true"
        self.monitoring.log_level = os.getenv("CES_LOG_LEVEL", self.monitoring.log_level)

        # Security configuration
        self.security.enable_authentication = os.getenv("CES_ENABLE_AUTH", "false").lower() == "true"
        self.security.jwt_secret_key = os.getenv("CES_JWT_SECRET", self.security.jwt_secret_key)
        self.security.session_secret = os.getenv("CES_SESSION_SECRET", self.security.session_secret)

    def _ensure_secrets(self):
        """Ensure security secrets are set."""
        import secrets

        if not self.security.jwt_secret_key:
            self.security.jwt_secret_key = secrets.token_hex(32)

        if not self.security.session_secret:
            self.security.session_secret = secrets.token_hex(32)

    def save_config(self):
        """Save current configuration to file."""
        try:
            config = {
                "web": asdict(self.web),
                "ai": asdict(self.ai),
                "database": asdict(self.database),
                "monitoring": asdict(self.monitoring),
                "security": asdict(self.security)
            }

            # Remove sensitive information before saving
            if "groq_api_key" in config["ai"]:
                config["ai"]["groq_api_key"] = "***masked***" if config["ai"]["groq_api_key"] else None
            if "openrouter_api_key" in config["ai"]:
                config["ai"]["openrouter_api_key"] = "***masked***" if config["ai"]["openrouter_api_key"] else None
            if "google_api_key" in config["ai"]:
                config["ai"]["google_api_key"] = "***masked***" if config["ai"]["google_api_key"] else None

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved configuration to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_web_config(self) -> WebConfig:
        """Get web configuration."""
        return self.web

    def get_ai_config(self) -> AIConfig:
        """Get AI configuration."""
        return self.ai

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.database

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.monitoring

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.security

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.web.debug

    def get_cors_origins(self) -> list:
        """Get CORS origins for web interface."""
        return self.web.cors_origins

    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results."""
        issues = []

        # Check required AI API keys
        if not self.ai.groq_api_key and not self.ai.openrouter_api_key and not self.ai.google_api_key:
            issues.append({
                "severity": "error",
                "category": "ai",
                "message": "At least one AI API key must be configured"
            })

        # Check web configuration
        if self.web.port < 1024 and not self.is_production():
            issues.append({
                "severity": "warning",
                "category": "web",
                "message": "Using privileged port in development mode"
            })

        # Check database path
        db_path = Path(self.database.path)
        if not db_path.parent.exists():
            issues.append({
                "severity": "warning",
                "category": "database",
                "message": f"Database directory does not exist: {db_path.parent}"
            })

        # Check security settings
        if self.is_production() and not self.security.enable_https:
            issues.append({
                "severity": "warning",
                "category": "security",
                "message": "HTTPS should be enabled in production"
            })

        return {
            "valid": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues,
            "timestamp": json.dumps({"timestamp": "placeholder"})[13:-2]  # Simple ISO timestamp
        }

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return {
            "web": asdict(self.web),
            "ai": asdict(self.ai),
            "database": asdict(self.database),
            "monitoring": asdict(self.monitoring),
            "security": asdict(self.security)
        }

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for section, section_updates in updates.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in section_updates.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

        # Save updated configuration
        self.save_config()
        logger.info("Configuration updated")

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.web = WebConfig()
        self.ai = AIConfig()
        self.database = DatabaseConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()

        # Reload from environment
        self._load_from_env()
        self._ensure_secrets()

        logger.info("Configuration reset to defaults")