"""
CES Plugin Base Classes

Defines the base interfaces and classes for CES plugins.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PluginInfo:
    """Information about a plugin"""
    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: Optional[str] = None
    repository: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []


@dataclass
class PluginContext:
    """Context information passed to plugins"""
    ces_version: str
    config: Dict[str, Any]
    logger: logging.Logger
    data_directory: str
    temp_directory: str


class PluginHook:
    """Base class for plugin hooks"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.handlers: List[Callable] = []

    def register(self, handler: Callable):
        """Register a handler for this hook"""
        if handler not in self.handlers:
            self.handlers.append(handler)

    def unregister(self, handler: Callable):
        """Unregister a handler for this hook"""
        if handler in self.handlers:
            self.handlers.remove(handler)

    async def trigger(self, *args, **kwargs):
        """Trigger all handlers for this hook"""
        results = []
        for handler in self.handlers:
            try:
                if hasattr(handler, '__call__'):
                    result = handler(*args, **kwargs)
                    if hasattr(result, '__await__'):
                        result = await result
                    results.append(result)
            except Exception as e:
                # Log error but continue with other handlers
                logging.getLogger(__name__).error(f"Plugin hook handler error: {e}")
        return results


class CESPlugin(ABC):
    """
    Base class for CES plugins

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self):
        self.info: Optional[PluginInfo] = None
        self.context: Optional[PluginContext] = None
        self.logger: Optional[logging.Logger] = None
        self.is_enabled = False
        self.hooks: Dict[str, PluginHook] = {}

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin information"""
        pass

    def initialize(self, context: PluginContext) -> bool:
        """
        Initialize the plugin

        Args:
            context: Plugin context with CES information

        Returns:
            True if initialization successful, False otherwise
        """
        self.context = context
        self.logger = context.logger
        self.info = self.get_info()
        self.is_enabled = True

        self.logger.info(f"Plugin {self.info.name} v{self.info.version} initialized")
        return True

    def shutdown(self):
        """Shutdown the plugin"""
        self.is_enabled = False
        if self.logger:
            self.logger.info(f"Plugin {self.info.name} shutdown")

    def is_compatible(self, ces_version: str) -> bool:
        """
        Check if plugin is compatible with CES version

        Args:
            ces_version: CES version string

        Returns:
            True if compatible, False otherwise
        """
        # Default implementation - override for version-specific compatibility
        return True

    def get_dependencies(self) -> List[str]:
        """Get list of plugin dependencies"""
        return self.info.dependencies if self.info else []

    def get_hooks(self) -> Dict[str, PluginHook]:
        """Get plugin hooks"""
        return self.hooks

    def register_hook(self, hook: PluginHook):
        """Register a plugin hook"""
        self.hooks[hook.name] = hook

    def unregister_hook(self, hook_name: str):
        """Unregister a plugin hook"""
        if hook_name in self.hooks:
            del self.hooks[hook_name]

    # Plugin lifecycle methods - override as needed
    def on_enable(self):
        """Called when plugin is enabled"""
        pass

    def on_disable(self):
        """Called when plugin is disabled"""
        pass

    def on_config_change(self, new_config: Dict[str, Any]):
        """Called when configuration changes"""
        pass

    def get_config_schema(self) -> Optional[Dict[str, Any]]:
        """Get plugin configuration schema"""
        return None

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration"""
        return True


# Standard CES Plugin Hooks
class TaskExecutionHook(PluginHook):
    """Hook for task execution events"""

    def __init__(self):
        super().__init__("task_execution", "Triggered during task execution")

class AIInteractionHook(PluginHook):
    """Hook for AI assistant interactions"""

    def __init__(self):
        super().__init__("ai_interaction", "Triggered during AI interactions")

class FeedbackSubmissionHook(PluginHook):
    """Hook for feedback submission events"""

    def __init__(self):
        super().__init__("feedback_submission", "Triggered when feedback is submitted")

class AnalyticsEventHook(PluginHook):
    """Hook for analytics events"""

    def __init__(self):
        super().__init__("analytics_event", "Triggered for analytics events")

class SystemHealthHook(PluginHook):
    """Hook for system health monitoring"""

    def __init__(self):
        super().__init__("system_health", "Triggered for system health checks")