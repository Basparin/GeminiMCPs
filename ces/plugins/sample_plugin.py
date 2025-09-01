"""
CES Sample Plugin

Demonstrates the CES plugin architecture with basic functionality.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .base import CESPlugin, PluginInfo, TaskExecutionHook, AIInteractionHook


class SamplePlugin(CESPlugin):
    """
    Sample plugin demonstrating CES plugin capabilities
    """

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="CES Sample Plugin",
            version="1.0.0",
            description="A sample plugin demonstrating CES plugin architecture",
            author="CES Development Team",
            license="MIT",
            homepage="https://github.com/ces-project/sample-plugin",
            tags=["sample", "demo", "example"]
        )

    def initialize(self, context):
        """Initialize the sample plugin"""
        super().initialize(context)

        # Register hooks
        task_hook = TaskExecutionHook()
        task_hook.register(self.on_task_execution)

        ai_hook = AIInteractionHook()
        ai_hook.register(self.on_ai_interaction)

        self.register_hook(task_hook)
        self.register_hook(ai_hook)

        # Plugin-specific initialization
        self.task_count = 0
        self.ai_interactions = 0

        self.logger.info("Sample plugin initialized with hooks registered")
        return True

    def on_task_execution(self, task_description: str, assistant: str, result: Dict[str, Any]):
        """Handle task execution events"""
        self.task_count += 1
        self.logger.info(f"Sample plugin: Task executed by {assistant} - {task_description[:50]}...")

        # Could perform additional processing here
        # For example: logging, metrics collection, notifications, etc.

    def on_ai_interaction(self, assistant: str, interaction_type: str, data: Dict[str, Any]):
        """Handle AI interaction events"""
        self.ai_interactions += 1
        self.logger.info(f"Sample plugin: AI interaction with {assistant} ({interaction_type})")

        # Could perform additional processing here
        # For example: conversation analysis, quality monitoring, etc.

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            "tasks_processed": self.task_count,
            "ai_interactions": self.ai_interactions,
            "uptime": str(datetime.now() - self.info.created_at) if self.info else "Unknown"
        }

    def on_enable(self):
        """Called when plugin is enabled"""
        self.logger.info("Sample plugin enabled")

    def on_disable(self):
        """Called when plugin is disabled"""
        self.logger.info("Sample plugin disabled")

    def get_config_schema(self) -> Dict[str, Any]:
        """Get plugin configuration schema"""
        return {
            "type": "object",
            "properties": {
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "INFO"
                },
                "enable_metrics": {
                    "type": "boolean",
                    "default": True
                }
            }
        }