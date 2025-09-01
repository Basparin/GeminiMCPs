# CES Plugin Development Guide

## Overview

The CES Plugin Architecture provides a powerful and flexible system for extending CES functionality. This guide covers everything you need to know to develop, test, and distribute CES plugins.

## Plugin Architecture

### Core Concepts

#### Plugin Base Class
All CES plugins must inherit from the `CESPlugin` base class:

```python
from ces.plugins.base import CESPlugin, PluginInfo

class MyPlugin(CESPlugin):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="My Custom Plugin",
            version="1.0.0",
            description="Description of what the plugin does",
            author="Your Name",
            license="MIT",
            homepage="https://github.com/your/plugin-repo",
            tags=["tag1", "tag2"]
        )
```

#### Plugin Context
Plugins receive a context object with CES information:

```python
def initialize(self, context):
    super().initialize(context)

    # Access CES configuration
    ces_config = context.config

    # Access logger
    logger = context.logger

    # Access data directory (plugin-specific)
    data_dir = context.data_directory

    return True
```

### Plugin Lifecycle

#### 1. Discovery
CES discovers plugins from these locations:
- `./plugins/` (project plugins directory)
- `./ces/plugins/` (built-in plugins)
- `~/.ces/plugins/` (user plugins)

#### 2. Loading
```python
# Plugin is loaded by the plugin manager
plugin_manager.load_plugin("my_plugin")
```

#### 3. Initialization
```python
def initialize(self, context):
    # Plugin initialization code
    super().initialize(context)

    # Register hooks
    # Set up data structures
    # Initialize connections

    return True  # Return False to fail initialization
```

#### 4. Enabling
```python
def on_enable(self):
    # Plugin activation code
    # Start background tasks
    # Register event handlers
    pass
```

#### 5. Runtime
Plugin runs and responds to hooks and events.

#### 6. Disabling
```python
def on_disable(self):
    # Plugin deactivation code
    # Stop background tasks
    # Clean up resources
    pass
```

#### 7. Unloading
Plugin is removed from memory.

## Hook System

### Available Hooks

#### TaskExecutionHook
Triggered when tasks are executed:

```python
from ces.plugins.base import TaskExecutionHook

def on_task_execution(self, task_description: str, assistant: str, result: dict):
    """Handle task execution events"""
    self.logger.info(f"Task executed: {task_description}")
    # Custom logic here

# Register the hook
hook = TaskExecutionHook()
hook.register(self.on_task_execution)
self.register_hook(hook)
```

#### AIInteractionHook
Triggered during AI assistant interactions:

```python
from ces.plugins.base import AIInteractionHook

def on_ai_interaction(self, assistant: str, interaction_type: str, data: dict):
    """Handle AI interactions"""
    if interaction_type == "response":
        # Process AI response
        pass

hook = AIInteractionHook()
hook.register(self.on_ai_interaction)
self.register_hook(hook)
```

#### FeedbackSubmissionHook
Triggered when user feedback is submitted:

```python
from ces.plugins.base import FeedbackSubmissionHook

def on_feedback_submission(self, feedback_data: dict):
    """Handle feedback submission"""
    # Process feedback
    # Send notifications
    # Update metrics
    pass

hook = FeedbackSubmissionHook()
hook.register(self.on_feedback_submission)
self.register_hook(hook)
```

#### AnalyticsEventHook
Triggered for analytics events:

```python
from ces.plugins.base import AnalyticsEventHook

def on_analytics_event(self, event_type: str, data: dict):
    """Handle analytics events"""
    # Process analytics data
    # Update custom metrics
    pass

hook = AnalyticsEventHook()
hook.register(self.on_analytics_event)
self.register_hook(hook)
```

#### SystemHealthHook
Triggered for system health checks:

```python
from ces.plugins.base import SystemHealthHook

def on_system_health(self, health_data: dict):
    """Handle system health events"""
    # Monitor system health
    # Send alerts if needed
    pass

hook = SystemHealthHook()
hook.register(self.on_system_health)
self.register_hook(hook)
```

### Custom Hooks
You can create custom hooks for your plugins:

```python
from ces.plugins.base import PluginHook

class CustomHook(PluginHook):
    def __init__(self):
        super().__init__("my_custom_event", "Custom plugin event")

# Register and trigger custom hooks
custom_hook = CustomHook()
custom_hook.register(self.my_handler)
self.register_hook(custom_hook)

# Trigger the hook
await self.trigger_hook("my_custom_event", data)
```

## Configuration

### Plugin Configuration Schema
```python
def get_config_schema(self) -> dict:
    """Return plugin configuration schema"""
    return {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "description": "API key for external service"
            },
            "enable_feature": {
                "type": "boolean",
                "default": True,
                "description": "Enable advanced features"
            },
            "max_retries": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
                "description": "Maximum retry attempts"
            }
        },
        "required": ["api_key"]
    }
```

### Accessing Configuration
```python
def initialize(self, context):
    super().initialize(context)

    # Get plugin configuration
    config = context.config

    # Access specific settings
    api_key = config.get("api_key")
    enable_feature = config.get("enable_feature", True)
```

### Configuration Changes
```python
def on_config_change(self, new_config: dict):
    """Handle configuration changes"""
    # Update plugin settings
    self.api_key = new_config.get("api_key")
    self.enable_feature = new_config.get("enable_feature", True)

    # Reinitialize connections if needed
    if self.connection:
        self.connection.update_config(new_config)
```

## Data Persistence

### Plugin Data Directory
Each plugin gets its own data directory:

```python
def initialize(self, context):
    super().initialize(context)

    # Plugin-specific data directory
    self.data_dir = context.data_directory

    # Create subdirectories as needed
    self.cache_dir = os.path.join(self.data_dir, "cache")
    self.logs_dir = os.path.join(self.data_dir, "logs")
    os.makedirs(self.cache_dir, exist_ok=True)
    os.makedirs(self.logs_dir, exist_ok=True)
```

### Storing Data
```python
import json
import os

def save_plugin_data(self, data: dict, filename: str):
    """Save plugin data to file"""
    filepath = os.path.join(self.data_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_plugin_data(self, filename: str) -> dict:
    """Load plugin data from file"""
    filepath = os.path.join(self.data_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}
```

## Error Handling

### Plugin Exceptions
```python
class PluginError(Exception):
    """Base plugin exception"""
    pass

class ConfigurationError(PluginError):
    """Configuration-related errors"""
    pass

class ConnectionError(PluginError):
    """Connection-related errors"""
    pass
```

### Error Handling in Hooks
```python
def on_task_execution(self, task_description: str, assistant: str, result: dict):
    try:
        # Plugin logic here
        self.process_task(task_description, assistant, result)
    except Exception as e:
        # Log error but don't crash the system
        self.logger.error(f"Plugin error in task execution: {e}")
        # Optionally re-raise if critical
        # raise PluginError(f"Task processing failed: {e}")
```

## Testing Plugins

### Unit Tests
```python
import unittest
from unittest.mock import Mock, patch
from ces.plugins.base import PluginContext

class TestMyPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = MyPlugin()
        self.mock_context = Mock(spec=PluginContext)
        self.mock_context.config = {"api_key": "test_key"}
        self.mock_context.logger = Mock()
        self.mock_context.data_directory = "/tmp/test_plugin"

    def test_initialization(self):
        """Test plugin initialization"""
        result = self.plugin.initialize(self.mock_context)
        self.assertTrue(result)
        self.assertEqual(self.plugin.context, self.mock_context)

    def test_hook_registration(self):
        """Test hook registration"""
        self.plugin.initialize(self.mock_context)
        hooks = self.plugin.get_hooks()
        self.assertIn("task_execution", hooks)

    @patch('my_plugin.ExternalAPI')
    def test_task_processing(self, mock_api):
        """Test task processing functionality"""
        # Test implementation
        pass
```

### Integration Tests
```python
import asyncio
from ces.plugins.manager import plugin_manager

class TestPluginIntegration(unittest.TestCase):
    def setUp(self):
        self.plugin_manager = plugin_manager

    def test_plugin_loading(self):
        """Test plugin loading and enabling"""
        # Load plugin
        result = self.plugin_manager.load_plugin("my_plugin")
        self.assertTrue(result)

        # Enable plugin
        result = self.plugin_manager.enable_plugin("my_plugin")
        self.assertTrue(result)

        # Verify plugin is active
        plugins = self.plugin_manager.list_plugins()
        self.assertIn("my_plugin", plugins)
        self.assertTrue(plugins["my_plugin"]["enabled"])
```

## Plugin Distribution

### Directory Structure
```
my_plugin/
├── __init__.py          # Plugin class definition
├── plugin.json          # Plugin metadata (optional)
├── requirements.txt     # Plugin dependencies (optional)
├── README.md           # Plugin documentation
├── tests/              # Plugin tests
│   ├── __init__.py
│   └── test_plugin.py
└── docs/               # Additional documentation
    └── usage.md
```

### Plugin Metadata (plugin.json)
```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "My custom CES plugin",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/your/my-plugin",
  "repository": "https://github.com/your/my-plugin.git",
  "dependencies": ["requests", "pandas"],
  "tags": ["analytics", "api"],
  "min_ces_version": "0.4.0",
  "max_ces_version": "1.0.0"
}
```

### Installation
1. **Manual Installation**
   ```bash
   # Copy plugin to user plugins directory
   cp -r my_plugin ~/.ces/plugins/

   # Or to project plugins directory
   cp -r my_plugin ./plugins/
   ```

2. **CLI Installation**
   ```bash
   # Discover plugins
   ces plugin discover

   # Load and enable
   ces plugin load my_plugin
   ces plugin enable my_plugin
   ```

## Best Practices

### Code Quality
1. **Follow PEP 8**: Use consistent Python style
2. **Add Type Hints**: Include type annotations
3. **Write Documentation**: Document all public methods
4. **Handle Errors Gracefully**: Don't crash the main system

### Performance
1. **Async Operations**: Use async/await for I/O operations
2. **Resource Management**: Clean up resources properly
3. **Caching**: Cache expensive operations when possible
4. **Background Tasks**: Use appropriate threading/async patterns

### Security
1. **Input Validation**: Validate all inputs
2. **Secure Storage**: Don't store sensitive data in plain text
3. **API Keys**: Use secure key management
4. **Network Security**: Use HTTPS for external connections

### Compatibility
1. **Version Checking**: Check CES version compatibility
2. **Dependency Management**: Specify and check dependencies
3. **Backward Compatibility**: Maintain compatibility when possible
4. **Migration Paths**: Provide migration guides for breaking changes

## Example Plugin

### Complete Example
```python
"""
Example CES Plugin - Task Notifier

Sends notifications when tasks are completed.
"""

import asyncio
import json
import os
from typing import Dict, Any
from ces.plugins.base import CESPlugin, PluginInfo, TaskExecutionHook

class TaskNotifierPlugin(CESPlugin):
    """Plugin that sends notifications for completed tasks"""

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="Task Notifier",
            version="1.0.0",
            description="Sends notifications when tasks are completed",
            author="CES Team",
            license="MIT",
            tags=["notification", "task", "productivity"]
        )

    def initialize(self, context):
        super().initialize(context)

        # Plugin configuration
        self.notification_url = context.config.get("notification_url")
        self.enable_email = context.config.get("enable_email", False)

        # Register hooks
        task_hook = TaskExecutionHook()
        task_hook.register(self.on_task_execution)
        self.register_hook(task_hook)

        # Load notification history
        self.history_file = os.path.join(context.data_directory, "notification_history.json")
        self.notification_history = self.load_history()

        self.logger.info("Task Notifier plugin initialized")
        return True

    def on_task_execution(self, task_description: str, assistant: str, result: Dict[str, Any]):
        """Handle task execution completion"""
        try:
            # Check if task was successful
            if result.get("status") == "completed":
                # Send notification
                self.send_notification(task_description, assistant, result)

                # Record in history
                self.record_notification(task_description, assistant, result)

        except Exception as e:
            self.logger.error(f"Error in task notification: {e}")

    def send_notification(self, task: str, assistant: str, result: Dict[str, Any]):
        """Send notification about completed task"""
        message = f"Task completed: {task[:50]}..."
        details = f"Assistant: {assistant}, Time: {result.get('execution_time', 'N/A')}s"

        if self.notification_url:
            # Send HTTP notification
            asyncio.create_task(self.send_http_notification(message, details))
        elif self.enable_email:
            # Send email notification
            asyncio.create_task(self.send_email_notification(message, details))
        else:
            # Log notification
            self.logger.info(f"Task notification: {message} - {details}")

    async def send_http_notification(self, message: str, details: str):
        """Send HTTP notification"""
        try:
            import aiohttp

            payload = {
                "message": message,
                "details": details,
                "timestamp": str(asyncio.get_event_loop().time())
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.notification_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("HTTP notification sent successfully")
                    else:
                        self.logger.error(f"HTTP notification failed: {response.status}")

        except Exception as e:
            self.logger.error(f"HTTP notification error: {e}")

    async def send_email_notification(self, message: str, details: str):
        """Send email notification (placeholder)"""
        # Implement email sending logic here
        self.logger.info(f"Email notification: {message}")

    def record_notification(self, task: str, assistant: str, result: Dict[str, Any]):
        """Record notification in history"""
        notification = {
            "task": task,
            "assistant": assistant,
            "execution_time": result.get("execution_time"),
            "timestamp": str(asyncio.get_event_loop().time())
        }

        self.notification_history.append(notification)

        # Keep only last 1000 notifications
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]

        # Save to file
        self.save_history()

    def load_history(self) -> list:
        """Load notification history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading notification history: {e}")
        return []

    def save_history(self):
        """Save notification history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.notification_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving notification history: {e}")

    def get_config_schema(self) -> Dict[str, Any]:
        """Get plugin configuration schema"""
        return {
            "type": "object",
            "properties": {
                "notification_url": {
                    "type": "string",
                    "description": "Webhook URL for notifications"
                },
                "enable_email": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable email notifications"
                }
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            "total_notifications": len(self.notification_history),
            "recent_notifications": self.notification_history[-10:] if self.notification_history else []
        }
```

## Plugin Registry

### Official Plugin Repository
CES maintains an official plugin repository at:
- **GitHub**: `https://github.com/ces-project/plugins`
- **Documentation**: Plugin registry with examples and templates

### Community Plugins
- **Community Repository**: User-contributed plugins
- **Plugin Marketplace**: Web interface for plugin discovery
- **Rating System**: Community ratings and reviews

## Support and Resources

### Getting Help
1. **Plugin Development Forum**: Community discussions
2. **GitHub Issues**: Bug reports and feature requests
3. **Plugin Examples**: Official examples repository
4. **Documentation**: This development guide

### Resources
- **Plugin Template**: Start developing with a template
- **Testing Framework**: Plugin testing utilities
- **CI/CD Integration**: Automated testing and deployment
- **Security Guidelines**: Plugin security best practices

---

*This guide covers CES plugin development for Phase 0.4. Check the official documentation for updates and additional resources.*