"""CES Plugin Architecture.

Provides a comprehensive plugin system for extending CES functionality,
including plugin discovery, loading, lifecycle management, and security.
"""

import asyncio
import importlib
import inspect
import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, asdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class PluginMetadata:
    """Plugin metadata structure."""
    name: str
    version: str
    description: str
    author: str
    license: str
    dependencies: List[str]
    capabilities: List[str]
    entry_point: str
    config_schema: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class PluginInstance:
    """Plugin instance with metadata and runtime information."""
    metadata: PluginMetadata
    instance: Any
    status: str  # loaded, active, inactive, error
    loaded_at: str
    config: Dict[str, Any]
    error_message: Optional[str] = None

class PluginInterface(ABC):
    """Abstract base class for CES plugins."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin gracefully."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of plugin capabilities."""
        pass

    @abstractmethod
    def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute a plugin action."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status and health information."""
        pass

class PluginManager:
    """Manages plugin lifecycle, discovery, and execution."""

    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.plugins: Dict[str, PluginInstance] = {}
        self.capability_registry: Dict[str, List[str]] = {}  # capability -> plugin_names
        self.event_listeners: Dict[str, List[Callable]] = {}
        self.security_manager = PluginSecurityManager()

        # Create plugin directory if it doesn't exist
        self.plugin_directory.mkdir(exist_ok=True)

        # Create subdirectories
        (self.plugin_directory / "enabled").mkdir(exist_ok=True)
        (self.plugin_directory / "disabled").mkdir(exist_ok=True)
        (self.plugin_directory / "config").mkdir(exist_ok=True)

    def is_healthy(self) -> bool:
        """Check if plugin manager is healthy."""
        return True

    async def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugin directory."""
        discovered_plugins = []

        # Look for plugin.json files in enabled directory
        for plugin_dir in (self.plugin_directory / "enabled").iterdir():
            if plugin_dir.is_dir():
                metadata_file = plugin_dir / "plugin.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)

                        # Validate metadata
                        if self._validate_plugin_metadata(metadata_dict):
                            plugin_name = metadata_dict["name"]
                            discovered_plugins.append(plugin_name)
                            logger.info(f"Discovered plugin: {plugin_name}")
                        else:
                            logger.warning(f"Invalid plugin metadata in {plugin_dir}")

                    except Exception as e:
                        logger.error(f"Error reading plugin metadata from {plugin_dir}: {e}")

        return discovered_plugins

    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name."""
        try:
            plugin_dir = self.plugin_directory / "enabled" / plugin_name
            metadata_file = plugin_dir / "plugin.json"

            if not metadata_file.exists():
                logger.error(f"Plugin metadata not found: {plugin_name}")
                return False

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)

            metadata = PluginMetadata(**metadata_dict)

            # Check dependencies
            if not await self._check_dependencies(metadata.dependencies):
                logger.error(f"Dependencies not satisfied for plugin: {plugin_name}")
                return False

            # Security check
            if not self.security_manager.validate_plugin(plugin_dir, metadata):
                logger.error(f"Security validation failed for plugin: {plugin_name}")
                return False

            # Load plugin module
            plugin_instance = await self._load_plugin_module(plugin_dir, metadata)

            if not plugin_instance:
                logger.error(f"Failed to load plugin module: {plugin_name}")
                return False

            # Load configuration
            config = await self._load_plugin_config(plugin_name, metadata)

            # Initialize plugin
            if not await plugin_instance.initialize(config):
                logger.error(f"Plugin initialization failed: {plugin_name}")
                return False

            # Register plugin
            plugin_wrapper = PluginInstance(
                metadata=metadata,
                instance=plugin_instance,
                status="active",
                loaded_at=datetime.now().isoformat(),
                config=config
            )

            self.plugins[plugin_name] = plugin_wrapper

            # Register capabilities
            for capability in metadata.capabilities:
                if capability not in self.capability_registry:
                    self.capability_registry[capability] = []
                self.capability_registry[capability].append(plugin_name)

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False

        try:
            plugin_wrapper = self.plugins[plugin_name]

            # Shutdown plugin
            if not await plugin_wrapper.instance.shutdown():
                logger.warning(f"Plugin shutdown returned false: {plugin_name}")

            # Unregister capabilities
            for capability in plugin_wrapper.metadata.capabilities:
                if capability in self.capability_registry:
                    if plugin_name in self.capability_registry[capability]:
                        self.capability_registry[capability].remove(plugin_name)

            # Remove from registry
            del self.plugins[plugin_name]

            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def execute_plugin_action(self, plugin_name: str, action: str,
                                  parameters: Dict[str, Any]) -> Any:
        """Execute an action on a plugin."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not loaded: {plugin_name}")

        plugin_wrapper = self.plugins[plugin_name]

        if plugin_wrapper.status != "active":
            raise ValueError(f"Plugin not active: {plugin_name}")

        try:
            # Security check
            if not self.security_manager.validate_action(plugin_name, action, parameters):
                raise ValueError(f"Action not allowed: {action}")

            # Execute action
            result = await plugin_wrapper.instance.execute(action, parameters)

            # Emit event
            await self._emit_event("plugin_action_executed", {
                "plugin_name": plugin_name,
                "action": action,
                "parameters": parameters,
                "result": result
            })

            return result

        except Exception as e:
            logger.error(f"Error executing plugin action {plugin_name}.{action}: {e}")
            raise

    def get_plugin_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific plugin."""
        if plugin_name not in self.plugins:
            return None

        plugin_wrapper = self.plugins[plugin_name]
        status_info = plugin_wrapper.instance.get_status()

        return {
            "name": plugin_name,
            "status": plugin_wrapper.status,
            "version": plugin_wrapper.metadata.version,
            "capabilities": plugin_wrapper.metadata.capabilities,
            "loaded_at": plugin_wrapper.loaded_at,
            "health": status_info
        }

    def get_all_plugins_status(self) -> Dict[str, Any]:
        """Get status of all plugins."""
        plugin_statuses = {}

        for plugin_name, plugin_wrapper in self.plugins.items():
            plugin_statuses[plugin_name] = self.get_plugin_status(plugin_name)

        return {
            "total_plugins": len(plugin_statuses),
            "active_plugins": len([p for p in plugin_statuses.values() if p and p["status"] == "active"]),
            "plugins": plugin_statuses,
            "capabilities": self.capability_registry
        }

    def get_plugins_by_capability(self, capability: str) -> List[str]:
        """Get list of plugins that provide a specific capability."""
        return self.capability_registry.get(capability, [])

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        if not await self.unload_plugin(plugin_name):
            return False

        return await self.load_plugin(plugin_name)

    async def install_plugin(self, plugin_archive_path: str) -> bool:
        """Install a plugin from archive."""
        # This would extract the plugin archive and move it to the enabled directory
        # Implementation would depend on the archive format (zip, tar, etc.)
        logger.info(f"Installing plugin from: {plugin_archive_path}")
        # TODO: Implement plugin installation logic
        return False

    async def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        try:
            # Unload if loaded
            if plugin_name in self.plugins:
                await self.unload_plugin(plugin_name)

            # Move to disabled directory
            enabled_dir = self.plugin_directory / "enabled" / plugin_name
            disabled_dir = self.plugin_directory / "disabled" / plugin_name

            if enabled_dir.exists():
                import shutil
                shutil.move(str(enabled_dir), str(disabled_dir))

            logger.info(f"Successfully uninstalled plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_name}: {e}")
            return False

    def register_event_listener(self, event_type: str, callback: Callable):
        """Register an event listener."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(callback)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered listeners."""
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                try:
                    await callback(event_type, data)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")

    async def _load_plugin_module(self, plugin_dir: Path, metadata: PluginMetadata) -> Optional[PluginInterface]:
        """Load plugin module from directory."""
        try:
            # Add plugin directory to Python path
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))

            # Import plugin module
            module = importlib.import_module(metadata.entry_point)

            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginInterface) and
                    obj != PluginInterface):
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.error(f"No plugin class found in {metadata.entry_point}")
                return None

            # Instantiate plugin
            plugin_instance = plugin_class()

            return plugin_instance

        except Exception as e:
            logger.error(f"Error loading plugin module {metadata.entry_point}: {e}")
            return None

    async def _load_plugin_config(self, plugin_name: str, metadata: PluginMetadata) -> Dict[str, Any]:
        """Load plugin configuration."""
        config_file = self.plugin_directory / "config" / f"{plugin_name}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading plugin config for {plugin_name}: {e}")

        # Return default config
        return {}

    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are satisfied."""
        # Simple dependency checking - could be enhanced
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"Dependency not satisfied: {dep}")
                return False
        return True

    def _validate_plugin_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate plugin metadata structure."""
        required_fields = ["name", "version", "description", "author", "entry_point", "capabilities"]

        for field in required_fields:
            if field not in metadata:
                logger.error(f"Missing required field in plugin metadata: {field}")
                return False

        # Validate capabilities is a list
        if not isinstance(metadata.get("capabilities", []), list):
            logger.error("Plugin capabilities must be a list")
            return False

        return True

class PluginSecurityManager:
    """Manages plugin security and sandboxing."""

    def __init__(self):
        self.allowed_imports = set([
            "os", "sys", "json", "datetime", "typing", "abc",
            "asyncio", "logging", "pathlib", "dataclasses"
        ])
        self.forbidden_operations = set([
            "eval", "exec", "compile", "__import__", "open", "file"
        ])

    def validate_plugin(self, plugin_dir: Path, metadata: PluginMetadata) -> bool:
        """Validate plugin for security issues."""
        try:
            # Check for malicious files
            forbidden_files = [".exe", ".bat", ".sh", ".dll", ".so"]
            for file_path in plugin_dir.rglob("*"):
                if file_path.is_file():
                    if any(file_path.name.endswith(ext) for ext in forbidden_files):
                        logger.error(f"Forbidden file type in plugin: {file_path}")
                        return False

            # Check plugin metadata for suspicious content
            if self._contains_suspicious_content(metadata.description):
                logger.error("Suspicious content detected in plugin description")
                return False

            return True

        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return False

    def validate_action(self, plugin_name: str, action: str, parameters: Dict[str, Any]) -> bool:
        """Validate plugin action for security."""
        # Check for dangerous actions
        dangerous_actions = ["system", "exec", "shell", "delete", "remove"]

        if any(dangerous in action.lower() for dangerous in dangerous_actions):
            logger.warning(f"Potentially dangerous action requested: {plugin_name}.{action}")
            # Could implement additional validation here

        # Validate parameters don't contain dangerous content
        for key, value in parameters.items():
            if isinstance(value, str) and self._contains_suspicious_content(value):
                logger.error(f"Suspicious content in parameter {key}")
                return False

        return True

    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        suspicious_patterns = [
            "import os", "import sys", "eval(", "exec(", "compile(",
            "__import__", "open(", "file(", "subprocess", "system("
        ]

        content_lower = content.lower()
        return any(pattern in content_lower for pattern in suspicious_patterns)

class PluginTemplateGenerator:
    """Generates plugin templates for developers."""

    @staticmethod
    def generate_plugin_template(plugin_name: str, capabilities: List[str]) -> str:
        """Generate a basic plugin template."""
        template = f'''"""CES Plugin: {plugin_name}

A CES plugin that provides {", ".join(capabilities)} capabilities.
"""

from typing import Dict, List, Any
from ces.plugins.plugin_manager import PluginInterface

class {plugin_name.replace("_", "").title()}Plugin(PluginInterface):
    """{plugin_name} plugin implementation."""

    def __init__(self):
        self.config = {{}}
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self.config = config
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Initialization error: {{e}}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        try:
            self.is_initialized = False
            return True
        except Exception as e:
            print(f"Shutdown error: {{e}}")
            return False

    def get_capabilities(self) -> List[str]:
        """Return plugin capabilities."""
        return {capabilities!r}

    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute a plugin action."""
        if not self.is_initialized:
            raise RuntimeError("Plugin not initialized")

        if action == "example_action":
            return {{"message": "Hello from {plugin_name} plugin!", "parameters": parameters}}
        else:
            raise ValueError(f"Unknown action: {{action}}")

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status."""
        return {{
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "version": "1.0.0"
        }}
'''
        return template

    @staticmethod
    def generate_plugin_metadata(plugin_name: str, capabilities: List[str]) -> Dict[str, Any]:
        """Generate plugin metadata."""
        return {
            "name": plugin_name,
            "version": "1.0.0",
            "description": f"A CES plugin that provides {', '.join(capabilities)} capabilities",
            "author": "Plugin Developer",
            "license": "MIT",
            "dependencies": [],
            "capabilities": capabilities,
            "entry_point": f"{plugin_name}_plugin",
            "config_schema": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable or disable the plugin"
                    }
                }
            }
        }