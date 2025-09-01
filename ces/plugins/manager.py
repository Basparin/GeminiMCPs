"""
CES Plugin Manager

Manages plugin loading, lifecycle, and coordination.
"""

import logging
import asyncio
import importlib
import inspect
import os
import json
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from datetime import datetime

from .base import CESPlugin, PluginContext, PluginInfo, PluginHook


class PluginManager:
    """
    Manages CES plugins and their lifecycle
    """

    def __init__(self, plugin_directories: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)

        # Plugin directories
        self.plugin_directories = plugin_directories or [
            "./plugins",
            "./ces/plugins",
            os.path.expanduser("~/.ces/plugins")
        ]

        # Plugin storage
        self.loaded_plugins: Dict[str, CESPlugin] = {}
        self.enabled_plugins: Dict[str, CESPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.global_hooks: Dict[str, PluginHook] = {}

        # CES context
        self.ces_config = config or {}
        self.ces_version = "0.4.0"

        # Plugin data directory
        self.data_directory = "./data/plugins"
        os.makedirs(self.data_directory, exist_ok=True)

        # Load plugin configurations
        self._load_plugin_configs()

    def _load_plugin_configs(self):
        """Load plugin configurations from storage"""
        config_file = f"{self.data_directory}/plugin_configs.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.plugin_configs = json.load(f)
                self.logger.info("Loaded plugin configurations")
            except Exception as e:
                self.logger.error(f"Error loading plugin configs: {e}")

    def _save_plugin_configs(self):
        """Save plugin configurations to storage"""
        try:
            with open(f"{self.data_directory}/plugin_configs.json", 'w') as f:
                json.dump(self.plugin_configs, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving plugin configs: {e}")

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directories

        Returns:
            List of discovered plugin names
        """
        discovered_plugins = []

        for plugin_dir in self.plugin_directories:
            if not os.path.exists(plugin_dir):
                continue

            # Look for Python files and directories
            for item in os.listdir(plugin_dir):
                plugin_path = os.path.join(plugin_dir, item)

                # Check for plugin directories with __init__.py
                if os.path.isdir(plugin_path) and os.path.exists(os.path.join(plugin_path, "__init__.py")):
                    if self._is_valid_plugin_directory(plugin_path):
                        discovered_plugins.append(item)

                # Check for single plugin files
                elif item.endswith(".py") and item != "__init__.py":
                    plugin_name = item[:-3]  # Remove .py extension
                    if self._is_valid_plugin_file(plugin_path):
                        discovered_plugins.append(plugin_name)

        return list(set(discovered_plugins))  # Remove duplicates

    def _is_valid_plugin_directory(self, plugin_path: str) -> bool:
        """Check if a directory contains a valid plugin"""
        try:
            # Try to import the plugin
            plugin_name = os.path.basename(plugin_path)
            spec = importlib.util.spec_from_file_location(
                plugin_name,
                os.path.join(plugin_path, "__init__.py")
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check if it has a plugin class
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, CESPlugin) and
                        obj != CESPlugin):
                        return True
        except Exception as e:
            self.logger.debug(f"Invalid plugin directory {plugin_path}: {e}")

        return False

    def _is_valid_plugin_file(self, plugin_path: str) -> bool:
        """Check if a file contains a valid plugin"""
        try:
            plugin_name = os.path.basename(plugin_path)[:-3]  # Remove .py
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check if it has a plugin class
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, CESPlugin) and
                        obj != CESPlugin):
                        return True
        except Exception as e:
            self.logger.debug(f"Invalid plugin file {plugin_path}: {e}")

        return False

    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a plugin by name

        Args:
            plugin_name: Name of the plugin to load

        Returns:
            True if loaded successfully, False otherwise
        """
        if plugin_name in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} already loaded")
            return True

        plugin_instance = None

        # Try to find and load the plugin
        for plugin_dir in self.plugin_directories:
            plugin_path = None

            # Check for directory plugin
            dir_path = os.path.join(plugin_dir, plugin_name)
            if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "__init__.py")):
                plugin_path = os.path.join(dir_path, "__init__.py")

            # Check for file plugin
            file_path = os.path.join(plugin_dir, f"{plugin_name}.py")
            if os.path.isfile(file_path):
                plugin_path = file_path

            if plugin_path:
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Find the plugin class
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, CESPlugin) and
                                obj != CESPlugin):
                                plugin_instance = obj()
                                break

                        if plugin_instance:
                            break

                except Exception as e:
                    self.logger.error(f"Error loading plugin {plugin_name}: {e}")
                    continue

        if not plugin_instance:
            self.logger.error(f"Plugin {plugin_name} not found or invalid")
            return False

        # Create plugin context
        context = PluginContext(
            ces_version=self.ces_version,
            config=self.plugin_configs.get(plugin_name, {}),
            logger=self.logger.getChild(f"plugin.{plugin_name}"),
            data_directory=f"{self.data_directory}/{plugin_name}",
            temp_directory=f"{self.data_directory}/temp"
        )

        # Ensure plugin data directory exists
        os.makedirs(context.data_directory, exist_ok=True)

        # Initialize plugin
        if plugin_instance.initialize(context):
            self.loaded_plugins[plugin_name] = plugin_instance

            # Register plugin hooks
            for hook_name, hook in plugin_instance.get_hooks().items():
                if hook_name not in self.global_hooks:
                    self.global_hooks[hook_name] = hook
                else:
                    # Merge handlers
                    for handler in hook.handlers:
                        self.global_hooks[hook_name].register(handler)

            self.logger.info(f"Plugin {plugin_name} loaded successfully")
            return True
        else:
            self.logger.error(f"Failed to initialize plugin {plugin_name}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin by name

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if unloaded successfully, False otherwise
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} not loaded")
            return True

        plugin = self.loaded_plugins[plugin_name]

        # Disable if enabled
        if plugin_name in self.enabled_plugins:
            self.disable_plugin(plugin_name)

        # Shutdown plugin
        try:
            plugin.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down plugin {plugin_name}: {e}")

        # Remove from loaded plugins
        del self.loaded_plugins[plugin_name]

        self.logger.info(f"Plugin {plugin_name} unloaded")
        return True

    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a loaded plugin

        Args:
            plugin_name: Name of the plugin to enable

        Returns:
            True if enabled successfully, False otherwise
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.error(f"Plugin {plugin_name} not loaded")
            return False

        if plugin_name in self.enabled_plugins:
            self.logger.warning(f"Plugin {plugin_name} already enabled")
            return True

        plugin = self.loaded_plugins[plugin_name]

        try:
            plugin.on_enable()
            self.enabled_plugins[plugin_name] = plugin
            self.logger.info(f"Plugin {plugin_name} enabled")
            return True
        except Exception as e:
            self.logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable an enabled plugin

        Args:
            plugin_name: Name of the plugin to disable

        Returns:
            True if disabled successfully, False otherwise
        """
        if plugin_name not in self.enabled_plugins:
            self.logger.warning(f"Plugin {plugin_name} not enabled")
            return True

        plugin = self.enabled_plugins[plugin_name]

        try:
            plugin.on_disable()
            del self.enabled_plugins[plugin_name]
            self.logger.info(f"Plugin {plugin_name} disabled")
            return True
        except Exception as e:
            self.logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """
        Get information about a plugin

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin information or None if not found
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name].info
        return None

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded plugins with their status

        Returns:
            Dictionary of plugin information
        """
        result = {}

        for name, plugin in self.loaded_plugins.items():
            result[name] = {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "description": plugin.info.description,
                "author": plugin.info.author,
                "enabled": name in self.enabled_plugins,
                "hooks": list(plugin.get_hooks().keys())
            }

        return result

    async def trigger_hook(self, hook_name: str, *args, **kwargs):
        """
        Trigger a global hook across all enabled plugins

        Args:
            hook_name: Name of the hook to trigger
            *args: Positional arguments for hook handlers
            **kwargs: Keyword arguments for hook handlers
        """
        if hook_name in self.global_hooks:
            await self.global_hooks[hook_name].trigger(*args, **kwargs)

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a plugin

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin configuration
        """
        return self.plugin_configs.get(plugin_name, {})

    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]):
        """
        Set configuration for a plugin

        Args:
            plugin_name: Name of the plugin
            config: Plugin configuration
        """
        self.plugin_configs[plugin_name] = config
        self._save_plugin_configs()

        # Notify plugin of config change
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            try:
                plugin.on_config_change(config)
            except Exception as e:
                self.logger.error(f"Error notifying plugin {plugin_name} of config change: {e}")

    def install_plugin_from_path(self, plugin_path: str) -> bool:
        """
        Install a plugin from a file path

        Args:
            plugin_path: Path to the plugin file or directory

        Returns:
            True if installed successfully, False otherwise
        """
        # This would copy the plugin to the plugin directory
        # Implementation depends on the deployment strategy
        self.logger.info(f"Plugin installation from path not yet implemented: {plugin_path}")
        return False

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """
        Uninstall a plugin

        Args:
            plugin_name: Name of the plugin to uninstall

        Returns:
            True if uninstalled successfully, False otherwise
        """
        # Unload the plugin first
        self.unload_plugin(plugin_name)

        # Remove plugin files
        # This would remove the plugin from the plugin directory
        self.logger.info(f"Plugin uninstallation not yet implemented: {plugin_name}")
        return False

    def check_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """
        Check if plugin dependencies are satisfied

        Args:
            plugin_name: Name of the plugin

        Returns:
            List of missing dependencies
        """
        if plugin_name not in self.loaded_plugins:
            return ["Plugin not loaded"]

        plugin = self.loaded_plugins[plugin_name]
        dependencies = plugin.get_dependencies()

        missing = []
        for dep in dependencies:
            if dep not in self.loaded_plugins:
                missing.append(dep)

        return missing

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get plugin system status

        Returns:
            System status information
        """
        return {
            "total_plugins": len(self.loaded_plugins),
            "enabled_plugins": len(self.enabled_plugins),
            "plugin_directories": self.plugin_directories,
            "available_hooks": list(self.global_hooks.keys()),
            "data_directory": self.data_directory
        }


# Global plugin manager instance
plugin_manager = PluginManager()