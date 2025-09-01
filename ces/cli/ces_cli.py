"""
CES Command Line Interface

Provides CLI commands for interacting with the Cognitive Enhancement System.
"""

import argparse
import sys
import logging
from typing import Optional
from pathlib import Path

from ..core.cognitive_agent import CognitiveAgent
from ..config.ces_config import CESConfig
from ..utils.helpers import setup_logging


class CESCLI:
    """
    Command Line Interface for CES operations.

    Provides commands for:
    - Task execution
    - System status
    - Configuration management
    - Memory operations
    """

    def __init__(self):
        self.config = CESConfig()
        self.logger = setup_logging(self.config.log_level, self.config.debug_mode)
        self.agent: Optional[CognitiveAgent] = None

    def initialize_agent(self):
        """Initialize the cognitive agent"""
        if self.agent is None:
            self.agent = CognitiveAgent(self.config)
        return self.agent

    def run(self):
        """Run the CLI application"""
        parser = self._create_parser()
        args = parser.parse_args()

        if not hasattr(args, 'command'):
            parser.print_help()
            return

        # Execute the appropriate command
        try:
            if args.command == 'task':
                self._handle_task_command(args)
            elif args.command == 'status':
                self._handle_status_command(args)
            elif args.command == 'config':
                self._handle_config_command(args)
            elif args.command == 'memory':
                self._handle_memory_command(args)
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            sys.exit(1)

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""
        parser = argparse.ArgumentParser(
            description="Cognitive Enhancement System (CES) CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Task command
        task_parser = subparsers.add_parser('task', help='Execute a task')
        task_parser.add_argument('description', help='Task description')
        task_parser.add_argument('--assistant', help='Preferred AI assistant')
        task_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

        # Status command
        status_parser = subparsers.add_parser('status', help='Show system status')
        status_parser.add_argument('--detailed', '-d', action='store_true', help='Detailed status')

        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')

        # Config show
        config_subparsers.add_parser('show', help='Show current configuration')

        # Config set
        config_set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        config_set_parser.add_argument('key', help='Configuration key')
        config_set_parser.add_argument('value', help='Configuration value')

        # Memory command
        memory_parser = subparsers.add_parser('memory', help='Memory operations')
        memory_subparsers = memory_parser.add_subparsers(dest='memory_action')

        # Memory stats
        memory_subparsers.add_parser('stats', help='Show memory statistics')

        # Memory clear
        memory_subparsers.add_parser('clear', help='Clear memory data')

        return parser

    def _handle_task_command(self, args):
        """Handle task execution command"""
        agent = self.initialize_agent()

        # Execute the task
        result = agent.execute_task(args.description)

        # Display results
        print(f"Task: {args.description}")
        print(f"Status: {result['status']}")
        print(f"Assistant: {result.get('assistant_used', 'N/A')}")

        if args.verbose and 'analysis' in result:
            analysis = result['analysis']
            print(f"Complexity: {analysis.complexity_score}")
            print(f"Skills: {', '.join(analysis.required_skills)}")
            print(f"Estimated duration: {analysis.estimated_duration} minutes")

        if result['status'] == 'completed':
            print(f"Result: {result.get('result', 'No result')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    def _handle_status_command(self, args):
        """Handle status command"""
        agent = self.initialize_agent()

        status = agent.get_status()

        print("CES System Status")
        print("=" * 20)
        print(f"Overall Status: {status['status']}")

        if args.detailed:
            print("\nComponents:")
            for component, comp_status in status['components'].items():
                print(f"  {component}: {comp_status.get('status', 'unknown')}")

        print(f"\nTimestamp: {status['timestamp']}")

    def _handle_config_command(self, args):
        """Handle configuration command"""
        if args.config_action == 'show':
            self._show_config()
        elif args.config_action == 'set':
            self._set_config_value(args.key, args.value)
        else:
            print("Invalid config action. Use 'show' or 'set'.")

    def _show_config(self):
        """Show current configuration"""
        print("CES Configuration")
        print("=" * 18)

        config_dict = self.config.to_dict()
        for key, value in config_dict.items():
            if 'api_key' in key.lower() and value:
                print(f"{key}: ***masked***")
            else:
                print(f"{key}: {value}")

    def _set_config_value(self, key: str, value: str):
        """Set a configuration value"""
        # Convert value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)

        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.config.save_to_file()
            print(f"Configuration updated: {key} = {value}")
        else:
            print(f"Unknown configuration key: {key}")

    def _handle_memory_command(self, args):
        """Handle memory command"""
        agent = self.initialize_agent()

        if args.memory_action == 'stats':
            memory_status = agent.memory_manager.get_status()
            print("Memory Statistics")
            print("=" * 17)
            for key, value in memory_status.items():
                print(f"{key}: {value}")

        elif args.memory_action == 'clear':
            # This would be implemented with confirmation
            print("Memory clearing not yet implemented")
        else:
            print("Invalid memory action. Use 'stats' or 'clear'.")


def main():
    """Main entry point for CES CLI"""
    cli = CESCLI()
    cli.run()


if __name__ == "__main__":
    main()