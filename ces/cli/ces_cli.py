"""
CES Command Line Interface - Phase 0.3 Enhanced

Provides comprehensive CLI commands for interacting with the Cognitive Enhancement System.
Includes end-to-end task execution workflow, performance monitoring, and integration validation.
"""

import argparse
import sys
import logging
import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json

from ..core.cognitive_agent import CognitiveAgent
from ..config.ces_config import CESConfig
from ..utils.helpers import setup_logging, format_task_summary
from ..codesage_integration import CodeSageIntegration
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table


class CESCLI:
    """
    Enhanced Command Line Interface for CES Phase 0.3 operations.

    Provides comprehensive commands for:
    - End-to-end task execution workflow
    - System status and health monitoring
    - Configuration management
    - Memory operations
    - Performance monitoring
    - Integration validation
    - CodeSage tool integration
    """

    def __init__(self):
        self.config = CESConfig()
        self.logger = setup_logging(self.config.log_level, self.config.debug_mode)
        self.agent: Optional[CognitiveAgent] = None
        self.codesage: Optional[CodeSageIntegration] = None
        self.console = Console()
        self.performance_metrics: Dict[str, Any] = {}

    def initialize_agent(self):
        """Initialize the cognitive agent"""
        if self.agent is None:
            self.agent = CognitiveAgent(self.config)
        return self.agent

    async def initialize_codesage(self):
        """Initialize CodeSage integration"""
        if self.codesage is None:
            self.codesage = CodeSageIntegration()
            connected = await self.codesage.connect()
            if connected:
                self.logger.info("CodeSage integration initialized successfully")
            else:
                self.logger.warning("CodeSage integration failed to connect")
        return self.codesage

    async def execute_task_enhanced(self, task_description: str, assistant: Optional[str] = None,
                                  verbose: bool = False) -> Dict[str, Any]:
        """
        Enhanced task execution with progress monitoring and CodeSage integration

        Args:
            task_description: Description of the task to execute
            assistant: Preferred AI assistant
            verbose: Enable verbose output

        Returns:
            Task execution result
        """
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            # Initialize components
            task = progress.add_task("Initializing CES components...", total=3)
            agent = self.initialize_agent()
            progress.update(task, advance=1)

            codesage = await self.initialize_codesage()
            progress.update(task, advance=1)

            # Analyze task
            progress.update(task, description="Analyzing task requirements...")
            analysis = agent.analyze_task(task_description)
            progress.update(task, advance=1, description="Task analysis complete")

            # Execute task
            task_exec = progress.add_task("Executing task with AI assistant...", total=1)
            result = agent.execute_task(task_description)
            progress.update(task_exec, completed=1)

            # Record performance metrics
            total_time = time.time() - start_time
            self.performance_metrics = {
                "total_execution_time": total_time,
                "task_complexity": analysis.complexity_score,
                "assistant_used": result.get("result", {}).get("assistant_used", "unknown"),
                "timestamp": datetime.now().isoformat()
            }

            return result

    def validate_integration(self) -> Dict[str, Any]:
        """
        Validate integration between all CES components

        Returns:
            Integration validation results
        """
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "overall_status": "unknown"
        }

        try:
            # Test cognitive agent
            agent = self.initialize_agent()
            agent_status = agent.get_status()
            validation_results["components"]["cognitive_agent"] = {
                "status": "healthy" if agent_status["status"] == "operational" else "unhealthy",
                "details": agent_status
            }

            # Test memory system
            memory_status = agent.memory_manager.get_status()
            validation_results["components"]["memory_system"] = {
                "status": "healthy" if memory_status["status"] == "operational" else "unhealthy",
                "details": memory_status
            }

            # Test AI orchestrator
            orchestrator_status = agent.ai_orchestrator.get_status()
            validation_results["components"]["ai_orchestrator"] = {
                "status": "healthy" if orchestrator_status["status"] == "operational" else "unhealthy",
                "details": orchestrator_status
            }

            # Test ethical controller
            ethical_status = agent.ethical_controller.get_status()
            validation_results["components"]["ethical_controller"] = {
                "status": "healthy" if ethical_status["status"] == "operational" else "unhealthy",
                "details": ethical_status
            }

            # Determine overall status
            all_healthy = all(
                comp["status"] == "healthy"
                for comp in validation_results["components"].values()
            )
            validation_results["overall_status"] = "healthy" if all_healthy else "degraded"

        except Exception as e:
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            self.logger.error(f"Integration validation failed: {e}")

        return validation_results

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report for the current session

        Returns:
            Performance metrics report
        """
        report = {
            "session_metrics": self.performance_metrics,
            "system_info": {
                "config": self.config.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        }

        if self.agent:
            # Get component-specific metrics
            agent_status = self.agent.get_status()
            report["component_metrics"] = {
                "cognitive_agent": agent_status,
                "memory_system": self.agent.memory_manager.get_status(),
                "ai_orchestrator": self.agent.ai_orchestrator.get_status(),
                "ethical_controller": self.agent.ethical_controller.get_status()
            }

        return report

    def run(self):
        """Run the enhanced CLI application for Phase 0.3"""
        parser = self._create_parser()
        args = parser.parse_args()

        if not hasattr(args, 'command'):
            parser.print_help()
            return

        # Execute the appropriate command
        try:
            if args.command == 'task':
                asyncio.run(self._handle_task_command_enhanced(args))
            elif args.command == 'status':
                self._handle_status_command_enhanced(args)
            elif args.command == 'validate':
                self._handle_validate_command(args)
            elif args.command == 'performance':
                self._handle_performance_command(args)
            elif args.command == 'codesage':
                asyncio.run(self._handle_codesage_command(args))
            elif args.command == 'config':
                self._handle_config_command(args)
            elif args.command == 'memory':
                self._handle_memory_command(args)
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            self.console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the enhanced argument parser for Phase 0.3"""
        parser = argparse.ArgumentParser(
            description="Cognitive Enhancement System (CES) CLI - Phase 0.3",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ces task "Implement user authentication" --verbose
  ces status --detailed
  ces validate
  ces performance
  ces codesage analyze --path /path/to/codebase
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Task command - Enhanced for Phase 0.3
        task_parser = subparsers.add_parser('task', help='Execute a task with full CES workflow')
        task_parser.add_argument('description', help='Task description')
        task_parser.add_argument('--assistant', help='Preferred AI assistant (grok, qwen, gemini)')
        task_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with analysis details')
        task_parser.add_argument('--json', action='store_true', help='Output results in JSON format')

        # Status command - Enhanced
        status_parser = subparsers.add_parser('status', help='Show comprehensive system status')
        status_parser.add_argument('--detailed', '-d', action='store_true', help='Detailed component status')
        status_parser.add_argument('--health', action='store_true', help='Include health check results')

        # Validate command - New for Phase 0.3
        validate_parser = subparsers.add_parser('validate', help='Validate CES component integration')
        validate_parser.add_argument('--json', action='store_true', help='Output validation results in JSON')

        # Performance command - New for Phase 0.3
        perf_parser = subparsers.add_parser('performance', help='Show performance metrics and baselines')
        perf_parser.add_argument('--baseline', action='store_true', help='Show baseline performance metrics')
        perf_parser.add_argument('--json', action='store_true', help='Output in JSON format')

        # CodeSage command - New for Phase 0.3
        codesage_parser = subparsers.add_parser('codesage', help='CodeSage MCP server integration')
        codesage_subparsers = codesage_parser.add_subparsers(dest='codesage_action')

        # CodeSage status
        codesage_subparsers.add_parser('status', help='Show CodeSage server status')

        # CodeSage analyze
        analyze_parser = codesage_subparsers.add_parser('analyze', help='Analyze codebase with CodeSage')
        analyze_parser.add_argument('--path', required=True, help='Path to codebase to analyze')
        analyze_parser.add_argument('--tools', nargs='*', help='Specific tools to use for analysis')

        # Config command - Existing
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        config_subparsers.add_parser('show', help='Show current configuration')
        config_set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        config_set_parser.add_argument('key', help='Configuration key')
        config_set_parser.add_argument('value', help='Configuration value')

        # Memory command - Existing
        memory_parser = subparsers.add_parser('memory', help='Memory operations')
        memory_subparsers = memory_parser.add_subparsers(dest='memory_action')
        memory_subparsers.add_parser('stats', help='Show memory statistics')
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

    async def _handle_task_command_enhanced(self, args):
        """Handle enhanced task command with Phase 0.3 features"""
        try:
            result = await self.execute_task_enhanced(
                args.description,
                assistant=args.assistant,
                verbose=args.verbose
            )

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                self._display_task_result_enhanced(result, args.verbose)

        except Exception as e:
            self.console.print(f"[red]Task execution failed:[/red] {e}")
            if args.verbose:
                import traceback
                self.console.print(traceback.format_exc())

    def _handle_status_command_enhanced(self, args):
        """Handle enhanced status command"""
        agent = self.initialize_agent()

        if args.health:
            # Include health check
            validation = self.validate_integration()
            self._display_health_status(validation)
        else:
            # Standard status
            status = agent.get_status()
            self._display_status_enhanced(status, args.detailed)

    def _handle_validate_command(self, args):
        """Handle integration validation command"""
        validation = self.validate_integration()

        if args.json:
            print(json.dumps(validation, indent=2))
        else:
            self._display_validation_results(validation)

    def _handle_performance_command(self, args):
        """Handle performance monitoring command"""
        if args.baseline:
            # Show baseline metrics
            baseline_metrics = self._get_baseline_metrics()
            if args.json:
                print(json.dumps(baseline_metrics, indent=2))
            else:
                self._display_baseline_metrics(baseline_metrics)
        else:
            # Show current performance
            report = self.get_performance_report()
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                self._display_performance_report(report)

    async def _handle_codesage_command(self, args):
        """Handle CodeSage integration commands"""
        if args.codesage_action == 'status':
            codesage = await self.initialize_codesage()
            if codesage:
                status = await codesage.get_server_status()
                self._display_codesage_status(status)
            else:
                self.console.print("[red]CodeSage integration not available[/red]")

        elif args.codesage_action == 'analyze':
            await self._handle_codesage_analyze(args)

    async def _handle_codesage_analyze(self, args):
        """Handle CodeSage codebase analysis"""
        codesage = await self.initialize_codesage()
        if not codesage:
            self.console.print("[red]CodeSage integration not available[/red]")
            return

        try:
            # Perform analysis
            analysis_result = await codesage.execute_tool(
                "analyze_codebase_structure",
                {"codebase_path": args.path}
            )

            self._display_codesage_analysis(analysis_result)

        except Exception as e:
            self.console.print(f"[red]CodeSage analysis failed:[/red] {e}")

    def _display_task_result_enhanced(self, result: Dict[str, Any], verbose: bool):
        """Display enhanced task results with rich formatting"""
        status = result.get('status', 'unknown')

        if status == 'completed':
            self.console.print("[green]✓ Task completed successfully[/green]")
        elif status == 'rejected':
            self.console.print("[red]✗ Task rejected[/red]")
        else:
            self.console.print(f"[yellow]⚠ Task {status}[/yellow]")

        # Show analysis if verbose
        if verbose and 'analysis' in result:
            analysis = result['analysis']
            table = Table(title="Task Analysis")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Complexity Score", f"{analysis.complexity_score:.1f}")
            table.add_row("Required Skills", ", ".join(analysis.required_skills))
            table.add_row("Estimated Duration", f"{analysis.estimated_duration} min")
            table.add_row("Recommended Assistant", analysis.recommended_assistants[0] if analysis.recommended_assistants else "None")

            self.console.print(table)

        # Show result
        if 'result' in result and result['result']:
            result_data = result['result']
            if isinstance(result_data, dict):
                assistant = result_data.get('assistant_used', 'Unknown')
                execution_time = result_data.get('execution_time', 0)
                self.console.print(f"\n[blue]Assistant:[/blue] {assistant}")
                self.console.print(f"[blue]Execution Time:[/blue] {execution_time:.2f}s")

                if 'result' in result_data:
                    self.console.print(f"\n[green]Result:[/green]\n{result_data['result']}")
            else:
                self.console.print(f"\n[green]Result:[/green]\n{result_data}")

    def _display_status_enhanced(self, status: Dict[str, Any], detailed: bool):
        """Display enhanced status with rich formatting"""
        table = Table(title="CES System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        table.add_row("Overall", status.get('status', 'unknown').upper(), "")

        if detailed and 'components' in status:
            for component, comp_status in status['components'].items():
                status_color = "green" if comp_status.get('status') == 'operational' else "red"
                table.add_row(
                    component.replace('_', ' ').title(),
                    f"[{status_color}]{comp_status.get('status', 'unknown')}[/{status_color}]",
                    str(comp_status.get('details', ''))
                )

        self.console.print(table)

    def _display_health_status(self, validation: Dict[str, Any]):
        """Display health check results"""
        overall_status = validation.get('overall_status', 'unknown')

        if overall_status == 'healthy':
            self.console.print("[green]✓ All components are healthy[/green]")
        else:
            self.console.print(f"[red]⚠ System status: {overall_status}[/red]")

        table = Table(title="Component Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        for component, health in validation.get('components', {}).items():
            status = health.get('status', 'unknown')
            status_color = "green" if status == "healthy" else "red" if status == "unhealthy" else "yellow"
            table.add_row(
                component.replace('_', ' ').title(),
                f"[{status_color}]{status}[/{status_color}]",
                str(health.get('details', ''))
            )

        self.console.print(table)

    def _display_validation_results(self, validation: Dict[str, Any]):
        """Display integration validation results"""
        self._display_health_status(validation)

    def _display_performance_report(self, report: Dict[str, Any]):
        """Display performance report"""
        if 'session_metrics' in report and report['session_metrics']:
            metrics = report['session_metrics']
            table = Table(title="Session Performance Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Execution Time", f"{metrics.get('total_execution_time', 0):.2f}s")
            table.add_row("Task Complexity", f"{metrics.get('task_complexity', 0):.1f}")
            table.add_row("Assistant Used", metrics.get('assistant_used', 'Unknown'))

            self.console.print(table)
        else:
            self.console.print("[yellow]No performance metrics available for current session[/yellow]")

    def _get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline performance metrics"""
        return {
            "response_time_p95": "<500ms",
            "throughput": "100+ tasks/minute",
            "memory_usage": "<256MB",
            "cpu_usage": "<30%",
            "error_rate": "<1%",
            "cache_hit_rate": ">95%"
        }

    def _display_baseline_metrics(self, metrics: Dict[str, Any]):
        """Display baseline performance metrics"""
        table = Table(title="Performance Baselines")
        table.add_column("Metric", style="cyan")
        table.add_column("Target", style="green")

        for metric, target in metrics.items():
            table.add_row(metric.replace('_', ' ').title(), target)

        self.console.print(table)

    def _display_codesage_status(self, status: Dict[str, Any]):
        """Display CodeSage server status"""
        if status.get('status') == 'connected':
            self.console.print("[green]✓ CodeSage server connected[/green]")
            self.console.print(f"Tools available: {status.get('tools_count', 0)}")
        else:
            self.console.print("[red]✗ CodeSage server not connected[/red]")
            self.console.print(f"Error: {status.get('error', 'Unknown')}")

    def _display_codesage_analysis(self, analysis: Dict[str, Any]):
        """Display CodeSage analysis results"""
        if analysis.get('status') == 'success':
            self.console.print("[green]✓ CodeSage analysis completed[/green]")
            if 'result' in analysis:
                result = analysis['result']
                if isinstance(result, dict):
                    for key, value in result.items():
                        self.console.print(f"[cyan]{key}:[/cyan] {value}")
                else:
                    self.console.print(f"[green]Analysis Result:[/green]\n{result}")
        else:
            self.console.print(f"[red]✗ CodeSage analysis failed: {analysis.get('error', 'Unknown error')}[/red]")


def main():
    """Main entry point for CES CLI"""
    cli = CESCLI()
    cli.run()


if __name__ == "__main__":
    main()