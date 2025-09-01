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
            elif args.command == 'ai':
                self._handle_ai_command(args)
            elif args.command == 'analytics':
                self._handle_analytics_command(args)
            elif args.command == 'feedback':
                self._handle_feedback_command(args)
            elif args.command == 'plugin':
                self._handle_plugin_command(args)
            elif args.command == 'config':
                self._handle_config_command(args)
            elif args.command == 'memory':
                self._handle_memory_command(args)
            elif args.command == 'dashboard':
                self._handle_dashboard_command(args)
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
   ces ai analyze "Write a Python function to sort a list"
   ces ai status
   ces ai performance
   ces analytics usage --days 7
   ces analytics tasks
   ces analytics realtime
   ces analytics user user123
   ces feedback submit --type bug --title "App crashes" --message "Details..."
   ces feedback list --type bug --limit 5
   ces feedback summary --days 30
   ces feedback update feedback_123 --status reviewed --notes "Reviewed"
   ces plugin list
   ces plugin load sample_plugin
   ces plugin enable sample_plugin
   ces plugin info sample_plugin
   ces dashboard --port 8000
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

        # AI command - New for Phase 0.4
        ai_parser = subparsers.add_parser('ai', help='AI assistant management and specialization')
        ai_subparsers = ai_parser.add_subparsers(dest='ai_action')

        # AI analyze
        ai_analyze_parser = ai_subparsers.add_parser('analyze', help='Analyze task and get AI recommendations')
        ai_analyze_parser.add_argument('task', help='Task description to analyze')

        # AI status
        ai_subparsers.add_parser('status', help='Show AI assistant status with specialization info')

        # AI performance
        ai_subparsers.add_parser('performance', help='Show AI performance and specialization metrics')

        # Analytics command - New for Phase 0.4
        analytics_parser = subparsers.add_parser('analytics', help='Analytics and usage insights')
        analytics_subparsers = analytics_parser.add_subparsers(dest='analytics_action')

        # Analytics usage
        analytics_usage_parser = analytics_subparsers.add_parser('usage', help='Show usage analytics report')
        analytics_usage_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')

        # Analytics tasks
        analytics_subparsers.add_parser('tasks', help='Show task analytics report')

        # Analytics realtime
        analytics_subparsers.add_parser('realtime', help='Show real-time analytics metrics')

        # Analytics user
        analytics_user_parser = analytics_subparsers.add_parser('user', help='Show analytics for a specific user')
        analytics_user_parser.add_argument('user_id', help='User ID to analyze')

        # Feedback command - New for Phase 0.4
        feedback_parser = subparsers.add_parser('feedback', help='User feedback management')
        feedback_subparsers = feedback_parser.add_subparsers(dest='feedback_action')

        # Feedback submit
        feedback_submit_parser = feedback_subparsers.add_parser('submit', help='Submit user feedback')
        feedback_submit_parser.add_argument('--type', required=True,
                                          choices=['bug', 'feature', 'improvement', 'general'],
                                          help='Type of feedback')
        feedback_submit_parser.add_argument('--title', required=True, help='Feedback title')
        feedback_submit_parser.add_argument('--message', required=True, help='Feedback message')
        feedback_submit_parser.add_argument('--rating', type=int, choices=range(1, 6),
                                          help='Rating (1-5)')

        # Feedback list
        feedback_list_parser = feedback_subparsers.add_parser('list', help='List feedback entries')
        feedback_list_parser.add_argument('--type', choices=['bug', 'feature', 'improvement', 'general'],
                                        help='Filter by feedback type')
        feedback_list_parser.add_argument('--status', choices=['new', 'reviewed', 'addressed', 'closed'],
                                        help='Filter by status')
        feedback_list_parser.add_argument('--limit', type=int, default=10, help='Maximum entries to show')

        # Feedback summary
        feedback_summary_parser = feedback_subparsers.add_parser('summary', help='Show feedback summary')
        feedback_summary_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')

        # Feedback update
        feedback_update_parser = feedback_subparsers.add_parser('update', help='Update feedback status')
        feedback_update_parser.add_argument('feedback_id', help='Feedback entry ID')
        feedback_update_parser.add_argument('--status', required=True,
                                          choices=['new', 'reviewed', 'addressed', 'closed'],
                                          help='New status')
        feedback_update_parser.add_argument('--notes', help='Review notes')

        # Plugin command - New for Phase 0.4
        plugin_parser = subparsers.add_parser('plugin', help='Plugin management')
        plugin_subparsers = plugin_parser.add_subparsers(dest='plugin_action')

        # Plugin list
        plugin_subparsers.add_parser('list', help='List available plugins')

        # Plugin load
        plugin_load_parser = plugin_subparsers.add_parser('load', help='Load a plugin')
        plugin_load_parser.add_argument('name', help='Plugin name to load')

        # Plugin unload
        plugin_unload_parser = plugin_subparsers.add_parser('unload', help='Unload a plugin')
        plugin_unload_parser.add_argument('name', help='Plugin name to unload')

        # Plugin enable
        plugin_enable_parser = plugin_subparsers.add_parser('enable', help='Enable a plugin')
        plugin_enable_parser.add_argument('name', help='Plugin name to enable')

        # Plugin disable
        plugin_disable_parser = plugin_subparsers.add_parser('disable', help='Disable a plugin')
        plugin_disable_parser.add_argument('name', help='Plugin name to disable')

        # Plugin info
        plugin_info_parser = plugin_subparsers.add_parser('info', help='Show plugin information')
        plugin_info_parser.add_argument('name', help='Plugin name')

        # Plugin discover
        plugin_subparsers.add_parser('discover', help='Discover available plugins')

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

        # Dashboard command - New for Phase 0.4
        dashboard_parser = subparsers.add_parser('dashboard', help='Start web dashboard')
        dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host to bind dashboard server')
        dashboard_parser.add_argument('--port', type=int, default=8000, help='Port for dashboard server')

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

    def _handle_dashboard_command(self, args):
        """Handle dashboard command"""
        try:
            from ..web.dashboard import dashboard
            self.console.print(f"[green]Starting CES Dashboard on {args.host}:{args.port}[/green]")
            self.console.print("[blue]Open your browser to http://localhost:8000[/blue]")
            self.console.print("[yellow]Press Ctrl+C to stop the dashboard[/yellow]")
            dashboard.run(host=args.host, port=args.port)
        except ImportError:
            self.console.print("[red]Dashboard module not available. Make sure all dependencies are installed.[/red]")
        except Exception as e:
            self.console.print(f"[red]Failed to start dashboard: {e}[/red]")

    def _handle_ai_command(self, args):
        """Handle AI command with specialization features"""
        if args.ai_action == 'analyze':
            self._handle_ai_analyze(args)
        elif args.ai_action == 'status':
            self._handle_ai_status()
        elif args.ai_action == 'performance':
            self._handle_ai_performance()
        else:
            self.console.print("[red]Invalid AI action. Use 'analyze', 'status', or 'performance'.[/red]")

    def _handle_ai_analyze(self, args):
        """Handle AI task analysis"""
        agent = self.initialize_agent()

        if not hasattr(agent, 'ai_orchestrator'):
            self.console.print("[red]AI orchestrator not available[/red]")
            return

        try:
            recommendations = agent.ai_orchestrator.get_task_recommendations(args.task)

            # Display analysis
            analysis = recommendations.get('task_analysis', {})
            table = Table(title=f"Task Analysis: {args.task[:50]}...")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Task Type", analysis.get('task_type', 'unknown'))
            table.add_row("Complexity", analysis.get('complexity', 'unknown'))
            table.add_row("Estimated Duration", f"{analysis.get('estimated_duration', 0)}s")
            table.add_row("Confidence", f"{analysis.get('confidence', 0):.2f}")

            self.console.print(table)

            # Display assistant recommendations
            assistants = recommendations.get('assistant_recommendations', [])
            if assistants:
                self.console.print("\n[green]Recommended AI Assistants:[/green]")
                assistant_table = Table(title="Assistant Recommendations")
                assistant_table.add_column("Assistant", style="cyan")
                assistant_table.add_column("Score", style="green")
                assistant_table.add_column("Reason", style="yellow")

                for rec in assistants[:3]:  # Show top 3
                    assistant_table.add_row(
                        rec.get('display_name', rec.get('name', 'Unknown')),
                        f"{rec.get('score', 0):.2f}",
                        rec.get('reason', 'N/A')
                    )

                self.console.print(assistant_table)
            else:
                self.console.print("[yellow]No assistant recommendations available[/yellow]")

        except Exception as e:
            self.console.print(f"[red]AI analysis failed: {e}[/red]")

    def _handle_ai_status(self):
        """Handle AI status command with specialization info"""
        agent = self.initialize_agent()

        if not hasattr(agent, 'ai_orchestrator'):
            self.console.print("[red]AI orchestrator not available[/red]")
            return

        try:
            status = agent.ai_orchestrator.get_status()

            # Display basic status
            table = Table(title="AI Assistant Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Assistants", str(status.get('total_assistants', 0)))
            table.add_row("Available Assistants", str(status.get('available_assistants', 0)))

            specialization = status.get('specialization_status', {})
            table.add_row("Tasks Processed", str(specialization.get('total_tasks_processed', 0)))
            table.add_row("Learning Enabled", str(specialization.get('learning_enabled', False)))

            self.console.print(table)

            # Display available assistants
            assistants = agent.ai_orchestrator.get_available_assistants()
            if assistants:
                self.console.print("\n[green]Available Assistants:[/green]")
                assistant_table = Table(title="Assistant Details")
                assistant_table.add_column("Name", style="cyan")
                assistant_table.add_column("Performance", style="green")
                assistant_table.add_column("Tasks", style="yellow")
                assistant_table.add_column("Success Rate", style="magenta")

                for assistant in assistants:
                    assistant_table.add_row(
                        assistant.get('display_name', assistant.get('name', 'Unknown')),
                        f"{assistant.get('performance_score', 0):.2f}",
                        str(assistant.get('task_count', 0)),
                        f"{assistant.get('success_rate', 0):.2%}"
                    )

                self.console.print(assistant_table)

        except Exception as e:
            self.console.print(f"[red]AI status check failed: {e}[/red]")

    def _handle_ai_performance(self):
        """Handle AI performance command"""
        agent = self.initialize_agent()

        if not hasattr(agent, 'ai_orchestrator'):
            self.console.print("[red]AI orchestrator not available[/red]")
            return

        try:
            report = agent.ai_orchestrator.get_performance_report()

            # Display overall metrics
            metrics = report.get('specialization_metrics', {}).get('overall_metrics', {})
            table = Table(title="AI Performance Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Tasks", str(metrics.get('total_tasks', 0)))
            table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.2%}")
            table.add_row("Avg Response Time", f"{metrics.get('average_response_time', 0):.2f}s")
            table.add_row("User Satisfaction", f"{metrics.get('user_satisfaction', 0):.2f}/5")

            self.console.print(table)

            # Display recommendations
            recommendations = report.get('specialization_metrics', {}).get('recommendations', [])
            if recommendations:
                self.console.print("\n[green]System Recommendations:[/green]")
                for rec in recommendations:
                    if rec['type'] == 'top_performers':
                        self.console.print(f"🏆 Top performing assistants: {', '.join(rec['assistants'])}")
                    elif rec['type'] == 'needs_improvement':
                        self.console.print(f"⚠️  Assistants needing improvement: {', '.join(rec['assistants'])}")

        except Exception as e:
            self.console.print(f"[red]AI performance report failed: {e}[/red]")

    def _handle_analytics_command(self, args):
        """Handle analytics command"""
        if args.analytics_action == 'usage':
            self._handle_analytics_usage(args)
        elif args.analytics_action == 'tasks':
            self._handle_analytics_tasks()
        elif args.analytics_action == 'realtime':
            self._handle_analytics_realtime()
        elif args.analytics_action == 'user':
            self._handle_analytics_user(args)
        else:
            self.console.print("[red]Invalid analytics action. Use 'usage', 'tasks', 'realtime', or 'user'.[/red]")

    def _handle_analytics_usage(self, args):
        """Handle usage analytics"""
        try:
            from ..analytics.analytics_engine import analytics_engine
            report = analytics_engine.generate_usage_report(args.days)

            # Display summary
            summary = report.get('summary', {})
            table = Table(title=f"Usage Analytics (Last {args.days} days)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Events", str(summary.get('total_events', 0)))
            table.add_row("Unique Users", str(summary.get('unique_users', 0)))
            table.add_row("Avg Events/User", f"{summary.get('average_events_per_user', 0):.1f}")
            table.add_row("Task Success Rate", f"{summary.get('task_success_rate', 0):.1%}")

            self.console.print(table)

            # Display insights
            insights = report.get('insights', [])
            if insights:
                self.console.print("\n[green]System Insights:[/green]")
                for insight in insights:
                    self.console.print(f"• {insight}")

        except Exception as e:
            self.console.print(f"[red]Usage analytics failed: {e}[/red]")

    def _handle_analytics_tasks(self):
        """Handle task analytics"""
        try:
            from ..analytics.analytics_engine import analytics_engine
            report = analytics_engine.generate_task_analytics_report()

            # Display task types
            task_types = report.get('task_types', {})
            if task_types:
                table = Table(title="Task Type Analytics")
                table.add_column("Task Type", style="cyan")
                table.add_column("Total", style="yellow")
                table.add_column("Success Rate", style="green")
                table.add_column("Avg Time", style="magenta")

                for task_type, stats in task_types.items():
                    table.add_row(
                        task_type.replace('_', ' ').title(),
                        str(stats.get('total_tasks', 0)),
                        f"{stats.get('success_rate', 0):.1%}",
                        f"{stats.get('average_execution_time', 0):.2f}s"
                    )

                self.console.print(table)

            # Display assistant performance
            assistant_perf = report.get('assistant_performance', {})
            if assistant_perf:
                self.console.print("\n[green]Assistant Performance:[/green]")
                perf_table = Table(title="Assistant Performance")
                perf_table.add_column("Assistant", style="cyan")
                perf_table.add_column("Tasks", style="yellow")
                perf_table.add_column("Success Rate", style="green")
                perf_table.add_column("Avg Time", style="magenta")

                for assistant, perf in assistant_perf.items():
                    perf_table.add_row(
                        assistant.upper(),
                        str(perf.get('total_tasks', 0)),
                        f"{perf.get('success_rate', 0):.1%}",
                        f"{perf.get('average_execution_time', 0):.2f}s"
                    )

                self.console.print(perf_table)

        except Exception as e:
            self.console.print(f"[red]Task analytics failed: {e}[/red]")

    def _handle_analytics_realtime(self):
        """Handle real-time analytics"""
        try:
            from ..analytics.analytics_engine import analytics_engine
            metrics = analytics_engine.get_real_time_metrics()

            table = Table(title="Real-time Analytics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Active Users", str(metrics.get('active_users', 0)))
            table.add_row("Current Tasks", str(metrics.get('current_tasks', 0)))
            table.add_row("System Load", f"{metrics.get('system_load', 0):.1%}")
            table.add_row("Events Today", str(metrics.get('total_events_today', 0)))

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Real-time analytics failed: {e}[/red]")

    def _handle_analytics_user(self, args):
        """Handle user analytics"""
        try:
            from ..analytics.analytics_engine import analytics_engine
            user_data = analytics_engine.get_user_analytics(args.user_id)

            if 'message' in user_data:
                self.console.print(f"[yellow]{user_data['message']}[/yellow]")
                return

            table = Table(title=f"User Analytics: {args.user_id}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Events", str(user_data.get('total_events', 0)))
            table.add_row("Total Tasks", str(user_data.get('total_tasks', 0)))
            table.add_row("Task Success Rate", f"{user_data.get('task_success_rate', 0):.1%}")
            table.add_row("Days Active", str(user_data.get('days_active', 0)))

            self.console.print(table)

            # Display event distribution
            event_dist = user_data.get('event_distribution', {})
            if event_dist:
                self.console.print("\n[green]Event Distribution:[/green]")
                for event_type, count in event_dist.items():
                    self.console.print(f"• {event_type}: {count}")

        except Exception as e:
            self.console.print(f"[red]User analytics failed: {e}[/red]")

    def _handle_feedback_command(self, args):
        """Handle feedback command"""
        if args.feedback_action == 'submit':
            self._handle_feedback_submit(args)
        elif args.feedback_action == 'list':
            self._handle_feedback_list(args)
        elif args.feedback_action == 'summary':
            self._handle_feedback_summary(args)
        elif args.feedback_action == 'update':
            self._handle_feedback_update(args)
        else:
            self.console.print("[red]Invalid feedback action. Use 'submit', 'list', 'summary', or 'update'.[/red]")

    def _handle_feedback_submit(self, args):
        """Handle feedback submission"""
        try:
            from ..feedback.feedback_manager import feedback_manager

            feedback_id = feedback_manager.submit_feedback(
                user_id="cli_user",
                feedback_type=args.type,
                title=args.title,
                message=args.message,
                rating=args.rating
            )

            self.console.print(f"[green]✓ Feedback submitted successfully![/green]")
            self.console.print(f"[blue]Feedback ID:[/blue] {feedback_id}")

        except Exception as e:
            self.console.print(f"[red]Failed to submit feedback: {e}[/red]")

    def _handle_feedback_list(self, args):
        """Handle feedback listing"""
        try:
            from ..feedback.feedback_manager import feedback_manager

            entries = feedback_manager.get_feedback_entries(
                feedback_type=args.type,
                status=args.status,
                limit=args.limit
            )

            if not entries:
                self.console.print("[yellow]No feedback entries found.[/yellow]")
                return

            table = Table(title="User Feedback")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Type", style="green")
            table.add_column("Title", style="yellow")
            table.add_column("Status", style="magenta")
            table.add_column("Priority", style="red")
            table.add_column("Created", style="blue")

            for entry in entries:
                created = datetime.fromisoformat(entry['created_at']).strftime("%Y-%m-%d %H:%M")
                rating = f" ({entry.get('rating', 'N/A')}/5)" if entry.get('rating') else ""
                table.add_row(
                    entry['id'][:8] + "...",
                    entry['feedback_type'].upper(),
                    entry['title'][:30] + "..." if len(entry['title']) > 30 else entry['title'],
                    entry['status'],
                    entry['priority'],
                    created
                )

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Failed to list feedback: {e}[/red]")

    def _handle_feedback_summary(self, args):
        """Handle feedback summary"""
        try:
            from ..feedback.feedback_manager import feedback_manager

            summary = feedback_manager.get_feedback_summary(args.days)

            # Display summary
            table = Table(title=f"Feedback Summary (Last {args.days} days)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Feedback", str(summary.get('total_feedback', 0)))
            if summary.get('average_rating', 0) > 0:
                table.add_row("Average Rating", f"{summary['average_rating']:.1f}/5")

            self.console.print(table)

            # Display distributions
            if summary.get('feedback_types'):
                self.console.print("\n[green]Feedback Types:[/green]")
                for ftype, count in summary['feedback_types'].items():
                    self.console.print(f"• {ftype.title()}: {count}")

            if summary.get('categories'):
                self.console.print("\n[green]Categories:[/green]")
                for category, count in summary['categories'].items():
                    self.console.print(f"• {category.title()}: {count}")

            # Display urgent issues
            if summary.get('urgent_issues'):
                self.console.print("\n[red]Urgent Issues:[/red]")
                for issue in summary['urgent_issues']:
                    self.console.print(f"• {issue}")

        except Exception as e:
            self.console.print(f"[red]Failed to get feedback summary: {e}[/red]")

    def _handle_feedback_update(self, args):
        """Handle feedback status update"""
        try:
            from ..feedback.feedback_manager import feedback_manager

            success = feedback_manager.update_feedback_status(
                feedback_id=args.feedback_id,
                status=args.status,
                reviewed_by="cli_user",
                review_notes=args.notes
            )

            if success:
                self.console.print(f"[green]✓ Feedback {args.feedback_id} status updated to {args.status}[/green]")
            else:
                self.console.print(f"[red]✗ Failed to update feedback {args.feedback_id}[/red]")

        except Exception as e:
            self.console.print(f"[red]Failed to update feedback: {e}[/red]")

    def _handle_plugin_command(self, args):
        """Handle plugin command"""
        if args.plugin_action == 'list':
            self._handle_plugin_list()
        elif args.plugin_action == 'load':
            self._handle_plugin_load(args)
        elif args.plugin_action == 'unload':
            self._handle_plugin_unload(args)
        elif args.plugin_action == 'enable':
            self._handle_plugin_enable(args)
        elif args.plugin_action == 'disable':
            self._handle_plugin_disable(args)
        elif args.plugin_action == 'info':
            self._handle_plugin_info(args)
        elif args.plugin_action == 'discover':
            self._handle_plugin_discover()
        else:
            self.console.print("[red]Invalid plugin action. Use 'list', 'load', 'unload', 'enable', 'disable', 'info', or 'discover'.[/red]")

    def _handle_plugin_list(self):
        """Handle plugin listing"""
        try:
            from ..plugins.manager import plugin_manager
            plugins = plugin_manager.list_plugins()

            if not plugins:
                self.console.print("[yellow]No plugins loaded.[/yellow]")
                return

            table = Table(title="CES Plugins")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Description", style="magenta")

            for name, info in plugins.items():
                status = "Enabled" if info["enabled"] else "Disabled"
                table.add_row(
                    info["name"],
                    info["version"],
                    status,
                    info["description"][:50] + "..." if len(info["description"]) > 50 else info["description"]
                )

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Failed to list plugins: {e}[/red]")

    def _handle_plugin_load(self, args):
        """Handle plugin loading"""
        try:
            from ..plugins.manager import plugin_manager

            if plugin_manager.load_plugin(args.name):
                self.console.print(f"[green]✓ Plugin '{args.name}' loaded successfully[/green]")
            else:
                self.console.print(f"[red]✗ Failed to load plugin '{args.name}'[/red]")

        except Exception as e:
            self.console.print(f"[red]Failed to load plugin: {e}[/red]")

    def _handle_plugin_unload(self, args):
        """Handle plugin unloading"""
        try:
            from ..plugins.manager import plugin_manager

            if plugin_manager.unload_plugin(args.name):
                self.console.print(f"[green]✓ Plugin '{args.name}' unloaded successfully[/green]")
            else:
                self.console.print(f"[red]✗ Failed to unload plugin '{args.name}'[/red]")

        except Exception as e:
            self.console.print(f"[red]Failed to unload plugin: {e}[/red]")

    def _handle_plugin_enable(self, args):
        """Handle plugin enabling"""
        try:
            from ..plugins.manager import plugin_manager

            if plugin_manager.enable_plugin(args.name):
                self.console.print(f"[green]✓ Plugin '{args.name}' enabled successfully[/green]")
            else:
                self.console.print(f"[red]✗ Failed to enable plugin '{args.name}'[/red]")

        except Exception as e:
            self.console.print(f"[red]Failed to enable plugin: {e}[/red]")

    def _handle_plugin_disable(self, args):
        """Handle plugin disabling"""
        try:
            from ..plugins.manager import plugin_manager

            if plugin_manager.disable_plugin(args.name):
                self.console.print(f"[green]✓ Plugin '{args.name}' disabled successfully[/green]")
            else:
                self.console.print(f"[red]✗ Failed to disable plugin '{args.name}'[/red]")

        except Exception as e:
            self.console.print(f"[red]Failed to disable plugin: {e}[/red]")

    def _handle_plugin_info(self, args):
        """Handle plugin information display"""
        try:
            from ..plugins.manager import plugin_manager
            info = plugin_manager.get_plugin_info(args.name)

            if not info:
                self.console.print(f"[red]Plugin '{args.name}' not found[/red]")
                return

            table = Table(title=f"Plugin Information: {args.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Name", info.name)
            table.add_row("Version", info.version)
            table.add_row("Description", info.description)
            table.add_row("Author", info.author)
            table.add_row("License", info.license)
            table.add_row("Homepage", info.homepage or "N/A")
            table.add_row("Tags", ", ".join(info.tags) if info.tags else "None")

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Failed to get plugin info: {e}[/red]")

    def _handle_plugin_discover(self):
        """Handle plugin discovery"""
        try:
            from ..plugins.manager import plugin_manager
            plugins = plugin_manager.discover_plugins()

            if not plugins:
                self.console.print("[yellow]No plugins discovered.[/yellow]")
                return

            self.console.print(f"[green]Discovered {len(plugins)} plugin(s):[/green]")
            for plugin in plugins:
                self.console.print(f"• {plugin}")

        except Exception as e:
            self.console.print(f"[red]Failed to discover plugins: {e}[/red]")


def main():
    """Main entry point for CES CLI"""
    cli = CESCLI()
    cli.run()


if __name__ == "__main__":
    main()