"""
Cognitive Agent - Main CES Orchestrator

The Cognitive Agent is the central intelligence of the CES system. It analyzes tasks,
manages context, coordinates AI assistants, and ensures ethical operation.
Enhanced with CodeSage MCP server capabilities for advanced task orchestration.
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, Callable
from dataclasses import dataclass
from datetime import datetime
import time
import threading
from collections import deque
import statistics

from ..ai_orchestrator.ai_assistant import AIOrchestrator
from ..core.memory_manager import MemoryManager
from ..core.ethical_controller import EthicalController
from ..config.ces_config import CESConfig
from ..codesage_integration import CodeSageIntegration
from ..core.performance_monitor import get_performance_monitor, get_usage_analyzer
from ..core.tools import get_ces_tools
from ..core.human_ai_interaction import HumanAIInteractionManager
from ..core.error_recovery import ErrorRecoveryManager

if TYPE_CHECKING:
    from ..core.task_workflow import get_workflow_orchestrator


@dataclass
class TaskAnalysis:
    """Analysis result for a given task"""
    complexity_score: float
    required_skills: List[str]
    estimated_duration: int  # minutes
    recommended_assistants: List[str]
    ethical_concerns: List[str]
    context_requirements: List[str]
    mcp_tools_required: List[str] = None  # MCP tools needed for task
    performance_requirements: Dict[str, Any] = None  # Performance constraints


@dataclass
class MCPToolExecution:
    """Result of MCP tool execution"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    execution_time_ms: float
    success: bool
    timestamp: datetime


class CognitiveAgent:
    """
    Main cognitive agent that orchestrates CES operations.

    This agent analyzes tasks, manages context, coordinates AI assistants,
    and ensures all operations align with ethical guidelines.
    Enhanced with CodeSage MCP server capabilities for advanced orchestration.
    """

    def __init__(self, config: Optional[CESConfig] = None):
        self.config = config or CESConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.memory_manager = MemoryManager()
        self.ai_orchestrator = AIOrchestrator()
        self.ethical_controller = EthicalController()

        # Initialize MCP integration (CodeSage)
        self.codesage_integration = CodeSageIntegration()

        # Performance monitoring (adapted from CodeSage)
        self.performance_monitor = get_performance_monitor()
        self.usage_analyzer = get_usage_analyzer()

        # CES Tools for AI orchestration (adapted from CodeSage)
        self.ces_tools = get_ces_tools()

        # Workflow orchestrator (adapted from CodeSage)
        from ..core.task_workflow import get_workflow_orchestrator
        self.workflow_orchestrator = get_workflow_orchestrator(self, self.codesage_integration)

        # Human-AI interaction manager for real-time collaboration
        self.interaction_manager = HumanAIInteractionManager(self.memory_manager, self.ai_orchestrator)

        # Error recovery manager for fault tolerance
        self.error_recovery = ErrorRecoveryManager()

        # MCP tool execution history
        self.mcp_execution_history = deque(maxlen=1000)

        self.logger.info("Cognitive Agent initialized with MCP capabilities and human-AI interaction protocols")

    async def analyze_task(self, task_description: str) -> TaskAnalysis:
        """
        Analyze a task using CodeSage tools for intelligent decomposition and complexity assessment.
        Includes performance monitoring for P95 < 500ms target.

        Args:
            task_description: Natural language description of the task

        Returns:
            TaskAnalysis: Detailed analysis of the task
        """
        self.logger.info(f"Analyzing task with CodeSage integration: {task_description[:50]}...")

        # Enhanced task analysis using CodeSage tools with performance tracking
        analysis_start = time.time()

        try:
            # Use CodeSage for comprehensive analysis
            codesage_analysis = await self._perform_codesage_analysis(task_description)

            # Extract complexity from CodeSage analysis
            complexity_score = codesage_analysis.get('complexity_score', self._calculate_complexity(task_description))

            # Extract skills from CodeSage analysis
            required_skills = codesage_analysis.get('required_skills', self._identify_required_skills(task_description))

            # Enhanced duration estimation with CodeSage insights
            estimated_duration = self._estimate_duration_with_codesage(complexity_score, required_skills, codesage_analysis)

            # Get recommendations from AI orchestrator with enhanced context
            recommended_assistants = self.ai_orchestrator.recommend_assistants(
                task_description, required_skills
            )

            # Check ethical concerns
            ethical_concerns = self.ethical_controller.check_task_ethics(task_description)

            # Determine context requirements with CodeSage insights
            context_requirements = self.memory_manager.analyze_context_needs(task_description)
            if codesage_analysis.get('context_hints'):
                context_requirements.extend(codesage_analysis['context_hints'])

            # Analyze MCP tools required with CodeSage recommendations
            mcp_tools_required = self._analyze_mcp_tools_needed_enhanced(task_description, required_skills, codesage_analysis)

            # Determine performance requirements with CodeSage insights
            performance_requirements = self._determine_performance_requirements_enhanced(complexity_score, codesage_analysis)

            analysis_time = (time.time() - analysis_start) * 1000

            # Record performance metrics
            self.performance_monitor.record_response_time(
                analysis_time,
                "task_analysis",
                {
                    'complexity_score': complexity_score,
                    'skills_count': len(required_skills),
                    'tools_count': len(mcp_tools_required) if mcp_tools_required else 0
                }
            )

            # Check P95 target achievement
            if analysis_time > 500:
                self.logger.warning(f"Task analysis exceeded P95 target: {analysis_time:.2f}ms > 500ms")

            self.logger.info(f"Task analysis completed in {analysis_time:.2f}ms with complexity score: {complexity_score}")

            return TaskAnalysis(
                complexity_score=complexity_score,
                required_skills=required_skills,
                estimated_duration=estimated_duration,
                recommended_assistants=recommended_assistants,
                ethical_concerns=ethical_concerns,
                context_requirements=context_requirements,
                mcp_tools_required=mcp_tools_required,
                performance_requirements=performance_requirements
            )

        except Exception as e:
            analysis_time = (time.time() - analysis_start) * 1000
            self.logger.error(f"Task analysis failed after {analysis_time:.2f}ms: {e}")

            # Record failed analysis performance
            self.performance_monitor.record_response_time(
                analysis_time,
                "task_analysis",
                {'error': str(e), 'failed': True}
            )

            # Return basic analysis as fallback
            return TaskAnalysis(
                complexity_score=self._calculate_complexity(task_description),
                required_skills=self._identify_required_skills(task_description),
                estimated_duration=self._estimate_duration(
                    self._calculate_complexity(task_description),
                    self._identify_required_skills(task_description)
                ),
                recommended_assistants=['grok'],
                ethical_concerns=[],
                context_requirements=['task_history'],
                mcp_tools_required=None,
                performance_requirements=self._determine_performance_requirements(
                    self._calculate_complexity(task_description)
                )
            )

    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a task using the optimal AI assistant configuration with enhanced CodeSage integration.
        Includes comprehensive performance monitoring for P95 < 500ms target.

        Args:
            task_description: Task to execute

        Returns:
            Dict containing execution results
        """
        execution_start = time.time()
        self.logger.info(f"Executing task: {task_description[:50]}...")

        try:
            # Analyze the task first (now async with CodeSage integration)
            analysis = await self.analyze_task(task_description)

            # Check if task is ethically acceptable
            if analysis.ethical_concerns:
                self.logger.warning(f"Ethical concerns identified: {analysis.ethical_concerns}")
                if not self.ethical_controller.approve_task(analysis.ethical_concerns):
                    execution_time = (time.time() - execution_start) * 1000
                    self.performance_monitor.record_task_completion(
                        execution_time,
                        "rejected_task",
                        {'ethical_rejection': True}
                    )
                    return {
                        "status": "rejected",
                        "reason": "Ethical concerns",
                        "details": analysis.ethical_concerns,
                        "execution_time_ms": execution_time
                    }

            # Retrieve relevant context with enhanced retention
            context_start = time.time()
            context = await self.memory_manager.retrieve_context(
                task_description, analysis.context_requirements
            )
            context_time = (time.time() - context_start) * 1000
            self.performance_monitor.record_context_retrieval(context_time, "task_context")

            # Execute tools (both CES and MCP) if available and needed
            tool_results = []
            mcp_results = []
            if analysis.mcp_tools_required:
                self.logger.info(f"Executing {len(analysis.mcp_tools_required)} tools for task")
                for tool_name in analysis.mcp_tools_required:
                    try:
                        # Prepare tool arguments based on task
                        tool_args = self._prepare_tool_arguments(tool_name, task_description, context)
                        if tool_args:
                            # Try CES tools first, then MCP tools
                            ces_tool_mapping = ['read_code_file', 'search_codebase', 'get_file_structure',
                                              'count_lines_of_code', 'get_dependencies_overview', 'analyze_codebase_improvements']

                            if tool_name in ces_tool_mapping:
                                tool_result = await self.execute_ces_tool(tool_name, tool_args)
                                tool_results.append(tool_result)
                                self.logger.info(f"CES tool {tool_name} executed in {tool_result['execution_time_ms']:.2f}ms")
                            elif self.codesage_integration.connected:
                                tool_result = await self.execute_mcp_tool(tool_name, tool_args)
                                tool_results.append(tool_result.__dict__)
                                mcp_results.append(tool_result)
                                self.logger.info(f"MCP tool {tool_name} executed in {tool_result.execution_time_ms:.2f}ms")
                            else:
                                self.logger.warning(f"Tool {tool_name} not available (CES or MCP)")
                    except Exception as e:
                        self.logger.error(f"Failed to execute tool {tool_name}: {e}")

            # Execute task with selected assistant, enhanced with MCP results
            enhanced_context = context.copy()
            if mcp_results:
                enhanced_context['mcp_tool_results'] = [result.__dict__ for result in mcp_results]

            # Execute with performance monitoring
            task_execution_start = time.time()
            result = await self.ai_orchestrator.execute_task(
                task_description=task_description,
                context=enhanced_context,
                assistant_preferences=analysis.recommended_assistants
            )
            task_execution_time = (time.time() - task_execution_start) * 1000
            self.performance_monitor.record_task_completion(
                task_execution_time,
                "ai_orchestration",
                {
                    'assistants_used': len(result.get('assistants_used', [])),
                    'execution_mode': result.get('execution_mode', 'unknown')
                }
            )

            # Store execution result in memory
            self.memory_manager.store_task_result(task_description, result)

            # Calculate total execution time
            total_execution_time = (time.time() - execution_start) * 1000

            # Record overall task execution performance
            self.performance_monitor.record_response_time(
                total_execution_time,
                "task_execution",
                {
                    'complexity_score': analysis.complexity_score,
                    'tools_executed': len(tool_results),
                    'assistants_used': len(result.get('assistants_used', []))
                }
            )

            # Check P95 target achievement
            if total_execution_time > 500:
                self.logger.warning(f"Task execution exceeded P95 target: {total_execution_time:.2f}ms > 500ms")

            # Record operation success for circuit breaker
            self.error_recovery.record_operation_success("cognitive_agent")

            self.logger.info(f"Task executed successfully in {total_execution_time:.2f}ms")

            return {
                "status": "completed",
                "analysis": analysis,
                "result": result,
                "tool_results": tool_results,
                "performance_metrics": self.performance_monitor.get_current_metrics(),
                "execution_time_ms": total_execution_time,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            total_execution_time = (time.time() - execution_start) * 1000
            self.logger.error(f"Task execution failed after {total_execution_time:.2f}ms: {e}")

            # Record operation failure for circuit breaker
            self.error_recovery.record_operation_failure("cognitive_agent")

            # Handle error with recovery system
            recovery_result = await self.error_recovery.handle_error(
                e, "cognitive_agent", "execute_task",
                context={
                    'task_description': task_description[:100],
                    'execution_time_ms': total_execution_time,
                    'analysis_completed': 'analysis' in locals()
                }
            )

            # Record failed execution performance
            self.performance_monitor.record_task_completion(
                total_execution_time,
                "failed_task",
                {
                    'error': str(e),
                    'recovery_strategy': recovery_result.strategy_used.value,
                    'recovery_success': recovery_result.success
                }
            )

            # Return appropriate response based on recovery
            if recovery_result.success and recovery_result.fallback_used:
                return {
                    "status": "completed_with_fallback",
                    "warning": "Task completed using fallback mechanisms due to errors",
                    "error": str(e),
                    "recovery_strategy": recovery_result.strategy_used.value,
                    "execution_time_ms": total_execution_time,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "failed",
                    "error": str(e),
                    "recovery_strategy": recovery_result.strategy_used.value,
                    "recovery_success": recovery_result.success,
                    "execution_time_ms": total_execution_time,
                    "timestamp": datetime.now().isoformat()
                }

    async def _perform_codesage_analysis(self, task_description: str) -> Dict[str, Any]:
        """Perform comprehensive analysis using CodeSage tools"""
        analysis_results = {
            'complexity_score': 5.0,
            'required_skills': [],
            'context_hints': [],
            'performance_bottlenecks': [],
            'dependencies': [],
            'improvement_suggestions': []
        }

        try:
            # Use CodeSage comprehensive analysis if available
            if self.codesage_integration.connected:
                comprehensive_result = await self.execute_mcp_tool('run_comprehensive_advanced_analysis', {
                    'file_path': '.',  # Analyze current codebase context
                    'task_description': task_description
                })

                if comprehensive_result.success:
                    result_data = comprehensive_result.result
                    if isinstance(result_data, dict):
                        # Extract complexity from performance bottlenecks
                        if 'performance_bottlenecks' in result_data:
                            bottleneck_count = len(result_data['performance_bottlenecks'])
                            analysis_results['complexity_score'] = min(10.0, 5.0 + bottleneck_count * 0.5)

                        # Extract skills from dependencies
                        if 'dependencies' in result_data:
                            analysis_results['required_skills'] = self._extract_skills_from_dependencies(result_data['dependencies'])

                        # Add context hints
                        if 'context_requirements' in result_data:
                            analysis_results['context_hints'] = result_data['context_requirements']

                        # Store improvement suggestions
                        if 'improvements' in result_data:
                            analysis_results['improvement_suggestions'] = result_data['improvements']

            # Fallback to basic analysis if CodeSage not available
            else:
                analysis_results['complexity_score'] = self._calculate_complexity(task_description)
                analysis_results['required_skills'] = self._identify_required_skills(task_description)

        except Exception as e:
            self.logger.warning(f"CodeSage analysis failed, using fallback: {e}")
            analysis_results['complexity_score'] = self._calculate_complexity(task_description)
            analysis_results['required_skills'] = self._identify_required_skills(task_description)

        return analysis_results

    def _calculate_complexity(self, task_description: str) -> float:
        """Calculate task complexity score (0-10)"""
        # Enhanced implementation with more sophisticated analysis
        length_score = min(len(task_description) / 500, 1.0) * 3

        # Complex keywords with weights
        complexity_keywords = {
            'complex': 2, 'advanced': 2, 'optimize': 1.5, 'architecture': 2,
            'integrate': 1.5, 'refactor': 1.5, 'debug': 1, 'performance': 1.5,
            'security': 2, 'testing': 1, 'deployment': 1.5, 'scaling': 2
        }

        keyword_score = 0
        task_lower = task_description.lower()
        for keyword, weight in complexity_keywords.items():
            if keyword in task_lower:
                keyword_score += weight

        # Multi-step indicators
        multi_step_indicators = ['step', 'phase', 'stage', 'multiple', 'several', 'various']
        multi_step_score = sum(1 for indicator in multi_step_indicators if indicator in task_lower) * 0.5

        # Technical depth indicators
        technical_indicators = ['algorithm', 'data structure', 'concurrency', 'distributed', 'microservice']
        technical_score = sum(1 for indicator in technical_indicators if indicator in task_lower) * 1.5

        total_score = length_score + keyword_score + multi_step_score + technical_score
        return min(total_score, 10.0)

    def _extract_skills_from_dependencies(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """Extract required skills from dependency analysis"""
        skills = set()

        for dep in dependencies:
            dep_type = dep.get('type', '').lower()
            if 'function' in dep_type:
                skills.add('programming')
            elif 'class' in dep_type:
                skills.add('object_oriented_design')
            elif 'module' in dep_type:
                skills.add('modular_design')
            elif 'external' in dep_type:
                skills.add('integration')
            elif 'database' in dep_type:
                skills.add('data_management')

        return list(skills) or ['general']

    def _identify_required_skills(self, task_description: str) -> List[str]:
        """Identify skills required for the task with enhanced keyword matching"""
        skills = []
        task_lower = task_description.lower()

        # Programming and development skills
        if any(word in task_lower for word in ['python', 'code', 'program', 'implement', 'develop']):
            skills.append('programming')

        # Design and architecture skills
        if any(word in task_lower for word in ['design', 'architecture', 'structure', 'plan', 'organize']):
            skills.append('system_design')

        # Testing and debugging skills
        if any(word in task_lower for word in ['test', 'debug', 'fix', 'error', 'bug', 'validate']):
            skills.append('testing')

        # Documentation skills
        if any(word in task_lower for word in ['document', 'readme', 'comment', 'explain', 'describe']):
            skills.append('documentation')

        # Analysis and optimization skills
        if any(word in task_lower for word in ['analyze', 'optimize', 'performance', 'review', 'assess']):
            skills.append('analysis')

        # Integration and deployment skills
        if any(word in task_lower for word in ['integrate', 'deploy', 'configure', 'setup']):
            skills.append('integration')

        return skills or ['general']

    def _estimate_duration_with_codesage(self, complexity: float, skills: List[str], codesage_analysis: Dict[str, Any]) -> int:
        """Estimate task duration using CodeSage insights"""
        base_duration = 30  # 30 minutes base

        # Complexity multiplier (enhanced)
        complexity_multiplier = 1 + (complexity / 10)  # 1-2x multiplier

        # Skill multiplier with CodeSage insights
        skill_multiplier = 1 + (len(skills) * 0.2)

        # Adjust for performance bottlenecks
        bottleneck_multiplier = 1.0
        if codesage_analysis.get('performance_bottlenecks'):
            bottleneck_count = len(codesage_analysis['performance_bottlenecks'])
            bottleneck_multiplier = 1 + (bottleneck_count * 0.1)

        # Adjust for dependencies
        dependency_multiplier = 1.0
        if codesage_analysis.get('dependencies'):
            dep_count = len(codesage_analysis['dependencies'])
            dependency_multiplier = 1 + (dep_count * 0.05)

        # Adjust for improvement suggestions (indicates complexity)
        improvement_multiplier = 1.0
        if codesage_analysis.get('improvement_suggestions'):
            imp_count = len(codesage_analysis['improvement_suggestions'])
            improvement_multiplier = 1 + (imp_count * 0.1)

        total_multiplier = complexity_multiplier * skill_multiplier * bottleneck_multiplier * dependency_multiplier * improvement_multiplier

        estimated = int(base_duration * total_multiplier)

        # Cap at reasonable limits
        return max(15, min(estimated, 480))  # 15 minutes to 8 hours

    def _estimate_duration(self, complexity: float, skills: List[str]) -> int:
        """Estimate task duration in minutes (fallback method)"""
        base_duration = 30  # 30 minutes base
        complexity_multiplier = 1 + (complexity / 10)  # 1-2x multiplier
        skill_multiplier = 1 + (len(skills) * 0.2)  # Additional time per skill

        return int(base_duration * complexity_multiplier * skill_multiplier)

    def _analyze_mcp_tools_needed_enhanced(self, task_description: str, skills: List[str], codesage_analysis: Dict[str, Any]) -> List[str]:
        """Analyze which MCP tools are needed with CodeSage insights"""
        tools_needed = []
        task_lower = task_description.lower()

        # Enhanced skill to tool mapping
        skill_tool_mapping = {
            'programming': ['read_code_file', 'search_codebase', 'analyze_codebase_improvements', 'run_comprehensive_advanced_analysis'],
            'system_design': ['get_file_structure', 'analyze_codebase_improvements', 'analyze_function_dependencies'],
            'testing': ['generate_unit_tests', 'profile_code_performance', 'run_comprehensive_advanced_analysis'],
            'documentation': ['auto_document_tool', 'list_undocumented_functions', 'summarize_code_section'],
            'analysis': ['analyze_function_dependencies', 'predict_performance_bottlenecks', 'run_comprehensive_advanced_analysis'],
            'integration': ['get_dependencies_overview', 'analyze_external_library_usage'],
            'object_oriented_design': ['analyze_function_dependencies', 'get_file_structure'],
            'modular_design': ['get_file_structure', 'analyze_codebase_improvements'],
            'data_management': ['analyze_external_library_usage', 'get_dependencies_overview']
        }

        # Add tools based on skills
        for skill in skills:
            if skill in skill_tool_mapping:
                tools_needed.extend(skill_tool_mapping[skill])

        # CodeSage-driven tool recommendations
        if codesage_analysis.get('performance_bottlenecks'):
            tools_needed.extend(['predict_performance_bottlenecks', 'profile_code_performance', 'get_performance_metrics'])

        if codesage_analysis.get('dependencies'):
            tools_needed.extend(['analyze_function_dependencies', 'get_dependencies_overview'])

        if codesage_analysis.get('improvement_suggestions'):
            tools_needed.append('analyze_codebase_improvements')

        # Task-specific tool analysis (enhanced)
        if any(word in task_lower for word in ['analyze', 'review', 'assess', 'evaluate']):
            tools_needed.extend(['analyze_codebase_improvements', 'run_comprehensive_advanced_analysis', 'analyze_function_dependencies'])

        if 'test' in task_lower or 'testing' in task_lower:
            tools_needed.extend(['generate_unit_tests', 'profile_code_performance'])

        if any(word in task_lower for word in ['document', 'docstring', 'readme']):
            tools_needed.extend(['auto_document_tool', 'list_undocumented_functions'])

        if any(word in task_lower for word in ['performance', 'optimize', 'speed', 'efficiency']):
            tools_needed.extend(['profile_code_performance', 'get_performance_metrics', 'predict_performance_bottlenecks'])

        if any(word in task_lower for word in ['debug', 'fix', 'error', 'bug']):
            tools_needed.extend(['analyze_function_dependencies', 'run_comprehensive_advanced_analysis'])

        if any(word in task_lower for word in ['integrate', 'api', 'external']):
            tools_needed.extend(['analyze_external_library_usage', 'get_dependencies_overview'])

        return list(set(tools_needed))  # Remove duplicates

    def _analyze_mcp_tools_needed(self, task_description: str, skills: List[str]) -> List[str]:
        """Analyze which MCP tools are needed for the task (fallback method)"""
        tools_needed = []
        task_lower = task_description.lower()

        # Map skills to MCP tools
        skill_tool_mapping = {
            'programming': ['read_code_file', 'search_codebase', 'analyze_codebase_improvements'],
            'system_design': ['get_file_structure', 'analyze_codebase_improvements'],
            'testing': ['generate_unit_tests', 'profile_code_performance'],
            'documentation': ['auto_document_tool', 'list_undocumented_functions'],
            'analysis': ['analyze_function_dependencies', 'predict_performance_bottlenecks']
        }

        for skill in skills:
            if skill in skill_tool_mapping:
                tools_needed.extend(skill_tool_mapping[skill])

        # Task-specific tool analysis
        if 'analyze' in task_lower or 'review' in task_lower:
            tools_needed.extend(['analyze_codebase_improvements', 'run_comprehensive_advanced_analysis'])
        if 'test' in task_lower:
            tools_needed.append('generate_unit_tests')
        if 'document' in task_lower:
            tools_needed.append('auto_document_tool')
        if 'performance' in task_lower or 'optimize' in task_lower:
            tools_needed.extend(['profile_code_performance', 'get_performance_metrics'])

        return list(set(tools_needed))  # Remove duplicates

    def _determine_performance_requirements_enhanced(self, complexity_score: float, codesage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine performance requirements with CodeSage insights"""
        base_requirements = self._determine_performance_requirements(complexity_score)

        # Adjust based on CodeSage analysis
        adjustments = {
            'max_response_time_ms': base_requirements['max_response_time_ms'],
            'memory_limit_mb': base_requirements['memory_limit_mb'],
            'cpu_limit_percent': base_requirements['cpu_limit_percent']
        }

        # Increase requirements for performance-critical tasks
        if codesage_analysis.get('performance_bottlenecks'):
            bottleneck_count = len(codesage_analysis['performance_bottlenecks'])
            # Reduce response time requirements for performance tasks
            adjustments['max_response_time_ms'] = max(1000, adjustments['max_response_time_ms'] - (bottleneck_count * 1000))
            adjustments['cpu_limit_percent'] = min(95, adjustments['cpu_limit_percent'] + (bottleneck_count * 5))

        # Increase memory for complex dependency analysis
        if codesage_analysis.get('dependencies'):
            dep_count = len(codesage_analysis['dependencies'])
            adjustments['memory_limit_mb'] = min(2048, adjustments['memory_limit_mb'] + (dep_count * 50))

        # Stricter requirements for high-complexity tasks with many improvements needed
        if codesage_analysis.get('improvement_suggestions'):
            imp_count = len(codesage_analysis['improvement_suggestions'])
            if imp_count > 5:
                adjustments['max_response_time_ms'] = max(2000, adjustments['max_response_time_ms'] * 0.8)
                adjustments['memory_limit_mb'] = min(4096, adjustments['memory_limit_mb'] * 1.5)

        # Month 1 target: P95 < 500ms for task analysis
        adjustments['target_p95_response_time_ms'] = 500
        adjustments['monitoring_enabled'] = True
        adjustments['performance_tracking'] = True

        return adjustments

    def _determine_performance_requirements(self, complexity_score: float) -> Dict[str, Any]:
        """Determine performance requirements based on task complexity (fallback method)"""
        if complexity_score < 3:
            return {
                'max_response_time_ms': 5000,
                'memory_limit_mb': 256,
                'cpu_limit_percent': 50
            }
        elif complexity_score < 7:
            return {
                'max_response_time_ms': 15000,
                'memory_limit_mb': 512,
                'cpu_limit_percent': 70
            }
        else:
            return {
                'max_response_time_ms': 30000,
                'memory_limit_mb': 1024,
                'cpu_limit_percent': 90
            }

    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolExecution:
        """Execute an MCP tool and record performance metrics"""
        start_time = time.time()

        try:
            # Execute tool via CodeSage integration
            result = await self.codesage_integration.execute_tool(tool_name, arguments)
            success = result.get('status') == 'success'

            execution_time = (time.time() - start_time) * 1000

            # Record performance metrics
            self.performance_monitor.record_mcp_tool_execution(tool_name, execution_time, success)

            # Record usage
            self.usage_analyzer.record_user_action("anonymous", f"mcp_tool_{tool_name}", arguments)

            execution_result = MCPToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                result=result.get('result') if success else result.get('error'),
                execution_time_ms=execution_time,
                success=success,
                timestamp=datetime.now()
            )

            # Record operation success for circuit breaker
            self.error_recovery.record_operation_success("codesage_mcp")

            # Store in execution history
            self.mcp_execution_history.append(execution_result)

            return execution_result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"MCP tool execution failed: {e}")

            # Handle error with recovery system
            recovery_result = await self.error_recovery.handle_error(
                e, "codesage_integration", f"execute_{tool_name}",
                context={
                    'tool_name': tool_name,
                    'arguments': arguments,
                    'execution_time_ms': execution_time
                }
            )

            # Record operation failure for circuit breaker
            self.error_recovery.record_operation_failure("codesage_mcp")

            execution_result = MCPToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                result=str(e),
                execution_time_ms=execution_time,
                success=False,
                timestamp=datetime.now()
            )

            self.mcp_execution_history.append(execution_result)
            return execution_result

    async def execute_ces_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CES tool directly (not through MCP)"""
        start_time = time.time()

        try:
            # Map tool names to CES tools methods
            tool_method_mapping = {
                'read_code_file': self.ces_tools.read_code_file,
                'search_codebase': self.ces_tools.search_codebase,
                'get_file_structure': self.ces_tools.get_file_structure,
                'count_lines_of_code': self.ces_tools.count_lines_of_code,
                'get_dependencies_overview': self.ces_tools.get_dependencies_overview,
                'analyze_codebase_improvements': self.ces_tools.analyze_codebase_improvements
            }

            if tool_name not in tool_method_mapping:
                return {
                    "status": "error",
                    "error": f"CES tool '{tool_name}' not available",
                    "execution_time_ms": 0
                }

            # Execute the tool
            method = tool_method_mapping[tool_name]
            result = method(**arguments)

            execution_time = (time.time() - start_time) * 1000

            # Record performance metrics
            success = result.get('status') == 'success'
            self.performance_monitor.record_mcp_tool_execution(f"ces_{tool_name}", execution_time, success)

            return {
                **result,
                "execution_time_ms": execution_time,
                "tool_type": "ces"
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"CES tool execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": execution_time,
                "tool_type": "ces"
            }

    def _prepare_tool_arguments(self, tool_name: str, task_description: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare arguments for tool execution based on task and context"""
        # Extract codebase path from context or task description
        codebase_path = context.get('codebase_path', '.')

        # CES tools mapping (direct execution)
        ces_tool_mapping = {
            'read_code_file': lambda: {'file_path': self._extract_file_path(task_description)},
            'search_codebase': lambda: {'pattern': self._extract_search_pattern(task_description), 'file_types': ['.py', '.js', '.ts']},
            'analyze_codebase_improvements': lambda: {},
            'get_file_structure': lambda: {'file_path': self._extract_file_path(task_description) or 'main.py'},
            'count_lines_of_code': lambda: {},
            'get_dependencies_overview': lambda: {}
        }

        # MCP tools mapping (remote execution)
        mcp_tool_mapping = {
            'generate_unit_tests': lambda: {'file_path': self._extract_file_path(task_description)},
            'auto_document_tool': lambda: {'tool_name': 'analyze_codebase'},
            'profile_code_performance': lambda: {'file_path': self._extract_file_path(task_description)},
            'get_performance_metrics': lambda: {},
            'analyze_function_dependencies': lambda: {'file_path': self._extract_file_path(task_description)},
            'predict_performance_bottlenecks': lambda: {'file_path': self._extract_file_path(task_description)}
        }

        # Try CES tools first
        if tool_name in ces_tool_mapping:
            try:
                return ces_tool_mapping[tool_name]()
            except Exception as e:
                self.logger.warning(f"Failed to prepare CES tool arguments for {tool_name}: {e}")

        # Fall back to MCP tools
        if tool_name in mcp_tool_mapping:
            try:
                return mcp_tool_mapping[tool_name]()
            except Exception as e:
                self.logger.warning(f"Failed to prepare MCP tool arguments for {tool_name}: {e}")

        return None

    def _extract_file_path(self, task_description: str) -> str:
        """Extract file path from task description"""
        # Simple extraction - could be enhanced with NLP
        import re
        file_match = re.search(r'file[:\s]+([^\s]+\.[a-zA-Z]+)', task_description, re.IGNORECASE)
        if file_match:
            return file_match.group(1)
        return 'main.py'  # Default fallback

    def _extract_search_pattern(self, task_description: str) -> str:
        """Extract search pattern from task description"""
        # Look for quoted strings or specific keywords
        import re
        pattern_match = re.search(r'["\']([^"\']+)["\']', task_description)
        if pattern_match:
            return pattern_match.group(1)

        # Look for function or class names
        func_match = re.search(r'function[:\s]+(\w+)', task_description, re.IGNORECASE)
        if func_match:
            return func_match.group(1)

        return 'def '  # Default pattern for function search

    async def execute_workflow(self, task_description: str,
                             progress_callback: Optional[Callable] = None,
                             workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a complete end-to-end workflow for complex tasks

        Args:
            task_description: Natural language task description
            progress_callback: Optional callback for progress updates
            workflow_id: Optional custom workflow ID

        Returns:
            Complete workflow execution results
        """
        self.logger.info(f"Executing workflow for task: {task_description[:50]}...")

        try:
            # Execute workflow using the orchestrator
            workflow_execution = await self.workflow_orchestrator.execute_workflow(
                task_description=task_description,
                progress_callback=progress_callback,
                workflow_id=workflow_id
            )

            # Convert to dict for JSON serialization
            result = {
                "status": "completed",
                "workflow_id": workflow_execution.workflow_id,
                "task_description": workflow_execution.task_description,
                "analysis": workflow_execution.analysis.__dict__,
                "current_stage": workflow_execution.current_stage.value,
                "progress_percentage": workflow_execution.progress_percentage,
                "status": workflow_execution.status,
                "results": workflow_execution.results,
                "performance_metrics": workflow_execution.performance_metrics,
                "start_time": workflow_execution.start_time.isoformat(),
                "end_time": workflow_execution.end_time.isoformat() if workflow_execution.end_time else None,
                "total_execution_time": (workflow_execution.end_time - workflow_execution.start_time).total_seconds() if workflow_execution.end_time else 0,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Workflow {workflow_execution.workflow_id} completed with status: {workflow_execution.status}")
            return result

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""
        workflow = self.workflow_orchestrator.get_workflow_status(workflow_id)
        if workflow:
            return {
                "workflow_id": workflow.workflow_id,
                "task_description": workflow.task_description,
                "current_stage": workflow.current_stage.value,
                "progress_percentage": workflow.progress_percentage,
                "status": workflow.status,
                "start_time": workflow.start_time.isoformat(),
                "end_time": workflow.end_time.isoformat() if workflow.end_time else None
            }
        return None

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflows"""
        workflows = self.workflow_orchestrator.get_active_workflows()
        return {
            wid: {
                "workflow_id": w.workflow_id,
                "task_description": w.task_description,
                "current_stage": w.current_stage.value,
                "progress_percentage": w.progress_percentage,
                "status": w.status,
                "start_time": w.start_time.isoformat(),
                "end_time": w.end_time.isoformat() if w.end_time else None
            } for wid, w in workflows.items()
        }

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        return self.workflow_orchestrator.cancel_workflow(workflow_id)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics adapted from CodeSage"""
        return self.performance_monitor.get_current_metrics()

    # Human-AI Interaction Methods
    async def start_interaction_session(self, user_id: str, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new human-AI interaction session"""
        return await self.interaction_manager.start_session(user_id, initial_context)

    async def send_interaction_message(self, session_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an interaction session"""
        from ..core.human_ai_interaction import InteractionMessage

        message = InteractionMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            sender=message_data.get('sender', 'human'),
            message_type=message_data.get('message_type', 'query'),
            content=message_data['content'],
            timestamp=datetime.now(),
            metadata=message_data.get('metadata'),
            context=message_data.get('context'),
            requires_response=message_data.get('requires_response', True)
        )

        return await self.interaction_manager.send_message(session_id, message)

    async def join_interaction_session(self, session_id: str, user_id: str) -> bool:
        """Join an existing interaction session for concurrent collaboration"""
        return await self.interaction_manager.join_session(session_id, user_id)

    async def leave_interaction_session(self, session_id: str, user_id: str) -> bool:
        """Leave an interaction session"""
        return await self.interaction_manager.leave_session(session_id, user_id)

    def get_interaction_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an interaction session"""
        return self.interaction_manager.get_session_status(session_id)

    def get_active_interaction_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active interaction sessions"""
        return self.interaction_manager.get_active_sessions()

    def get_interaction_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for human-AI interactions"""
        return self.interaction_manager.get_performance_metrics()

    async def cleanup_interaction_sessions(self):
        """Clean up inactive interaction sessions"""
        await self.interaction_manager.cleanup_inactive_sessions()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive agent"""
        return {
            "status": "operational",
            "components": {
                "memory_manager": self.memory_manager.get_status(),
                "ai_orchestrator": self.ai_orchestrator.get_status(),
                "ethical_controller": self.ethical_controller.get_status(),
                "codesage_integration": {
                    "connected": self.codesage_integration.connected,
                    "server_url": self.codesage_integration.server_url,
                    "available_tools_count": len(self.codesage_integration.available_tools)
                },
                "human_ai_interaction": {
                    "active_sessions": len(self.interaction_manager.get_active_sessions()),
                    "performance_metrics": self.interaction_manager.get_performance_metrics()
                },
                "error_recovery": {
                    "health_status": self.error_recovery.get_health_status(),
                    "error_statistics": self.error_recovery.get_error_statistics()
                }
            },
            "performance_metrics": self.get_performance_metrics(),
            "mcp_execution_history_size": len(self.mcp_execution_history),
            "timestamp": datetime.now().isoformat()
        }