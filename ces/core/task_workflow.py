"""
CES Task Workflow - End-to-End Task Execution Orchestration

Provides comprehensive workflow orchestration for CES tasks, integrating
cognitive analysis, tool execution, progress monitoring, and result synthesis.
Adapted from CodeSage MCP workflow capabilities.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time

from .cognitive_agent import CognitiveAgent, TaskAnalysis
from .performance_monitor import get_performance_monitor
from ..codesage_integration import CodeSageIntegration, CESToolExtensions


class WorkflowStage(Enum):
    """Workflow execution stages"""
    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """Individual step in the workflow"""
    name: str
    stage: WorkflowStage
    description: str
    tools_required: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    result: Optional[Any] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Complete workflow execution context"""
    workflow_id: str
    task_description: str
    analysis: TaskAnalysis
    steps: List[WorkflowStep] = field(default_factory=list)
    current_stage: WorkflowStage = WorkflowStage.INITIALIZED
    progress_percentage: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"
    results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class CESTaskWorkflow:
    """CES End-to-End Task Workflow Orchestrator"""

    def __init__(self, cognitive_agent: CognitiveAgent,
                 codesage_integration: Optional[CodeSageIntegration] = None):
        self.cognitive_agent = cognitive_agent
        self.codesage_integration = codesage_integration
        self.ces_tools_extensions = CESToolExtensions(codesage_integration) if codesage_integration else None
        self.performance_monitor = get_performance_monitor()
        self.logger = logging.getLogger(__name__)

        # Workflow management
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_callbacks: Dict[str, List[Callable]] = {}

    async def execute_workflow(self, task_description: str,
                             progress_callback: Optional[Callable] = None,
                             workflow_id: Optional[str] = None) -> WorkflowExecution:
        """
        Execute complete end-to-end workflow for a task

        Args:
            task_description: Natural language task description
            progress_callback: Optional callback for progress updates
            workflow_id: Optional custom workflow ID

        Returns:
            Complete workflow execution results
        """
        if workflow_id is None:
            workflow_id = f"workflow_{int(time.time())}_{hash(task_description) % 10000}"

        # Initialize workflow
        workflow = WorkflowExecution(
            workflow_id=workflow_id,
            task_description=task_description,
            analysis=TaskAnalysis(
                complexity_score=0,
                required_skills=[],
                estimated_duration=0,
                recommended_assistants=[],
                ethical_concerns=[],
                context_requirements=[]
            )
        )

        self.active_workflows[workflow_id] = workflow
        if progress_callback:
            self.workflow_callbacks[workflow_id] = [progress_callback]

        try:
            # Stage 1: Task Analysis
            await self._update_workflow_progress(workflow, WorkflowStage.ANALYZING, 10)
            workflow.analysis = self.cognitive_agent.analyze_task(task_description)
            self.logger.info(f"Task analysis completed for workflow {workflow_id}")

            # Stage 2: Workflow Planning
            await self._update_workflow_progress(workflow, WorkflowStage.PLANNING, 20)
            workflow.steps = self._plan_workflow_steps(workflow.analysis)
            self.logger.info(f"Workflow planning completed with {len(workflow.steps)} steps")

            # Stage 3: Tool Execution
            await self._update_workflow_progress(workflow, WorkflowStage.EXECUTING, 30)
            await self._execute_workflow_steps(workflow)

            # Stage 4: Result Synthesis
            await self._update_workflow_progress(workflow, WorkflowStage.SYNTHESIZING, 80)
            workflow.results = await self._synthesize_results(workflow)

            # Stage 5: Completion
            await self._update_workflow_progress(workflow, WorkflowStage.COMPLETED, 100)
            workflow.status = "completed"
            workflow.end_time = datetime.now()

            # Record final performance metrics
            workflow.performance_metrics = self.performance_monitor.get_performance_report()

            self.logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            workflow.status = "failed"
            workflow.end_time = datetime.now()
            await self._update_workflow_progress(workflow, WorkflowStage.FAILED, 0)

        finally:
            # Cleanup
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            if workflow_id in self.workflow_callbacks:
                del self.workflow_callbacks[workflow_id]

        return workflow

    def _plan_workflow_steps(self, analysis: TaskAnalysis) -> List[WorkflowStep]:
        """Plan workflow steps based on task analysis"""
        steps = []

        # Analysis step
        steps.append(WorkflowStep(
            name="task_analysis",
            stage=WorkflowStage.ANALYZING,
            description="Analyze task requirements and complexity",
            tools_required=[],
            timeout_seconds=60
        ))

        # Tool execution steps based on required tools
        if analysis.mcp_tools_required:
            for i, tool_name in enumerate(analysis.mcp_tools_required):
                steps.append(WorkflowStep(
                    name=f"tool_execution_{i}",
                    stage=WorkflowStage.EXECUTING,
                    description=f"Execute {tool_name} tool",
                    tools_required=[tool_name],
                    timeout_seconds=120,
                    dependencies=["task_analysis"]
                ))

        # AI orchestration step
        steps.append(WorkflowStep(
            name="ai_orchestration",
            stage=WorkflowStage.EXECUTING,
            description="Execute task with AI assistants",
            tools_required=[],
            timeout_seconds=300,
            dependencies=["task_analysis"] + [f"tool_execution_{i}" for i in range(len(analysis.mcp_tools_required or []))]
        ))

        # Result synthesis step
        steps.append(WorkflowStep(
            name="result_synthesis",
            stage=WorkflowStage.SYNTHESIZING,
            description="Synthesize and format final results",
            tools_required=[],
            timeout_seconds=60,
            dependencies=["ai_orchestration"]
        ))

        return steps

    async def _execute_workflow_steps(self, workflow: WorkflowExecution) -> None:
        """Execute all workflow steps in dependency order"""
        executed_steps = set()
        step_results = {}

        while len(executed_steps) < len(workflow.steps):
            # Find steps ready for execution
            ready_steps = []
            for step in workflow.steps:
                if step.name not in executed_steps:
                    # Check if all dependencies are satisfied
                    deps_satisfied = all(dep in executed_steps for dep in step.dependencies)
                    if deps_satisfied:
                        ready_steps.append(step)

            if not ready_steps:
                # No steps ready - possible circular dependency or error
                self.logger.error(f"No steps ready for execution in workflow {workflow.workflow_id}")
                break

            # Execute ready steps concurrently
            tasks = []
            for step in ready_steps:
                task = self._execute_workflow_step(workflow, step, step_results)
                tasks.append(task)

            # Wait for all steps in this batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Step {step.name} failed: {result}")
                    step.status = "failed"
                    step.error_message = str(result)
                else:
                    step.status = "completed"
                    step.result = result
                    step.execution_time = time.time() - time.time()  # Would need to track actual time

                executed_steps.add(step.name)
                step_results[step.name] = step.result

            # Update progress
            progress = (len(executed_steps) / len(workflow.steps)) * 70 + 30  # 30-100%
            await self._update_workflow_progress(workflow, WorkflowStage.EXECUTING, progress)

    async def _execute_workflow_step(self, workflow: WorkflowExecution,
                                   step: WorkflowStep, step_results: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""
        try:
            if step.name == "task_analysis":
                # Already done during analysis phase
                return workflow.analysis

            elif step.name == "ai_orchestration":
                # Execute main task with cognitive agent
                result = await self.cognitive_agent.execute_task(workflow.task_description)
                return result

            elif step.name.startswith("tool_execution_"):
                # Execute specific tools
                if not self.codesage_integration or not self.codesage_integration.connected:
                    raise Exception("CodeSage integration not available for tool execution")

                tool_name = step.tools_required[0]
                args = self.cognitive_agent._prepare_tool_arguments(tool_name, workflow.task_description, {})

                if tool_name in ['read_code_file', 'search_codebase', 'get_file_structure',
                               'count_lines_of_code', 'get_dependencies_overview', 'analyze_codebase_improvements']:
                    # Use CES tools
                    result = await self.cognitive_agent.execute_ces_tool(tool_name, args or {})
                else:
                    # Use MCP tools
                    execution_result = await self.cognitive_agent.execute_mcp_tool(tool_name, args or {})
                    result = execution_result.result

                return result

            elif step.name == "result_synthesis":
                # Synthesize final results
                return await self._synthesize_step_results(workflow, step_results)

            else:
                raise Exception(f"Unknown step type: {step.name}")

        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            raise

    async def _synthesize_results(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """Synthesize final workflow results"""
        synthesis = {
            "workflow_id": workflow.workflow_id,
            "task_description": workflow.task_description,
            "analysis": workflow.analysis.__dict__,
            "step_results": {step.name: {
                "status": step.status,
                "result": step.result,
                "execution_time": step.execution_time,
                "error": step.error_message
            } for step in workflow.steps},
            "performance_metrics": workflow.performance_metrics,
            "total_execution_time": (workflow.end_time - workflow.start_time).total_seconds() if workflow.end_time else 0,
            "timestamp": datetime.now().isoformat()
        }

        # Add intelligent insights
        synthesis["insights"] = self._generate_workflow_insights(workflow)

        return synthesis

    async def _synthesize_step_results(self, workflow: WorkflowExecution, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from individual steps"""
        synthesis = {
            "step_count": len(step_results),
            "successful_steps": sum(1 for step in workflow.steps if step.status == "completed"),
            "failed_steps": sum(1 for step in workflow.steps if step.status == "failed"),
            "tool_executions": [],
            "ai_interactions": []
        }

        # Extract tool execution results
        for step_name, result in step_results.items():
            if step_name.startswith("tool_execution_"):
                synthesis["tool_executions"].append({
                    "step": step_name,
                    "result": result
                })
            elif step_name == "ai_orchestration":
                synthesis["ai_interactions"].append(result)

        return synthesis

    def _generate_workflow_insights(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """Generate intelligent insights about the workflow execution"""
        insights = {
            "efficiency_score": 0.0,
            "bottlenecks": [],
            "recommendations": [],
            "success_patterns": []
        }

        # Calculate efficiency score
        if workflow.end_time and workflow.start_time:
            planned_time = workflow.analysis.estimated_duration * 60  # Convert to seconds
            actual_time = (workflow.end_time - workflow.start_time).total_seconds()
            if planned_time > 0:
                insights["efficiency_score"] = min(1.0, planned_time / actual_time)

        # Identify bottlenecks
        for step in workflow.steps:
            if step.execution_time > step.timeout_seconds * 0.8:  # Over 80% of timeout
                insights["bottlenecks"].append({
                    "step": step.name,
                    "execution_time": step.execution_time,
                    "timeout": step.timeout_seconds
                })

        # Generate recommendations
        if insights["efficiency_score"] < 0.7:
            insights["recommendations"].append("Consider optimizing tool execution order")
        if len(insights["bottlenecks"]) > 0:
            insights["recommendations"].append("Review timeout settings for bottleneck steps")

        return insights

    async def _update_workflow_progress(self, workflow: WorkflowExecution,
                                      stage: WorkflowStage, percentage: float) -> None:
        """Update workflow progress and notify callbacks"""
        workflow.current_stage = stage
        workflow.progress_percentage = percentage

        # Notify callbacks
        if workflow.workflow_id in self.workflow_callbacks:
            for callback in self.workflow_callbacks[workflow.workflow_id]:
                try:
                    await callback(workflow.workflow_id, stage.value, percentage, workflow)
                except Exception as e:
                    self.logger.error(f"Progress callback failed: {e}")

    def get_active_workflows(self) -> Dict[str, WorkflowExecution]:
        """Get all active workflows"""
        return self.active_workflows.copy()

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get status of a specific workflow"""
        return self.active_workflows.get(workflow_id)

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "cancelled"
            workflow.end_time = datetime.now()
            del self.active_workflows[workflow_id]
            return True
        return False


# Global workflow orchestrator instance
_workflow_orchestrator = None


def get_workflow_orchestrator(cognitive_agent: CognitiveAgent,
                            codesage_integration: Optional[CodeSageIntegration] = None) -> CESTaskWorkflow:
    """Get the global CES workflow orchestrator instance"""
    global _workflow_orchestrator
    if _workflow_orchestrator is None:
        _workflow_orchestrator = CESTaskWorkflow(cognitive_agent, codesage_integration)
    return _workflow_orchestrator