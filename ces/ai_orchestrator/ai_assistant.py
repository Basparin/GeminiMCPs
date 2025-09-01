"""
AI Orchestrator - CES AI Assistant Management

Manages integration with multiple AI assistants (Grok, qwen-cli-coder, gemini-cli)
and coordinates task delegation based on capabilities and performance.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import subprocess
import json

from .cli_integration import AIAssistantManager, APIResult, CLIResult
from .specialization import specialization_manager


class AIOrchestrator:
    """
    Orchestrates AI assistant interactions for CES.

    Manages multiple AI providers, handles task delegation,
    and optimizes assistant selection based on task requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize real CLI integration manager
        self.cli_manager = AIAssistantManager()

        # Available AI assistants and their capabilities (for backward compatibility)
        self.assistants = {
            'grok': {
                'name': 'Grok CLI',
                'capabilities': ['general_reasoning', 'coding', 'analysis'],
                'command': 'grok',
                'strengths': ['reasoning', 'general_knowledge']
            },
            'qwen': {
                'name': 'qwen-cli-coder',
                'capabilities': ['coding', 'code_generation', 'debugging'],
                'command': 'qwen-cli-coder',
                'strengths': ['code_generation', 'technical_tasks']
            },
            'gemini': {
                'name': 'Gemini CLI',
                'capabilities': ['analysis', 'documentation', 'review'],
                'command': 'gemini-cli',
                'strengths': ['code_analysis', 'documentation']
            }
        }

        self.logger.info("AI Orchestrator initialized with real CLI integration")

    def recommend_assistants(self, task_description: str, required_skills: List[str]) -> List[str]:
        """
        Recommend AI assistants for a task based on description and skills

        Args:
            task_description: Description of the task
            required_skills: List of required skills

        Returns:
            List of recommended assistant names
        """
        recommendations = []
        task_lower = task_description.lower()

        # Match skills to assistant capabilities
        for assistant_name, assistant_info in self.assistants.items():
            capability_match = any(skill in assistant_info['capabilities']
                                 for skill in required_skills)

            # Additional keyword matching
            keyword_match = False
            for keyword in assistant_info.get('keywords', []):
                if keyword in task_lower:
                    keyword_match = True
                    break

            if capability_match or keyword_match:
                recommendations.append(assistant_name)

        # If no specific matches, return general-purpose assistants
        if not recommendations:
            recommendations = ['grok', 'qwen']

        return recommendations

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None,
                           assistant_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a task using the most appropriate AI assistant with specialization

        Args:
            task_description: Task to execute
            context: Additional context for the task
            assistant_preferences: Preferred assistants to use

        Returns:
            Task execution result
        """
        start_time = datetime.now()

        # Analyze task with specialization manager
        task_analysis = specialization_manager.analyze_task(task_description)

        # Get available assistants
        available_assistants = [name for name, info in self.assistants.items()
                               if self._is_assistant_available(name)]

        # Select optimal assistant using specialization
        if assistant_preferences:
            # Filter by preferences but use specialization for ranking
            preferred_available = [pref for pref in assistant_preferences if pref in available_assistants]
            if preferred_available:
                assistant = specialization_manager.select_optimal_assistant(task_analysis, preferred_available)
            else:
                assistant = specialization_manager.select_optimal_assistant(task_analysis, available_assistants)
        else:
            assistant = specialization_manager.select_optimal_assistant(task_analysis, available_assistants)

        if not assistant:
            return {
                "status": "failed",
                "error": "No suitable AI assistant available",
                "task_analysis": task_analysis,
                "timestamp": start_time.isoformat()
            }

        # Execute with selected assistant
        try:
            result = await self._execute_with_assistant(assistant, task_description, context)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Record result for learning
            success = "error" not in str(result).lower() and len(str(result)) > 10
            await specialization_manager.record_task_result(
                task_description=task_description,
                assistant_name=assistant,
                success=success,
                response_time=execution_time
            )

            return {
                "status": "completed",
                "assistant_used": assistant,
                "result": result,
                "execution_time": execution_time,
                "task_analysis": task_analysis,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Task execution failed with {assistant}: {e}")

            # Record failed result for learning
            await specialization_manager.record_task_result(
                task_description=task_description,
                assistant_name=assistant,
                success=False,
                response_time=execution_time
            )

            return {
                "status": "failed",
                "assistant_used": assistant,
                "error": str(e),
                "execution_time": execution_time,
                "task_analysis": task_analysis,
                "timestamp": start_time.isoformat()
            }

    def _select_assistant(self, task_description: str,
                         preferences: Optional[List[str]] = None) -> Optional[str]:
        """Select the most appropriate assistant for a task"""
        if preferences:
            # Try preferred assistants first
            for pref in preferences:
                if pref in self.assistants and self._is_assistant_available(pref):
                    return pref

        # Fallback to recommendation logic
        recommendations = self.recommend_assistants(task_description, [])
        for rec in recommendations:
            if self._is_assistant_available(rec):
                return rec

        return None

    def _is_assistant_available(self, assistant_name: str) -> bool:
        """Check if an AI assistant is available"""
        if assistant_name not in self.assistants:
            return False

        # Check availability using real CLI manager
        assistant = self.cli_manager.get_assistant(assistant_name)
        if assistant:
            return assistant.is_available()

        return False

    def _prepare_prompt(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare a comprehensive prompt with context"""
        prompt_parts = [f"Task: {task_description}"]

        if context:
            if 'task_history' in context and context['task_history']:
                prompt_parts.append("\nRecent similar tasks:")
                for task in context['task_history'][:3]:
                    prompt_parts.append(f"- {task['description'][:100]}...")

            if 'user_preferences' in context and context['user_preferences']:
                prompt_parts.append(f"\nUser preferences: {context['user_preferences']}")

        return "\n".join(prompt_parts)

    async def _execute_with_assistant(self, assistant_name: str, task_description: str, context: Optional[Dict[str, Any]] = None) -> Union[APIResult, CLIResult]:
        """
        Execute a task with a specific AI assistant using real CLI integration

        Args:
            assistant_name: Name of the assistant to use
            task_description: Task description
            context: Additional context

        Returns:
            Result from assistant execution
        """
        self.logger.info(f"Executing task with {assistant_name}")

        try:
            # Use the real CLI manager to execute the task
            result = await self.cli_manager.execute_with_assistant(assistant_name, task_description, context)

            if isinstance(result, APIResult):
                if result.success:
                    return result.response
                else:
                    self.logger.error(f"API call failed for {assistant_name}: {result.error}")
                    return f"Error: {result.error}"
            elif isinstance(result, CLIResult):
                if result.success:
                    return result.output
                else:
                    self.logger.error(f"CLI execution failed for {assistant_name}: {result.error}")
                    return f"Error: {result.error}"
            else:
                return f"Unknown result type from {assistant_name}"

        except Exception as e:
            self.logger.error(f"Failed to execute with {assistant_name}: {e}")
            return f"Execution failed: {str(e)}"

    def get_available_assistants(self) -> List[Dict[str, Any]]:
        """Get list of available AI assistants with specialization info"""
        available = []
        for name, info in self.assistants.items():
            if self._is_assistant_available(name):
                # Get specialization data
                profile = specialization_manager.assistant_profiles.get(name)
                assistant_info = {
                    "name": name,
                    "display_name": info["name"],
                    "capabilities": info["capabilities"],
                    "strengths": info["strengths"]
                }

                if profile:
                    assistant_info.update({
                        "performance_score": profile.performance_score,
                        "task_count": profile.task_count,
                        "success_rate": profile.success_rate,
                        "average_response_time": profile.average_response_time,
                        "specializations": profile.specializations
                    })

                available.append(assistant_info)
        return available

    def get_task_recommendations(self, task_description: str) -> Dict[str, Any]:
        """
        Get detailed recommendations for a task including specialization analysis

        Args:
            task_description: Description of the task

        Returns:
            Detailed task analysis and recommendations
        """
        analysis = specialization_manager.analyze_task(task_description)
        recommendations = specialization_manager.get_assistant_recommendations(task_description)

        return {
            "task_analysis": analysis,
            "assistant_recommendations": recommendations,
            "specialization_status": specialization_manager.get_specialization_status()
        }

    async def test_assistant_connection(self, assistant_name: str) -> Dict[str, Any]:
        """
        Test connection to an AI assistant

        Args:
            assistant_name: Name of the assistant to test

        Returns:
            Test result
        """
        if assistant_name not in self.assistants:
            return {"status": "error", "message": "Assistant not found"}

        try:
            # Test execution with real CLI manager
            test_result = await self._execute_with_assistant(assistant_name, "Test connection - please respond with 'Connection successful'")
            return {
                "status": "success",
                "assistant": assistant_name,
                "response": test_result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "assistant": assistant_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_status(self) -> Dict[str, Any]:
        """Get AI orchestrator status with specialization info"""
        available_count = len(self.get_available_assistants())
        cli_status = self.cli_manager.get_all_status()
        specialization_status = specialization_manager.get_specialization_status()

        return {
            "status": "operational",
            "total_assistants": len(self.assistants),
            "available_assistants": available_count,
            "assistants": list(self.assistants.keys()),
            "cli_integration_status": cli_status,
            "specialization_status": specialization_status,
            "last_check": datetime.now().isoformat()
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report including specialization metrics

        Returns:
            Performance report with specialization data
        """
        specialization_report = specialization_manager.get_performance_report()

        return {
            "orchestrator_status": self.get_status(),
            "specialization_metrics": specialization_report,
            "generated_at": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "component": "AI Orchestrator",
            "timestamp": datetime.now().isoformat(),
            "orchestrator_status": "healthy",
            "checks": {}
        }

        # Check CLI manager health
        cli_health = self.cli_manager.health_check()
        health_status["checks"]["cli_manager"] = cli_health

        # Check assistant availability
        available_assistants = self.get_available_assistants()
        health_status["checks"]["assistant_availability"] = {
            "status": "healthy" if len(available_assistants) > 0 else "warning",
            "details": f"{len(available_assistants)} assistants available"
        }

        # Overall health
        all_healthy = all(
            check.get("status") in ["healthy", "skipped"]
            for check in health_status["checks"].values()
            if isinstance(check, dict)
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"

        return health_status