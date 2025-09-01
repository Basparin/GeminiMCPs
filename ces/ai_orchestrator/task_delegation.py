"""
Task Delegation - CES Intelligent Task Routing

Handles intelligent routing of tasks to appropriate AI assistants based on
task complexity, required skills, and assistant performance history.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .ai_assistant import AIOrchestrator


class TaskDelegator:
    """
    Intelligently delegates tasks to AI assistants based on various factors.

    Considers:
    - Task complexity and requirements
    - Assistant capabilities and performance
    - Current workload and availability
    - User preferences and history
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize AI orchestrator for real task execution
        self.ai_orchestrator = AIOrchestrator()

        # Task delegation rules and patterns
        self.delegation_rules = {
            'simple': ['grok', 'qwen'],
            'complex': ['qwen', 'gemini', 'grok'],
            'analysis': ['gemini', 'grok'],
            'coding': ['qwen', 'grok'],
            'documentation': ['gemini', 'grok']
        }

        # Performance tracking
        self.delegation_history = []

        self.logger.info("Task Delegator initialized with real AI integration")

    async def delegate_task(self, task_analysis: Dict[str, Any],
                           available_assistants: Optional[List[str]] = None,
                           user_preferences: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Delegate a task to the most appropriate assistant and execute it

        Args:
            task_analysis: Analysis of the task requirements
            available_assistants: List of currently available assistants (optional)
            user_preferences: User preferences for assistant selection
            context: Additional context for task execution

        Returns:
            Delegation and execution result
        """
        start_time = datetime.now()

        # Get available assistants from orchestrator if not provided
        if available_assistants is None:
            available_assistants = self.ai_orchestrator.get_available_assistants()
            available_assistants = [a["name"] for a in available_assistants]

        complexity = task_analysis.get('complexity_score', 5)
        required_skills = task_analysis.get('required_skills', [])
        task_description = task_analysis.get('description', '')
        task_type = self._determine_task_type(required_skills)

        # Get candidate assistants based on task type
        candidates = self.delegation_rules.get(task_type, ['grok'])

        # Filter by availability
        available_candidates = [c for c in candidates if c in available_assistants]

        if not available_candidates:
            return {
                "decision": "fallback",
                "assistant": available_assistants[0] if available_assistants else None,
                "reasoning": "No preferred assistants available",
                "confidence": 0.3,
                "execution_result": None,
                "timestamp": start_time.isoformat()
            }

        # Select best assistant
        selected_assistant = self._select_best_assistant(
            available_candidates, complexity, user_preferences
        )

        # Execute the task
        try:
            execution_result = await self.ai_orchestrator.execute_task(
                task_description, context, [selected_assistant]
            )

            delegation_result = {
                "decision": "delegated",
                "assistant": selected_assistant,
                "reasoning": f"Selected {selected_assistant} for {task_type} task",
                "confidence": 0.8,
                "alternatives": available_candidates[1:] if len(available_candidates) > 1 else [],
                "execution_result": execution_result,
                "task_type": task_type,
                "complexity": complexity,
                "timestamp": start_time.isoformat()
            }

            # Record delegation history
            self.delegation_history.append(delegation_result)

            return delegation_result

        except Exception as e:
            self.logger.error(f"Task delegation execution failed: {e}")
            return {
                "decision": "failed",
                "assistant": selected_assistant,
                "reasoning": f"Selected {selected_assistant} but execution failed",
                "confidence": 0.8,
                "error": str(e),
                "execution_result": None,
                "timestamp": start_time.isoformat()
            }

    def _determine_task_type(self, skills: List[str]) -> str:
        """Determine task type based on required skills"""
        if 'programming' in skills or 'coding' in skills:
            return 'coding'
        elif 'analysis' in skills or 'testing' in skills:
            return 'analysis'
        elif 'documentation' in skills:
            return 'documentation'
        elif len(skills) > 2:
            return 'complex'
        else:
            return 'simple'

    def _select_best_assistant(self, candidates: List[str],
                              complexity: float,
                              user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Select the best assistant from candidates using real availability checks"""
        # Check user preferences first
        if user_preferences and 'preferred_assistant' in user_preferences:
            preferred = user_preferences['preferred_assistant']
            if preferred in candidates:
                # Verify the preferred assistant is actually available
                if self._is_assistant_available(preferred):
                    return preferred

        # Default selection logic based on complexity and real availability
        if complexity > 7:
            # For complex tasks, prefer specialized assistants
            for candidate in ['qwen', 'gemini']:
                if candidate in candidates and self._is_assistant_available(candidate):
                    return candidate

        # For simpler tasks, prefer general-purpose assistants
        for candidate in ['grok', 'qwen']:
            if candidate in candidates and self._is_assistant_available(candidate):
                return candidate

        # Fallback to first available candidate
        for candidate in candidates:
            if self._is_assistant_available(candidate):
                return candidate

        # Ultimate fallback
        return candidates[0] if candidates else 'grok'

    def _is_assistant_available(self, assistant_name: str) -> bool:
        """Check if an assistant is available using the orchestrator"""
        available_assistants = self.ai_orchestrator.get_available_assistants()
        available_names = [a["name"] for a in available_assistants]
        return assistant_name in available_names

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get statistics about task delegation patterns"""
        total_delegations = len(self.delegation_history)
        successful_delegations = len([d for d in self.delegation_history
                                    if d.get("execution_result", {}).get("status") == "completed"])

        # Calculate assistant usage statistics
        assistant_usage = {}
        for delegation in self.delegation_history:
            assistant = delegation.get("assistant")
            if assistant:
                assistant_usage[assistant] = assistant_usage.get(assistant, 0) + 1

        # Calculate task type statistics
        task_type_usage = {}
        for delegation in self.delegation_history:
            task_type = delegation.get("task_type")
            if task_type:
                task_type_usage[task_type] = task_type_usage.get(task_type, 0) + 1

        return {
            "total_delegations": total_delegations,
            "successful_delegations": successful_delegations,
            "success_rate": successful_delegations / total_delegations if total_delegations > 0 else 0,
            "rules_defined": len(self.delegation_rules),
            "task_types_supported": list(self.delegation_rules.keys()),
            "assistant_usage": assistant_usage,
            "task_type_usage": task_type_usage,
            "last_update": datetime.now().isoformat()
        }

    def update_delegation_rules(self, task_type: str, assistants: List[str]):
        """Update delegation rules for a task type"""
        self.delegation_rules[task_type] = assistants
        self.logger.info(f"Updated delegation rules for {task_type}: {assistants}")

    def get_status(self) -> Dict[str, Any]:
        """Get task delegator status"""
        stats = self.get_delegation_stats()

        return {
            "status": "operational",
            "delegation_rules": len(self.delegation_rules),
            "supported_task_types": list(self.delegation_rules.keys()),
            "total_delegations": stats["total_delegations"],
            "success_rate": stats["success_rate"],
            "ai_orchestrator_status": self.ai_orchestrator.get_status(),
            "last_update": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on task delegation system"""
        health_status = {
            "component": "Task Delegator",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check AI orchestrator health
        orchestrator_health = self.ai_orchestrator.health_check()
        health_status["checks"]["ai_orchestrator"] = orchestrator_health

        # Check delegation rules
        rules_count = len(self.delegation_rules)
        health_status["checks"]["delegation_rules"] = {
            "status": "healthy" if rules_count > 0 else "warning",
            "details": f"{rules_count} delegation rules configured"
        }

        # Check delegation history
        history_size = len(self.delegation_history)
        health_status["checks"]["delegation_history"] = {
            "status": "healthy",
            "details": f"{history_size} delegations recorded"
        }

        # Overall health
        all_healthy = all(
            check.get("status") in ["healthy", "skipped"]
            for check in health_status["checks"].values()
            if isinstance(check, dict)
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"

        return health_status