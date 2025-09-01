"""
Task Delegation - CES Intelligent Task Routing

Handles intelligent routing of tasks to appropriate AI assistants based on
task complexity, required skills, and assistant performance history.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


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

        # Task delegation rules and patterns
        self.delegation_rules = {
            'simple': ['grok', 'qwen'],
            'complex': ['qwen', 'gemini', 'grok'],
            'analysis': ['gemini', 'grok'],
            'coding': ['qwen', 'grok'],
            'documentation': ['gemini', 'grok']
        }

        self.logger.info("Task Delegator initialized")

    def delegate_task(self, task_analysis: Dict[str, Any],
                     available_assistants: List[str],
                     user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Delegate a task to the most appropriate assistant

        Args:
            task_analysis: Analysis of the task requirements
            available_assistants: List of currently available assistants
            user_preferences: User preferences for assistant selection

        Returns:
            Delegation decision with reasoning
        """
        complexity = task_analysis.get('complexity_score', 5)
        required_skills = task_analysis.get('required_skills', [])
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
                "confidence": 0.3
            }

        # Select best assistant
        selected_assistant = self._select_best_assistant(
            available_candidates, complexity, user_preferences
        )

        return {
            "decision": "delegated",
            "assistant": selected_assistant,
            "reasoning": f"Selected {selected_assistant} for {task_type} task",
            "confidence": 0.8,
            "alternatives": available_candidates[1:] if len(available_candidates) > 1 else []
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
        """Select the best assistant from candidates"""
        if user_preferences and 'preferred_assistant' in user_preferences:
            preferred = user_preferences['preferred_assistant']
            if preferred in candidates:
                return preferred

        # Default selection logic
        if complexity > 7:
            # For complex tasks, prefer specialized assistants
            for candidate in ['qwen', 'gemini']:
                if candidate in candidates:
                    return candidate

        # For simpler tasks, use general-purpose
        if 'grok' in candidates:
            return 'grok'

        # Fallback to first available
        return candidates[0]

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get statistics about task delegation patterns"""
        return {
            "rules_defined": len(self.delegation_rules),
            "task_types_supported": list(self.delegation_rules.keys()),
            "last_update": datetime.now().isoformat()
        }

    def update_delegation_rules(self, task_type: str, assistants: List[str]):
        """Update delegation rules for a task type"""
        self.delegation_rules[task_type] = assistants
        self.logger.info(f"Updated delegation rules for {task_type}: {assistants}")

    def get_status(self) -> Dict[str, Any]:
        """Get task delegator status"""
        return {
            "status": "operational",
            "delegation_rules": len(self.delegation_rules),
            "supported_task_types": list(self.delegation_rules.keys()),
            "last_update": datetime.now().isoformat()
        }