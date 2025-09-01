"""
AI Orchestrator - CES AI Assistant Management

Manages integration with multiple AI assistants (Grok, qwen-cli-coder, gemini-cli)
and coordinates task delegation based on capabilities and performance.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import json


class AIOrchestrator:
    """
    Orchestrates AI assistant interactions for CES.

    Manages multiple AI providers, handles task delegation,
    and optimizes assistant selection based on task requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Available AI assistants and their capabilities
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
                'name': 'gemini-cli',
                'capabilities': ['analysis', 'documentation', 'review'],
                'command': 'gemini-cli',
                'strengths': ['code_analysis', 'documentation']
            }
        }

        self.logger.info("AI Orchestrator initialized")

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

    def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None,
                    assistant_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a task using the most appropriate AI assistant

        Args:
            task_description: Task to execute
            context: Additional context for the task
            assistant_preferences: Preferred assistants to use

        Returns:
            Task execution result
        """
        # Select assistant
        assistant = self._select_assistant(task_description, assistant_preferences)

        if not assistant:
            return {
                "status": "failed",
                "error": "No suitable AI assistant available",
                "timestamp": datetime.now().isoformat()
            }

        # Prepare task prompt with context
        prompt = self._prepare_prompt(task_description, context)

        # Execute with selected assistant
        try:
            result = self._execute_with_assistant(assistant, prompt)

            return {
                "status": "completed",
                "assistant_used": assistant,
                "result": result,
                "execution_time": 0,  # Would be calculated in real implementation
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Task execution failed with {assistant}: {e}")
            return {
                "status": "failed",
                "assistant_used": assistant,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
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

        # In a real implementation, this would check if the CLI tool is installed
        # For now, assume all configured assistants are available
        return True

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

    def _execute_with_assistant(self, assistant_name: str, prompt: str) -> str:
        """
        Execute a task with a specific AI assistant

        Note: This is a placeholder implementation. In production, this would
        interface with actual AI CLI tools or APIs.
        """
        assistant_info = self.assistants[assistant_name]

        # Placeholder for actual AI execution
        # In real implementation, this would:
        # 1. Format the prompt for the specific assistant
        # 2. Execute the CLI command or API call
        # 3. Parse and return the response

        self.logger.info(f"Executing task with {assistant_info['name']}")

        # Mock response based on assistant type
        if assistant_name == 'grok':
            return f"Grok analysis: {prompt[:100]}... Task appears to be well-structured."
        elif assistant_name == 'qwen':
            return f"qwen-cli-coder: Generated code solution for: {prompt[:100]}..."
        elif assistant_name == 'gemini':
            return f"gemini-cli analysis: Code review completed for: {prompt[:100]}..."
        else:
            return f"Task executed: {prompt[:100]}..."

    def get_available_assistants(self) -> List[Dict[str, Any]]:
        """Get list of available AI assistants"""
        available = []
        for name, info in self.assistants.items():
            if self._is_assistant_available(name):
                available.append({
                    "name": name,
                    "display_name": info["name"],
                    "capabilities": info["capabilities"],
                    "strengths": info["strengths"]
                })
        return available

    def test_assistant_connection(self, assistant_name: str) -> Dict[str, Any]:
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
            # Simple test execution
            test_result = self._execute_with_assistant(assistant_name, "Test connection")
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
        """Get AI orchestrator status"""
        available_count = len(self.get_available_assistants())

        return {
            "status": "operational",
            "total_assistants": len(self.assistants),
            "available_assistants": available_count,
            "assistants": list(self.assistants.keys()),
            "last_check": datetime.now().isoformat()
        }