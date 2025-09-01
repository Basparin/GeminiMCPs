"""
CES AI Orchestrator

Manages AI assistant integration and task delegation:
- AI Assistant: Interface to various AI providers
- Task Delegation: Intelligent routing of tasks to appropriate assistants
"""

from .ai_assistant import AIOrchestrator
from .task_delegation import TaskDelegator

__all__ = ["AIOrchestrator", "TaskDelegator"]