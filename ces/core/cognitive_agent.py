"""
Cognitive Agent - Main CES Orchestrator

The Cognitive Agent is the central intelligence of the CES system. It analyzes tasks,
manages context, coordinates AI assistants, and ensures ethical operation.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..ai_orchestrator.ai_assistant import AIOrchestrator
from ..core.memory_manager import MemoryManager
from ..core.ethical_controller import EthicalController
from ..config.ces_config import CESConfig


@dataclass
class TaskAnalysis:
    """Analysis result for a given task"""
    complexity_score: float
    required_skills: List[str]
    estimated_duration: int  # minutes
    recommended_assistants: List[str]
    ethical_concerns: List[str]
    context_requirements: List[str]


class CognitiveAgent:
    """
    Main cognitive agent that orchestrates CES operations.

    This agent analyzes tasks, manages context, coordinates AI assistants,
    and ensures all operations align with ethical guidelines.
    """

    def __init__(self, config: Optional[CESConfig] = None):
        self.config = config or CESConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.memory_manager = MemoryManager()
        self.ai_orchestrator = AIOrchestrator()
        self.ethical_controller = EthicalController()

        self.logger.info("Cognitive Agent initialized")

    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """
        Analyze a task to determine complexity, requirements, and optimal approach.

        Args:
            task_description: Natural language description of the task

        Returns:
            TaskAnalysis: Detailed analysis of the task
        """
        self.logger.info(f"Analyzing task: {task_description[:50]}...")

        # Basic task analysis (placeholder for more sophisticated analysis)
        complexity_score = self._calculate_complexity(task_description)
        required_skills = self._identify_required_skills(task_description)
        estimated_duration = self._estimate_duration(complexity_score, required_skills)

        # Get recommendations from AI orchestrator
        recommended_assistants = self.ai_orchestrator.recommend_assistants(
            task_description, required_skills
        )

        # Check ethical concerns
        ethical_concerns = self.ethical_controller.check_task_ethics(task_description)

        # Determine context requirements
        context_requirements = self.memory_manager.analyze_context_needs(task_description)

        return TaskAnalysis(
            complexity_score=complexity_score,
            required_skills=required_skills,
            estimated_duration=estimated_duration,
            recommended_assistants=recommended_assistants,
            ethical_concerns=ethical_concerns,
            context_requirements=context_requirements
        )

    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a task using the optimal AI assistant configuration.

        Args:
            task_description: Task to execute

        Returns:
            Dict containing execution results
        """
        self.logger.info(f"Executing task: {task_description[:50]}...")

        # Analyze the task first
        analysis = self.analyze_task(task_description)

        # Check if task is ethically acceptable
        if analysis.ethical_concerns:
            self.logger.warning(f"Ethical concerns identified: {analysis.ethical_concerns}")
            if not self.ethical_controller.approve_task(analysis.ethical_concerns):
                return {
                    "status": "rejected",
                    "reason": "Ethical concerns",
                    "details": analysis.ethical_concerns
                }

        # Retrieve relevant context
        context = self.memory_manager.retrieve_context(
            task_description, analysis.context_requirements
        )

        # Execute task with selected assistant
        result = self.ai_orchestrator.execute_task(
            task_description=task_description,
            context=context,
            assistant_preferences=analysis.recommended_assistants
        )

        # Store execution result in memory
        self.memory_manager.store_task_result(task_description, result)

        return {
            "status": "completed",
            "analysis": analysis,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_complexity(self, task_description: str) -> float:
        """Calculate task complexity score (0-10)"""
        # Placeholder implementation - would use NLP analysis
        length_score = min(len(task_description) / 500, 1.0) * 3
        keyword_score = sum(1 for keyword in ['complex', 'advanced', 'optimize', 'architecture']
                          if keyword in task_description.lower()) * 2
        return min(length_score + keyword_score, 10.0)

    def _identify_required_skills(self, task_description: str) -> List[str]:
        """Identify skills required for the task"""
        skills = []
        task_lower = task_description.lower()

        if 'python' in task_lower or 'code' in task_lower:
            skills.append('programming')
        if 'design' in task_lower or 'architecture' in task_lower:
            skills.append('system_design')
        if 'test' in task_lower or 'debug' in task_lower:
            skills.append('testing')
        if 'document' in task_lower or 'readme' in task_lower:
            skills.append('documentation')

        return skills or ['general']

    def _estimate_duration(self, complexity: float, skills: List[str]) -> int:
        """Estimate task duration in minutes"""
        base_duration = 30  # 30 minutes base
        complexity_multiplier = 1 + (complexity / 10)  # 1-2x multiplier
        skill_multiplier = 1 + (len(skills) * 0.2)  # Additional time per skill

        return int(base_duration * complexity_multiplier * skill_multiplier)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive agent"""
        return {
            "status": "operational",
            "components": {
                "memory_manager": self.memory_manager.get_status(),
                "ai_orchestrator": self.ai_orchestrator.get_status(),
                "ethical_controller": self.ethical_controller.get_status()
            },
            "timestamp": datetime.now().isoformat()
        }