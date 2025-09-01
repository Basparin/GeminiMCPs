"""
CES AI Assistant Specialization Module

Provides advanced AI assistant specialization, dynamic task routing,
and performance-based optimization for enhanced tool capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


@dataclass
class AssistantProfile:
    """Profile for an AI assistant with capabilities and performance metrics"""
    name: str
    display_name: str
    capabilities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    task_count: int = 0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    is_available: bool = True


@dataclass
class TaskProfile:
    """Profile for a task type with requirements and optimal assistants"""
    task_type: str
    keywords: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    preferred_assistants: List[str] = field(default_factory=list)
    complexity_level: str = "medium"  # low, medium, high
    estimated_duration: int = 300  # seconds
    success_patterns: List[str] = field(default_factory=list)


@dataclass
class SpecializationMetrics:
    """Metrics for assistant specialization performance"""
    total_tasks: int = 0
    successful_tasks: int = 0
    average_response_time: float = 0.0
    specialization_accuracy: float = 0.0
    user_satisfaction: float = 0.0
    last_updated: Optional[datetime] = None


class AISpecializationManager:
    """
    Manages AI assistant specialization and dynamic task routing
    """

    def __init__(self, storage_path: str = "./data/ai_specialization"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Assistant profiles
        self.assistant_profiles: Dict[str, AssistantProfile] = {}

        # Task profiles
        self.task_profiles: Dict[str, TaskProfile] = {}

        # Performance metrics
        self.metrics = SpecializationMetrics()

        # Task history for learning
        self.task_history: List[Dict[str, Any]] = []

        # Initialize storage and load existing data
        self._initialized = False
        self._load_default_profiles()

    async def ensure_initialized(self):
        """Ensure the manager is initialized"""
        if not self._initialized:
            await self._initialize_storage()
            self._initialized = True

    def _load_default_profiles(self):
        """Load default assistant and task profiles"""
        # Default assistant profiles
        self.assistant_profiles = {
            'grok': AssistantProfile(
                name='grok',
                display_name='Grok CLI',
                capabilities=['general_reasoning', 'coding', 'analysis', 'creative_writing'],
                strengths=['reasoning', 'general_knowledge', 'code_explanation'],
                weaknesses=['specialized_technical_tasks', 'complex_mathematical_calculations'],
                specializations=['general_assistance', 'code_review', 'problem_solving'],
                performance_score=0.95
            ),
            'qwen': AssistantProfile(
                name='qwen',
                display_name='Qwen CLI Coder',
                capabilities=['coding', 'code_generation', 'debugging', 'refactoring'],
                strengths=['code_generation', 'technical_tasks', 'algorithm_implementation'],
                weaknesses=['creative_writing', 'general_reasoning'],
                specializations=['code_generation', 'debugging', 'refactoring'],
                performance_score=0.92
            ),
            'gemini': AssistantProfile(
                name='gemini',
                display_name='Gemini CLI',
                capabilities=['analysis', 'documentation', 'review', 'data_processing'],
                strengths=['code_analysis', 'documentation', 'data_insights'],
                weaknesses=['code_generation', 'real_time_interaction'],
                specializations=['code_analysis', 'documentation', 'review'],
                performance_score=0.90
            )
        }

        # Default task profiles
        self.task_profiles = {
            'code_generation': TaskProfile(
                task_type='code_generation',
                keywords=['write', 'create', 'generate', 'implement', 'build'],
                required_capabilities=['coding', 'code_generation'],
                preferred_assistants=['qwen', 'grok'],
                complexity_level='medium',
                estimated_duration=600
            ),
            'code_analysis': TaskProfile(
                task_type='code_analysis',
                keywords=['analyze', 'review', 'examine', 'check', 'inspect'],
                required_capabilities=['analysis', 'review'],
                preferred_assistants=['gemini', 'grok'],
                complexity_level='low',
                estimated_duration=300
            ),
            'debugging': TaskProfile(
                task_type='debugging',
                keywords=['debug', 'fix', 'error', 'bug', 'issue'],
                required_capabilities=['debugging', 'analysis'],
                preferred_assistants=['qwen', 'gemini'],
                complexity_level='high',
                estimated_duration=900
            ),
            'documentation': TaskProfile(
                task_type='documentation',
                keywords=['document', 'docstring', 'comment', 'readme'],
                required_capabilities=['documentation', 'writing'],
                preferred_assistants=['gemini', 'grok'],
                complexity_level='low',
                estimated_duration=450
            ),
            'refactoring': TaskProfile(
                task_type='refactoring',
                keywords=['refactor', 'optimize', 'improve', 'clean'],
                required_capabilities=['refactoring', 'analysis'],
                preferred_assistants=['qwen', 'gemini'],
                complexity_level='medium',
                estimated_duration=750
            )
        }

    async def _initialize_storage(self):
        """Initialize storage directory and load existing data"""
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing profiles and metrics
        await self._load_profiles()
        await self._load_metrics()
        await self._load_task_history()

    async def _load_profiles(self):
        """Load assistant and task profiles from storage"""
        profiles_file = f"{self.storage_path}/profiles.json"
        if os.path.exists(profiles_file):
            try:
                async with open(profiles_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Load assistant profiles
                    for name, profile_data in data.get('assistants', {}).items():
                        profile_data['last_used'] = datetime.fromisoformat(profile_data['last_used']) if profile_data.get('last_used') else None
                        self.assistant_profiles[name] = AssistantProfile(**profile_data)
                    # Load task profiles
                    for task_type, profile_data in data.get('tasks', {}).items():
                        self.task_profiles[task_type] = TaskProfile(**profile_data)
                self.logger.info("Loaded existing profiles from storage")
            except Exception as e:
                self.logger.error(f"Error loading profiles: {e}")

    async def _load_metrics(self):
        """Load performance metrics from storage"""
        metrics_file = f"{self.storage_path}/metrics.json"
        if os.path.exists(metrics_file):
            try:
                async with open(metrics_file, 'r') as f:
                    data = json.loads(await f.read())
                    data['last_updated'] = datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None
                    self.metrics = SpecializationMetrics(**data)
                self.logger.info("Loaded existing metrics from storage")
            except Exception as e:
                self.logger.error(f"Error loading metrics: {e}")

    async def _load_task_history(self):
        """Load task history from storage"""
        history_file = f"{self.storage_path}/task_history.json"
        if os.path.exists(history_file):
            try:
                async with open(history_file, 'r') as f:
                    self.task_history = json.loads(await f.read())
                self.logger.info(f"Loaded {len(self.task_history)} task history entries")
            except Exception as e:
                self.logger.error(f"Error loading task history: {e}")

    async def _save_profiles(self):
        """Save profiles to storage"""
        await self.ensure_initialized()
        try:
            data = {
                'assistants': {name: {
                    **profile.__dict__,
                    'last_used': profile.last_used.isoformat() if profile.last_used else None
                } for name, profile in self.assistant_profiles.items()},
                'tasks': {task_type: profile.__dict__ for task_type, profile in self.task_profiles.items()}
            }
            async with open(f"{self.storage_path}/profiles.json", 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving profiles: {e}")

    async def _save_metrics(self):
        """Save metrics to storage"""
        await self.ensure_initialized()
        try:
            data = {
                **self.metrics.__dict__,
                'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
            }
            async with open(f"{self.storage_path}/metrics.json", 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    async def _save_task_history(self):
        """Save task history to storage"""
        await self.ensure_initialized()
        try:
            async with open(f"{self.storage_path}/task_history.json", 'w') as f:
                await f.write(json.dumps(self.task_history[-1000:], indent=2))  # Keep last 1000 entries
        except Exception as e:
            self.logger.error(f"Error saving task history: {e}")

    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze a task to determine its type, requirements, and optimal assistants

        Args:
            task_description: Description of the task

        Returns:
            Task analysis with recommended assistants and complexity
        """
        task_lower = task_description.lower()

        # Find matching task profiles
        matching_profiles = []
        for profile in self.task_profiles.values():
            keyword_matches = sum(1 for keyword in profile.keywords if keyword in task_lower)
            if keyword_matches > 0:
                matching_profiles.append((profile, keyword_matches))

        # Sort by keyword matches
        matching_profiles.sort(key=lambda x: x[1], reverse=True)

        if matching_profiles:
            best_match = matching_profiles[0][0]

            # Get recommended assistants based on profile
            recommended_assistants = []
            for assistant_name in best_match.preferred_assistants:
                if assistant_name in self.assistant_profiles:
                    assistant = self.assistant_profiles[assistant_name]
                    if assistant.is_available:
                        recommended_assistants.append({
                            'name': assistant_name,
                            'score': assistant.performance_score,
                            'reason': f"Specialized in {best_match.task_type}"
                        })

            # Sort by performance score
            recommended_assistants.sort(key=lambda x: x['score'], reverse=True)

            return {
                'task_type': best_match.task_type,
                'complexity': best_match.complexity_level,
                'estimated_duration': best_match.estimated_duration,
                'required_capabilities': best_match.required_capabilities,
                'recommended_assistants': recommended_assistants,
                'confidence': min(1.0, matching_profiles[0][1] / len(best_match.keywords))
            }

        # Fallback analysis
        return {
            'task_type': 'general',
            'complexity': 'medium',
            'estimated_duration': 300,
            'required_capabilities': ['general_reasoning'],
            'recommended_assistants': [
                {'name': 'grok', 'score': 0.95, 'reason': 'General purpose assistant'}
            ],
            'confidence': 0.5
        }

    def select_optimal_assistant(self, task_analysis: Dict[str, Any],
                                available_assistants: List[str]) -> Optional[str]:
        """
        Select the optimal assistant for a task based on analysis and availability

        Args:
            task_analysis: Task analysis from analyze_task
            available_assistants: List of currently available assistants

        Returns:
            Name of the optimal assistant or None
        """
        recommended = task_analysis.get('recommended_assistants', [])

        # Filter by availability
        available_recommended = [rec for rec in recommended if rec['name'] in available_assistants]

        if available_recommended:
            # Return the highest scoring available assistant
            return max(available_recommended, key=lambda x: x['score'])['name']

        # Fallback to any available assistant
        if available_assistants:
            return available_assistants[0]

        return None

    async def record_task_result(self, task_description: str, assistant_name: str,
                                success: bool, response_time: float,
                                user_feedback: Optional[int] = None):
        """
        Record the result of a task execution for learning

        Args:
            task_description: Original task description
            assistant_name: Assistant that executed the task
            success: Whether the task was successful
            response_time: Time taken to complete the task
            user_feedback: User satisfaction rating (1-5)
        """
        task_record = {
            'timestamp': datetime.now().isoformat(),
            'task_description': task_description,
            'assistant_name': assistant_name,
            'success': success,
            'response_time': response_time,
            'user_feedback': user_feedback
        }

        self.task_history.append(task_record)

        # Update assistant profile
        if assistant_name in self.assistant_profiles:
            profile = self.assistant_profiles[assistant_name]
            profile.task_count += 1
            profile.last_used = datetime.now()

            # Update success rate
            if success:
                profile.success_rate = (profile.success_rate * (profile.task_count - 1) + 1) / profile.task_count
            else:
                profile.success_rate = (profile.success_rate * (profile.task_count - 1)) / profile.task_count

            # Update average response time
            profile.average_response_time = (
                (profile.average_response_time * (profile.task_count - 1) + response_time) / profile.task_count
            )

            # Update performance score based on success rate and response time
            time_score = max(0, 1 - (response_time / 60))  # Prefer faster responses
            profile.performance_score = (profile.success_rate + time_score) / 2

        # Update overall metrics
        self.metrics.total_tasks += 1
        if success:
            self.metrics.successful_tasks += 1

        self.metrics.average_response_time = (
            (self.metrics.average_response_time * (self.metrics.total_tasks - 1) + response_time) /
            self.metrics.total_tasks
        )

        if user_feedback:
            self.metrics.user_satisfaction = (
                (self.metrics.user_satisfaction * (self.metrics.total_tasks - 1) + user_feedback / 5) /
                self.metrics.total_tasks
            )

        self.metrics.last_updated = datetime.now()

        # Save updated data
        await self._save_profiles()
        await self._save_metrics()
        await self._save_task_history()

    def get_assistant_recommendations(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Get detailed recommendations for assistants to handle a task

        Args:
            task_description: Description of the task

        Returns:
            List of assistant recommendations with scores and reasons
        """
        analysis = self.analyze_task(task_description)
        recommendations = analysis.get('recommended_assistants', [])

        # Enhance recommendations with additional context
        for rec in recommendations:
            assistant_name = rec['name']
            if assistant_name in self.assistant_profiles:
                profile = self.assistant_profiles[assistant_name]
                rec.update({
                    'display_name': profile.display_name,
                    'capabilities': profile.capabilities,
                    'strengths': profile.strengths,
                    'task_count': profile.task_count,
                    'success_rate': profile.success_rate,
                    'average_response_time': profile.average_response_time
                })

        return recommendations

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for the specialization system

        Returns:
            Comprehensive performance report
        """
        report = {
            'overall_metrics': {
                'total_tasks': self.metrics.total_tasks,
                'success_rate': self.metrics.successful_tasks / max(1, self.metrics.total_tasks),
                'average_response_time': self.metrics.average_response_time,
                'user_satisfaction': self.metrics.user_satisfaction,
                'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
            },
            'assistant_performance': {},
            'task_type_performance': {},
            'recommendations': []
        }

        # Assistant performance
        for name, profile in self.assistant_profiles.items():
            report['assistant_performance'][name] = {
                'display_name': profile.display_name,
                'performance_score': profile.performance_score,
                'task_count': profile.task_count,
                'success_rate': profile.success_rate,
                'average_response_time': profile.average_response_time,
                'last_used': profile.last_used.isoformat() if profile.last_used else None,
                'is_available': profile.is_available
            }

        # Task type performance
        task_type_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'total_time': 0})
        for task in self.task_history:
            task_type = self.analyze_task(task['task_description'])['task_type']
            task_type_stats[task_type]['total'] += 1
            task_type_stats[task_type]['total_time'] += task['response_time']
            if task['success']:
                task_type_stats[task_type]['successful'] += 1

        for task_type, stats in task_type_stats.items():
            report['task_type_performance'][task_type] = {
                'total_tasks': stats['total'],
                'success_rate': stats['successful'] / max(1, stats['total']),
                'average_response_time': stats['total_time'] / max(1, stats['total'])
            }

        # Generate recommendations
        if self.metrics.total_tasks > 10:
            # Recommend best performing assistants
            best_assistants = sorted(
                report['assistant_performance'].items(),
                key=lambda x: x[1]['performance_score'],
                reverse=True
            )[:3]

            report['recommendations'].append({
                'type': 'top_performers',
                'assistants': [name for name, _ in best_assistants]
            })

            # Identify underperforming assistants
            underperformers = [
                name for name, perf in report['assistant_performance'].items()
                if perf['task_count'] > 5 and perf['success_rate'] < 0.8
            ]

            if underperformers:
                report['recommendations'].append({
                    'type': 'needs_improvement',
                    'assistants': underperformers
                })

        return report

    async def update_assistant_availability(self, assistant_name: str, is_available: bool):
        """
        Update the availability status of an assistant

        Args:
            assistant_name: Name of the assistant
            is_available: Whether the assistant is available
        """
        if assistant_name in self.assistant_profiles:
            self.assistant_profiles[assistant_name].is_available = is_available
            await self._save_profiles()

    async def add_custom_task_profile(self, task_profile: TaskProfile):
        """
        Add a custom task profile for specialization

        Args:
            task_profile: Custom task profile to add
        """
        self.task_profiles[task_profile.task_type] = task_profile
        await self._save_profiles()

    def get_specialization_status(self) -> Dict[str, Any]:
        """
        Get the current status of the specialization system

        Returns:
            Status information
        """
        return {
            'total_assistants': len(self.assistant_profiles),
            'total_task_profiles': len(self.task_profiles),
            'total_tasks_processed': self.metrics.total_tasks,
            'learning_enabled': True,
            'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None,
            'available_assistants': [
                name for name, profile in self.assistant_profiles.items()
                if profile.is_available
            ]
        }


# Global specialization manager instance
specialization_manager = AISpecializationManager()