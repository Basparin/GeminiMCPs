"""
Adaptive Learner - CES Learning and Improvement Engine

Learns from user interactions, task outcomes, and system performance to
continuously improve CES effectiveness and adapt to user preferences.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json


class AdaptiveLearner:
    """
    Learns patterns and preferences to improve CES performance over time.

    Key functions:
    - User preference learning
    - Task outcome analysis
    - Performance pattern recognition
    - Recommendation optimization
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Learning data storage
        self.user_patterns = defaultdict(lambda: defaultdict(int))
        self.task_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
        self.assistant_performance = defaultdict(lambda: {'score': 0, 'count': 0})

        self.logger.info("Adaptive Learner initialized")

    def learn_from_task(self, task_description: str, result: Dict[str, Any]):
        """
        Learn from task execution results

        Args:
            task_description: Description of the completed task
            result: Execution result and metadata
        """
        # Extract task type/category
        task_type = self._categorize_task(task_description)

        # Update success rates
        success = result.get('status') == 'completed'
        self.task_success_rates[task_type]['total'] += 1
        if success:
            self.task_success_rates[task_type]['success'] += 1

        # Learn from assistant performance
        assistant = result.get('assistant_used', 'unknown')
        if assistant != 'unknown':
            performance_score = self._calculate_performance_score(result)
            self.assistant_performance[assistant]['score'] += performance_score
            self.assistant_performance[assistant]['count'] += 1

        # Learn user preferences from task patterns
        self._learn_user_patterns(task_description, result)

        self.logger.debug(f"Learned from task: {task_type} - Success: {success}")

    def get_task_success_rate(self, task_type: str) -> float:
        """Get success rate for a task type"""
        stats = self.task_success_rates[task_type]
        if stats['total'] == 0:
            return 0.0
        return stats['success'] / stats['total']

    def get_assistant_recommendation(self, task_description: str) -> List[str]:
        """
        Recommend assistants based on learned patterns

        Args:
            task_description: Task to recommend assistants for

        Returns:
            List of recommended assistant names
        """
        task_type = self._categorize_task(task_description)

        # Get assistants sorted by performance for this task type
        assistant_scores = {}
        for assistant, stats in self.assistant_performance.items():
            if stats['count'] > 0:
                avg_score = stats['score'] / stats['count']
                assistant_scores[assistant] = avg_score

        # Sort by performance and return top recommendations
        sorted_assistants = sorted(assistant_scores.items(), key=lambda x: x[1], reverse=True)
        return [assistant for assistant, _ in sorted_assistants[:3]]

    def get_user_preferences(self, context: str) -> Dict[str, Any]:
        """
        Get learned user preferences for a context

        Args:
            context: Context to get preferences for

        Returns:
            Dictionary of learned preferences
        """
        if context in self.user_patterns:
            return dict(self.user_patterns[context])
        return {}

    def _categorize_task(self, task_description: str) -> str:
        """Categorize a task based on its description"""
        task_lower = task_description.lower()

        if any(word in task_lower for word in ['code', 'python', 'programming', 'function']):
            return 'coding'
        elif any(word in task_lower for word in ['test', 'testing', 'debug', 'fix']):
            return 'testing'
        elif any(word in task_lower for word in ['design', 'architecture', 'structure']):
            return 'design'
        elif any(word in task_lower for word in ['document', 'readme', 'write']):
            return 'documentation'
        else:
            return 'general'

    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate performance score for a task result"""
        score = 0.0

        # Base score for completion
        if result.get('status') == 'completed':
            score += 1.0

        # Bonus for fast execution
        execution_time = result.get('execution_time', 0)
        if execution_time > 0 and execution_time < 300:  # Less than 5 minutes
            score += 0.5

        # Bonus for high quality
        if result.get('quality_score', 0) > 0.8:
            score += 0.3

        return min(score, 2.0)  # Cap at 2.0

    def _learn_user_patterns(self, task_description: str, result: Dict[str, Any]):
        """Learn user behavior patterns"""
        # Learn preferred task types
        task_type = self._categorize_task(task_description)
        self.user_patterns['task_types'][task_type] += 1

        # Learn preferred times (if timestamp available)
        if 'timestamp' in result:
            try:
                dt = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                hour = dt.hour
                time_of_day = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
                self.user_patterns['time_preferences'][time_of_day] += 1
            except:
                pass

        # Learn success patterns
        if result.get('status') == 'completed':
            self.user_patterns['success_patterns'][task_type] += 1

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        return {
            "task_types_learned": dict(self.user_patterns['task_types']),
            "total_tasks_processed": sum(stats['total'] for stats in self.task_success_rates.values()),
            "assistants_evaluated": len(self.assistant_performance),
            "success_rates": {
                task_type: self.get_task_success_rate(task_type)
                for task_type in self.task_success_rates.keys()
            }
        }

    def reset_learning_data(self):
        """Reset all learned data (for testing or fresh start)"""
        self.user_patterns.clear()
        self.task_success_rates.clear()
        self.assistant_performance.clear()
        self.logger.info("Learning data reset")

    def export_learning_data(self) -> str:
        """Export learning data as JSON string"""
        data = {
            "user_patterns": dict(self.user_patterns),
            "task_success_rates": dict(self.task_success_rates),
            "assistant_performance": dict(self.assistant_performance),
            "export_timestamp": datetime.now().isoformat()
        }
        return json.dumps(data, indent=2)

    def import_learning_data(self, json_data: str):
        """Import learning data from JSON string"""
        try:
            data = json.loads(json_data)
            self.user_patterns.update(data.get('user_patterns', {}))
            self.task_success_rates.update(data.get('task_success_rates', {}))
            self.assistant_performance.update(data.get('assistant_performance', {}))
            self.logger.info("Learning data imported successfully")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to import learning data: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get adaptive learner status"""
        return {
            "status": "operational",
            "patterns_learned": len(self.user_patterns),
            "task_types_tracked": len(self.task_success_rates),
            "assistants_evaluated": len(self.assistant_performance),
            "last_update": datetime.now().isoformat()
        }