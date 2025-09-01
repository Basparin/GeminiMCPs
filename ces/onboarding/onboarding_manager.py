"""
CES Onboarding Manager

Manages user onboarding, tutorials, and guided experiences
for new CES users.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TutorialStep:
    """Represents a single step in a tutorial"""
    id: str
    title: str
    description: str
    content: str
    action_type: str  # click, input, navigate, api_call
    target_element: Optional[str] = None
    expected_input: Optional[str] = None
    api_endpoint: Optional[str] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)
    success_message: str = "Step completed successfully!"
    estimated_time: int = 30  # seconds


@dataclass
class Tutorial:
    """Represents a complete tutorial"""
    id: str
    title: str
    description: str
    category: str  # beginner, intermediate, advanced
    estimated_duration: int  # minutes
    prerequisites: List[str] = field(default_factory=list)
    steps: List[TutorialStep] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class UserProgress:
    """Tracks user progress through tutorials"""
    user_id: str
    tutorial_id: str
    current_step: int = 0
    completed_steps: List[int] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    time_spent: int = 0  # seconds
    attempts: Dict[int, int] = field(default_factory=dict)  # step -> attempts
    hints_used: Dict[int, int] = field(default_factory=dict)  # step -> hints used
    score: int = 0  # completion score


class OnboardingManager:
    """
    Manages user onboarding and tutorial experiences
    """

    def __init__(self, data_path: str = "./data/onboarding"):
        self.data_path = data_path
        self.tutorials: Dict[str, Tutorial] = {}
        self.user_progress: Dict[str, Dict[str, UserProgress]] = {}

        # Create data directory
        os.makedirs(data_path, exist_ok=True)

        # Load tutorials and user progress
        self._load_tutorials()
        self._load_user_progress()

    def _load_tutorials(self):
        """Load tutorial definitions"""
        tutorial_file = f"{self.data_path}/tutorials.json"

        # Create default tutorials if file doesn't exist
        if not os.path.exists(tutorial_file):
            self._create_default_tutorials()
            self._save_tutorials()
        else:
            try:
                with open(tutorial_file, 'r') as f:
                    data = json.load(f)
                    for tutorial_data in data.get('tutorials', []):
                        # Convert step data to TutorialStep objects
                        steps = []
                        for step_data in tutorial_data.get('steps', []):
                            steps.append(TutorialStep(**step_data))

                        tutorial_data['steps'] = steps
                        tutorial = Tutorial(**tutorial_data)
                        self.tutorials[tutorial.id] = tutorial
            except Exception as e:
                print(f"Error loading tutorials: {e}")
                self._create_default_tutorials()

    def _save_tutorials(self):
        """Save tutorial definitions"""
        try:
            data = {'tutorials': []}
            for tutorial in self.tutorials.values():
                tutorial_dict = {
                    **tutorial.__dict__,
                    'steps': [step.__dict__ for step in tutorial.steps]
                }
                data['tutorials'].append(tutorial_dict)

            with open(f"{self.data_path}/tutorials.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving tutorials: {e}")

    def _load_user_progress(self):
        """Load user progress data"""
        progress_file = f"{self.data_path}/user_progress.json"

        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                    for user_id, tutorials in data.items():
                        self.user_progress[user_id] = {}
                        for tutorial_id, progress_data in tutorials.items():
                            # Convert datetime strings
                            if progress_data.get('started_at'):
                                progress_data['started_at'] = datetime.fromisoformat(progress_data['started_at'])
                            if progress_data.get('completed_at'):
                                progress_data['completed_at'] = datetime.fromisoformat(progress_data['completed_at'])

                            progress = UserProgress(**progress_data)
                            self.user_progress[user_id][tutorial_id] = progress
            except Exception as e:
                print(f"Error loading user progress: {e}")

    def _save_user_progress(self):
        """Save user progress data"""
        try:
            data = {}
            for user_id, tutorials in self.user_progress.items():
                data[user_id] = {}
                for tutorial_id, progress in tutorials.items():
                    progress_dict = progress.__dict__.copy()
                    # Convert datetime objects to strings
                    if progress.started_at:
                        progress_dict['started_at'] = progress.started_at.isoformat()
                    if progress.completed_at:
                        progress_dict['completed_at'] = progress.completed_at.isoformat()

                    data[user_id][tutorial_id] = progress_dict

            with open(f"{self.data_path}/user_progress.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving user progress: {e}")

    def _create_default_tutorials(self):
        """Create default tutorial content"""
        # Welcome Tutorial
        welcome_tutorial = Tutorial(
            id="welcome",
            title="Welcome to CES",
            description="Get started with the Cognitive Enhancement System",
            category="beginner",
            estimated_duration=5,
            learning_objectives=[
                "Understand CES capabilities",
                "Navigate the dashboard",
                "Create your first task"
            ],
            tags=["getting-started", "dashboard", "basics"]
        )

        welcome_tutorial.steps = [
            TutorialStep(
                id="welcome_1",
                title="Welcome to CES",
                description="Let's get you started with CES!",
                content="CES is a powerful AI-assisted development platform. This tutorial will guide you through the basics.",
                action_type="click",
                target_element="#dashboard-header",
                success_message="Great! You've found the dashboard header."
            ),
            TutorialStep(
                id="welcome_2",
                title="Check System Status",
                description="Let's check the current system status",
                content="The dashboard shows real-time system status. Look for the status indicator in the header.",
                action_type="navigate",
                target_element="#system-status",
                success_message="Perfect! You can see the system is healthy."
            ),
            TutorialStep(
                id="welcome_3",
                title="Create Your First Task",
                description="Time to create your first task",
                content="Click the 'Create New Task' section to submit your first task to CES.",
                action_type="click",
                target_element="#task-form",
                success_message="Excellent! Now you can create tasks."
            ),
            TutorialStep(
                id="welcome_4",
                title="Submit a Task",
                description="Submit a task for CES to process",
                content="Enter a task description like 'Help me understand Python classes' and submit it.",
                action_type="input",
                target_element="#task-description",
                expected_input="Help me understand Python classes",
                success_message="Fantastic! CES will now process your task."
            )
        ]

        # Dashboard Tutorial
        dashboard_tutorial = Tutorial(
            id="dashboard_basics",
            title="Dashboard Basics",
            description="Master the CES dashboard interface",
            category="beginner",
            estimated_duration=10,
            prerequisites=["welcome"],
            learning_objectives=[
                "Navigate dashboard sections",
                "Monitor system performance",
                "Track AI assistant status",
                "View analytics data"
            ],
            tags=["dashboard", "monitoring", "analytics"]
        )

        dashboard_tutorial.steps = [
            TutorialStep(
                id="dashboard_1",
                title="Performance Metrics",
                description="Understanding system performance",
                content="The dashboard shows CPU, memory, and disk usage. These metrics help you understand system health.",
                action_type="click",
                target_element="#performance-metrics",
                success_message="Good! You can monitor system performance here."
            ),
            TutorialStep(
                id="dashboard_2",
                title="AI Assistant Status",
                description="Check AI assistant availability",
                content="This section shows the status of all AI assistants (Grok, Qwen, Gemini).",
                action_type="click",
                target_element="#ai-assistants",
                success_message="Perfect! All assistants are ready to help."
            ),
            TutorialStep(
                id="dashboard_3",
                title="Active Tasks",
                description="Monitor running tasks",
                content="See all currently active tasks and their progress.",
                action_type="click",
                target_element="#active-tasks",
                success_message="Great! You can track task progress here."
            ),
            TutorialStep(
                id="dashboard_4",
                title="Recent Activity",
                description="View system activity",
                content="This shows recent system events and user actions.",
                action_type="click",
                target_element="#recent-activity",
                success_message="Excellent! Stay updated with system activity."
            )
        ]

        # Feedback Tutorial
        feedback_tutorial = Tutorial(
            id="feedback_system",
            title="Using the Feedback System",
            description="Learn how to provide feedback and improve CES",
            category="intermediate",
            estimated_duration=8,
            learning_objectives=[
                "Submit different types of feedback",
                "View feedback analytics",
                "Track feedback resolution"
            ],
            tags=["feedback", "improvement", "analytics"]
        )

        feedback_tutorial.steps = [
            TutorialStep(
                id="feedback_1",
                title="Access Feedback Section",
                description="Navigate to the feedback area",
                content="Click on the feedback section to access feedback features.",
                action_type="click",
                target_element="#feedback-section",
                success_message="Found the feedback section!"
            ),
            TutorialStep(
                id="feedback_2",
                title="Submit Bug Report",
                description="Report a potential issue",
                content="Select 'Bug Report' and describe an issue you've encountered.",
                action_type="input",
                target_element="#feedback-form",
                success_message="Bug report submitted successfully!"
            ),
            TutorialStep(
                id="feedback_3",
                title="Submit Feature Request",
                description="Request a new feature",
                content="Select 'Feature Request' and describe a feature you'd like to see.",
                action_type="input",
                target_element="#feedback-form",
                success_message="Feature request submitted!"
            ),
            TutorialStep(
                id="feedback_4",
                title="View Feedback Analytics",
                description="Check feedback statistics",
                content="Look at the feedback summary to see trends and insights.",
                action_type="click",
                target_element="#feedback-summary",
                success_message="Great! You can see feedback analytics here."
            )
        ]

        # Analytics Tutorial
        analytics_tutorial = Tutorial(
            id="analytics_insights",
            title="Analytics and Insights",
            description="Understand usage patterns and system insights",
            category="intermediate",
            estimated_duration=12,
            learning_objectives=[
                "Interpret usage analytics",
                "Understand system insights",
                "Monitor performance trends",
                "Track user engagement"
            ],
            tags=["analytics", "insights", "performance"]
        )

        analytics_tutorial.steps = [
            TutorialStep(
                id="analytics_1",
                title="Access Analytics",
                description="Navigate to analytics section",
                content="Click on the analytics section to view usage data and insights.",
                action_type="click",
                target_element="#analytics-dashboard",
                success_message="Analytics dashboard accessed!"
            ),
            TutorialStep(
                id="analytics_2",
                title="Real-time Metrics",
                description="Monitor live system metrics",
                content="View real-time metrics like active users, current tasks, and system load.",
                action_type="click",
                target_element="#realtime-metrics",
                success_message="Real-time metrics are updating!"
            ),
            TutorialStep(
                id="analytics_3",
                title="Usage Summary",
                description="Review usage statistics",
                content="Check total events, unique users, and task success rates.",
                action_type="click",
                target_element="#usage-summary",
                success_message="Usage patterns are clear!"
            ),
            TutorialStep(
                id="analytics_4",
                title="System Insights",
                description="View automated insights",
                content="Read system-generated insights and recommendations.",
                action_type="click",
                target_element="#system-insights",
                success_message="AI-generated insights are helpful!"
            )
        ]

        # Store tutorials
        self.tutorials = {
            "welcome": welcome_tutorial,
            "dashboard_basics": dashboard_tutorial,
            "feedback_system": feedback_tutorial,
            "analytics_insights": analytics_tutorial
        }

    def get_available_tutorials(self, user_id: Optional[str] = None,
                               category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available tutorials, optionally filtered by category

        Args:
            user_id: User ID to check completion status
            category: Filter by tutorial category

        Returns:
            List of available tutorials
        """
        tutorials = []

        for tutorial in self.tutorials.values():
            if category and tutorial.category != category:
                continue

            tutorial_dict = {
                **tutorial.__dict__,
                'steps': [step.__dict__ for step in tutorial.steps],
                'completed': False,
                'in_progress': False,
                'progress': 0
            }

            if user_id and user_id in self.user_progress:
                user_tutorials = self.user_progress[user_id]
                if tutorial.id in user_tutorials:
                    progress = user_tutorials[tutorial.id]
                    tutorial_dict['completed'] = progress.completed_at is not None
                    tutorial_dict['in_progress'] = progress.started_at is not None and not tutorial_dict['completed']
                    tutorial_dict['progress'] = len(progress.completed_steps) / len(tutorial.steps) if tutorial.steps else 0

            tutorials.append(tutorial_dict)

        return tutorials

    def start_tutorial(self, user_id: str, tutorial_id: str) -> Optional[Dict[str, Any]]:
        """
        Start a tutorial for a user

        Args:
            user_id: User ID
            tutorial_id: Tutorial ID

        Returns:
            Tutorial data or None if not found
        """
        if tutorial_id not in self.tutorials:
            return None

        tutorial = self.tutorials[tutorial_id]

        # Initialize user progress if not exists
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        if tutorial_id not in self.user_progress[user_id]:
            progress = UserProgress(
                user_id=user_id,
                tutorial_id=tutorial_id,
                started_at=datetime.now()
            )
            self.user_progress[user_id][tutorial_id] = progress
            self._save_user_progress()

        # Return tutorial with progress
        progress = self.user_progress[user_id][tutorial_id]
        tutorial_dict = {
            **tutorial.__dict__,
            'steps': [step.__dict__ for step in tutorial.steps],
            'current_step': progress.current_step,
            'completed_steps': progress.completed_steps,
            'progress': len(progress.completed_steps) / len(tutorial.steps) if tutorial.steps else 0
        }

        return tutorial_dict

    def complete_tutorial_step(self, user_id: str, tutorial_id: str,
                              step_index: int, success: bool = True) -> Dict[str, Any]:
        """
        Mark a tutorial step as completed

        Args:
            user_id: User ID
            tutorial_id: Tutorial ID
            step_index: Step index
            success: Whether step was completed successfully

        Returns:
            Updated progress information
        """
        if (user_id not in self.user_progress or
            tutorial_id not in self.user_progress[user_id]):
            return {"error": "Tutorial not started"}

        progress = self.user_progress[user_id][tutorial_id]
        tutorial = self.tutorials[tutorial_id]

        if step_index < 0 or step_index >= len(tutorial.steps):
            return {"error": "Invalid step index"}

        if step_index not in progress.completed_steps:
            progress.completed_steps.append(step_index)

        progress.current_step = step_index + 1

        # Check if tutorial is completed
        if len(progress.completed_steps) == len(tutorial.steps):
            progress.completed_at = datetime.now()
            progress.score = 100  # Full completion

        self._save_user_progress()

        return {
            "step_completed": step_index,
            "tutorial_progress": len(progress.completed_steps) / len(tutorial.steps),
            "tutorial_completed": progress.completed_at is not None,
            "next_step": progress.current_step if progress.current_step < len(tutorial.steps) else None
        }

    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's tutorial progress

        Args:
            user_id: User ID

        Returns:
            User's tutorial progress
        """
        if user_id not in self.user_progress:
            return {"tutorials": {}}

        progress_data = {}
        for tutorial_id, progress in self.user_progress[user_id].items():
            if tutorial_id in self.tutorials:
                tutorial = self.tutorials[tutorial_id]
                progress_data[tutorial_id] = {
                    "title": tutorial.title,
                    "category": tutorial.category,
                    "current_step": progress.current_step,
                    "completed_steps": len(progress.completed_steps),
                    "total_steps": len(tutorial.steps),
                    "progress": len(progress.completed_steps) / len(tutorial.steps) if tutorial.steps else 0,
                    "completed": progress.completed_at is not None,
                    "started_at": progress.started_at.isoformat() if progress.started_at else None,
                    "completed_at": progress.completed_at.isoformat() if progress.completed_at else None,
                    "time_spent": progress.time_spent
                }

        return {"tutorials": progress_data}

    def get_tutorial_details(self, tutorial_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed tutorial information

        Args:
            tutorial_id: Tutorial ID

        Returns:
            Tutorial details or None if not found
        """
        if tutorial_id not in self.tutorials:
            return None

        tutorial = self.tutorials[tutorial_id]
        return {
            **tutorial.__dict__,
            'steps': [step.__dict__ for step in tutorial.steps]
        }

    def reset_tutorial_progress(self, user_id: str, tutorial_id: str) -> bool:
        """
        Reset tutorial progress for a user

        Args:
            user_id: User ID
            tutorial_id: Tutorial ID

        Returns:
            True if reset successful, False otherwise
        """
        if (user_id in self.user_progress and
            tutorial_id in self.user_progress[user_id]):
            del self.user_progress[user_id][tutorial_id]
            self._save_user_progress()
            return True

        return False

    def get_onboarding_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get overall onboarding status for a user

        Args:
            user_id: User ID

        Returns:
            Onboarding status summary
        """
        progress = self.get_user_progress(user_id)
        tutorials = progress["tutorials"]

        total_tutorials = len(self.tutorials)
        completed_tutorials = sum(1 for t in tutorials.values() if t["completed"])
        in_progress_tutorials = sum(1 for t in tutorials.values() if t["progress"] > 0 and not t["completed"])

        # Calculate overall progress
        total_progress = sum(t["progress"] for t in tutorials.values())
        overall_progress = total_progress / total_tutorials if total_tutorials > 0 else 0

        # Determine onboarding level
        if overall_progress < 0.25:
            level = "beginner"
            next_steps = ["Complete the 'Welcome to CES' tutorial"]
        elif overall_progress < 0.5:
            level = "novice"
            next_steps = ["Learn dashboard basics", "Try creating tasks"]
        elif overall_progress < 0.75:
            level = "intermediate"
            next_steps = ["Explore analytics", "Learn feedback system"]
        else:
            level = "advanced"
            next_steps = ["Explore advanced features", "Consider plugin development"]

        return {
            "user_id": user_id,
            "level": level,
            "overall_progress": overall_progress,
            "completed_tutorials": completed_tutorials,
            "in_progress_tutorials": in_progress_tutorials,
            "total_tutorials": total_tutorials,
            "next_steps": next_steps,
            "recommendations": self._get_personalized_recommendations(user_id)
        }

    def _get_personalized_recommendations(self, user_id: str) -> List[str]:
        """
        Get personalized tutorial recommendations

        Args:
            user_id: User ID

        Returns:
            List of recommendations
        """
        recommendations = []
        progress = self.get_user_progress(user_id)["tutorials"]

        # Check if welcome tutorial is completed
        if "welcome" not in progress or not progress["welcome"]["completed"]:
            recommendations.append("Start with the 'Welcome to CES' tutorial")

        # Check dashboard knowledge
        if "dashboard_basics" not in progress or not progress["dashboard_basics"]["completed"]:
            recommendations.append("Learn dashboard basics for better navigation")

        # Check feedback system
        if "feedback_system" not in progress or not progress["feedback_system"]["completed"]:
            recommendations.append("Learn how to provide feedback to improve CES")

        # Check analytics
        if "analytics_insights" not in progress or not progress["analytics_insights"]["completed"]:
            recommendations.append("Explore analytics to understand usage patterns")

        # Default recommendations
        if not recommendations:
            recommendations = [
                "Try advanced features like collaborative sessions",
                "Explore plugin development",
                "Help improve CES by providing feedback"
            ]

        return recommendations


# Global onboarding manager instance
onboarding_manager = OnboardingManager()