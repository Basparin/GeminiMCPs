"""CES User Onboarding Manager.

Provides comprehensive user onboarding flow with tutorials, guided experiences,
progress tracking, and personalized learning paths for the Cognitive Enhancement System.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class OnboardingStep:
    """Represents a single onboarding step."""
    id: str
    title: str
    description: str
    content: Dict[str, Any]
    prerequisites: List[str]
    estimated_duration: int  # minutes
    difficulty: str  # beginner, intermediate, advanced
    category: str  # getting_started, features, advanced, best_practices
    interactive: bool = False
    completion_criteria: Dict[str, Any] = None

@dataclass
class UserProgress:
    """Tracks user onboarding progress."""
    user_id: str
    current_step: str
    completed_steps: List[str]
    started_at: str
    last_activity: str
    total_time_spent: int  # minutes
    skill_assessment: Dict[str, Any]
    preferences: Dict[str, Any]
    achievements: List[str]

@dataclass
class Tutorial:
    """Represents a complete tutorial."""
    id: str
    title: str
    description: str
    steps: List[OnboardingStep]
    target_audience: str  # beginner, intermediate, advanced, expert
    estimated_duration: int  # minutes
    prerequisites: List[str]
    learning_objectives: List[str]
    tags: List[str]

class OnboardingManager:
    """Manages user onboarding experiences and tutorials."""

    def __init__(self):
        self.tutorials: Dict[str, Tutorial] = {}
        self.user_progress: Dict[str, UserProgress] = {}
        self.onboarding_steps: Dict[str, OnboardingStep] = {}
        self.achievement_system = AchievementSystem()
        self.personalization_engine = PersonalizationEngine()

        # Initialize default tutorials and steps
        self._initialize_default_content()

    def is_healthy(self) -> bool:
        """Check if onboarding manager is healthy."""
        return True

    def _initialize_default_content(self):
        """Initialize default onboarding content."""
        # Define onboarding steps
        self.onboarding_steps = {
            "welcome": OnboardingStep(
                id="welcome",
                title="Welcome to CES",
                description="Get started with the Cognitive Enhancement System",
                content={
                    "type": "interactive",
                    "text": "Welcome to CES! Let's get you started with the most powerful AI-assisted development platform.",
                    "actions": ["next"]
                },
                prerequisites=[],
                estimated_duration=2,
                difficulty="beginner",
                category="getting_started",
                interactive=True
            ),
            "create_first_task": OnboardingStep(
                id="create_first_task",
                title="Create Your First Task",
                description="Learn how to create and manage tasks with AI assistance",
                content={
                    "type": "tutorial",
                    "text": "Tasks are the core of CES. Let's create your first task and see how AI can help you.",
                    "demo": {
                        "endpoint": "/api/tasks",
                        "method": "POST",
                        "payload": {"description": "Implement a simple function", "priority": "medium"}
                    }
                },
                prerequisites=["welcome"],
                estimated_duration=5,
                difficulty="beginner",
                category="getting_started",
                interactive=True,
                completion_criteria={"task_created": True}
            ),
            "explore_ai_assistants": OnboardingStep(
                id="explore_ai_assistants",
                title="Meet Your AI Assistants",
                description="Discover the different AI assistants available to help you",
                content={
                    "type": "interactive",
                    "text": "CES comes with multiple specialized AI assistants. Let's explore what each one does.",
                    "assistants": [
                        {"name": "Grok", "specialty": "General reasoning and problem solving"},
                        {"name": "Qwen Coder", "specialty": "Code generation and analysis"},
                        {"name": "Gemini", "specialty": "Code analysis and debugging"}
                    ]
                },
                prerequisites=["create_first_task"],
                estimated_duration=8,
                difficulty="beginner",
                category="features",
                interactive=True
            ),
            "collaborative_sessions": OnboardingStep(
                id="collaborative_sessions",
                title="Collaborative Workspaces",
                description="Learn how to create and manage collaborative sessions",
                content={
                    "type": "tutorial",
                    "text": "CES supports real-time collaboration. Let's create a session and invite others to join.",
                    "features": ["Real-time collaboration", "Session management", "User permissions"]
                },
                prerequisites=["explore_ai_assistants"],
                estimated_duration=10,
                difficulty="intermediate",
                category="features",
                interactive=True,
                completion_criteria={"session_created": True}
            ),
            "advanced_analytics": OnboardingStep(
                id="advanced_analytics",
                title="Analytics and Insights",
                description="Explore advanced analytics and performance insights",
                content={
                    "type": "tutorial",
                    "text": "CES provides comprehensive analytics to help you understand your productivity patterns.",
                    "analytics": ["Performance metrics", "Usage patterns", "AI assistant effectiveness"]
                },
                prerequisites=["collaborative_sessions"],
                estimated_duration=12,
                difficulty="intermediate",
                category="advanced",
                interactive=True
            ),
            "plugin_system": OnboardingStep(
                id="plugin_system",
                title="Extending CES with Plugins",
                description="Learn how to extend CES functionality with plugins",
                content={
                    "type": "tutorial",
                    "text": "CES has a powerful plugin system that allows you to add custom functionality.",
                    "topics": ["Plugin architecture", "Installing plugins", "Creating custom plugins"]
                },
                prerequisites=["advanced_analytics"],
                estimated_duration=15,
                difficulty="advanced",
                category="advanced",
                interactive=True
            ),
            "best_practices": OnboardingStep(
                id="best_practices",
                title="CES Best Practices",
                description="Learn best practices for getting the most out of CES",
                content={
                    "type": "tutorial",
                    "text": "Master the art of effective AI-assisted development with these proven strategies.",
                    "practices": [
                        "Writing clear task descriptions",
                        "Choosing the right AI assistant",
                        "Reviewing and refining AI suggestions",
                        "Collaborating effectively in sessions"
                    ]
                },
                prerequisites=["plugin_system"],
                estimated_duration=10,
                difficulty="intermediate",
                category="best_practices",
                interactive=True
            )
        }

        # Define tutorials
        self.tutorials = {
            "getting_started": Tutorial(
                id="getting_started",
                title="Getting Started with CES",
                description="Complete beginner's guide to CES",
                steps=["welcome", "create_first_task", "explore_ai_assistants"],
                target_audience="beginner",
                estimated_duration=15,
                prerequisites=[],
                learning_objectives=[
                    "Understand CES core concepts",
                    "Create and manage tasks",
                    "Work with AI assistants"
                ],
                tags=["beginner", "essentials", "getting-started"]
            ),
            "collaboration_mastery": Tutorial(
                id="collaboration_mastery",
                title="Collaboration Mastery",
                description="Master collaborative features and team workflows",
                steps=["collaborative_sessions", "advanced_analytics"],
                target_audience="intermediate",
                estimated_duration=22,
                prerequisites=["getting_started"],
                learning_objectives=[
                    "Create and manage collaborative sessions",
                    "Understand user roles and permissions",
                    "Analyze team productivity"
                ],
                tags=["collaboration", "teamwork", "intermediate"]
            ),
            "power_user": Tutorial(
                id="power_user",
                title="Power User Guide",
                description="Advanced features and customization",
                steps=["plugin_system", "best_practices"],
                target_audience="advanced",
                estimated_duration=25,
                prerequisites=["collaboration_mastery"],
                learning_objectives=[
                    "Extend CES with plugins",
                    "Implement best practices",
                    "Customize workflows"
                ],
                tags=["advanced", "plugins", "customization"]
            )
        }

    async def start_user_onboarding(self, user_id: str, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start onboarding process for a new user."""
        try:
            # Assess user skill level
            skill_level = self._assess_user_skill_level(user_profile or {})

            # Create personalized learning path
            learning_path = self.personalization_engine.create_learning_path(skill_level, user_profile or {})

            # Initialize user progress
            progress = UserProgress(
                user_id=user_id,
                current_step=learning_path[0] if learning_path else "welcome",
                completed_steps=[],
                started_at=datetime.now().isoformat(),
                last_activity=datetime.now().isoformat(),
                total_time_spent=0,
                skill_assessment={"level": skill_level, "confidence": 0.5},
                preferences=user_profile or {},
                achievements=[]
            )

            self.user_progress[user_id] = progress

            # Get first step content
            first_step = self.onboarding_steps.get(progress.current_step)
            if not first_step:
                return {"error": "Invalid starting step"}

            logger.info(f"Started onboarding for user {user_id} at skill level {skill_level}")
            return {
                "status": "started",
                "user_id": user_id,
                "current_step": asdict(first_step),
                "learning_path": learning_path,
                "estimated_completion_time": self._calculate_path_duration(learning_path)
            }

        except Exception as e:
            logger.error(f"Error starting onboarding for user {user_id}: {e}")
            return {"error": str(e)}

    async def get_next_step(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the next onboarding step for a user."""
        if user_id not in self.user_progress:
            return None

        progress = self.user_progress[user_id]
        current_step_id = progress.current_step

        # Check if current step is completed
        if current_step_id not in progress.completed_steps:
            step = self.onboarding_steps.get(current_step_id)
            return asdict(step) if step else None

        # Find next step in learning path
        learning_path = self.personalization_engine.get_user_learning_path(user_id)
        if not learning_path:
            return None

        try:
            current_index = learning_path.index(current_step_id)
            if current_index + 1 < len(learning_path):
                next_step_id = learning_path[current_index + 1]
                step = self.onboarding_steps.get(next_step_id)
                return asdict(step) if step else None
        except ValueError:
            pass

        return None

    async def complete_step(self, user_id: str, step_id: str, completion_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mark a step as completed and update progress."""
        if user_id not in self.user_progress:
            return {"error": "User not found"}

        progress = self.user_progress[user_id]

        if step_id not in self.onboarding_steps:
            return {"error": "Invalid step"}

        if step_id in progress.completed_steps:
            return {"error": "Step already completed"}

        # Validate completion criteria
        step = self.onboarding_steps[step_id]
        if not self._validate_completion_criteria(step, completion_data):
            return {"error": "Completion criteria not met"}

        # Mark step as completed
        progress.completed_steps.append(step_id)
        progress.last_activity = datetime.now().isoformat()
        progress.total_time_spent += step.estimated_duration

        # Check for achievements
        new_achievements = self.achievement_system.check_achievements(user_id, progress)
        progress.achievements.extend(new_achievements)

        # Update skill assessment
        progress.skill_assessment = self._update_skill_assessment(progress)

        # Get next step
        next_step = await self.get_next_step(user_id)
        if next_step:
            progress.current_step = next_step["id"]

        # Calculate progress percentage
        learning_path = self.personalization_engine.get_user_learning_path(user_id)
        progress_percentage = len(progress.completed_steps) / len(learning_path) * 100 if learning_path else 0

        logger.info(f"User {user_id} completed step {step_id}")

        return {
            "status": "completed",
            "step_id": step_id,
            "next_step": next_step,
            "progress_percentage": round(progress_percentage, 2),
            "new_achievements": new_achievements,
            "total_time_spent": progress.total_time_spent
        }

    async def get_user_progress(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user progress information."""
        if user_id not in self.user_progress:
            return None

        progress = self.user_progress[user_id]
        learning_path = self.personalization_engine.get_user_learning_path(user_id)

        return {
            "user_id": user_id,
            "current_step": progress.current_step,
            "completed_steps": progress.completed_steps,
            "total_steps": len(learning_path) if learning_path else 0,
            "progress_percentage": len(progress.completed_steps) / len(learning_path) * 100 if learning_path else 0,
            "started_at": progress.started_at,
            "last_activity": progress.last_activity,
            "total_time_spent": progress.total_time_spent,
            "skill_assessment": progress.skill_assessment,
            "achievements": progress.achievements,
            "learning_path": learning_path
        }

    async def get_available_tutorials(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get list of available tutorials, optionally filtered by user."""
        tutorials = []

        for tutorial in self.tutorials.values():
            tutorial_dict = asdict(tutorial)

            # Check if user meets prerequisites
            if user_id:
                meets_prerequisites = self._check_prerequisites(user_id, tutorial.prerequisites)
                tutorial_dict["available"] = meets_prerequisites
                tutorial_dict["recommended"] = self._is_tutorial_recommended(user_id, tutorial)
            else:
                tutorial_dict["available"] = True
                tutorial_dict["recommended"] = False

            tutorials.append(tutorial_dict)

        return tutorials

    async def start_tutorial(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Start a specific tutorial for a user."""
        if tutorial_id not in self.tutorials:
            return {"error": "Tutorial not found"}

        tutorial = self.tutorials[tutorial_id]

        # Check prerequisites
        if not self._check_prerequisites(user_id, tutorial.prerequisites):
            return {"error": "Prerequisites not met"}

        # Create learning path from tutorial steps
        learning_path = tutorial.steps
        self.personalization_engine.set_user_learning_path(user_id, learning_path)

        # Reset or update user progress
        if user_id in self.user_progress:
            progress = self.user_progress[user_id]
            progress.current_step = learning_path[0]
            progress.last_activity = datetime.now().isoformat()
        else:
            # Start new onboarding with this tutorial
            return await self.start_user_onboarding(user_id)

        logger.info(f"User {user_id} started tutorial {tutorial_id}")
        return {
            "status": "started",
            "tutorial_id": tutorial_id,
            "tutorial": asdict(tutorial),
            "first_step": asdict(self.onboarding_steps[learning_path[0]])
        }

    async def get_tutorial_progress(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Get progress for a specific tutorial."""
        if tutorial_id not in self.tutorials:
            return {"error": "Tutorial not found"}

        tutorial = self.tutorials[tutorial_id]
        progress = self.user_progress.get(user_id)

        if not progress:
            return {"error": "User progress not found"}

        completed_in_tutorial = [step for step in progress.completed_steps if step in tutorial.steps]
        progress_percentage = len(completed_in_tutorial) / len(tutorial.steps) * 100

        return {
            "tutorial_id": tutorial_id,
            "completed_steps": completed_in_tutorial,
            "total_steps": len(tutorial.steps),
            "progress_percentage": round(progress_percentage, 2),
            "estimated_time_remaining": (len(tutorial.steps) - len(completed_in_tutorial)) * 10  # Rough estimate
        }

    def _assess_user_skill_level(self, user_profile: Dict[str, Any]) -> str:
        """Assess user's skill level based on profile."""
        # Simple assessment based on profile data
        experience_years = user_profile.get("experience_years", 0)
        previous_tools = user_profile.get("previous_tools", [])
        coding_experience = user_profile.get("coding_experience", "beginner")

        if experience_years >= 5 or "expert" in coding_experience.lower():
            return "advanced"
        elif experience_years >= 2 or "intermediate" in coding_experience.lower():
            return "intermediate"
        else:
            return "beginner"

    def _validate_completion_criteria(self, step: OnboardingStep, completion_data: Dict[str, Any]) -> bool:
        """Validate that step completion criteria are met."""
        if not step.completion_criteria:
            return True

        # Check completion criteria
        for criterion, expected_value in step.completion_criteria.items():
            actual_value = completion_data.get(criterion)
            if actual_value != expected_value:
                return False

        return True

    def _update_skill_assessment(self, progress: UserProgress) -> Dict[str, Any]:
        """Update user's skill assessment based on progress."""
        completed_count = len(progress.completed_steps)
        total_time = progress.total_time_spent

        # Simple skill progression logic
        if completed_count >= 10:
            level = "advanced"
            confidence = 0.9
        elif completed_count >= 5:
            level = "intermediate"
            confidence = 0.7
        else:
            level = "beginner"
            confidence = 0.5

        return {"level": level, "confidence": confidence}

    def _check_prerequisites(self, user_id: str, prerequisites: List[str]) -> bool:
        """Check if user meets tutorial prerequisites."""
        if not prerequisites:
            return True

        progress = self.user_progress.get(user_id)
        if not progress:
            return False

        return all(prereq in progress.completed_steps for prereq in prerequisites)

    def _is_tutorial_recommended(self, user_id: str, tutorial: Tutorial) -> bool:
        """Check if tutorial is recommended for user."""
        progress = self.user_progress.get(user_id)
        if not progress:
            return tutorial.target_audience == "beginner"

        user_level = progress.skill_assessment.get("level", "beginner")
        return tutorial.target_audience == user_level

    def _calculate_path_duration(self, learning_path: List[str]) -> int:
        """Calculate total duration for a learning path."""
        total_duration = 0
        for step_id in learning_path:
            step = self.onboarding_steps.get(step_id)
            if step:
                total_duration += step.estimated_duration
        return total_duration

class AchievementSystem:
    """Manages user achievements and gamification."""

    def __init__(self):
        self.achievements = {
            "first_task": {
                "name": "First Steps",
                "description": "Created your first task",
                "icon": "🎯",
                "criteria": {"completed_steps": ["create_first_task"]}
            },
            "collaboration_master": {
                "name": "Team Player",
                "description": "Created and managed collaborative sessions",
                "icon": "🤝",
                "criteria": {"completed_steps": ["collaborative_sessions"]}
            },
            "analytics_explorer": {
                "name": "Data Driven",
                "description": "Explored advanced analytics features",
                "icon": "📊",
                "criteria": {"completed_steps": ["advanced_analytics"]}
            },
            "plugin_developer": {
                "name": "Extensibility Expert",
                "description": "Learned about plugin system",
                "icon": "🔌",
                "criteria": {"completed_steps": ["plugin_system"]}
            },
            "speed_learner": {
                "name": "Quick Study",
                "description": "Completed onboarding in record time",
                "icon": "⚡",
                "criteria": {"time_threshold": 30}  # minutes
            }
        }

    def check_achievements(self, user_id: str, progress: UserProgress) -> List[str]:
        """Check for newly unlocked achievements."""
        new_achievements = []

        for achievement_id, achievement in self.achievements.items():
            if achievement_id in progress.achievements:
                continue

            if self._check_achievement_criteria(achievement, progress):
                new_achievements.append(achievement_id)

        return new_achievements

    def _check_achievement_criteria(self, achievement: Dict[str, Any], progress: UserProgress) -> bool:
        """Check if achievement criteria are met."""
        criteria = achievement["criteria"]

        # Check completed steps
        if "completed_steps" in criteria:
            required_steps = criteria["completed_steps"]
            if not all(step in progress.completed_steps for step in required_steps):
                return False

        # Check time threshold
        if "time_threshold" in criteria:
            if progress.total_time_spent > criteria["time_threshold"]:
                return False

        return True

class PersonalizationEngine:
    """Personalizes learning paths based on user characteristics."""

    def __init__(self):
        self.user_learning_paths: Dict[str, List[str]] = {}

    def create_learning_path(self, skill_level: str, user_profile: Dict[str, Any]) -> List[str]:
        """Create a personalized learning path."""
        base_paths = {
            "beginner": ["welcome", "create_first_task", "explore_ai_assistants", "collaborative_sessions"],
            "intermediate": ["collaborative_sessions", "advanced_analytics", "best_practices"],
            "advanced": ["plugin_system", "best_practices"]
        }

        base_path = base_paths.get(skill_level, base_paths["beginner"])

        # Add personalized elements based on profile
        if user_profile.get("interested_in_collaboration"):
            if "collaborative_sessions" not in base_path:
                base_path.append("collaborative_sessions")

        if user_profile.get("interested_in_analytics"):
            if "advanced_analytics" not in base_path:
                base_path.append("advanced_analytics")

        return base_path

    def set_user_learning_path(self, user_id: str, learning_path: List[str]):
        """Set learning path for a user."""
        self.user_learning_paths[user_id] = learning_path

    def get_user_learning_path(self, user_id: str) -> List[str]:
        """Get learning path for a user."""
        return self.user_learning_paths.get(user_id, [])