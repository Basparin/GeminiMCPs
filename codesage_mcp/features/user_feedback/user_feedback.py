"""
User Feedback Collection and Analysis Module for CodeSage MCP Server.

This module provides comprehensive user feedback collection, analysis, and integration
capabilities to drive continuous improvement based on user needs and satisfaction.

Classes:
    UserFeedbackCollector: Collects and manages user feedback
    FeedbackAnalyzer: Analyzes feedback patterns and generates insights
    UserSatisfactionTracker: Tracks user satisfaction metrics
    FeedbackIntegrationManager: Integrates feedback with development processes
"""

import logging
import json
import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import statistics
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    PERFORMANCE_ISSUE = "performance_issue"
    USABILITY_FEEDBACK = "usability_feedback"
    GENERAL_FEEDBACK = "general_feedback"


class SatisfactionLevel(Enum):
    """User satisfaction levels."""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5


@dataclass
class UserFeedback:
    """Represents a single piece of user feedback."""
    feedback_id: str
    user_id: str
    feedback_type: FeedbackType
    satisfaction_level: Optional[SatisfactionLevel]
    title: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    priority_score: float = 0.0


@dataclass
class UserProfile:
    """Represents a user profile with usage patterns and preferences."""
    user_id: str
    first_seen: float
    last_seen: float
    total_sessions: int
    total_requests: int
    favorite_tools: List[str]
    common_workflows: List[str]
    satisfaction_history: List[SatisfactionLevel]
    feedback_count: int
    avg_response_time_preference: float
    performance_sensitivity: str  # "high", "medium", "low"


class UserFeedbackCollector:
    """Collects and manages user feedback from various sources."""

    def __init__(self, storage_path: str = ".codesage/user_feedback.json"):
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self.feedback_items: Dict[str, UserFeedback] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.feedback_callbacks: List[Callable] = []

        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Load existing feedback
        self._load_feedback()

    def collect_feedback(self, user_id: str, feedback_type: FeedbackType,
                        title: str, description: str,
                        satisfaction_level: Optional[SatisfactionLevel] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Collect a new piece of user feedback."""
        with self._lock:
            feedback_id = f"{user_id}_{int(time.time())}_{hash(title + description) % 10000}"

            feedback = UserFeedback(
                feedback_id=feedback_id,
                user_id=user_id,
                feedback_type=feedback_type,
                satisfaction_level=satisfaction_level,
                title=title,
                description=description,
                metadata=metadata or {},
                priority_score=self._calculate_priority_score(feedback_type, satisfaction_level, metadata)
            )

            self.feedback_items[feedback_id] = feedback

            # Update user profile
            self._update_user_profile(user_id, feedback)

            # Trigger callbacks
            for callback in self.feedback_callbacks:
                try:
                    callback(feedback)
                except Exception as e:
                    logger.error(f"Error in feedback callback: {e}")

            # Save to storage
            self._save_feedback()

            logger.info(f"Collected feedback from user {user_id}: {title}")
            return feedback_id

    def collect_implicit_feedback(self, user_id: str, action: str,
                                 response_time_ms: float, success: bool,
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect implicit feedback from user interactions."""
        with self._lock:
            # Analyze response time for performance feedback
            if response_time_ms > 1000:  # Slow response
                self.collect_feedback(
                    user_id=user_id,
                    feedback_type=FeedbackType.PERFORMANCE_ISSUE,
                    title="Slow Response Time",
                    description=f"Response time of {response_time_ms:.0f}ms for action: {action}",
                    satisfaction_level=SatisfactionLevel.DISSATISFIED,
                    metadata={
                        "action": action,
                        "response_time_ms": response_time_ms,
                        "success": success,
                        "implicit": True,
                        **(metadata or {})
                    }
                )
            elif not success:
                # Collect error feedback
                self.collect_feedback(
                    user_id=user_id,
                    feedback_type=FeedbackType.BUG_REPORT,
                    title="Operation Failed",
                    description=f"Failed to execute action: {action}",
                    satisfaction_level=SatisfactionLevel.DISSATISFIED,
                    metadata={
                        "action": action,
                        "response_time_ms": response_time_ms,
                        "success": success,
                        "implicit": True,
                        **(metadata or {})
                    }
                )

            # Update user profile with interaction data
            self._update_user_profile_interaction(user_id, action, response_time_ms, success)

    def _calculate_priority_score(self, feedback_type: FeedbackType,
                                satisfaction_level: Optional[SatisfactionLevel],
                                metadata: Dict[str, Any]) -> float:
        """Calculate priority score for feedback item."""
        score = 0.0

        # Base score by feedback type
        type_scores = {
            FeedbackType.BUG_REPORT: 8.0,
            FeedbackType.PERFORMANCE_ISSUE: 7.0,
            FeedbackType.FEATURE_REQUEST: 5.0,
            FeedbackType.USABILITY_FEEDBACK: 4.0,
            FeedbackType.GENERAL_FEEDBACK: 3.0
        }
        score += type_scores.get(feedback_type, 3.0)

        # Adjust by satisfaction level
        if satisfaction_level:
            if satisfaction_level in [SatisfactionLevel.VERY_DISSATISFIED, SatisfactionLevel.DISSATISFIED]:
                score += 3.0
            elif satisfaction_level == SatisfactionLevel.NEUTRAL:
                score += 1.0

        # Adjust by metadata indicators
        if metadata.get("blocking", False):
            score += 2.0
        if metadata.get("affects_multiple_users", False):
            score += 1.5
        if metadata.get("urgent", False):
            score += 2.5

        return min(score, 10.0)  # Cap at 10.0

    def _update_user_profile(self, user_id: str, feedback: UserFeedback) -> None:
        """Update user profile with feedback data."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                first_seen=time.time(),
                last_seen=time.time(),
                total_sessions=0,
                total_requests=0,
                favorite_tools=[],
                common_workflows=[],
                satisfaction_history=[],
                feedback_count=0,
                avg_response_time_preference=500,  # Default 500ms
                performance_sensitivity="medium"
            )

        profile = self.user_profiles[user_id]
        profile.last_seen = time.time()
        profile.feedback_count += 1

        if feedback.satisfaction_level:
            profile.satisfaction_history.append(feedback.satisfaction_level)

            # Keep only recent satisfaction scores
            if len(profile.satisfaction_history) > 20:
                profile.satisfaction_history = profile.satisfaction_history[-20:]

    def _update_user_profile_interaction(self, user_id: str, action: str,
                                       response_time_ms: float, success: bool) -> None:
        """Update user profile with interaction data."""
        if user_id not in self.user_profiles:
            return

        profile = self.user_profiles[user_id]
        profile.total_requests += 1

        # Update performance sensitivity based on response time
        if response_time_ms > 2000:
            profile.performance_sensitivity = "high"
        elif response_time_ms > 1000:
            profile.performance_sensitivity = "medium"
        else:
            profile.performance_sensitivity = "low"

        # Update average response time preference
        profile.avg_response_time_preference = (
            (profile.avg_response_time_preference * (profile.total_requests - 1)) +
            response_time_ms
        ) / profile.total_requests

    def get_feedback_summary(self, user_id: Optional[str] = None,
                           feedback_type: Optional[FeedbackType] = None) -> Dict[str, Any]:
        """Get summary of feedback data."""
        with self._lock:
            feedback_items = list(self.feedback_items.values())

            # Filter by user
            if user_id:
                feedback_items = [f for f in feedback_items if f.user_id == user_id]

            # Filter by type
            if feedback_type:
                feedback_items = [f for f in feedback_items if f.feedback_type == feedback_type]

            if not feedback_items:
                return {"total_feedback": 0, "summary": "No feedback available"}

            # Calculate statistics
            total_feedback = len(feedback_items)
            resolved_count = sum(1 for f in feedback_items if f.resolved)
            avg_priority = statistics.mean(f.priority_score for f in feedback_items)

            # Satisfaction distribution
            satisfaction_counts = defaultdict(int)
            for feedback in feedback_items:
                if feedback.satisfaction_level:
                    satisfaction_counts[feedback.satisfaction_level.value] += 1

            # Feedback type distribution
            type_counts = defaultdict(int)
            for feedback in feedback_items:
                type_counts[feedback.feedback_type.value] += 1

            # Recent feedback (last 7 days)
            week_ago = time.time() - (7 * 24 * 60 * 60)
            recent_feedback = [f for f in feedback_items if f.timestamp > week_ago]

            return {
                "total_feedback": total_feedback,
                "resolved_count": resolved_count,
                "unresolved_count": total_feedback - resolved_count,
                "average_priority": avg_priority,
                "satisfaction_distribution": dict(satisfaction_counts),
                "feedback_type_distribution": dict(type_counts),
                "recent_feedback_count": len(recent_feedback),
                "resolution_rate": (resolved_count / total_feedback) * 100 if total_feedback > 0 else 0
            }

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about a specific user."""
        with self._lock:
            if user_id not in self.user_profiles:
                return {"error": "User profile not found"}

            profile = self.user_profiles[user_id]

            # Calculate satisfaction trend
            satisfaction_scores = [s.value for s in profile.satisfaction_history[-10:]]
            satisfaction_trend = "stable"
            if len(satisfaction_scores) >= 2:
                first_half = satisfaction_scores[:len(satisfaction_scores)//2]
                second_half = satisfaction_scores[len(satisfaction_scores)//2:]
                if statistics.mean(second_half) > statistics.mean(first_half) + 0.5:
                    satisfaction_trend = "improving"
                elif statistics.mean(second_half) < statistics.mean(first_half) - 0.5:
                    satisfaction_trend = "declining"

            # Calculate engagement level
            days_active = (profile.last_seen - profile.first_seen) / (24 * 60 * 60)
            engagement_level = "low"
            if profile.total_requests > 100:
                engagement_level = "high"
            elif profile.total_requests > 50:
                engagement_level = "medium"

            return {
                "user_id": user_id,
                "days_active": days_active,
                "total_requests": profile.total_requests,
                "feedback_count": profile.feedback_count,
                "avg_satisfaction": statistics.mean([s.value for s in profile.satisfaction_history]) if profile.satisfaction_history else None,
                "satisfaction_trend": satisfaction_trend,
                "performance_sensitivity": profile.performance_sensitivity,
                "engagement_level": engagement_level,
                "avg_response_time_preference": profile.avg_response_time_preference
            }

    def add_feedback_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when feedback is collected."""
        self.feedback_callbacks.append(callback)

    def _save_feedback(self) -> None:
        """Save feedback data to storage."""
        try:
            data = {
                "feedback_items": {
                    fid: {
                        "feedback_id": f.feedback_id,
                        "user_id": f.user_id,
                        "feedback_type": f.feedback_type.value,
                        "satisfaction_level": f.satisfaction_level.value if f.satisfaction_level else None,
                        "title": f.title,
                        "description": f.description,
                        "metadata": f.metadata,
                        "timestamp": f.timestamp,
                        "resolved": f.resolved,
                        "resolution_notes": f.resolution_notes,
                        "priority_score": f.priority_score
                    }
                    for fid, f in self.feedback_items.items()
                },
                "user_profiles": {
                    uid: {
                        "user_id": p.user_id,
                        "first_seen": p.first_seen,
                        "last_seen": p.last_seen,
                        "total_sessions": p.total_sessions,
                        "total_requests": p.total_requests,
                        "favorite_tools": p.favorite_tools,
                        "common_workflows": p.common_workflows,
                        "satisfaction_history": [s.value for s in p.satisfaction_history],
                        "feedback_count": p.feedback_count,
                        "avg_response_time_preference": p.avg_response_time_preference,
                        "performance_sensitivity": p.performance_sensitivity
                    }
                    for uid, p in self.user_profiles.items()
                }
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")

    def _load_feedback(self) -> None:
        """Load feedback data from storage."""
        try:
            if not os.path.exists(self.storage_path):
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load feedback items
            for fid, f_data in data.get("feedback_items", {}).items():
                feedback = UserFeedback(
                    feedback_id=f_data["feedback_id"],
                    user_id=f_data["user_id"],
                    feedback_type=FeedbackType(f_data["feedback_type"]),
                    satisfaction_level=SatisfactionLevel(f_data["satisfaction_level"]) if f_data["satisfaction_level"] else None,
                    title=f_data["title"],
                    description=f_data["description"],
                    metadata=f_data["metadata"],
                    timestamp=f_data["timestamp"],
                    resolved=f_data["resolved"],
                    resolution_notes=f_data["resolution_notes"],
                    priority_score=f_data["priority_score"]
                )
                self.feedback_items[fid] = feedback

            # Load user profiles
            for uid, p_data in data.get("user_profiles", {}).items():
                profile = UserProfile(
                    user_id=p_data["user_id"],
                    first_seen=p_data["first_seen"],
                    last_seen=p_data["last_seen"],
                    total_sessions=p_data["total_sessions"],
                    total_requests=p_data["total_requests"],
                    favorite_tools=p_data["favorite_tools"],
                    common_workflows=p_data["common_workflows"],
                    satisfaction_history=[SatisfactionLevel(s) for s in p_data["satisfaction_history"]],
                    feedback_count=p_data["feedback_count"],
                    avg_response_time_preference=p_data["avg_response_time_preference"],
                    performance_sensitivity=p_data["performance_sensitivity"]
                )
                self.user_profiles[uid] = profile

        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")


class FeedbackAnalyzer:
    """Analyzes feedback patterns and generates insights."""

    def __init__(self, feedback_collector: UserFeedbackCollector):
        self.feedback_collector = feedback_collector
        self._lock = threading.RLock()

    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in user feedback."""
        with self._lock:
            feedback_items = list(self.feedback_collector.feedback_items.values())

            if not feedback_items:
                return {"analysis": "No feedback data available"}

            # Analyze feedback by type
            type_analysis = self._analyze_feedback_by_type(feedback_items)

            # Analyze satisfaction trends
            satisfaction_analysis = self._analyze_satisfaction_trends(feedback_items)

            # Identify common themes
            theme_analysis = self._identify_common_themes(feedback_items)

            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(
                type_analysis, satisfaction_analysis, theme_analysis
            )

            return {
                "feedback_count": len(feedback_items),
                "analysis_period_days": self._calculate_analysis_period(feedback_items),
                "type_analysis": type_analysis,
                "satisfaction_analysis": satisfaction_analysis,
                "theme_analysis": theme_analysis,
                "recommendations": recommendations,
                "generated_at": time.time()
            }

    def _analyze_feedback_by_type(self, feedback_items: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze feedback distribution by type."""
        type_counts = defaultdict(int)
        type_priorities = defaultdict(list)

        for feedback in feedback_items:
            type_counts[feedback.feedback_type.value] += 1
            type_priorities[feedback.feedback_type.value].append(feedback.priority_score)

        analysis = {}
        for feedback_type, count in type_counts.items():
            priorities = type_priorities[feedback_type]
            analysis[feedback_type] = {
                "count": count,
                "percentage": (count / len(feedback_items)) * 100,
                "avg_priority": statistics.mean(priorities),
                "max_priority": max(priorities),
                "min_priority": min(priorities)
            }

        return analysis

    def _analyze_satisfaction_trends(self, feedback_items: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze satisfaction trends over time."""
        # Group feedback by time periods (weeks)
        weekly_satisfaction = defaultdict(list)

        for feedback in feedback_items:
            if feedback.satisfaction_level:
                week_start = feedback.timestamp - (feedback.timestamp % (7 * 24 * 60 * 60))
                weekly_satisfaction[week_start].append(feedback.satisfaction_level.value)

        # Calculate weekly averages
        weekly_averages = {}
        for week, scores in weekly_satisfaction.items():
            weekly_averages[week] = statistics.mean(scores)

        # Calculate trend
        trend = "stable"
        if len(weekly_averages) >= 2:
            weeks = sorted(weekly_averages.keys())
            first_half = [weekly_averages[w] for w in weeks[:len(weeks)//2]]
            second_half = [weekly_averages[w] for w in weeks[len(weeks)//2:]]

            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)

                if second_avg > first_avg + 0.2:
                    trend = "improving"
                elif second_avg < first_avg - 0.2:
                    trend = "declining"

        return {
            "overall_average": statistics.mean([f.satisfaction_level.value for f in feedback_items if f.satisfaction_level]),
            "trend": trend,
            "weekly_averages": weekly_averages,
            "satisfaction_distribution": self._calculate_satisfaction_distribution(feedback_items)
        }

    def _calculate_satisfaction_distribution(self, feedback_items: List[UserFeedback]) -> Dict[str, int]:
        """Calculate satisfaction level distribution."""
        distribution = defaultdict(int)
        for feedback in feedback_items:
            if feedback.satisfaction_level:
                distribution[feedback.satisfaction_level.name] += 1
        return dict(distribution)

    def _identify_common_themes(self, feedback_items: List[UserFeedback]) -> Dict[str, Any]:
        """Identify common themes in feedback."""
        # Simple keyword-based theme identification
        themes = defaultdict(list)

        performance_keywords = ["slow", "fast", "performance", "speed", "response", "latency"]
        usability_keywords = ["interface", "ui", "ux", "easy", "difficult", "confusing", "intuitive"]
        feature_keywords = ["feature", "functionality", "capability", "tool", "missing"]
        bug_keywords = ["error", "bug", "crash", "fail", "broken", "issue"]

        for feedback in feedback_items:
            text = (feedback.title + " " + feedback.description).lower()

            if any(keyword in text for keyword in performance_keywords):
                themes["performance"].append(feedback)
            if any(keyword in text for keyword in usability_keywords):
                themes["usability"].append(feedback)
            if any(keyword in text for keyword in feature_keywords):
                themes["features"].append(feedback)
            if any(keyword in text for keyword in bug_keywords):
                themes["bugs"].append(feedback)

        theme_analysis = {}
        for theme, feedback_list in themes.items():
            theme_analysis[theme] = {
                "count": len(feedback_list),
                "avg_priority": statistics.mean([f.priority_score for f in feedback_list]),
                "satisfaction_impact": self._calculate_theme_satisfaction_impact(feedback_list)
            }

        return theme_analysis

    def _calculate_theme_satisfaction_impact(self, feedback_list: List[UserFeedback]) -> float:
        """Calculate how much a theme impacts user satisfaction."""
        satisfaction_scores = []
        for feedback in feedback_list:
            if feedback.satisfaction_level:
                # Adjust score based on priority (higher priority = more impact)
                adjusted_score = feedback.satisfaction_level.value * (feedback.priority_score / 5.0)
                satisfaction_scores.append(adjusted_score)

        return statistics.mean(satisfaction_scores) if satisfaction_scores else 3.0

    def _generate_improvement_recommendations(self, type_analysis: Dict,
                                           satisfaction_analysis: Dict,
                                           theme_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Performance recommendations
        if "performance_issue" in type_analysis:
            perf_data = type_analysis["performance_issue"]
            if perf_data["avg_priority"] > 6.0:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "description": f"High-priority performance issues found ({perf_data['count']} reports)",
                    "impact": "High user dissatisfaction due to slow response times",
                    "recommendation": "Prioritize performance optimization in next sprint"
                })

        # Feature recommendations
        if "feature_request" in type_analysis:
            feature_data = type_analysis["feature_request"]
            if feature_data["count"] > 5:
                recommendations.append({
                    "type": "feature_development",
                    "priority": "medium",
                    "description": f"Multiple feature requests ({feature_data['count']} total)",
                    "impact": "Users requesting additional functionality",
                    "recommendation": "Analyze feature requests for product roadmap planning"
                })

        # Usability recommendations
        if "usability" in theme_analysis:
            usability_data = theme_analysis["usability"]
            if usability_data["satisfaction_impact"] < 3.0:
                recommendations.append({
                    "type": "usability_improvement",
                    "priority": "medium",
                    "description": "Usability issues impacting user satisfaction",
                    "impact": f"Average satisfaction score: {usability_data['satisfaction_impact']:.1f}/5",
                    "recommendation": "Conduct usability testing and iterate on interface design"
                })

        # Bug fix recommendations
        if "bug_report" in type_analysis:
            bug_data = type_analysis["bug_report"]
            if bug_data["count"] > 3:
                recommendations.append({
                    "type": "bug_fixes",
                    "priority": "high",
                    "description": f"Multiple bug reports ({bug_data['count']} total)",
                    "impact": "Users experiencing technical issues",
                    "recommendation": "Prioritize critical bug fixes in next release"
                })

        return recommendations

    def _calculate_analysis_period(self, feedback_items: List[UserFeedback]) -> float:
        """Calculate the time period covered by the analysis."""
        if not feedback_items:
            return 0

        timestamps = [f.timestamp for f in feedback_items]
        return (max(timestamps) - min(timestamps)) / (24 * 60 * 60)  # Days


# Global instances
_user_feedback_collector: Optional[UserFeedbackCollector] = None
_feedback_analyzer: Optional[FeedbackAnalyzer] = None


def get_user_feedback_collector() -> UserFeedbackCollector:
    """Get the global user feedback collector instance."""
    global _user_feedback_collector
    if _user_feedback_collector is None:
        _user_feedback_collector = UserFeedbackCollector()
    return _user_feedback_collector


def get_feedback_analyzer() -> FeedbackAnalyzer:
    """Get the global feedback analyzer instance."""
    global _feedback_analyzer, _user_feedback_collector
    if _user_feedback_collector is None:
        _user_feedback_collector = UserFeedbackCollector()
    if _feedback_analyzer is None:
        _feedback_analyzer = FeedbackAnalyzer(_user_feedback_collector)
    return _feedback_analyzer