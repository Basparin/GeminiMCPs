"""
User Feedback Tools for CodeSage MCP Server.

This module provides tools for collecting, analyzing, and managing user feedback
to drive continuous improvement based on user needs and satisfaction.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.user_feedback.user_feedback import (
    get_user_feedback_collector,
    get_feedback_analyzer,
    FeedbackType,
    SatisfactionLevel
)

logger = logging.getLogger(__name__)


def collect_user_feedback_tool(feedback_type: str, title: str, description: str,
                              satisfaction_level: int = None, user_id: str = "anonymous",
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Collect user feedback for analysis and improvement.

    Args:
        feedback_type: Type of feedback ("bug_report", "feature_request", "performance_issue", "usability_feedback", "general_feedback")
        title: Brief title for the feedback
        description: Detailed description of the feedback
        satisfaction_level: User satisfaction level (1-5, where 1=Very Dissatisfied, 5=Very Satisfied)
        user_id: Identifier for the user providing feedback
        metadata: Additional context data about the feedback

    Returns:
        Dictionary containing feedback collection result with feedback ID and status
    """
    try:
        collector = get_user_feedback_collector()

        # Validate feedback type
        try:
            feedback_type_enum = FeedbackType(feedback_type)
        except ValueError:
            return {
                "error": f"Invalid feedback type: {feedback_type}",
                "valid_types": [ft.value for ft in FeedbackType]
            }

        # Validate satisfaction level
        satisfaction_enum = None
        if satisfaction_level is not None:
            try:
                satisfaction_enum = SatisfactionLevel(satisfaction_level)
            except ValueError:
                return {
                    "error": f"Invalid satisfaction level: {satisfaction_level}",
                    "valid_levels": {level.value: level.name for level in SatisfactionLevel}
                }

        # Collect the feedback
        feedback_id = collector.collect_feedback(
            user_id=user_id,
            feedback_type=feedback_type_enum,
            title=title,
            description=description,
            satisfaction_level=satisfaction_enum,
            metadata=metadata or {}
        )

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback collected successfully",
            "feedback_type": feedback_type,
            "user_id": user_id
        }

    except Exception as e:
        logger.error(f"Error collecting user feedback: {e}")
        return {
            "error": f"Failed to collect feedback: {str(e)}",
            "feedback_type": feedback_type,
            "user_id": user_id
        }


def get_feedback_summary_tool(user_id: str = None, feedback_type: str = None) -> Dict[str, Any]:
    """
    Get a summary of user feedback data.

    Args:
        user_id: Optional user ID to filter feedback by specific user
        feedback_type: Optional feedback type to filter by

    Returns:
        Dictionary containing feedback summary statistics and analysis
    """
    try:
        collector = get_user_feedback_collector()

        # Validate feedback type if provided
        feedback_type_enum = None
        if feedback_type:
            try:
                feedback_type_enum = FeedbackType(feedback_type)
            except ValueError:
                return {
                    "error": f"Invalid feedback type: {feedback_type}",
                    "valid_types": [ft.value for ft in FeedbackType]
                }

        summary = collector.get_feedback_summary(
            user_id=user_id,
            feedback_type=feedback_type_enum
        )

        return summary

    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        return {
            "error": f"Failed to get feedback summary: {str(e)}"
        }


def get_user_insights_tool(user_id: str) -> Dict[str, Any]:
    """
    Get insights about a specific user's behavior and satisfaction.

    Args:
        user_id: User ID to get insights for

    Returns:
        Dictionary containing user insights including satisfaction trends,
        engagement level, performance preferences, and usage patterns
    """
    try:
        collector = get_user_feedback_collector()

        insights = collector.get_user_insights(user_id)

        if "error" in insights:
            return insights

        # Add additional insights
        analyzer = get_feedback_analyzer()
        feedback_analysis = analyzer.analyze_feedback_patterns()

        insights["feedback_analysis"] = feedback_analysis

        return insights

    except Exception as e:
        logger.exception(f"Error getting user insights: {e}")
        return {
            "error": f"Failed to get user insights: {str(e)}",
            "user_id": user_id
        }


def analyze_feedback_patterns_tool() -> Dict[str, Any]:
    """
    Analyze patterns in user feedback to identify trends and improvement opportunities.

    Returns:
        Dictionary containing comprehensive feedback pattern analysis including:
        - Feedback distribution by type and priority
        - Satisfaction trends over time
        - Common themes and issues
        - Improvement recommendations
    """
    try:
        analyzer = get_feedback_analyzer()
        analysis = analyzer.analyze_feedback_patterns()

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing feedback patterns: {e}")
        return {
            "error": f"Failed to analyze feedback patterns: {str(e)}"
        }


def get_feedback_driven_recommendations_tool() -> Dict[str, Any]:
    """
    Generate recommendations based on user feedback analysis.

    Returns:
        Dictionary containing prioritized recommendations for product improvement
        based on user feedback patterns, satisfaction trends, and usage data
    """
    try:
        analyzer = get_feedback_analyzer()
        analysis = analyzer.analyze_feedback_patterns()

        # Extract and prioritize recommendations
        recommendations = analysis.get("recommendations", [])

        # Add additional context and prioritization
        for i, rec in enumerate(recommendations):
            rec["id"] = f"rec_{i+1}"
            rec["estimated_effort"] = _estimate_recommendation_effort(rec)
            rec["expected_impact"] = _estimate_recommendation_impact(rec, analysis)
            rec["timeline"] = _estimate_recommendation_timeline(rec)

        # Sort by priority and impact
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: (
            priority_order.get(x.get("priority", "low"), 1),
            x.get("expected_impact", 0)
        ), reverse=True)

        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority_count": sum(1 for r in recommendations if r.get("priority") == "high"),
            "analysis_summary": {
                "feedback_count": analysis.get("feedback_count", 0),
                "analysis_period_days": analysis.get("analysis_period_days", 0),
                "key_themes": list(analysis.get("theme_analysis", {}).keys())
            },
            "generated_at": analysis.get("generated_at")
        }

    except Exception as e:
        logger.error(f"Error generating feedback-driven recommendations: {e}")
        return {
            "error": f"Failed to generate recommendations: {str(e)}"
        }


def _estimate_recommendation_effort(recommendation: Dict[str, Any]) -> str:
    """Estimate the effort required for a recommendation."""
    rec_type = recommendation.get("type", "")

    effort_map = {
        "performance_optimization": "high",
        "bug_fixes": "medium",
        "feature_development": "high",
        "usability_improvement": "medium",
        "security_enhancement": "high",
        "documentation_improvement": "low",
        "monitoring_enhancement": "low"
    }

    return effort_map.get(rec_type, "medium")


def _estimate_recommendation_impact(recommendation: Dict[str, Any], analysis: Dict[str, Any]) -> float:
    """Estimate the impact of implementing a recommendation."""
    rec_type = recommendation.get("type", "")
    priority = recommendation.get("priority", "low")

    # Base impact scores
    base_impact = {
        "performance_optimization": 8.0,
        "bug_fixes": 7.0,
        "feature_development": 6.0,
        "usability_improvement": 5.0,
        "security_enhancement": 9.0,
        "documentation_improvement": 3.0,
        "monitoring_enhancement": 4.0
    }

    impact = base_impact.get(rec_type, 5.0)

    # Adjust based on priority
    if priority == "high":
        impact *= 1.3
    elif priority == "low":
        impact *= 0.8

    # Adjust based on feedback volume
    feedback_count = analysis.get("feedback_count", 0)
    if feedback_count > 50:
        impact *= 1.2
    elif feedback_count < 10:
        impact *= 0.9

    return min(impact, 10.0)


def _estimate_recommendation_timeline(recommendation: Dict[str, Any]) -> str:
    """Estimate the timeline for implementing a recommendation."""
    effort = _estimate_recommendation_effort(recommendation)
    priority = recommendation.get("priority", "low")

    if priority == "high":
        return "1-2 weeks"
    elif effort == "low":
        return "1 week"
    elif effort == "medium":
        return "2-3 weeks"
    else:  # high effort
        return "1-2 months"


def get_user_satisfaction_metrics_tool() -> Dict[str, Any]:
    """
    Get comprehensive user satisfaction metrics and trends.

    Returns:
        Dictionary containing user satisfaction analysis including:
        - Overall satisfaction scores and trends
        - Satisfaction distribution across user segments
        - Key drivers of satisfaction and dissatisfaction
        - Satisfaction correlation with performance metrics
    """
    try:
        collector = get_user_feedback_collector()
        analyzer = get_feedback_analyzer()

        # Get all feedback for satisfaction analysis
        all_feedback = list(collector.feedback_items.values())
        feedback_with_satisfaction = [f for f in all_feedback if f.satisfaction_level]

        if not feedback_with_satisfaction:
            return {
                "message": "No satisfaction data available",
                "total_feedback": len(all_feedback),
                "feedback_with_satisfaction": 0
            }

        # Calculate satisfaction metrics
        satisfaction_scores = [f.satisfaction_level.value for f in feedback_with_satisfaction]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)

        # Satisfaction distribution
        distribution = {}
        for level in SatisfactionLevel:
            count = sum(1 for f in feedback_with_satisfaction if f.satisfaction_level == level)
            distribution[level.name] = {
                "count": count,
                "percentage": (count / len(feedback_with_satisfaction)) * 100
            }

        # Satisfaction by feedback type
        satisfaction_by_type = {}
        for feedback_type in FeedbackType:
            type_feedback = [f for f in feedback_with_satisfaction if f.feedback_type == feedback_type]
            if type_feedback:
                type_scores = [f.satisfaction_level.value for f in type_feedback]
                satisfaction_by_type[feedback_type.value] = {
                    "count": len(type_feedback),
                    "avg_satisfaction": sum(type_scores) / len(type_scores),
                    "min_satisfaction": min(type_scores),
                    "max_satisfaction": max(type_scores)
                }

        # Identify satisfaction drivers
        high_satisfaction_feedback = [f for f in feedback_with_satisfaction if f.satisfaction_level.value >= 4]
        low_satisfaction_feedback = [f for f in feedback_with_satisfaction if f.satisfaction_level.value <= 2]

        satisfaction_drivers = {
            "positive_drivers": _extract_common_themes(high_satisfaction_feedback),
            "negative_drivers": _extract_common_themes(low_satisfaction_feedback)
        }

        return {
            "overall_satisfaction": {
                "average_score": avg_satisfaction,
                "total_responses": len(feedback_with_satisfaction),
                "score_interpretation": _interpret_satisfaction_score(avg_satisfaction)
            },
            "satisfaction_distribution": distribution,
            "satisfaction_by_type": satisfaction_by_type,
            "satisfaction_drivers": satisfaction_drivers,
            "trends": analyzer.analyze_feedback_patterns().get("satisfaction_analysis", {}),
            "generated_at": analyzer.analyze_feedback_patterns().get("generated_at")
        }

    except Exception as e:
        logger.error(f"Error getting user satisfaction metrics: {e}")
        return {
            "error": f"Failed to get satisfaction metrics: {str(e)}"
        }


def _interpret_satisfaction_score(score: float) -> str:
    """Interpret a satisfaction score into a human-readable description."""
    if score >= 4.5:
        return "Excellent - Users are very satisfied"
    elif score >= 4.0:
        return "Good - Users are generally satisfied"
    elif score >= 3.5:
        return "Fair - Users have mixed feelings"
    elif score >= 3.0:
        return "Poor - Users are somewhat dissatisfied"
    else:
        return "Critical - Users are very dissatisfied"


def _extract_common_themes(feedback_list: List) -> List[str]:
    """Extract common themes from feedback descriptions."""
    if not feedback_list:
        return []

    # Simple keyword extraction (could be enhanced with NLP)
    all_text = " ".join([f.title + " " + f.description for f in feedback_list]).lower()

    # Common themes to look for
    themes = {
        "performance": ["slow", "fast", "performance", "speed", "response", "latency"],
        "usability": ["interface", "ui", "ux", "easy", "difficult", "confusing", "intuitive"],
        "features": ["feature", "functionality", "capability", "tool", "missing"],
        "reliability": ["error", "bug", "crash", "fail", "broken", "issue", "stable"],
        "documentation": ["docs", "documentation", "help", "guide", "tutorial"]
    }

    found_themes = []
    for theme, keywords in themes.items():
        if any(keyword in all_text for keyword in keywords):
            found_themes.append(theme)

    return found_themes