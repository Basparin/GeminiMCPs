"""CES Analytics Manager.

Provides comprehensive analytics and insights for the Cognitive Enhancement System,
including usage patterns, performance metrics, user behavior analysis, and predictive analytics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from ..core.logging_config import get_logger

logger = get_logger(__name__)

class AnalyticsManager:
    """Manages analytics and insights for the CES system."""

    def __init__(self):
        self.usage_data = defaultdict(list)
        self.performance_data = defaultdict(list)
        self.user_sessions = {}
        self.task_metrics = defaultdict(list)
        self.ai_interactions = defaultdict(list)

    def is_healthy(self) -> bool:
        """Check if analytics manager is healthy."""
        return True

    async def record_usage_event(self, event_type: str, user_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a usage event for analytics."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "metadata": metadata or {}
        }
        self.usage_data[event_type].append(event)

        # Keep only last 1000 events per type
        if len(self.usage_data[event_type]) > 1000:
            self.usage_data[event_type] = self.usage_data[event_type][-1000:]

    async def record_performance_metric(self, metric_name: str, value: float, user_id: Optional[str] = None):
        """Record a performance metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "user_id": user_id
        }
        self.performance_data[metric_name].append(metric)

        # Keep only last 1000 metrics per type
        if len(self.performance_data[metric_name]) > 1000:
            self.performance_data[metric_name] = self.performance_data[metric_name][-1000:]

    async def record_task_completion(self, task_id: str, user_id: str, duration_ms: float, success: bool):
        """Record task completion metrics."""
        task_metric = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "user_id": user_id,
            "duration_ms": duration_ms,
            "success": success
        }
        self.task_metrics["completions"].append(task_metric)

    async def record_ai_interaction(self, assistant_type: str, user_id: str, response_time_ms: float, success: bool):
        """Record AI assistant interaction metrics."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "assistant_type": assistant_type,
            "user_id": user_id,
            "response_time_ms": response_time_ms,
            "success": success
        }
        self.ai_interactions[assistant_type].append(interaction)

    async def get_overview(self) -> Dict[str, Any]:
        """Get comprehensive analytics overview."""
        try:
            # Calculate time-based metrics (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Usage metrics
            total_events = sum(len(events) for events in self.usage_data.values())
            recent_events = []
            for event_list in self.usage_data.values():
                recent_events.extend([
                    event for event in event_list
                    if datetime.fromisoformat(event["timestamp"]) > cutoff_time
                ])

            # Performance metrics
            avg_response_time = 0
            if self.performance_data.get("response_time_ms"):
                recent_response_times = [
                    metric["value"] for metric in self.performance_data["response_time_ms"]
                    if datetime.fromisoformat(metric["timestamp"]) > cutoff_time
                ]
                if recent_response_times:
                    avg_response_time = sum(recent_response_times) / len(recent_response_times)

            # Task completion metrics
            recent_tasks = [
                task for task in self.task_metrics.get("completions", [])
                if datetime.fromisoformat(task["timestamp"]) > cutoff_time
            ]
            task_success_rate = 0
            if recent_tasks:
                successful_tasks = sum(1 for task in recent_tasks if task["success"])
                task_success_rate = (successful_tasks / len(recent_tasks)) * 100

            # AI interaction metrics
            ai_metrics = {}
            for assistant_type, interactions in self.ai_interactions.items():
                recent_interactions = [
                    interaction for interaction in interactions
                    if datetime.fromisoformat(interaction["timestamp"]) > cutoff_time
                ]
                if recent_interactions:
                    avg_response_time = sum(
                        i["response_time_ms"] for i in recent_interactions
                    ) / len(recent_interactions)
                    success_rate = (
                        sum(1 for i in recent_interactions if i["success"]) /
                        len(recent_interactions)
                    ) * 100
                    ai_metrics[assistant_type] = {
                        "total_interactions": len(recent_interactions),
                        "avg_response_time_ms": round(avg_response_time, 2),
                        "success_rate": round(success_rate, 2)
                    }

            # User engagement metrics
            user_activity = defaultdict(int)
            for event in recent_events:
                user_activity[event["user_id"]] += 1

            top_users = sorted(
                user_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                "summary": {
                    "total_events_24h": len(recent_events),
                    "active_users_24h": len(user_activity),
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "task_success_rate": round(task_success_rate, 2),
                    "total_tasks_24h": len(recent_tasks)
                },
                "performance": {
                    "response_time_trend": self._get_metric_trend("response_time_ms", hours=24),
                    "throughput_trend": self._get_metric_trend("requests_per_second", hours=24),
                    "error_rate_trend": self._get_metric_trend("error_rate_percent", hours=24)
                },
                "ai_assistants": ai_metrics,
                "user_engagement": {
                    "top_users": [{"user_id": uid, "activity_count": count} for uid, count in top_users],
                    "activity_distribution": self._get_activity_distribution(recent_events)
                },
                "tasks": {
                    "completion_trend": self._get_task_completion_trend(hours=24),
                    "success_rate_trend": self._get_task_success_trend(hours=24)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Analytics overview error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary for dashboard display."""
        try:
            overview = await self.get_overview()
            return {
                "total_users": overview["summary"]["active_users_24h"],
                "total_tasks": overview["summary"]["total_tasks_24h"],
                "avg_response_time": overview["summary"]["avg_response_time_ms"],
                "task_success_rate": overview["summary"]["task_success_rate"],
                "ai_assistants_count": len(overview["ai_assistants"]),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Analytics summary error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_metric_trend(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trend data for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics = self.performance_data.get(metric_name, [])

        # Group by hour
        hourly_data = defaultdict(list)
        for metric in metrics:
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time:
                hour = datetime.fromisoformat(metric["timestamp"]).replace(minute=0, second=0, microsecond=0)
                hourly_data[hour].append(metric["value"])

        # Calculate hourly averages
        trend = []
        for hour in sorted(hourly_data.keys()):
            avg_value = sum(hourly_data[hour]) / len(hourly_data[hour])
            trend.append({
                "timestamp": hour.isoformat(),
                "value": round(avg_value, 2)
            })

        return trend

    def _get_task_completion_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get task completion trend."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        tasks = self.task_metrics.get("completions", [])

        # Group by hour
        hourly_data = defaultdict(int)
        for task in tasks:
            if datetime.fromisoformat(task["timestamp"]) > cutoff_time:
                hour = datetime.fromisoformat(task["timestamp"]).replace(minute=0, second=0, microsecond=0)
                hourly_data[hour] += 1

        trend = []
        for hour in sorted(hourly_data.keys()):
            trend.append({
                "timestamp": hour.isoformat(),
                "completions": hourly_data[hour]
            })

        return trend

    def _get_task_success_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get task success rate trend."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        tasks = self.task_metrics.get("completions", [])

        # Group by hour
        hourly_data = defaultdict(lambda: {"total": 0, "successful": 0})
        for task in tasks:
            if datetime.fromisoformat(task["timestamp"]) > cutoff_time:
                hour = datetime.fromisoformat(task["timestamp"]).replace(minute=0, second=0, microsecond=0)
                hourly_data[hour]["total"] += 1
                if task["success"]:
                    hourly_data[hour]["successful"] += 1

        trend = []
        for hour in sorted(hourly_data.keys()):
            data = hourly_data[hour]
            success_rate = (data["successful"] / data["total"] * 100) if data["total"] > 0 else 0
            trend.append({
                "timestamp": hour.isoformat(),
                "success_rate": round(success_rate, 2),
                "total_tasks": data["total"]
            })

        return trend

    def _get_activity_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get activity distribution by event type."""
        event_types = [event["event_type"] for event in events]
        return dict(Counter(event_types))

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights for a specific user."""
        try:
            # Get user's events
            user_events = []
            for event_list in self.usage_data.values():
                user_events.extend([
                    event for event in event_list
                    if event["user_id"] == user_id
                ])

            # Get user's tasks
            user_tasks = [
                task for task in self.task_metrics.get("completions", [])
                if task["user_id"] == user_id
            ]

            # Get user's AI interactions
            user_ai_interactions = []
            for assistant_type, interactions in self.ai_interactions.items():
                user_ai_interactions.extend([
                    interaction for interaction in interactions
                    if interaction["user_id"] == user_id
                ])

            # Calculate metrics
            total_events = len(user_events)
            total_tasks = len(user_tasks)
            successful_tasks = sum(1 for task in user_tasks if task["success"])
            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

            # Activity by type
            activity_by_type = defaultdict(int)
            for event in user_events:
                activity_by_type[event["event_type"]] += 1

            # AI assistant preferences
            ai_preferences = defaultdict(int)
            for interaction in user_ai_interactions:
                ai_preferences[interaction["assistant_type"]] += 1

            return {
                "user_id": user_id,
                "metrics": {
                    "total_events": total_events,
                    "total_tasks": total_tasks,
                    "task_success_rate": round(success_rate, 2),
                    "avg_task_duration_ms": round(
                        sum(task["duration_ms"] for task in user_tasks) / len(user_tasks)
                        if user_tasks else 0, 2
                    )
                },
                "activity_breakdown": dict(activity_by_type),
                "ai_preferences": dict(ai_preferences),
                "recent_activity": user_events[-10:],  # Last 10 events
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"User insights error for {user_id}: {e}")
            return {"error": str(e), "user_id": user_id, "timestamp": datetime.now().isoformat()}