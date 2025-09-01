"""
CES Analytics Engine

Advanced analytics and insights system for tracking usage patterns,
performance metrics, and generating actionable insights.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import os
from pathlib import Path


class AnalyticsEngine:
    """
    Comprehensive analytics engine for CES usage tracking and insights
    """

    def __init__(self, storage_path: str = "./data/analytics"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Analytics data storage
        self.usage_events: List[Dict[str, Any]] = []
        self.performance_metrics: List[Dict[str, Any]] = []
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.task_analytics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Real-time metrics
        self.active_users = 0
        self.current_tasks = 0
        self.system_load = 0.0

        # Initialize storage
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the engine is initialized"""
        if not self._initialized:
            await self._initialize_storage()
            self._initialized = True

    async def _initialize_storage(self):
        """Initialize analytics storage"""
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing analytics data
        await self._load_usage_events()
        await self._load_performance_metrics()
        await self._load_user_sessions()

    async def _load_usage_events(self):
        """Load usage events from storage"""
        events_file = f"{self.storage_path}/usage_events.json"
        if os.path.exists(events_file):
            try:
                async with open(events_file, 'r') as f:
                    self.usage_events = json.loads(await f.read())
                self.logger.info(f"Loaded {len(self.usage_events)} usage events")
            except Exception as e:
                self.logger.error(f"Error loading usage events: {e}")

    async def _load_performance_metrics(self):
        """Load performance metrics from storage"""
        metrics_file = f"{self.storage_path}/performance_metrics.json"
        if os.path.exists(metrics_file):
            try:
                async with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.loads(await f.read())
                self.logger.info(f"Loaded {len(self.performance_metrics)} performance metrics")
            except Exception as e:
                self.logger.error(f"Error loading performance metrics: {e}")

    async def _load_user_sessions(self):
        """Load user sessions from storage"""
        sessions_file = f"{self.storage_path}/user_sessions.json"
        if os.path.exists(sessions_file):
            try:
                async with open(sessions_file, 'r') as f:
                    self.user_sessions = json.loads(await f.read())
                self.logger.info(f"Loaded {len(self.user_sessions)} user sessions")
            except Exception as e:
                self.logger.error(f"Error loading user sessions: {e}")

    async def _save_usage_events(self):
        """Save usage events to storage"""
        await self.ensure_initialized()
        try:
            async with open(f"{self.storage_path}/usage_events.json", 'w') as f:
                await f.write(json.dumps(self.usage_events[-10000:], indent=2))  # Keep last 10k events
        except Exception as e:
            self.logger.error(f"Error saving usage events: {e}")

    async def _save_performance_metrics(self):
        """Save performance metrics to storage"""
        await self.ensure_initialized()
        try:
            async with open(f"{self.storage_path}/performance_metrics.json", 'w') as f:
                await f.write(json.dumps(self.performance_metrics[-5000:], indent=2))  # Keep last 5k metrics
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")

    async def _save_user_sessions(self):
        """Save user sessions to storage"""
        await self.ensure_initialized()
        try:
            async with open(f"{self.storage_path}/user_sessions.json", 'w') as f:
                await f.write(json.dumps(self.user_sessions, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving user sessions: {e}")

    def track_usage_event(self, event_type: str, user_id: str = "anonymous",
                         session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Track a usage event for analytics

        Args:
            event_type: Type of event (task_created, task_completed, ai_interaction, etc.)
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional event metadata
        """
        event = {
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.usage_events.append(event)

        # Update real-time metrics
        if event_type == "user_login":
            self.active_users += 1
        elif event_type == "user_logout":
            self.active_users = max(0, self.active_users - 1)
        elif event_type == "task_started":
            self.current_tasks += 1
        elif event_type == "task_completed":
            self.current_tasks = max(0, self.current_tasks - 1)

        # Save periodically (every 100 events)
        if len(self.usage_events) % 100 == 0:
            asyncio.create_task(self._save_usage_events())

    def track_performance_metric(self, metric_type: str, value: float,
                                component: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Track a performance metric

        Args:
            metric_type: Type of metric (response_time, memory_usage, cpu_usage, etc.)
            value: Metric value
            component: Component that generated the metric
            metadata: Additional metric metadata
        """
        metric = {
            "metric_type": metric_type,
            "value": value,
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.performance_metrics.append(metric)

        # Update system load if it's a load metric
        if metric_type == "system_load":
            self.system_load = value

        # Save periodically (every 50 metrics)
        if len(self.performance_metrics) % 50 == 0:
            asyncio.create_task(self._save_performance_metrics())

    def track_task_execution(self, task_id: str, task_type: str, assistant_used: str,
                           execution_time: float, success: bool, user_id: str = "anonymous"):
        """
        Track task execution for analytics

        Args:
            task_id: Unique task identifier
            task_type: Type of task (code_generation, analysis, etc.)
            assistant_used: AI assistant that handled the task
            execution_time: Time taken to execute the task
            success: Whether the task was successful
            user_id: User who initiated the task
        """
        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "assistant_used": assistant_used,
            "execution_time": execution_time,
            "success": success,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

        self.task_analytics[task_type].append(task_data)

        # Also track as usage event
        self.track_usage_event(
            "task_execution",
            user_id=user_id,
            metadata={
                "task_type": task_type,
                "assistant_used": assistant_used,
                "execution_time": execution_time,
                "success": success
            }
        )

    def start_user_session(self, user_id: str, session_type: str = "interactive"):
        """
        Start tracking a user session

        Args:
            user_id: User identifier
            session_type: Type of session (interactive, api, background)
        """
        session = {
            "user_id": user_id,
            "session_type": session_type,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "events_count": 0,
            "tasks_completed": 0,
            "active": True
        }

        self.user_sessions[user_id] = session
        self.track_usage_event("user_login", user_id=user_id, metadata={"session_type": session_type})
        asyncio.create_task(self._save_user_sessions())

    def end_user_session(self, user_id: str):
        """
        End tracking a user session

        Args:
            user_id: User identifier
        """
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            end_time = datetime.now()
            start_time = datetime.fromisoformat(session["start_time"])

            session["end_time"] = end_time.isoformat()
            session["duration"] = (end_time - start_time).total_seconds()
            session["active"] = False

            self.track_usage_event("user_logout", user_id=user_id,
                                 metadata={"session_duration": session["duration"]})
            asyncio.create_task(self._save_user_sessions())

    def generate_usage_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive usage report

        Args:
            days: Number of days to include in the report

        Returns:
            Usage analytics report
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter events within the time range
        recent_events = [
            event for event in self.usage_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_date
        ]

        # Calculate basic metrics
        total_events = len(recent_events)
        unique_users = len(set(event["user_id"] for event in recent_events))

        # Event type distribution
        event_types = Counter(event["event_type"] for event in recent_events)

        # User activity
        user_activity = Counter(event["user_id"] for event in recent_events)

        # Task execution analysis
        task_events = [event for event in recent_events if event["event_type"] == "task_execution"]
        task_success_rate = 0.0
        if task_events:
            successful_tasks = sum(1 for event in task_events if event["metadata"].get("success", False))
            task_success_rate = successful_tasks / len(task_events)

        # Performance metrics analysis
        recent_metrics = [
            metric for metric in self.performance_metrics
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_date
        ]

        performance_summary = {}
        if recent_metrics:
            for metric_type in set(metric["metric_type"] for metric in recent_metrics):
                type_metrics = [m for m in recent_metrics if m["metric_type"] == metric_type]
                values = [m["value"] for m in type_metrics]
                performance_summary[metric_type] = {
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return {
            "report_period_days": days,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_events": total_events,
                "unique_users": unique_users,
                "average_events_per_user": total_events / max(1, unique_users),
                "task_success_rate": task_success_rate
            },
            "event_distribution": dict(event_types),
            "top_users": dict(user_activity.most_common(10)),
            "performance_metrics": performance_summary,
            "insights": self._generate_insights(recent_events, recent_metrics)
        }

    def generate_task_analytics_report(self) -> Dict[str, Any]:
        """
        Generate detailed task analytics report

        Returns:
            Task analytics report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "task_types": {},
            "assistant_performance": {},
            "trends": {}
        }

        # Analyze each task type
        for task_type, tasks in self.task_analytics.items():
            if not tasks:
                continue

            execution_times = [task["execution_time"] for task in tasks]
            success_count = sum(1 for task in tasks if task["success"])

            report["task_types"][task_type] = {
                "total_tasks": len(tasks),
                "success_rate": success_count / len(tasks),
                "average_execution_time": statistics.mean(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "assistants_used": list(set(task["assistant_used"] for task in tasks))
            }

        # Assistant performance across all tasks
        all_tasks = []
        for tasks in self.task_analytics.values():
            all_tasks.extend(tasks)

        if all_tasks:
            assistant_stats = defaultdict(lambda: {"total": 0, "successful": 0, "total_time": 0})

            for task in all_tasks:
                assistant = task["assistant_used"]
                assistant_stats[assistant]["total"] += 1
                assistant_stats[assistant]["total_time"] += task["execution_time"]
                if task["success"]:
                    assistant_stats[assistant]["successful"] += 1

            for assistant, stats in assistant_stats.items():
                report["assistant_performance"][assistant] = {
                    "total_tasks": stats["total"],
                    "success_rate": stats["successful"] / stats["total"],
                    "average_execution_time": stats["total_time"] / stats["total"]
                }

        return report

    def _generate_insights(self, events: List[Dict[str, Any]],
                          metrics: List[Dict[str, Any]]) -> List[str]:
        """
        Generate actionable insights from usage data

        Args:
            events: Recent usage events
            metrics: Recent performance metrics

        Returns:
            List of insights and recommendations
        """
        insights = []

        # Analyze user engagement
        event_types = Counter(event["event_type"] for event in events)
        total_events = len(events)

        if total_events > 0:
            # Task completion insights
            task_events = event_types.get("task_execution", 0)
            if task_events > 0:
                task_success_rate = sum(1 for event in events
                                      if event["event_type"] == "task_execution"
                                      and event["metadata"].get("success", False)) / task_events

                if task_success_rate < 0.8:
                    insights.append(f"Task success rate is {task_success_rate:.1%}, consider reviewing assistant selection logic")
                elif task_success_rate > 0.95:
                    insights.append(f"Excellent task success rate of {task_success_rate:.1%} indicates good assistant matching")

            # User activity insights
            user_activity = Counter(event["user_id"] for event in events)
            avg_events_per_user = total_events / len(user_activity)

            if avg_events_per_user < 5:
                insights.append("Low user engagement detected, consider improving user experience")
            elif avg_events_per_user > 20:
                insights.append("High user engagement indicates good system usability")

        # Performance insights
        if metrics:
            response_times = [m["value"] for m in metrics if m["metric_type"] == "response_time"]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                if avg_response_time > 5.0:
                    insights.append(f"Average response time of {avg_response_time:.2f}s is high, consider optimization")
                elif avg_response_time < 1.0:
                    insights.append(f"Excellent response time of {avg_response_time:.2f}s indicates good performance")
        # System load insights
        if self.system_load > 0.8:
            insights.append(f"High system load of {self.system_load:.1%}, consider scaling resources")
        elif self.system_load < 0.3:
            insights.append(f"Low system utilization of {self.system_load:.1%}, resources may be underutilized")
        # Default insights if none generated
        if not insights:
            insights.append("System operating within normal parameters")
            insights.append("Continue monitoring for optimization opportunities")

        return insights

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get current real-time system metrics

        Returns:
            Real-time metrics
        """
        return {
            "active_users": self.active_users,
            "current_tasks": self.current_tasks,
            "system_load": self.system_load,
            "total_events_today": len([
                event for event in self.usage_events
                if datetime.fromisoformat(event["timestamp"]).date() == datetime.now().date()
            ]),
            "timestamp": datetime.now().isoformat()
        }

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific user

        Args:
            user_id: User identifier

        Returns:
            User analytics data
        """
        user_events = [event for event in self.usage_events if event["user_id"] == user_id]

        if not user_events:
            return {"user_id": user_id, "message": "No data available for user"}

        # Calculate user metrics
        first_event = min(user_events, key=lambda x: x["timestamp"])
        last_event = max(user_events, key=lambda x: x["timestamp"])

        event_types = Counter(event["event_type"] for event in user_events)
        task_events = [event for event in user_events if event["event_type"] == "task_execution"]

        total_tasks = len(task_events)
        successful_tasks = sum(1 for event in task_events if event["metadata"].get("success", False))

        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "total_tasks": total_tasks,
            "task_success_rate": successful_tasks / max(1, total_tasks),
            "event_distribution": dict(event_types),
            "first_activity": first_event["timestamp"],
            "last_activity": last_event["timestamp"],
            "days_active": (datetime.fromisoformat(last_event["timestamp"]) -
                          datetime.fromisoformat(first_event["timestamp"])).days + 1
        }

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old analytics data

        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Clean up usage events
        original_count = len(self.usage_events)
        self.usage_events = [
            event for event in self.usage_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_date
        ]

        # Clean up performance metrics
        original_metrics_count = len(self.performance_metrics)
        self.performance_metrics = [
            metric for metric in self.performance_metrics
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_date
        ]

        # Clean up task analytics
        for task_type in self.task_analytics:
            self.task_analytics[task_type] = [
                task for task in self.task_analytics[task_type]
                if datetime.fromisoformat(task["timestamp"]) > cutoff_date
            ]

        # Save cleaned data
        await self._save_usage_events()
        await self._save_performance_metrics()

        self.logger.info(f"Cleaned up analytics data: {original_count - len(self.usage_events)} events, "
                        f"{original_metrics_count - len(self.performance_metrics)} metrics removed")


# Global analytics engine instance
analytics_engine = AnalyticsEngine()