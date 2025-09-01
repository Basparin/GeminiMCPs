"""
Performance Monitor Module for CodeSage MCP Server.

This module provides comprehensive performance monitoring capabilities including
real-time metrics collection, usage pattern analysis, and predictive analytics.
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric measurement with timestamp and metadata."""
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Container for current performance metrics."""
    response_time_ms: Optional[MetricValue] = None
    throughput_rps: Optional[MetricValue] = None
    memory_usage_percent: Optional[MetricValue] = None
    cpu_usage_percent: Optional[MetricValue] = None
    error_rate_percent: Optional[MetricValue] = None
    active_connections: Optional[MetricValue] = None


class PerformanceMonitor:
    """Main performance monitoring class."""

    def __init__(self, max_history_size: int = 1000):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=max_history_size)
        self.request_timestamps = deque(maxlen=max_history_size)
        self._lock = threading.Lock()

    def record_request(self, response_time_ms: float, success: bool, endpoint: str, user_id: str):
        """Record a request for performance monitoring."""
        with self._lock:
            self.request_count += 1
            if not success:
                self.error_count += 1

            current_time = time.time()
            self.response_times.append(response_time_ms)
            self.request_timestamps.append(current_time)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            current_time = time.time()

            # Calculate response time (last 5 minutes average)
            recent_responses = [rt for rt, ts in zip(self.response_times, self.request_timestamps)
                              if current_time - ts < 300]  # Last 5 minutes
            avg_response_time = statistics.mean(recent_responses) if recent_responses else 0

            # Calculate throughput (requests per second in last minute)
            recent_requests = [ts for ts in self.request_timestamps
                             if current_time - ts < 60]  # Last minute
            throughput = len(recent_requests) / 60.0 if recent_requests else 0

            # Calculate error rate
            recent_errors = sum(1 for rt, ts in zip(self.response_times, self.request_timestamps)
                              if current_time - ts < 300 and rt > 5000)  # Rough error detection
            error_rate = (recent_errors / len(recent_responses)) * 100 if recent_responses else 0

            # System metrics
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)

            return {
                "response_time_ms": {
                    "value": avg_response_time,
                    "timestamp": current_time,
                    "unit": "ms"
                },
                "throughput_rps": {
                    "value": throughput,
                    "timestamp": current_time,
                    "unit": "requests/second"
                },
                "memory_usage_percent": {
                    "value": memory_percent,
                    "timestamp": current_time,
                    "unit": "percent"
                },
                "cpu_usage_percent": {
                    "value": cpu_percent,
                    "timestamp": current_time,
                    "unit": "percent"
                },
                "error_rate_percent": {
                    "value": error_rate,
                    "timestamp": current_time,
                    "unit": "percent"
                }
            }

    def get_metrics_summary(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # Filter data for the time window
            window_responses = [rt for rt, ts in zip(self.response_times, self.request_timestamps)
                              if ts > cutoff_time]
            window_requests = len([ts for ts in self.request_timestamps if ts > cutoff_time])

            if not window_responses:
                return {"message": "No data available for the specified time window"}

            summary = {
                "window_seconds": window_seconds,
                "request_count": window_requests,
                "avg_response_time_ms": statistics.mean(window_responses),
                "min_response_time_ms": min(window_responses),
                "max_response_time_ms": max(window_responses),
                "p95_response_time_ms": statistics.quantiles(window_responses, n=20)[18],  # 95th percentile
                "p99_response_time_ms": statistics.quantiles(window_responses, n=100)[98],  # 99th percentile
                "throughput_rps": window_requests / window_seconds,
                "timestamp": current_time
            }

            return summary

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        summary_5m = self.get_metrics_summary(300)

        # Calculate performance score (0-100)
        response_time_score = max(0, 100 - (current_metrics["response_time_ms"]["value"] / 10))  # Penalize > 1s
        throughput_score = min(100, current_metrics["throughput_rps"]["value"] * 10)  # Reward higher throughput
        error_score = max(0, 100 - current_metrics["error_rate_percent"]["value"])
        resource_score = 100 - ((current_metrics["memory_usage_percent"]["value"] +
                               current_metrics["cpu_usage_percent"]["value"]) / 2)

        performance_score = (response_time_score + throughput_score + error_score + resource_score) / 4

        # Generate alerts
        alerts = []
        if current_metrics["memory_usage_percent"]["value"] > 85:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "message": ".1f",
                "timestamp": time.time()
            })

        if current_metrics["cpu_usage_percent"]["value"] > 90:
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": ".1f",
                "timestamp": time.time()
            })

        if current_metrics["error_rate_percent"]["value"] > 5:
            alerts.append({
                "type": "high_error_rate",
                "severity": "error",
                "message": ".1f",
                "timestamp": time.time()
            })

        return {
            "performance_score": round(performance_score, 2),
            "current_metrics": current_metrics,
            "summary_5m": summary_5m,
            "recent_alerts": alerts,
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "timestamp": time.time()
        }


class UsageAnalyzer:
    """Analyzes usage patterns and user behavior."""

    def __init__(self):
        self.user_actions = []
        self._lock = threading.Lock()

    def record_user_action(self, user_id: str, action: str, metadata: Dict[str, Any]):
        """Record a user action for pattern analysis."""
        with self._lock:
            self.user_actions.append({
                "user_id": user_id,
                "action": action,
                "timestamp": time.time(),
                "metadata": metadata
            })

    def analyze_patterns(self) -> List[Dict[str, Any]]:
        """Analyze usage patterns."""
        with self._lock:
            if not self.user_actions:
                return []

            # Simple pattern analysis - group by action type
            action_counts = {}
            for action_data in self.user_actions:
                action = action_data["action"]
                action_counts[action] = action_counts.get(action, 0) + 1

            patterns = []
            total_actions = len(self.user_actions)

            for action, count in action_counts.items():
                frequency = count / total_actions
                patterns.append({
                    "pattern_id": f"pattern_{action}",
                    "pattern_type": "action_frequency",
                    "frequency": frequency,
                    "avg_duration": 0,  # Not implemented yet
                    "user_profiles": ["default"],  # Not implemented yet
                    "resource_impact": {"cpu": 0.1, "memory": 0.05},  # Placeholder
                    "optimization_potential": min(0.9, frequency * 0.5),
                    "last_observed": max(action["timestamp"] for action in self.user_actions
                                       if action["action"] == action)
                })

            return patterns


class PredictiveAnalytics:
    """Provides predictive analytics for performance optimization."""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self._lock = threading.Lock()

    def predict_resource_usage(self) -> Dict[str, Any]:
        """Predict future resource usage."""
        with self._lock:
            # Simple linear prediction based on recent trends
            if len(self.metrics_history) < 10:
                return {"message": "Insufficient data for prediction"}

            recent_memory = [m.get("memory_usage_percent", {}).get("value", 0)
                           for m in list(self.metrics_history)[-10:]]
            recent_cpu = [m.get("cpu_usage_percent", {}).get("value", 0)
                         for m in list(self.metrics_history)[-10:]]

            predictions = {}

            if recent_memory:
                memory_trend = statistics.linear_regression(range(len(recent_memory)), recent_memory)
                predictions["memory_usage_percent"] = {
                    "predicted_values": [memory_trend[0] + memory_trend[1] * i for i in range(5)],
                    "slope": memory_trend[1],
                    "trend": "increasing" if memory_trend[1] > 0.1 else "stable",
                    "confidence": 0.7
                }

            if recent_cpu:
                cpu_trend = statistics.linear_regression(range(len(recent_cpu)), recent_cpu)
                predictions["cpu_usage_percent"] = {
                    "predicted_values": [cpu_trend[0] + cpu_trend[1] * i for i in range(5)],
                    "slope": cpu_trend[1],
                    "trend": "increasing" if cpu_trend[1] > 0.1 else "stable",
                    "confidence": 0.7
                }

            return predictions

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        with self._lock:
            if len(self.metrics_history) < 20:
                return []

            anomalies = []
            recent_metrics = list(self.metrics_history)[-20:]

            # Check for memory anomalies
            memory_values = [m.get("memory_usage_percent", {}).get("value", 0) for m in recent_metrics]
            if memory_values:
                mean_memory = statistics.mean(memory_values)
                stdev_memory = statistics.stdev(memory_values) if len(memory_values) > 1 else 0

                latest_memory = memory_values[-1]
                if stdev_memory > 0:
                    z_score = (latest_memory - mean_memory) / stdev_memory
                    if abs(z_score) > 2:  # 2 standard deviations
                        anomalies.append({
                            "metric_name": "memory_usage_percent",
                            "value": latest_memory,
                            "expected_value": mean_memory,
                            "z_score": z_score,
                            "severity": "high" if abs(z_score) > 3 else "medium",
                            "description": ".1f"
                        })

            return anomalies

    def recommend_optimizations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        predictions = self.predict_resource_usage()
        anomalies = self.detect_anomalies()

        recommendations = []

        # Memory optimization recommendations
        memory_pred = predictions.get("memory_usage_percent", {})
        if memory_pred.get("trend") == "increasing":
            recommendations.append({
                "type": "memory_optimization",
                "priority": "high",
                "description": "Memory usage trending upward - consider optimization",
                "actions": ["Implement memory pooling", "Add garbage collection tuning", "Review cache sizes"]
            })

        # Anomaly-based recommendations
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                recommendations.append({
                    "type": "anomaly_investigation",
                    "priority": "critical",
                    "description": f"Investigate {anomaly['metric_name']} anomaly",
                    "actions": ["Check system logs", "Review recent changes", "Monitor closely"]
                })

        return recommendations


# Global instances
_performance_monitor = None
_usage_analyzer = None
_predictive_analytics = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_usage_analyzer() -> UsageAnalyzer:
    """Get the global usage analyzer instance."""
    global _usage_analyzer
    if _usage_analyzer is None:
        _usage_analyzer = UsageAnalyzer()
    return _usage_analyzer


def get_predictive_analytics() -> PredictiveAnalytics:
    """Get the global predictive analytics instance."""
    global _predictive_analytics
    if _predictive_analytics is None:
        _predictive_analytics = PredictiveAnalytics()
    return _predictive_analytics
