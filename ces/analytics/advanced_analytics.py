"""CES Advanced Analytics Engine.

Provides advanced analytics capabilities including predictive analytics,
user behavior pattern recognition, performance optimization insights,
and intelligent recommendations for the Cognitive Enhancement System.
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from ..core.logging_config import get_logger

logger = get_logger(__name__)

class AdvancedAnalyticsEngine:
    """Advanced analytics engine with predictive capabilities and insights."""

    def __init__(self):
        self.behavior_patterns = defaultdict(list)
        self.performance_predictions = {}
        self.user_segments = {}
        self.anomaly_detection = {}
        self.trend_analysis = defaultdict(list)
        self.correlation_matrix = {}

    def is_healthy(self) -> bool:
        """Check if advanced analytics engine is healthy."""
        return True

    async def analyze_user_behavior_patterns(self, user_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavior patterns and provide insights."""
        try:
            # Extract temporal patterns
            hourly_patterns = self._extract_hourly_patterns(events)
            daily_patterns = self._extract_daily_patterns(events)
            task_completion_patterns = self._extract_task_patterns(events)

            # Identify peak productivity times
            peak_hours = self._identify_peak_productivity(hourly_patterns)

            # Calculate engagement metrics
            engagement_score = self._calculate_engagement_score(events)

            # Predict future behavior
            behavior_predictions = self._predict_user_behavior(user_id, events)

            # Generate personalized recommendations
            recommendations = self._generate_personalized_recommendations(
                user_id, hourly_patterns, task_completion_patterns
            )

            return {
                "user_id": user_id,
                "temporal_patterns": {
                    "hourly_distribution": hourly_patterns,
                    "daily_distribution": daily_patterns,
                    "peak_productivity_hours": peak_hours
                },
                "engagement_metrics": {
                    "overall_score": engagement_score,
                    "consistency_score": self._calculate_consistency_score(events),
                    "diversity_score": self._calculate_diversity_score(events)
                },
                "task_patterns": task_completion_patterns,
                "predictions": behavior_predictions,
                "recommendations": recommendations,
                "insights": self._generate_behavior_insights(
                    hourly_patterns, task_completion_patterns, engagement_score
                ),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"User behavior analysis error for {user_id}: {e}")
            return {"error": str(e), "user_id": user_id}

    async def predict_performance_trends(self, metric_name: str, historical_data: List[Dict[str, Any]],
                                       prediction_hours: int = 24) -> Dict[str, Any]:
        """Predict future performance trends using time series analysis."""
        try:
            if len(historical_data) < 10:
                return {"error": "Insufficient data for prediction"}

            # Extract time series data
            timestamps = []
            values = []

            for data_point in historical_data[-100:]:  # Use last 100 data points
                try:
                    timestamp = datetime.fromisoformat(data_point["timestamp"])
                    value = data_point.get("value", data_point.get("duration_ms", 0))
                    timestamps.append(timestamp)
                    values.append(value)
                except (ValueError, KeyError):
                    continue

            if len(values) < 5:
                return {"error": "Insufficient valid data points"}

            # Calculate trend
            trend_direction, trend_slope = self._calculate_trend(values)

            # Simple moving average prediction
            predictions = self._generate_predictions(values, prediction_hours)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(values, predictions)

            # Identify anomalies
            anomalies = self._detect_anomalies(values)

            return {
                "metric_name": metric_name,
                "current_trend": {
                    "direction": trend_direction,
                    "slope": round(trend_slope, 4),
                    "confidence": self._calculate_trend_confidence(values)
                },
                "predictions": {
                    "values": predictions,
                    "confidence_intervals": confidence_intervals,
                    "time_horizon_hours": prediction_hours
                },
                "anomalies": anomalies,
                "insights": self._generate_trend_insights(trend_direction, trend_slope, anomalies),
                "recommendations": self._generate_trend_recommendations(
                    trend_direction, anomalies, metric_name
                ),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Performance trend prediction error for {metric_name}: {e}")
            return {"error": str(e), "metric_name": metric_name}

    async def analyze_system_bottlenecks(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance to identify bottlenecks and optimization opportunities."""
        try:
            bottlenecks = []

            # Analyze response time bottlenecks
            if "response_times" in performance_data:
                response_bottlenecks = self._analyze_response_time_bottlenecks(
                    performance_data["response_times"]
                )
                bottlenecks.extend(response_bottlenecks)

            # Analyze resource utilization bottlenecks
            if "resource_usage" in performance_data:
                resource_bottlenecks = self._analyze_resource_bottlenecks(
                    performance_data["resource_usage"]
                )
                bottlenecks.extend(resource_bottlenecks)

            # Analyze throughput bottlenecks
            if "throughput" in performance_data:
                throughput_bottlenecks = self._analyze_throughput_bottlenecks(
                    performance_data["throughput"]
                )
                bottlenecks.extend(throughput_bottlenecks)

            # Calculate overall system health score
            health_score = self._calculate_system_health_score(bottlenecks, performance_data)

            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(bottlenecks)

            return {
                "bottlenecks": bottlenecks,
                "system_health_score": health_score,
                "severity_breakdown": self._categorize_bottleneck_severity(bottlenecks),
                "optimization_recommendations": recommendations,
                "estimated_improvement": self._estimate_optimization_impact(recommendations),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"System bottleneck analysis error: {e}")
            return {"error": str(e)}

    async def generate_user_segments(self, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate user segments based on behavior patterns and usage characteristics."""
        try:
            segments = {
                "power_users": [],
                "casual_users": [],
                "explorers": [],
                "specialists": [],
                "inactive": []
            }

            for user in user_data:
                user_id = user.get("user_id", "unknown")
                metrics = user.get("metrics", {})

                # Classification logic
                if metrics.get("total_tasks", 0) > 50:
                    segments["power_users"].append(user_id)
                elif metrics.get("total_tasks", 0) > 10:
                    segments["casual_users"].append(user_id)
                elif len(user.get("ai_preferences", {})) > 2:
                    segments["explorers"].append(user_id)
                elif max(user.get("ai_preferences", {}).values() or [0]) > 10:
                    segments["specialists"].append(user_id)
                else:
                    segments["inactive"].append(user_id)

            # Calculate segment statistics
            segment_stats = {}
            for segment_name, users in segments.items():
                segment_stats[segment_name] = {
                    "user_count": len(users),
                    "percentage": round(len(users) / len(user_data) * 100, 2) if user_data else 0
                }

            return {
                "segments": segments,
                "statistics": segment_stats,
                "segment_characteristics": self._describe_segment_characteristics(),
                "targeting_recommendations": self._generate_segment_targeting_recommendations(segments),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"User segmentation error: {e}")
            return {"error": str(e)}

    async def detect_anomalies(self, data_stream: List[Dict[str, Any]], sensitivity: float = 0.8) -> Dict[str, Any]:
        """Detect anomalies in data streams using statistical methods."""
        try:
            if len(data_stream) < 10:
                return {"error": "Insufficient data for anomaly detection"}

            # Extract values
            values = []
            for data_point in data_stream:
                if isinstance(data_point, dict) and "value" in data_point:
                    values.append(data_point["value"])
                elif isinstance(data_point, (int, float)):
                    values.append(data_point)

            if len(values) < 5:
                return {"error": "Insufficient numeric values"}

            # Calculate statistical measures
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0

            # Detect anomalies using z-score
            anomalies = []
            for i, value in enumerate(values):
                if stdev > 0:
                    z_score = abs(value - mean) / stdev
                    if z_score > sensitivity * 3:  # 3-sigma rule with sensitivity adjustment
                        anomalies.append({
                            "index": i,
                            "value": value,
                            "z_score": round(z_score, 2),
                            "timestamp": data_stream[i].get("timestamp", datetime.now().isoformat()),
                            "severity": "high" if z_score > 4 else "medium"
                        })

            # Calculate anomaly statistics
            anomaly_rate = len(anomalies) / len(values) * 100

            return {
                "anomalies_detected": len(anomalies),
                "anomaly_rate_percent": round(anomaly_rate, 2),
                "anomalies": anomalies[:20],  # Return top 20 anomalies
                "statistical_summary": {
                    "mean": round(mean, 2),
                    "standard_deviation": round(stdev, 2),
                    "min_value": min(values),
                    "max_value": max(values)
                },
                "insights": self._generate_anomaly_insights(anomalies, anomaly_rate),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"error": str(e)}

    def _extract_hourly_patterns(self, events: List[Dict[str, Any]]) -> Dict[int, int]:
        """Extract hourly activity patterns."""
        hourly_counts = defaultdict(int)

        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                hour = timestamp.hour
                hourly_counts[hour] += 1
            except (ValueError, KeyError):
                continue

        return dict(hourly_counts)

    def _extract_daily_patterns(self, events: List[Dict[str, Any]]) -> Dict[int, int]:
        """Extract daily activity patterns."""
        daily_counts = defaultdict(int)

        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                day = timestamp.weekday()  # 0=Monday, 6=Sunday
                daily_counts[day] += 1
            except (ValueError, KeyError):
                continue

        return dict(daily_counts)

    def _extract_task_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract task completion patterns."""
        task_events = [e for e in events if e.get("event_type") == "task_completed"]

        if not task_events:
            return {"total_tasks": 0, "success_rate": 0, "avg_duration": 0}

        success_count = sum(1 for e in task_events if e.get("metadata", {}).get("success", False))
        durations = []

        for event in task_events:
            duration = event.get("metadata", {}).get("duration_ms")
            if duration:
                durations.append(duration)

        return {
            "total_tasks": len(task_events),
            "success_rate": round(success_count / len(task_events) * 100, 2) if task_events else 0,
            "avg_duration_ms": round(statistics.mean(durations), 2) if durations else 0,
            "task_types": self._categorize_tasks(task_events)
        }

    def _identify_peak_productivity(self, hourly_patterns: Dict[int, int]) -> List[int]:
        """Identify peak productivity hours."""
        if not hourly_patterns:
            return []

        max_count = max(hourly_patterns.values())
        peak_hours = [hour for hour, count in hourly_patterns.items() if count >= max_count * 0.8]

        return sorted(peak_hours)

    def _calculate_engagement_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate user engagement score."""
        if not events:
            return 0.0

        # Factors: frequency, diversity, recency
        total_events = len(events)

        # Recency factor (events in last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_events = sum(1 for e in events if datetime.fromisoformat(e["timestamp"]) > week_ago)
        recency_score = recent_events / total_events if total_events > 0 else 0

        # Diversity factor (unique event types)
        event_types = set(e.get("event_type", "unknown") for e in events)
        diversity_score = len(event_types) / 10  # Normalize to max expected types

        # Frequency factor (events per day over last 30 days)
        month_ago = datetime.now() - timedelta(days=30)
        monthly_events = sum(1 for e in events if datetime.fromisoformat(e["timestamp"]) > month_ago)
        frequency_score = min(monthly_events / 30, 10) / 10  # Cap at 10 events/day

        # Weighted combination
        engagement_score = (recency_score * 0.4 + diversity_score * 0.3 + frequency_score * 0.3)

        return round(engagement_score * 100, 2)

    def _calculate_consistency_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate consistency of user activity."""
        if len(events) < 7:
            return 0.0

        # Group events by day
        daily_counts = defaultdict(int)
        for event in events:
            try:
                date = datetime.fromisoformat(event["timestamp"]).date()
                daily_counts[date] += 1
            except (ValueError, KeyError):
                continue

        if not daily_counts:
            return 0.0

        # Calculate coefficient of variation
        counts = list(daily_counts.values())
        if len(counts) < 2:
            return 100.0

        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts)

        cv = (stdev / mean) * 100 if mean > 0 else 0

        # Convert to consistency score (lower CV = higher consistency)
        consistency_score = max(0, 100 - cv)

        return round(consistency_score, 2)

    def _calculate_diversity_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate diversity of user activities."""
        event_types = Counter(e.get("event_type", "unknown") for e in events)

        if not event_types:
            return 0.0

        # Shannon diversity index
        total = sum(event_types.values())
        diversity = 0

        for count in event_types.values():
            if count > 0:
                p = count / total
                diversity -= p * math.log2(p)

        # Normalize to 0-100 scale
        max_diversity = math.log2(len(event_types)) if event_types else 1
        diversity_score = (diversity / max_diversity) * 100 if max_diversity > 0 else 0

        return round(diversity_score, 2)

    def _predict_user_behavior(self, user_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict future user behavior patterns."""
        if len(events) < 10:
            return {"error": "Insufficient data for prediction"}

        # Simple prediction based on recent patterns
        recent_events = events[-20:]  # Last 20 events

        # Predict activity level
        recent_daily_count = len(recent_events) / 7  # Events per day over last week
        predicted_daily_activity = recent_daily_count * 1.1  # 10% growth assumption

        # Predict preferred times
        recent_hourly = self._extract_hourly_patterns(recent_events)
        preferred_hours = sorted(recent_hourly.keys(), key=lambda h: recent_hourly[h], reverse=True)[:3]

        return {
            "predicted_daily_activity": round(predicted_daily_activity, 1),
            "preferred_hours": preferred_hours,
            "confidence_level": "medium",
            "prediction_basis": "recent_trends"
        }

    def _generate_personalized_recommendations(self, user_id: str, hourly_patterns: Dict[int, int],
                                             task_patterns: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations for the user."""
        recommendations = []

        # Time-based recommendations
        if hourly_patterns:
            peak_hour = max(hourly_patterns.keys(), key=lambda h: hourly_patterns[h])
            if peak_hour < 9 or peak_hour > 17:
                recommendations.append(f"Consider scheduling complex tasks during your peak hour: {peak_hour}:00")

        # Task-based recommendations
        success_rate = task_patterns.get("success_rate", 0)
        if success_rate < 70:
            recommendations.append("Consider breaking down complex tasks into smaller, manageable steps")

        if task_patterns.get("total_tasks", 0) > 20:
            recommendations.append("Great job staying productive! Consider taking short breaks between tasks")

        return recommendations

    def _generate_behavior_insights(self, hourly_patterns: Dict[int, int],
                                  task_patterns: Dict[str, Any], engagement_score: float) -> List[str]:
        """Generate insights about user behavior."""
        insights = []

        if engagement_score > 80:
            insights.append("High engagement level - you're actively using the system effectively")
        elif engagement_score > 60:
            insights.append("Good engagement level with room for increased activity")
        else:
            insights.append("Consider increasing your usage to get more value from the system")

        if hourly_patterns:
            peak_hour = max(hourly_patterns.keys(), key=lambda h: hourly_patterns[h])
            insights.append(f"Your most productive time appears to be around {peak_hour}:00")

        success_rate = task_patterns.get("success_rate", 0)
        if success_rate > 85:
            insights.append("Excellent task completion rate - you're very effective")
        elif success_rate > 70:
            insights.append("Good task completion rate with opportunities for improvement")

        return insights

    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and slope."""
        if len(values) < 2:
            return "stable", 0.0

        # Simple linear regression
        n = len(values)
        x = list(range(n))

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0

        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"

        return direction, slope

    def _generate_predictions(self, values: List[float], hours: int) -> List[float]:
        """Generate simple predictions using moving average."""
        if len(values) < 3:
            return [values[-1]] * hours

        # Use last 5 values for moving average
        window = min(5, len(values))
        recent_avg = sum(values[-window:]) / window

        # Simple trend continuation
        if len(values) >= 2:
            trend = values[-1] - values[-2]
            predictions = []
            current_value = values[-1]

            for i in range(hours):
                current_value += trend * 0.1  # Dampen trend
                predictions.append(max(0, current_value))  # Ensure non-negative

            return predictions

        return [recent_avg] * hours

    def _calculate_confidence_intervals(self, historical: List[float], predictions: List[float]) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        if len(historical) < 2:
            return [(p, p) for p in predictions]

        stdev = statistics.stdev(historical) if len(historical) > 1 else 0
        confidence_range = stdev * 1.96  # 95% confidence interval

        return [(max(0, p - confidence_range), p + confidence_range) for p in predictions]

    def _detect_anomalies(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        if len(values) < 5:
            return []

        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        anomalies = []
        for i, value in enumerate(values):
            if stdev > 0:
                z_score = abs(value - mean) / stdev
                if z_score > 3:  # 3-sigma rule
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "z_score": round(z_score, 2),
                        "deviation": "high" if value > mean else "low"
                    })

        return anomalies

    def _generate_trend_insights(self, direction: str, slope: float, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate insights about trends."""
        insights = []

        if direction == "increasing":
            insights.append(f"Performance is trending upward with slope {slope:.4f}")
        elif direction == "decreasing":
            insights.append(f"Performance is trending downward with slope {slope:.4f}")
        else:
            insights.append("Performance is stable with no significant trend")

        if anomalies:
            insights.append(f"Detected {len(anomalies)} anomalous data points that may require investigation")

        return insights

    def _generate_trend_recommendations(self, direction: str, anomalies: List[Dict[str, Any]],
                                      metric_name: str) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []

        if direction == "decreasing":
            recommendations.append(f"Investigate factors causing {metric_name} degradation")
            recommendations.append("Consider optimization strategies to reverse the trend")

        if anomalies:
            recommendations.append("Review anomalous data points for potential issues")
            recommendations.append("Implement monitoring alerts for similar anomalies")

        if direction == "stable" and not anomalies:
            recommendations.append("Current performance is stable and within normal ranges")

        return recommendations

    def _analyze_response_time_bottlenecks(self, response_times: List[float]) -> List[Dict[str, Any]]:
        """Analyze response time data for bottlenecks."""
        if not response_times:
            return []

        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)

        bottlenecks = []

        if p95 > 2000:  # 2 seconds
            bottlenecks.append({
                "type": "response_time",
                "severity": "high",
                "metric": "p95_response_time",
                "value": p95,
                "threshold": 2000,
                "description": f"P95 response time ({p95:.0f}ms) exceeds 2s threshold"
            })

        if p99 > 5000:  # 5 seconds
            bottlenecks.append({
                "type": "response_time",
                "severity": "critical",
                "metric": "p99_response_time",
                "value": p99,
                "threshold": 5000,
                "description": f"P99 response time ({p99:.0f}ms) exceeds 5s threshold"
            })

        return bottlenecks

    def _analyze_resource_bottlenecks(self, resource_usage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze resource usage for bottlenecks."""
        bottlenecks = []

        # CPU usage
        cpu_usage = resource_usage.get("cpu_percent", 0)
        if cpu_usage > 80:
            bottlenecks.append({
                "type": "resource",
                "resource": "cpu",
                "severity": "high",
                "value": cpu_usage,
                "threshold": 80,
                "description": f"CPU usage ({cpu_usage}%) is above 80% threshold"
            })

        # Memory usage
        memory_usage = resource_usage.get("memory_percent", 0)
        if memory_usage > 85:
            bottlenecks.append({
                "type": "resource",
                "resource": "memory",
                "severity": "high",
                "value": memory_usage,
                "threshold": 85,
                "description": f"Memory usage ({memory_usage}%) is above 85% threshold"
            })

        return bottlenecks

    def _analyze_throughput_bottlenecks(self, throughput: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze throughput for bottlenecks."""
        bottlenecks = []

        current_throughput = throughput.get("current_rps", 0)
        max_capacity = throughput.get("max_capacity_rps", 100)

        if current_throughput > max_capacity * 0.9:
            bottlenecks.append({
                "type": "throughput",
                "severity": "medium",
                "value": current_throughput,
                "threshold": max_capacity * 0.9,
                "description": f"Throughput ({current_throughput} RPS) approaching capacity limit"
            })

        return bottlenecks

    def _calculate_system_health_score(self, bottlenecks: List[Dict[str, Any]],
                                     performance_data: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        base_score = 100.0

        # Deduct points for each bottleneck
        severity_weights = {"low": 5, "medium": 10, "high": 20, "critical": 30}

        for bottleneck in bottlenecks:
            severity = bottleneck.get("severity", "medium")
            base_score -= severity_weights.get(severity, 10)

        # Ensure score stays within bounds
        return max(0.0, min(100.0, base_score))

    def _generate_optimization_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        for bottleneck in bottlenecks:
            rec_type = bottleneck["type"]

            if rec_type == "response_time":
                recommendations.append({
                    "category": "performance",
                    "priority": bottleneck["severity"],
                    "action": "Optimize database queries and implement caching",
                    "expected_impact": "20-40% response time improvement",
                    "effort": "medium"
                })

            elif rec_type == "resource" and bottleneck["resource"] == "cpu":
                recommendations.append({
                    "category": "infrastructure",
                    "priority": bottleneck["severity"],
                    "action": "Scale CPU resources or optimize compute-intensive operations",
                    "expected_impact": "30-50% CPU usage reduction",
                    "effort": "high"
                })

            elif rec_type == "resource" and bottleneck["resource"] == "memory":
                recommendations.append({
                    "category": "memory",
                    "priority": bottleneck["severity"],
                    "action": "Implement memory optimization and garbage collection tuning",
                    "expected_impact": "25-45% memory usage reduction",
                    "effort": "medium"
                })

        return recommendations

    def _estimate_optimization_impact(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate the overall impact of optimization recommendations."""
        total_impact = 0
        effort_breakdown = {"low": 0, "medium": 0, "high": 0}

        for rec in recommendations:
            # Parse expected impact (e.g., "20-40%" -> average 30%)
            impact_str = rec.get("expected_impact", "0%")
            if "%" in impact_str:
                try:
                    # Extract percentage range and take average
                    parts = impact_str.replace("%", "").split("-")
                    if len(parts) == 2:
                        impact = (float(parts[0]) + float(parts[1])) / 2
                    else:
                        impact = float(parts[0])
                    total_impact += impact
                except ValueError:
                    pass

            effort = rec.get("effort", "medium")
            effort_breakdown[effort] += 1

        return {
            "estimated_improvement_percent": round(total_impact, 1),
            "effort_distribution": effort_breakdown,
            "recommendation_count": len(recommendations)
        }

    def _categorize_bottleneck_severity(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize bottlenecks by severity."""
        severity_counts = defaultdict(int)

        for bottleneck in bottlenecks:
            severity = bottleneck.get("severity", "medium")
            severity_counts[severity] += 1

        return dict(severity_counts)

    def _categorize_tasks(self, task_events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize tasks by type."""
        task_types = defaultdict(int)

        for event in task_events:
            metadata = event.get("metadata", {})
            task_type = metadata.get("task_type", "general")
            task_types[task_type] += 1

        return dict(task_types)

    def _describe_segment_characteristics(self) -> Dict[str, str]:
        """Describe characteristics of each user segment."""
        return {
            "power_users": "High-frequency users with extensive task completion and diverse AI assistant usage",
            "casual_users": "Regular users with moderate activity and balanced AI assistant preferences",
            "explorers": "Users who experiment with multiple AI assistants and diverse task types",
            "specialists": "Users with strong preference for specific AI assistants and focused task types",
            "inactive": "Low-activity users who may need onboarding or re-engagement strategies"
        }

    def _generate_segment_targeting_recommendations(self, segments: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate targeting recommendations for each segment."""
        return {
            "power_users": [
                "Provide advanced features and customization options",
                "Offer premium support and early access to new features",
                "Create exclusive communities or beta testing opportunities"
            ],
            "casual_users": [
                "Send regular usage tips and feature highlights",
                "Provide gentle onboarding reminders",
                "Offer achievement badges and progress tracking"
            ],
            "explorers": [
                "Recommend new AI assistants and experimental features",
                "Provide comparative analysis of different assistants",
                "Create exploration challenges and discovery paths"
            ],
            "specialists": [
                "Offer deep customization for preferred AI assistants",
                "Provide advanced configuration options",
                "Create specialized workflows and templates"
            ],
            "inactive": [
                "Send re-engagement campaigns with usage reminders",
                "Offer simplified getting-started guides",
                "Provide personalized onboarding sessions"
            ]
        }

    def _generate_anomaly_insights(self, anomalies: List[Dict[str, Any]], anomaly_rate: float) -> List[str]:
        """Generate insights about detected anomalies."""
        insights = []

        if anomaly_rate > 10:
            insights.append(f"High anomaly rate ({anomaly_rate:.1f}%) suggests potential systemic issues")
        elif anomaly_rate > 5:
            insights.append(f"Moderate anomaly rate ({anomaly_rate:.1f}%) may indicate intermittent issues")
        else:
            insights.append(f"Low anomaly rate ({anomaly_rate:.1f}%) indicates stable system performance")

        if anomalies:
            high_severity = len([a for a in anomalies if a.get("severity") == "high"])
            if high_severity > 0:
                insights.append(f"Detected {high_severity} high-severity anomalies requiring immediate attention")

        return insights