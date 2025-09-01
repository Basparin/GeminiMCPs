"""
Trend Analysis Module for CodeSage MCP Server.

This module provides comprehensive trend analysis capabilities for tracking performance
metrics over time, identifying optimization opportunities, and generating predictive
insights for continuous improvement.

Classes:
    TrendAnalyzer: Core trend analysis functionality
    OptimizationRecommender: Generates optimization recommendations
    PerformancePredictor: Predicts future performance trends
    AnomalyDetector: Detects performance anomalies
"""

import logging
import time
import statistics
import threading
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Possible trend directions."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class OptimizationPriority(Enum):
    """Optimization priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PerformanceTrend:
    """Represents a performance trend analysis."""
    metric_name: str
    direction: TrendDirection
    slope: float
    confidence: float
    data_points: int
    time_window_days: float
    start_value: float
    end_value: float
    volatility: float
    analysis_timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationOpportunity:
    """Represents an optimization opportunity."""
    opportunity_id: str
    title: str
    description: str
    metric_affected: str
    current_impact: float
    potential_improvement: float
    priority: OptimizationPriority
    effort_estimate: str
    implementation_complexity: str
    expected_benefits: List[str]
    risks: List[str]
    prerequisites: List[str]
    timeline_estimate: str
    discovered_at: float = field(default_factory=time.time)


class TrendAnalyzer:
    """Analyzes performance trends over time."""

    def __init__(self, history_window_days: int = 30):
        self.history_window_days = history_window_days
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trend_cache: Dict[str, PerformanceTrend] = {}
        self._lock = threading.RLock()

    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value for trend analysis."""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            self.metric_history[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })

            # Clean old data
            cutoff_time = timestamp - (self.history_window_days * 24 * 60 * 60)
            while self.metric_history[metric_name] and self.metric_history[metric_name][0]['timestamp'] < cutoff_time:
                self.metric_history[metric_name].popleft()

    def analyze_trend(self, metric_name: str, analysis_window_days: Optional[int] = None) -> Optional[PerformanceTrend]:
        """Analyze the trend for a specific metric."""
        if analysis_window_days is None:
            analysis_window_days = self.history_window_days

        with self._lock:
            data = list(self.metric_history[metric_name])
            if len(data) < 5:  # Need minimum data points
                return None

            # Filter by time window
            cutoff_time = time.time() - (analysis_window_days * 24 * 60 * 60)
            recent_data = [d for d in data if d['timestamp'] >= cutoff_time]

            if len(recent_data) < 5:
                return None

            # Extract values and timestamps
            values = [d['value'] for d in recent_data]
            timestamps = [d['timestamp'] for d in recent_data]

            # Calculate trend using linear regression
            x = np.array([(t - timestamps[0]) / (24 * 60 * 60) for t in timestamps])  # Convert to days
            y = np.array(values)

            try:
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1]**2
                confidence = min(r_squared, 1.0)  # Confidence based on R-squared

                # Determine trend direction
                if abs(slope) < 0.01:  # Very small slope
                    direction = TrendDirection.STABLE
                elif slope > 0.05:  # Significant positive slope
                    direction = TrendDirection.DEGRADING
                elif slope < -0.05:  # Significant negative slope
                    direction = TrendDirection.IMPROVING
                else:
                    direction = TrendDirection.STABLE

                # Calculate volatility (coefficient of variation)
                if statistics.mean(values) != 0:
                    volatility = statistics.stdev(values) / abs(statistics.mean(values))
                else:
                    volatility = 0

                # Check for high volatility
                if volatility > 0.5 and direction == TrendDirection.STABLE:
                    direction = TrendDirection.VOLATILE

                trend = PerformanceTrend(
                    metric_name=metric_name,
                    direction=direction,
                    slope=slope,
                    confidence=confidence,
                    data_points=len(recent_data),
                    time_window_days=analysis_window_days,
                    start_value=values[0],
                    end_value=values[-1],
                    volatility=volatility
                )

                self.trend_cache[metric_name] = trend
                return trend

            except Exception as e:
                logger.warning(f"Error analyzing trend for {metric_name}: {e}")
                return None

    def get_all_trends(self, analysis_window_days: Optional[int] = None) -> Dict[str, PerformanceTrend]:
        """Get trend analysis for all metrics."""
        trends = {}
        metric_names = list(self.metric_history.keys())

        for metric_name in metric_names:
            trend = self.analyze_trend(metric_name, analysis_window_days)
            if trend:
                trends[metric_name] = trend

        return trends

    def identify_trending_issues(self) -> List[Dict[str, Any]]:
        """Identify metrics with concerning trends."""
        issues = []
        trends = self.get_all_trends()

        for metric_name, trend in trends.items():
            issue = None

            # Check for degrading performance
            if trend.direction == TrendDirection.DEGRADING:
                if "response_time" in metric_name:
                    issue = {
                        "type": "performance_degradation",
                        "severity": "high" if trend.slope > 0.1 else "medium",
                        "metric": metric_name,
                        "description": f"Response time increasing by {trend.slope:.3f} per day",
                        "trend": trend,
                        "recommendation": "Investigate performance bottlenecks and optimize slow operations"
                    }
                elif "error_rate" in metric_name:
                    issue = {
                        "type": "error_rate_increasing",
                        "severity": "high" if trend.slope > 0.01 else "medium",
                        "metric": metric_name,
                        "description": f"Error rate increasing by {trend.slope:.4f} per day",
                        "trend": trend,
                        "recommendation": "Investigate error sources and improve error handling"
                    }
                elif "memory_usage" in metric_name:
                    issue = {
                        "type": "memory_growth",
                        "severity": "medium" if trend.slope > 1 else "low",
                        "metric": metric_name,
                        "description": f"Memory usage increasing by {trend.slope:.1f}MB per day",
                        "trend": trend,
                        "recommendation": "Monitor memory usage and consider optimization or scaling"
                    }

            # Check for high volatility
            elif trend.direction == TrendDirection.VOLATILE:
                issue = {
                    "type": "performance_volatility",
                    "severity": "medium",
                    "metric": metric_name,
                    "description": f"High volatility detected (volatility: {trend.volatility:.2f})",
                    "trend": trend,
                    "recommendation": "Investigate source of performance variability"
                }

            if issue:
                issues.append(issue)

        return sorted(issues, key=lambda x: x["severity"] == "high", reverse=True)

    def predict_future_performance(self, metric_name: str, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future performance for a metric."""
        trend = self.analyze_trend(metric_name)
        if not trend:
            return {"error": "Insufficient data for prediction"}

        # Simple linear extrapolation
        current_value = trend.end_value
        predicted_change = trend.slope * days_ahead
        predicted_value = current_value + predicted_change

        # Calculate confidence interval (simplified)
        data = [d['value'] for d in self.metric_history[metric_name]]
        if len(data) > 1:
            stdev = statistics.stdev(data)
            margin_of_error = stdev * 1.96  # 95% confidence interval
        else:
            margin_of_error = 0

        return {
            "metric_name": metric_name,
            "current_value": current_value,
            "predicted_value": predicted_value,
            "predicted_change": predicted_change,
            "days_ahead": days_ahead,
            "confidence_interval": {
                "lower": predicted_value - margin_of_error,
                "upper": predicted_value + margin_of_error
            },
            "trend_direction": trend.direction.value,
            "prediction_confidence": trend.confidence,
            "based_on_data_points": trend.data_points
        }


class OptimizationRecommender:
    """Generates optimization recommendations based on trend analysis."""

    def __init__(self, trend_analyzer: TrendAnalyzer):
        self.trend_analyzer = trend_analyzer
        self.opportunities: Dict[str, OptimizationOpportunity] = {}
        self._lock = threading.RLock()

    def generate_recommendations(self) -> List[OptimizationOpportunity]:
        """Generate optimization recommendations based on current trends."""
        opportunities = []

        # Analyze performance trends
        trends = self.trend_analyzer.get_all_trends()

        # Generate recommendations based on trends
        for metric_name, trend in trends.items():
            if trend.direction == TrendDirection.DEGRADING:
                opportunities.extend(self._generate_degradation_recommendations(metric_name, trend))
            elif trend.direction == TrendDirection.VOLATILE:
                opportunities.extend(self._generate_volatility_recommendations(metric_name, trend))

        # Generate recommendations based on absolute performance levels
        opportunities.extend(self._generate_absolute_performance_recommendations(trends))

        # Generate cross-metric recommendations
        opportunities.extend(self._generate_cross_metric_recommendations(trends))

        # Remove duplicates and sort by priority
        unique_opportunities = self._deduplicate_opportunities(opportunities)
        return self._prioritize_opportunities(unique_opportunities)

    def _generate_degradation_recommendations(self, metric_name: str,
                                           trend: PerformanceTrend) -> List[OptimizationOpportunity]:
        """Generate recommendations for degrading performance."""
        opportunities = []

        if "response_time" in metric_name:
            if trend.slope > 0.1:  # Significant degradation
                opportunities.append(OptimizationOpportunity(
                    opportunity_id=f"perf_opt_{int(time.time())}_1",
                    title="Optimize Response Time Performance",
                    description=f"Response time is degrading by {trend.slope:.3f}ms per day. "
                               f"Current: {trend.end_value:.1f}ms, Predicted: {trend.end_value + trend.slope * 7:.1f}ms in 7 days.",
                    metric_affected=metric_name,
                    current_impact=trend.slope * 30,  # 30-day impact
                    potential_improvement=trend.end_value * 0.3,  # 30% improvement potential
                    priority=OptimizationPriority.HIGH if trend.slope > 0.2 else OptimizationPriority.MEDIUM,
                    effort_estimate="2-4 weeks",
                    implementation_complexity="Medium",
                    expected_benefits=[
                        "Reduce response time by 20-40%",
                        "Improve user satisfaction",
                        "Reduce resource consumption"
                    ],
                    risks=[
                        "Potential service disruption during optimization",
                        "May require code refactoring"
                    ],
                    prerequisites=[
                        "Performance profiling completed",
                        "Load testing environment available"
                    ],
                    timeline_estimate="2-3 weeks"
                ))

        elif "memory_usage" in metric_name:
            opportunities.append(OptimizationOpportunity(
                opportunity_id=f"mem_opt_{int(time.time())}_1",
                title="Optimize Memory Usage",
                description=f"Memory usage is growing by {trend.slope:.1f}MB per day. "
                           f"Current: {trend.end_value:.1f}MB, Predicted: {trend.end_value + trend.slope * 30:.1f}MB in 30 days.",
                metric_affected=metric_name,
                current_impact=trend.slope * 30,
                potential_improvement=trend.end_value * 0.2,
                priority=OptimizationPriority.MEDIUM,
                effort_estimate="1-2 weeks",
                implementation_complexity="Low to Medium",
                expected_benefits=[
                    "Reduce memory footprint by 15-25%",
                    "Improve application stability",
                    "Allow higher concurrent user load"
                ],
                risks=[
                    "May affect performance during transition",
                    "Requires careful testing"
                ],
                prerequisites=[
                    "Memory profiling completed",
                    "Alternative caching strategies identified"
                ],
                timeline_estimate="1-2 weeks"
            ))

        return opportunities

    def _generate_volatility_recommendations(self, metric_name: str,
                                          trend: PerformanceTrend) -> List[OptimizationOpportunity]:
        """Generate recommendations for volatile performance."""
        opportunities = []

        opportunities.append(OptimizationOpportunity(
            opportunity_id=f"stab_opt_{int(time.time())}_1",
            title="Improve Performance Stability",
            description=f"High performance volatility detected for {metric_name} "
                       f"(volatility: {trend.volatility:.2f}). This affects user experience consistency.",
            metric_affected=metric_name,
            current_impact=trend.volatility * 100,  # Impact as percentage
            potential_improvement=trend.volatility * 50,  # 50% reduction in volatility
            priority=OptimizationPriority.MEDIUM,
            effort_estimate="1-3 weeks",
            implementation_complexity="Medium",
            expected_benefits=[
                "More consistent user experience",
                "Better capacity planning",
                "Reduced support tickets"
            ],
            risks=[
                "May require architectural changes",
                "Could impact performance during stabilization"
            ],
            prerequisites=[
                "Root cause analysis completed",
                "Performance monitoring enhanced"
            ],
            timeline_estimate="2-4 weeks"
        ))

        return opportunities

    def _generate_absolute_performance_recommendations(self, trends: Dict[str, PerformanceTrend]) -> List[OptimizationOpportunity]:
        """Generate recommendations based on absolute performance levels."""
        opportunities = []

        # Check response time levels
        response_time_trend = trends.get('response_time_ms')
        if response_time_trend and response_time_trend.end_value > 100:
            opportunities.append(OptimizationOpportunity(
                opportunity_id=f"abs_perf_opt_{int(time.time())}_1",
                title="Improve Overall Response Time",
                description=f"Average response time ({response_time_trend.end_value:.1f}ms) exceeds recommended threshold (100ms).",
                metric_affected="response_time_ms",
                current_impact=response_time_trend.end_value - 100,
                potential_improvement=response_time_trend.end_value * 0.4,
                priority=OptimizationPriority.HIGH,
                effort_estimate="2-6 weeks",
                implementation_complexity="High",
                expected_benefits=[
                    "Faster user interactions",
                    "Better user satisfaction scores",
                    "Competitive advantage"
                ],
                risks=[
                    "May require significant architectural changes",
                    "Could affect functionality during optimization"
                ],
                prerequisites=[
                    "Comprehensive performance audit",
                    "User experience requirements documented"
                ],
                timeline_estimate="4-8 weeks"
            ))

        return opportunities

    def _generate_cross_metric_recommendations(self, trends: Dict[str, PerformanceTrend]) -> List[OptimizationOpportunity]:
        """Generate recommendations based on relationships between metrics."""
        opportunities = []

        # Check for memory-pressure related issues
        memory_trend = trends.get('memory_usage_percent')
        response_time_trend = trends.get('response_time_ms')

        if (memory_trend and response_time_trend and
            memory_trend.end_value > 80 and response_time_trend.end_value > 50):
            opportunities.append(OptimizationOpportunity(
                opportunity_id=f"cross_opt_{int(time.time())}_1",
                title="Optimize Memory Management Under Load",
                description="High memory usage correlated with increased response times. "
                           f"Memory: {memory_trend.end_value:.1f}%, Response Time: {response_time_trend.end_value:.1f}ms.",
                metric_affected="memory_usage_percent,response_time_ms",
                current_impact=(memory_trend.end_value - 70) + (response_time_trend.end_value - 30),
                potential_improvement=30,  # Combined improvement score
                priority=OptimizationPriority.HIGH,
                effort_estimate="3-5 weeks",
                implementation_complexity="High",
                expected_benefits=[
                    "Better performance under load",
                    "Improved scalability",
                    "Reduced infrastructure costs"
                ],
                risks=[
                    "Complex optimization requiring careful planning",
                    "May need infrastructure changes"
                ],
                prerequisites=[
                    "Load testing completed",
                    "Memory leak analysis done",
                    "Scaling strategy defined"
                ],
                timeline_estimate="4-6 weeks"
            ))

        return opportunities

    def _deduplicate_opportunities(self, opportunities: List[OptimizationOpportunity]) -> List[OptimizationOpportunity]:
        """Remove duplicate or very similar opportunities."""
        seen_titles = set()
        unique_opportunities = []

        for opp in opportunities:
            # Simple deduplication based on title similarity
            title_key = opp.title.lower().replace(" ", "")
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_opportunities.append(opp)

        return unique_opportunities

    def _prioritize_opportunities(self, opportunities: List[OptimizationOpportunity]) -> List[OptimizationOpportunity]:
        """Sort opportunities by priority and impact."""
        def sort_key(opp):
            priority_score = {
                OptimizationPriority.CRITICAL: 4,
                OptimizationPriority.HIGH: 3,
                OptimizationPriority.MEDIUM: 2,
                OptimizationPriority.LOW: 1
            }[opp.priority]

            impact_score = opp.current_impact * opp.potential_improvement

            return (priority_score, impact_score)

        return sorted(opportunities, key=sort_key, reverse=True)


class PerformancePredictor:
    """Predicts future performance trends using advanced analytics."""

    def __init__(self, trend_analyzer: TrendAnalyzer):
        self.trend_analyzer = trend_analyzer

    def predict_workload_capacity(self, target_response_time_ms: float = 100) -> Dict[str, Any]:
        """Predict maximum workload capacity for target response time."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated modeling

        response_time_trend = self.trend_analyzer.analyze_trend('response_time_ms')
        throughput_trend = self.trend_analyzer.analyze_trend('throughput_rps')

        if not response_time_trend or not throughput_trend:
            return {"error": "Insufficient data for capacity prediction"}

        # Simple capacity estimation based on current trends
        current_response_time = response_time_trend.end_value
        current_throughput = throughput_trend.end_value

        # Estimate capacity at target response time
        # This is a very simplified model - real implementation would use regression analysis
        response_time_ratio = current_response_time / target_response_time_ms
        estimated_capacity = current_throughput / response_time_ratio

        return {
            "target_response_time_ms": target_response_time_ms,
            "current_response_time_ms": current_response_time,
            "current_throughput_rps": current_throughput,
            "estimated_max_capacity_rps": estimated_capacity,
            "capacity_headroom_percent": ((estimated_capacity - current_throughput) / current_throughput) * 100,
            "confidence_level": min(response_time_trend.confidence, throughput_trend.confidence),
            "prediction_based_on_days": min(response_time_trend.time_window_days, throughput_trend.time_window_days)
        }

    def forecast_seasonal_patterns(self, metric_name: str, forecast_days: int = 90) -> Dict[str, Any]:
        """Forecast seasonal patterns in performance metrics."""
        # This would implement seasonal decomposition and forecasting
        # For now, return a placeholder implementation

        return {
            "metric_name": metric_name,
            "forecast_days": forecast_days,
            "status": "Seasonal forecasting requires extended historical data (3+ months)",
            "recommendation": "Continue collecting performance data for seasonal analysis"
        }


# Global instances
_trend_analyzer: Optional[TrendAnalyzer] = None
_optimization_recommender: Optional[OptimizationRecommender] = None
_performance_predictor: Optional[PerformancePredictor] = None


def get_trend_analyzer() -> TrendAnalyzer:
    """Get the global trend analyzer instance."""
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    return _trend_analyzer


def get_optimization_recommender() -> OptimizationRecommender:
    """Get the global optimization recommender instance."""
    global _optimization_recommender, _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    if _optimization_recommender is None:
        _optimization_recommender = OptimizationRecommender(_trend_analyzer)
    return _optimization_recommender


def get_performance_predictor() -> PerformancePredictor:
    """Get the global performance predictor instance."""
    global _performance_predictor, _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    if _performance_predictor is None:
        _performance_predictor = PerformancePredictor(_trend_analyzer)
    return _performance_predictor