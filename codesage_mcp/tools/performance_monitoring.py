"""
Performance Monitoring Tools for CodeSage MCP Server.

This module provides tools for monitoring and analyzing the performance of the CodeSage MCP Server,
including real-time metrics, usage patterns, and predictive analytics.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.performance_monitoring.performance_monitor import (
    get_performance_monitor,
    get_usage_analyzer,
    get_predictive_analytics
)

logger = logging.getLogger(__name__)


def get_performance_metrics_tool() -> Dict[str, Any]:
    """
    Get current real-time performance metrics.

    Returns:
        Dictionary containing current performance metrics including:
        - Response times (current, average, P95, P99)
        - Resource utilization (CPU, memory, disk, network)
        - Throughput metrics (RPS, concurrent connections)
        - Error rates and cache performance
        - System health indicators
    """
    try:
        monitor = get_performance_monitor()
        metrics = monitor.get_current_metrics()

        # Add additional computed metrics
        summary_5m = monitor.get_metrics_summary(300)

        result = {
            "current_metrics": metrics,
            "summary_5m": summary_5m,
            "timestamp": monitor.start_time,
            "uptime_seconds": monitor.start_time
        }

        return result

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {
            "error": f"Failed to retrieve performance metrics: {str(e)}",
            "timestamp": None
        }


def get_performance_report_tool() -> Dict[str, Any]:
    """
    Generate a comprehensive performance report.

    Returns:
        Comprehensive performance report including:
        - Current performance metrics
        - Baseline status and compliance
        - Recent alerts and anomalies
        - Performance score and recommendations
        - Trend analysis and forecasting
    """
    try:
        monitor = get_performance_monitor()
        report = monitor.get_performance_report()

        # Enhance report with additional analytics
        analyzer = get_usage_analyzer()
        predictive = get_predictive_analytics()

        # Add usage pattern insights
        usage_patterns = analyzer.analyze_patterns()
        report["usage_patterns"] = [pattern.__dict__ for pattern in usage_patterns[:5]]

        # Add predictive insights
        predictions = predictive.predict_resource_usage()
        anomalies = predictive.detect_anomalies()
        recommendations = predictive.recommend_optimizations()

        report["predictions"] = predictions
        report["anomalies"] = anomalies
        report["recommendations"] = recommendations

        return report

    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return {
            "error": f"Failed to generate performance report: {str(e)}",
            "timestamp": None
        }


def get_usage_patterns_tool() -> Dict[str, Any]:
    """
    Analyze and return usage patterns across different user profiles.

    Returns:
        Usage pattern analysis including:
        - Frequent action sequences
        - User behavior clusters
        - Resource consumption patterns
        - Optimization opportunities
        - Pattern strength metrics
    """
    try:
        analyzer = get_usage_analyzer()
        patterns = analyzer.analyze_patterns()

        # Convert patterns to dictionaries for JSON serialization
        pattern_data = []
        for pattern in patterns:
            pattern_dict = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "avg_duration": pattern.avg_duration,
                "user_profiles": pattern.user_profiles,
                "resource_impact": pattern.resource_impact,
                "optimization_potential": pattern.optimization_potential,
                "last_observed": pattern.last_observed
            }
            pattern_data.append(pattern_dict)

        # Add pattern statistics
        total_patterns = len(patterns)
        high_potential_patterns = [p for p in patterns if p.optimization_potential > 0.7]

        result = {
            "patterns": pattern_data,
            "statistics": {
                "total_patterns": total_patterns,
                "high_optimization_potential": len(high_potential_patterns),
                "avg_optimization_potential": sum(p.optimization_potential for p in patterns) / max(total_patterns, 1),
                "most_common_pattern_type": max([p.pattern_type for p in patterns], key=[p.pattern_type for p in patterns].count) if patterns else None
            },
            "insights": _generate_usage_insights(patterns)
        }

        return result

    except Exception as e:
        logger.error(f"Error analyzing usage patterns: {e}")
        return {
            "error": f"Failed to analyze usage patterns: {str(e)}",
            "patterns": [],
            "statistics": {}
        }


def get_predictive_analytics_tool() -> Dict[str, Any]:
    """
    Get predictive analytics for performance optimization.

    Returns:
        Predictive analytics including:
        - Resource usage forecasting
        - Anomaly detection results
        - Performance optimization recommendations
        - Capacity planning insights
        - Predictive maintenance alerts
    """
    try:
        predictive = get_predictive_analytics()
        monitor = get_performance_monitor()

        # Get predictions
        predictions = predictive.predict_resource_usage()

        # Get anomalies
        anomalies = predictive.detect_anomalies()

        # Get recommendations
        recommendations = predictive.recommend_optimizations()

        # Add capacity planning insights
        capacity_insights = _generate_capacity_insights(monitor, predictions)

        # Add predictive maintenance alerts
        maintenance_alerts = _generate_maintenance_alerts(monitor, predictions, anomalies)

        result = {
            "predictions": predictions,
            "anomalies": anomalies,
            "recommendations": recommendations,
            "capacity_insights": capacity_insights,
            "maintenance_alerts": maintenance_alerts,
            "generated_at": monitor.start_time
        }

        return result

    except Exception as e:
        logger.error(f"Error getting predictive analytics: {e}")
        return {
            "error": f"Failed to get predictive analytics: {str(e)}",
            "predictions": {},
            "anomalies": [],
            "recommendations": []
        }


def _generate_usage_insights(patterns: List) -> List[Dict[str, Any]]:
    """Generate insights from usage patterns."""
    insights = []

    if not patterns:
        return insights

    # Insight 1: High-frequency patterns
    high_freq_patterns = [p for p in patterns if p.frequency > 10]
    if high_freq_patterns:
        insights.append({
            "type": "high_frequency",
            "title": "High-Frequency Usage Patterns Detected",
            "description": f"Found {len(high_freq_patterns)} patterns with high usage frequency",
            "patterns": [p.pattern_id for p in high_freq_patterns],
            "recommendation": "Consider optimizing these frequent operations"
        })

    # Insight 2: High optimization potential
    high_potential = [p for p in patterns if p.optimization_potential > 0.8]
    if high_potential:
        insights.append({
            "type": "optimization_opportunity",
            "title": "High Optimization Potential",
            "description": f"Found {len(high_potential)} patterns with high optimization potential",
            "patterns": [p.pattern_id for p in high_potential],
            "recommendation": "Prioritize optimization of these patterns"
        })

    # Insight 3: Resource-intensive patterns
    resource_intensive = []
    for pattern in patterns:
        total_impact = sum(pattern.resource_impact.values())
        if total_impact > 1.0:  # Threshold for high resource usage
            resource_intensive.append(pattern)

    if resource_intensive:
        insights.append({
            "type": "resource_intensive",
            "title": "Resource-Intensive Patterns",
            "description": f"Found {len(resource_intensive)} patterns with high resource consumption",
            "patterns": [p.pattern_id for p in resource_intensive],
            "recommendation": "Monitor and optimize resource usage for these patterns"
        })

    return insights


def _generate_capacity_insights(monitor, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate capacity planning insights."""
    insights = []

    # Analyze prediction trends
    for metric_name, prediction in predictions.items():
        if prediction.get("trend") == "increasing":
            slope = prediction.get("slope", 0)
            if slope > 0.5:  # Significant upward trend
                insights.append({
                    "type": "capacity_warning",
                    "metric": metric_name,
                    "severity": "high",
                    "message": f"{metric_name} showing significant upward trend",
                    "trend": prediction["trend"],
                    "slope": slope,
                    "recommendation": "Plan capacity increase within next planning cycle"
                })
            elif slope > 0.2:
                insights.append({
                    "type": "capacity_monitor",
                    "metric": metric_name,
                    "severity": "medium",
                    "message": f"{metric_name} trending upward",
                    "trend": prediction["trend"],
                    "slope": slope,
                    "recommendation": "Monitor closely for capacity planning"
                })

    # Current capacity analysis
    current_metrics = monitor.get_current_metrics()

    # Memory capacity insight
    memory_usage = current_metrics.get("memory_usage_percent", {}).get("value")
    if memory_usage and memory_usage > 75:
        insights.append({
            "type": "memory_capacity",
            "severity": "high" if memory_usage > 85 else "medium",
            "message": f"Memory usage at {memory_usage:.1f}%",
            "current_usage": memory_usage,
            "recommendation": "Consider memory optimization or scaling"
        })

    # CPU capacity insight
    cpu_usage = current_metrics.get("cpu_usage_percent", {}).get("value")
    if cpu_usage and cpu_usage > 80:
        insights.append({
            "type": "cpu_capacity",
            "severity": "high" if cpu_usage > 90 else "medium",
            "message": f"CPU usage at {cpu_usage:.1f}%",
            "current_usage": cpu_usage,
            "recommendation": "Monitor CPU usage and consider optimization"
        })

    return insights


def _generate_maintenance_alerts(monitor, predictions: Dict[str, Any],
                               anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate predictive maintenance alerts."""
    alerts = []

    # Anomaly-based alerts
    for anomaly in anomalies:
        if anomaly.get("severity") in ["high", "critical"]:
            alerts.append({
                "type": "anomaly_alert",
                "severity": anomaly["severity"],
                "metric": anomaly["metric_name"],
                "message": anomaly["description"],
                "value": anomaly["value"],
                "expected": anomaly["expected_value"],
                "z_score": anomaly["z_score"],
                "recommendation": "Investigate anomaly and consider maintenance"
            })

    # Prediction-based maintenance alerts
    for metric_name, prediction in predictions.items():
        confidence = prediction.get("confidence", 0)
        if confidence > 0.8:  # High confidence prediction
            if prediction.get("trend") == "increasing":
                # Check if predicted value exceeds critical threshold
                predicted_values = prediction.get("predicted_values", [])
                if predicted_values:
                    max_predicted = max(predicted_values)
                    # This is a simplified check - in practice you'd compare against baselines
                    if max_predicted > 90:  # Assuming 90 is a critical threshold
                        alerts.append({
                            "type": "predictive_maintenance",
                            "severity": "medium",
                            "metric": metric_name,
                            "message": f"Predicted {metric_name} may exceed safe limits",
                            "predicted_max": max_predicted,
                            "confidence": confidence,
                            "recommendation": "Schedule maintenance before predicted threshold breach"
                        })

    # Time-based maintenance alerts (simplified)
    uptime_hours = monitor.start_time / 3600
    if uptime_hours > 168:  # Week of uptime
        alerts.append({
            "type": "routine_maintenance",
            "severity": "low",
            "message": f"System has been running for {uptime_hours:.1f} hours",
            "uptime_hours": uptime_hours,
            "recommendation": "Consider routine maintenance and restart"
        })

def detect_performance_regressions_tool(current_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Detect performance regressions by comparing current results against baseline.

    This tool analyzes benchmark results to identify performance regressions using
    statistical methods and configurable thresholds. It can trigger alerts, create
    GitHub issues, and recommend rollback actions for critical regressions.

    Args:
        current_results: Current benchmark results to analyze. If None, uses latest available results.

    Returns:
        Comprehensive regression analysis report including:
        - Detected regressions with severity levels
        - Statistical significance analysis
        - Recommendations for remediation
        - Alert status and actions taken
    """
    try:
        from codesage_mcp.features.performance_monitoring.regression_detector import get_regression_detector

        detector = get_regression_detector()

        # If no current results provided, try to get from recent benchmark results
        if current_results is None:
            # Try to load from benchmark_results directory
            from pathlib import Path
            import json

            results_dir = Path("benchmark_results")
            if results_dir.exists():
                # Look for the most recent benchmark report
                json_files = list(results_dir.glob("benchmark_report_*.json"))
                if json_files:
                    latest_report = max(json_files, key=lambda f: f.stat().st_mtime)
                    try:
                        with open(latest_report, 'r') as f:
                            report_data = json.load(f)
                            # Extract results from the report
                            current_results = {}
                            for result in report_data.get("results", []):
                                metric_name = result["metric_name"]
                                current_results[metric_name] = {
                                    "value": result["value"],
                                    "unit": result["unit"],
                                    "timestamp": report_data.get("timestamp")
                                }
                    except Exception as e:
                        logger.warning(f"Failed to load benchmark report: {e}")

        if not current_results:
            return {
                "error": "No current benchmark results available for regression analysis",
                "message": "Please run benchmarks first or provide current_results parameter",
                "timestamp": None
            }

        # Run regression detection
        report = detector.detect_regressions(current_results)

        # Convert dataclasses to dictionaries for JSON serialization
        result = {
            "test_run_id": report.test_run_id,
            "timestamp": report.timestamp,
            "regressions_detected": [
                {
                    "metric_name": r.metric_name,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "percentage_change": r.percentage_change,
                    "is_regression": r.is_regression,
                    "severity": r.severity,
                    "statistical_significance": r.statistical_significance,
                    "p_value": r.p_value,
                    "sample_size": r.sample_size,
                    "metadata": r.metadata
                }
                for r in report.regressions_detected
            ],
            "summary": report.summary,
            "recommendations": report.recommendations,
            "alerts_triggered": len(report.regressions_detected) > 0,
            "critical_regressions": len([r for r in report.regressions_detected if r.severity == "critical"])
        }

        # Add additional context
        result["analysis_complete"] = True
        result["baseline_available"] = report.summary.get("total_metrics_analyzed", 0) > 0

        return result

    except Exception as e:
        logger.error(f"Error detecting performance regressions: {e}")
        return {
            "error": f"Failed to detect performance regressions: {str(e)}",
            "analysis_complete": False,
            "regressions_detected": [],
            "recommendations": ["Investigate the error and try again"],
            "timestamp": None
        }
    return alerts