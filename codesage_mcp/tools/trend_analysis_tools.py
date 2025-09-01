"""
Trend Analysis Tools for CodeSage MCP Server.

This module provides tools for analyzing performance trends over time,
identifying optimization opportunities, and generating predictive insights.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.performance_monitoring.trend_analysis import (
    get_trend_analyzer,
    get_optimization_recommender,
    get_performance_predictor
)

logger = logging.getLogger(__name__)


def analyze_performance_trends_tool(metric_name: str = None, analysis_window_days: int = 30) -> Dict[str, Any]:
    """
    Analyze performance trends for specific metrics or all metrics.

    Args:
        metric_name: Optional specific metric to analyze (e.g., 'response_time_ms', 'memory_usage_percent')
        analysis_window_days: Number of days to analyze (default: 30)

    Returns:
        Dictionary containing trend analysis results including direction, slope,
        confidence, and performance predictions.
    """
    try:
        analyzer = get_trend_analyzer()

        if metric_name:
            # Analyze specific metric
            trend = analyzer.analyze_trend(metric_name, analysis_window_days)
            if trend:
                # Generate prediction
                predictor = get_performance_predictor()
                prediction = predictor.predict_future_performance(metric_name, 7)  # 7-day prediction

                return {
                    "metric_name": metric_name,
                    "trend_analysis": {
                        "direction": trend.direction.value,
                        "slope": trend.slope,
                        "confidence": trend.confidence,
                        "data_points": trend.data_points,
                        "time_window_days": trend.time_window_days,
                        "start_value": trend.start_value,
                        "end_value": trend.end_value,
                        "volatility": trend.volatility,
                        "analysis_timestamp": trend.analysis_timestamp
                    },
                    "prediction": prediction,
                    "interpretation": _interpret_trend(trend),
                    "recommendations": _generate_trend_recommendations(trend)
                }
            else:
                return {
                    "error": f"No trend data available for metric: {metric_name}",
                    "available_metrics": list(analyzer.metric_history.keys())
                }
        else:
            # Analyze all metrics
            trends = analyzer.get_all_trends(analysis_window_days)
            trending_issues = analyzer.identify_trending_issues()

            return {
                "analysis_window_days": analysis_window_days,
                "trends_analyzed": len(trends),
                "trends": {
                    metric_name: {
                        "direction": trend.direction.value,
                        "slope": trend.slope,
                        "confidence": trend.confidence,
                        "volatility": trend.volatility,
                        "interpretation": _interpret_trend(trend)
                    }
                    for metric_name, trend in trends.items()
                },
                "trending_issues": trending_issues,
                "summary": _generate_trends_summary(trends, trending_issues)
            }

    except Exception as e:
        logger.error(f"Error analyzing performance trends: {e}")
        return {
            "error": f"Failed to analyze performance trends: {str(e)}"
        }


def get_optimization_opportunities_tool() -> Dict[str, Any]:
    """
    Identify and prioritize optimization opportunities based on performance trends.

    Returns:
        Dictionary containing prioritized optimization opportunities with detailed
        analysis, implementation guidance, and expected benefits.
    """
    try:
        recommender = get_optimization_recommender()
        opportunities = recommender.generate_recommendations()

        # Convert opportunities to dictionaries for JSON serialization
        opportunities_data = []
        for opp in opportunities:
            opp_dict = {
                "opportunity_id": opp.opportunity_id,
                "title": opp.title,
                "description": opp.description,
                "metric_affected": opp.metric_affected,
                "current_impact": opp.current_impact,
                "potential_improvement": opp.potential_improvement,
                "priority": opp.priority.value,
                "effort_estimate": opp.effort_estimate,
                "implementation_complexity": opp.implementation_complexity,
                "expected_benefits": opp.expected_benefits,
                "risks": opp.risks,
                "prerequisites": opp.prerequisites,
                "timeline_estimate": opp.timeline_estimate,
                "discovered_at": opp.discovered_at
            }
            opportunities_data.append(opp_dict)

        # Group by priority
        priority_groups = {
            "critical": [opp for opp in opportunities_data if opp["priority"] == "critical"],
            "high": [opp for opp in opportunities_data if opp["priority"] == "high"],
            "medium": [opp for opp in opportunities_data if opp["priority"] == "medium"],
            "low": [opp for opp in opportunities_data if opp["priority"] == "low"]
        }

        return {
            "total_opportunities": len(opportunities_data),
            "opportunities_by_priority": priority_groups,
            "top_recommendations": opportunities_data[:5],  # Top 5 opportunities
            "implementation_roadmap": _generate_implementation_roadmap(opportunities_data),
            "expected_roi": _calculate_expected_roi(opportunities_data),
            "generated_at": opportunities[0].discovered_at if opportunities else None
        }

    except Exception as e:
        logger.exception(f"Error getting optimization opportunities: {e}")
        return {
            "error": f"Failed to get optimization opportunities: {str(e)}"
        }


def predict_performance_capacity_tool(target_response_time_ms: int = 100) -> Dict[str, Any]:
    """
    Predict maximum workload capacity for target performance levels.

    Args:
        target_response_time_ms: Target response time in milliseconds (default: 100ms)

    Returns:
        Dictionary containing capacity predictions, headroom analysis,
        and scaling recommendations.
    """
    try:
        predictor = get_performance_predictor()
        capacity_prediction = predictor.predict_workload_capacity(target_response_time_ms)

        if "error" in capacity_prediction:
            return capacity_prediction

        # Generate scaling recommendations
        scaling_recommendations = _generate_scaling_recommendations(capacity_prediction)

        # Generate capacity planning insights
        capacity_insights = _generate_capacity_planning_insights(capacity_prediction)

        return {
            "capacity_prediction": capacity_prediction,
            "scaling_recommendations": scaling_recommendations,
            "capacity_insights": capacity_insights,
            "planning_guidance": _generate_capacity_planning_guidance(capacity_prediction)
        }

    except Exception as e:
        logger.error(f"Error predicting performance capacity: {e}")
        return {
            "error": f"Failed to predict performance capacity: {str(e)}"
        }


def forecast_performance_trends_tool(metric_name: str, forecast_days: int = 30) -> Dict[str, Any]:
    """
    Forecast future performance trends using predictive analytics.

    Args:
        metric_name: Name of the metric to forecast
        forecast_days: Number of days to forecast (default: 30)

    Returns:
        Dictionary containing performance forecasts, confidence intervals,
        and trend analysis.
    """
    try:
        predictor = get_performance_predictor()
        analyzer = get_trend_analyzer()

        # Get current trend
        current_trend = analyzer.analyze_trend(metric_name)

        if not current_trend:
            return {
                "error": f"No historical data available for metric: {metric_name}",
                "available_metrics": list(analyzer.metric_history.keys())
            }

        # Generate forecast
        forecast = predictor.predict_future_performance(metric_name, forecast_days)

        # Generate seasonal forecast if enough data
        seasonal_forecast = predictor.forecast_seasonal_patterns(metric_name, forecast_days)

        # Generate forecast insights
        forecast_insights = _generate_forecast_insights(forecast, current_trend)

        return {
            "metric_name": metric_name,
            "current_trend": {
                "direction": current_trend.direction.value,
                "slope": current_trend.slope,
                "confidence": current_trend.confidence,
                "volatility": current_trend.volatility
            },
            "forecast": forecast,
            "seasonal_forecast": seasonal_forecast,
            "forecast_insights": forecast_insights,
            "recommendations": _generate_forecast_recommendations(forecast, current_trend)
        }

    except Exception as e:
        logger.error(f"Error forecasting performance trends: {e}")
        return {
            "error": f"Failed to forecast performance trends: {str(e)}"
        }


def get_performance_baseline_comparison_tool() -> Dict[str, Any]:
    """
    Compare current performance against established baselines.

    Returns:
        Dictionary containing baseline comparison analysis, compliance status,
        and recommendations for maintaining or improving baseline performance.
    """
    try:
        analyzer = get_trend_analyzer()
        trends = analyzer.get_all_trends()

        # Define baseline targets (these would typically come from configuration)
        baselines = {
            "response_time_ms": {"target": 25.0, "warning": 50.0, "critical": 100.0},
            "memory_usage_percent": {"target": 32.5, "warning": 70.0, "critical": 90.0},
            "cpu_usage_percent": {"target": 65.0, "warning": 85.0, "critical": 95.0},
            "cache_hit_rate": {"target": 95.0, "warning": 90.0, "critical": 80.0},
            "error_rate_percent": {"target": 0.05, "warning": 0.5, "critical": 1.0},
            "throughput_rps": {"target": 1500.0, "warning": 1800.0, "critical": 2000.0}
        }

        baseline_comparison = {}

        for metric_name, baseline in baselines.items():
            trend = trends.get(metric_name)
            if trend:
                current_value = trend.end_value
                status = "good"
                deviation = 0

                # Determine status based on baseline thresholds
                if metric_name in ["response_time_ms", "memory_usage_percent", "cpu_usage_percent", "error_rate_percent"]:
                    # Higher values are worse for these metrics
                    if current_value > baseline["critical"]:
                        status = "critical"
                        deviation = current_value - baseline["target"]
                    elif current_value > baseline["warning"]:
                        status = "warning"
                        deviation = current_value - baseline["target"]
                elif metric_name in ["cache_hit_rate", "throughput_rps"]:
                    # Lower values are worse for these metrics
                    if current_value < baseline["critical"]:
                        status = "critical"
                        deviation = baseline["target"] - current_value
                    elif current_value < baseline["warning"]:
                        status = "warning"
                        deviation = baseline["target"] - current_value

                baseline_comparison[metric_name] = {
                    "current_value": current_value,
                    "target": baseline["target"],
                    "warning_threshold": baseline["warning"],
                    "critical_threshold": baseline["critical"],
                    "status": status,
                    "deviation": deviation,
                    "deviation_percent": (deviation / baseline["target"]) * 100 if baseline["target"] != 0 else 0,
                    "trend_direction": trend.direction.value,
                    "compliance_score": _calculate_baseline_compliance(current_value, baseline, metric_name)
                }

        # Calculate overall baseline compliance
        compliance_scores = [data["compliance_score"] for data in baseline_comparison.values()]
        overall_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0

        # Generate baseline recommendations
        baseline_recommendations = _generate_baseline_recommendations(baseline_comparison)

        return {
            "baseline_comparison": baseline_comparison,
            "overall_compliance_score": overall_compliance,
            "compliance_rating": _interpret_compliance_score(overall_compliance),
            "baseline_recommendations": baseline_recommendations,
            "metrics_analyzed": len(baseline_comparison),
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error comparing performance baselines: {e}")
        return {
            "error": f"Failed to compare performance baselines: {str(e)}"
        }


def _interpret_trend(trend) -> str:
    """Interpret a performance trend in human-readable terms."""
    if trend.direction.value == "improving":
        if trend.slope < -0.1:
            return f"Strongly improving ({trend.slope:.3f} per day)"
        else:
            return f"Gradually improving ({trend.slope:.3f} per day)"
    elif trend.direction.value == "degrading":
        if trend.slope > 0.1:
            return f"Strongly degrading (+{trend.slope:.3f} per day)"
        else:
            return f"Gradually degrading (+{trend.slope:.3f} per day)"
    elif trend.direction.value == "volatile":
        return f"Highly volatile (volatility: {trend.volatility:.2f})"
    else:
        return f"Stable performance (slope: {trend.slope:.3f} per day)"


def _generate_trend_recommendations(trend) -> List[str]:
    """Generate recommendations based on trend analysis."""
    recommendations = []

    if trend.direction.value == "degrading":
        if "response_time" in trend.metric_name:
            recommendations.extend([
                "Profile application performance to identify bottlenecks",
                "Consider implementing caching for frequently accessed data",
                "Review database query optimization",
                "Evaluate need for horizontal scaling"
            ])
        elif "memory_usage" in trend.metric_name:
            recommendations.extend([
                "Monitor for memory leaks",
                "Implement memory-efficient data structures",
                "Consider adjusting cache sizes",
                "Review garbage collection settings"
            ])
        elif "error_rate" in trend.metric_name:
            recommendations.extend([
                "Analyze error patterns and root causes",
                "Improve error handling and recovery",
                "Review input validation",
                "Enhance monitoring and alerting"
            ])

    elif trend.direction.value == "volatile":
        recommendations.extend([
            "Investigate sources of performance variability",
            "Implement more consistent resource allocation",
            "Review background process scheduling",
            "Consider workload isolation techniques"
        ])

    return recommendations


def _generate_trends_summary(trends: Dict, issues: List) -> Dict[str, Any]:
    """Generate a summary of trend analysis results."""
    if not trends:
        return {"summary": "No trend data available"}

    improving_metrics = [name for name, trend in trends.items() if trend.direction.value == "improving"]
    degrading_metrics = [name for name, trend in trends.items() if trend.direction.value == "degrading"]
    volatile_metrics = [name for name, trend in trends.items() if trend.direction.value == "volatile"]

    summary = {
        "total_metrics_analyzed": len(trends),
        "improving_metrics": len(improving_metrics),
        "degrading_metrics": len(degrading_metrics),
        "volatile_metrics": len(volatile_metrics),
        "stable_metrics": len(trends) - len(improving_metrics) - len(degrading_metrics) - len(volatile_metrics),
        "active_issues": len(issues),
        "overall_health_score": _calculate_overall_health_score(trends, issues)
    }

    if improving_metrics:
        summary["top_improving"] = improving_metrics[:3]
    if degrading_metrics:
        summary["needs_attention"] = degrading_metrics[:3]
    if issues:
        summary["critical_issues"] = sum(1 for issue in issues if issue["severity"] == "high")

    return summary


def _calculate_overall_health_score(trends: Dict, issues: List) -> float:
    """Calculate an overall health score based on trends and issues."""
    if not trends:
        return 50.0

    # Base score
    score = 100.0

    # Deduct points for degrading trends
    degrading_count = sum(1 for trend in trends.values() if trend.direction.value == "degrading")
    score -= degrading_count * 10

    # Deduct points for volatile trends
    volatile_count = sum(1 for trend in trends.values() if trend.direction.value == "volatile")
    score -= volatile_count * 5

    # Deduct points for issues
    high_severity_issues = sum(1 for issue in issues if issue["severity"] == "high")
    medium_severity_issues = sum(1 for issue in issues if issue["severity"] == "medium")
    score -= high_severity_issues * 15
    score -= medium_severity_issues * 7

    return max(0.0, min(100.0, score))


def _generate_implementation_roadmap(opportunities: List[Dict]) -> Dict[str, Any]:
    """Generate an implementation roadmap for optimization opportunities."""
    # Group by timeline
    roadmap = {
        "immediate": [],  # 1-2 weeks
        "short_term": [],  # 1-2 months
        "medium_term": [],  # 3-6 months
        "long_term": []  # 6+ months
    }

    for opp in opportunities:
        timeline = opp["timeline_estimate"]

        if "week" in timeline:
            roadmap["immediate"].append(opp)
        elif "month" in timeline and "1" in timeline:
            roadmap["short_term"].append(opp)
        elif "month" in timeline and ("3" in timeline or "6" in timeline):
            roadmap["medium_term"].append(opp)
        else:
            roadmap["long_term"].append(opp)

    return {
        "roadmap": roadmap,
        "immediate_count": len(roadmap["immediate"]),
        "short_term_count": len(roadmap["short_term"]),
        "medium_term_count": len(roadmap["medium_term"]),
        "long_term_count": len(roadmap["long_term"]),
        "total_scheduled": sum(len(phase) for phase in roadmap.values())
    }


def _calculate_expected_roi(opportunities: List[Dict]) -> Dict[str, Any]:
    """Calculate expected ROI for optimization opportunities."""
    # Simplified ROI calculation
    total_investment_effort = sum(_estimate_effort_days(opp["effort_estimate"]) for opp in opportunities)
    total_expected_benefit = sum(opp["potential_improvement"] for opp in opportunities)

    # Assume each unit of improvement translates to business value
    roi_multiplier = 5  # 5x return on performance improvements
    expected_roi = (total_expected_benefit * roi_multiplier) / max(total_investment_effort, 1)

    return {
        "total_opportunities": len(opportunities),
        "estimated_effort_days": total_investment_effort,
        "expected_benefit_score": total_expected_benefit,
        "expected_roi_multiplier": expected_roi,
        "roi_interpretation": "Excellent" if expected_roi > 10 else "Good" if expected_roi > 5 else "Moderate" if expected_roi > 2 else "Low",
        "payback_period_months": total_investment_effort / 30  # Rough estimate
    }


def _estimate_effort_days(effort_estimate: str) -> float:
    """Convert effort estimate string to days."""
    if "week" in effort_estimate:
        weeks = float(effort_estimate.split("-")[0]) if "-" in effort_estimate else float(effort_estimate.split()[0])
        return weeks * 5  # Assume 5 working days per week
    elif "month" in effort_estimate:
        months = float(effort_estimate.split("-")[0]) if "-" in effort_estimate else float(effort_estimate.split()[0])
        return months * 20  # Assume 20 working days per month
    else:
        return 10  # Default estimate


def _generate_scaling_recommendations(capacity_prediction: Dict) -> List[Dict]:
    """Generate scaling recommendations based on capacity prediction."""
    recommendations = []

    headroom_percent = capacity_prediction.get("capacity_headroom_percent", 0)

    if headroom_percent < 10:
        recommendations.append({
            "type": "immediate_scaling",
            "priority": "critical",
            "description": "Immediate scaling required - very low capacity headroom",
            "action": "Implement horizontal scaling or optimize current resources",
            "timeline": "Within 1 week"
        })
    elif headroom_percent < 25:
        recommendations.append({
            "type": "near_term_scaling",
            "priority": "high",
            "description": "Plan scaling within next planning cycle",
            "action": "Prepare scaling infrastructure and procedures",
            "timeline": "Within 1 month"
        })
    elif headroom_percent > 100:
        recommendations.append({
            "type": "optimization_opportunity",
            "priority": "medium",
            "description": "Significant capacity headroom available",
            "action": "Consider resource optimization or consolidation",
            "timeline": "Next maintenance window"
        })

    return recommendations


def _generate_capacity_planning_insights(capacity_prediction: Dict) -> List[str]:
    """Generate capacity planning insights."""
    insights = []

    confidence = capacity_prediction.get("confidence_level", 0)
    headroom = capacity_prediction.get("capacity_headroom_percent", 0)

    if confidence > 0.8:
        insights.append("High confidence in capacity prediction - reliable for planning")
    elif confidence < 0.5:
        insights.append("Low confidence in capacity prediction - collect more data")

    if headroom > 50:
        insights.append("Good capacity headroom - can handle traffic spikes")
    elif headroom < 20:
        insights.append("Limited capacity headroom - monitor closely")

    return insights


def _generate_capacity_planning_guidance(capacity_prediction: Dict) -> Dict[str, Any]:
    """Generate capacity planning guidance."""
    current_throughput = capacity_prediction.get("current_throughput_rps", 0)
    max_capacity = capacity_prediction.get("estimated_max_capacity_rps", 0)

    return {
        "current_sustainable_load": current_throughput,
        "maximum_capacity": max_capacity,
        "recommended_buffer": max_capacity * 0.8,  # 80% of max capacity
        "scaling_trigger_points": {
            "warning": current_throughput * 1.5,
            "critical": current_throughput * 1.8
        },
        "planning_horizon_months": 3,
        "data_retention_days": 90
    }


def _generate_forecast_insights(forecast: Dict, current_trend) -> List[str]:
    """Generate insights from performance forecast."""
    insights = []

    if "predicted_change" in forecast:
        change = forecast["predicted_change"]
        if change > 10:
            insights.append(f"Significant performance degradation expected (+{change:.1f})")
        elif change < -10:
            insights.append(f"Significant performance improvement expected ({change:.1f})")
        else:
            insights.append("Stable performance expected")

    confidence = forecast.get("prediction_confidence", 0)
    if confidence > 0.8:
        insights.append("High confidence in forecast")
    elif confidence < 0.5:
        insights.append("Low confidence - more data needed")

    return insights


def _generate_forecast_recommendations(forecast: Dict, current_trend) -> List[str]:
    """Generate recommendations based on forecast."""
    recommendations = []

    if "predicted_change" in forecast:
        change = forecast["predicted_change"]
        if change > 20:
            recommendations.extend([
                "Prepare contingency plans for performance degradation",
                "Consider immediate optimization measures",
                "Review capacity planning assumptions"
            ])
        elif change < -20:
            recommendations.extend([
                "Monitor for continued improvement",
                "Consider resource reallocation",
                "Document successful optimization strategies"
            ])

    return recommendations


def _calculate_baseline_compliance(current_value: float, baseline: Dict, metric_name: str) -> float:
    """Calculate compliance score for a baseline metric."""
    target = baseline["target"]
    warning = baseline["warning"]
    critical = baseline["critical"]

    if metric_name in ["response_time_ms", "memory_usage_percent", "cpu_usage_percent", "error_rate_percent"]:
        # Higher values are worse
        if current_value <= target:
            return 100.0
        elif current_value <= warning:
            return 75.0 - ((current_value - target) / (warning - target)) * 25.0
        elif current_value <= critical:
            return 50.0 - ((current_value - warning) / (critical - warning)) * 50.0
        else:
            return max(0.0, 25.0 - ((current_value - critical) / critical) * 25.0)
    else:
        # Lower values are worse (cache hit rate, throughput)
        if current_value >= target:
            return 100.0
        elif current_value >= warning:
            return 75.0 - ((target - current_value) / (target - warning)) * 25.0
        elif current_value >= critical:
            return 50.0 - ((warning - current_value) / (warning - critical)) * 50.0
        else:
            return max(0.0, 25.0 - ((critical - current_value) / critical) * 25.0)


def _interpret_compliance_score(score: float) -> str:
    """Interpret a compliance score."""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Fair"
    elif score >= 60:
        return "Poor"
    else:
        return "Critical"


def _generate_baseline_recommendations(baseline_comparison: Dict) -> List[Dict]:
    """Generate recommendations for baseline compliance."""
    recommendations = []

    critical_metrics = [name for name, data in baseline_comparison.items() if data["status"] == "critical"]
    warning_metrics = [name for name, data in baseline_comparison.items() if data["status"] == "warning"]

    if critical_metrics:
        recommendations.append({
            "priority": "critical",
            "description": f"Address critical baseline violations in: {', '.join(critical_metrics)}",
            "action": "Immediate corrective action required",
            "timeline": "Within 24 hours"
        })

    if warning_metrics:
        recommendations.append({
            "priority": "high",
            "description": f"Review warning-level metrics: {', '.join(warning_metrics)}",
            "action": "Investigate and plan corrective measures",
            "timeline": "Within 1 week"
        })

    # Overall compliance recommendations
    overall_compliance = sum(data["compliance_score"] for data in baseline_comparison.values()) / len(baseline_comparison)

    if overall_compliance < 70:
        recommendations.append({
            "priority": "high",
            "description": "Overall baseline compliance is poor",
            "action": "Conduct comprehensive performance review",
            "timeline": "Within 2 weeks"
        })

    return recommendations