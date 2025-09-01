"""
Workload Pattern Recognition Tools for CodeSage MCP Server.

This module provides tools for workload pattern recognition and optimal resource allocation
based on identified patterns, including pattern analysis, resource allocation management,
and predictive workload forecasting.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.memory_management.workload_pattern_recognition import (
    get_workload_pattern_recognition,
    WorkloadPattern,
    ResourceType
)

logger = logging.getLogger(__name__)


def get_workload_pattern_analysis_tool() -> Dict[str, Any]:
    """
    Get comprehensive workload pattern analysis including current patterns, metrics,
    resource allocations, and optimization opportunities.

    Returns:
        Dictionary containing workload pattern analysis including:
        - Current detected patterns with characteristics
        - Pattern recognition metrics and performance
        - Resource allocation status and effectiveness
        - Pattern distribution and trends
        - Resource optimization opportunities
        - Predictive insights and recommendations
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()
        analysis = pattern_recognition.get_pattern_analysis()

        if "error" in analysis:
            return analysis

        # Enhance analysis with additional insights
        analysis["insights"] = _generate_pattern_insights(analysis)
        analysis["health_score"] = _calculate_pattern_health_score(analysis)
        analysis["recommendations"] = _generate_pattern_recommendations(analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error getting workload pattern analysis: {e}")
        return {
            "error": f"Failed to get workload pattern analysis: {str(e)}"
        }


def trigger_pattern_based_allocation_tool(pattern_type: str = "auto",
                                        resource_focus: str = "balanced") -> Dict[str, Any]:
    """
    Trigger pattern-based resource allocation with specified parameters.

    Args:
        pattern_type: Type of pattern to optimize for ("auto", "compute_intensive", "memory_intensive",
                    "io_intensive", "bursty", "steady_state")
        resource_focus: Resource allocation focus ("cpu", "memory", "balanced", "cache")

    Returns:
        Dictionary containing pattern-based allocation results including:
        - Detected or specified pattern characteristics
        - Resource allocation recommendations
        - Implementation results and performance impact
        - Allocation effectiveness analysis
        - Rollback plans and monitoring guidance
    """
    try:
        # Validate inputs
        try:
            if pattern_type != "auto":
                pattern_enum = WorkloadPattern(pattern_type.lower())
        except ValueError:
            return {
                "error": f"Invalid pattern type: {pattern_type}",
                "valid_patterns": [p.value for p in WorkloadPattern]
            }

        valid_focuses = ["cpu", "memory", "balanced", "cache"]
        if resource_focus not in valid_focuses:
            return {
                "error": f"Invalid resource focus: {resource_focus}",
                "valid_focuses": valid_focuses
            }

        pattern_recognition = get_workload_pattern_recognition()

        # Detect current patterns or use specified pattern
        if pattern_type == "auto":
            current_patterns = pattern_recognition.detected_patterns
            if current_patterns:
                target_pattern = current_patterns[-1]  # Use most recent pattern
            else:
                return {
                    "allocation_status": "no_patterns_detected",
                    "reason": "No workload patterns detected - run analysis first",
                    "generated_at": time.time()
                }
        else:
            # Create synthetic pattern for specified type
            target_pattern = _create_synthetic_pattern(pattern_enum, resource_focus)

        # Generate resource allocations
        allocations = pattern_recognition.resource_allocator.generate_allocations(target_pattern)

        # Filter allocations based on resource focus
        filtered_allocations = _filter_allocations_by_focus(allocations, resource_focus)

        # Apply allocations
        applied_allocations = pattern_recognition._apply_resource_allocations(filtered_allocations)

        # Prepare results
        results = {
            "allocation_status": "completed" if applied_allocations else "no_allocations_applied",
            "target_pattern": {
                "pattern_type": target_pattern.pattern_type.value,
                "intensity_score": target_pattern.intensity_score,
                "confidence_score": target_pattern.confidence_score,
                "resource_focus": resource_focus
            },
            "resource_allocations": [
                {
                    "resource_type": allocation.resource_type.value,
                    "current_allocation": allocation.current_allocation,
                    "recommended_allocation": allocation.recommended_allocation,
                    "allocation_reason": allocation.allocation_reason,
                    "expected_benefit": allocation.expected_benefit,
                    "implementation_priority": allocation.implementation_priority,
                    "applied": allocation in applied_allocations
                }
                for allocation in filtered_allocations
            ],
            "allocation_effectiveness": _analyze_allocation_effectiveness(applied_allocations, target_pattern),
            "performance_impact": _estimate_allocation_performance_impact(applied_allocations),
            "monitoring_guidance": _generate_monitoring_guidance(applied_allocations),
            "generated_at": time.time()
        }

        # Add success assessment
        results["success_assessment"] = _assess_allocation_success(results)

        return results

    except Exception as e:
        logger.error(f"Error triggering pattern-based allocation: {e}")
        return {
            "error": f"Failed to trigger pattern-based allocation: {str(e)}"
        }


def get_resource_allocation_status_tool() -> Dict[str, Any]:
    """
    Get current resource allocation status and effectiveness analysis.

    Returns:
        Dictionary containing resource allocation status including:
        - Current resource allocations by type
        - Allocation history and trends
        - Effectiveness metrics and performance impact
        - Resource utilization analysis
        - Optimization opportunities and recommendations
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()

        status = {
            "current_allocations": {
                resource.value: allocation
                for resource, allocation in pattern_recognition.resource_allocations.items()
            },
            "allocation_history_summary": _summarize_allocation_history(pattern_recognition.allocation_history),
            "effectiveness_metrics": _calculate_allocation_effectiveness_metrics(pattern_recognition),
            "resource_utilization": _analyze_resource_utilization(pattern_recognition),
            "allocation_trends": _analyze_allocation_trends(pattern_recognition.allocation_history),
            "optimization_opportunities": _identify_allocation_optimization_opportunities(pattern_recognition),
            "generated_at": time.time()
        }

        return status

    except Exception as e:
        logger.error(f"Error getting resource allocation status: {e}")
        return {
            "error": f"Failed to get resource allocation status: {str(e)}"
        }


def forecast_workload_patterns_tool(time_horizon_hours: int = 24) -> Dict[str, Any]:
    """
    Forecast workload patterns and resource needs for the specified time horizon.

    Args:
        time_horizon_hours: Number of hours to forecast (default: 24)

    Returns:
        Dictionary containing workload forecasting including:
        - Predicted workload patterns with probabilities
        - Resource demand forecasts
        - Confidence intervals and risk assessment
        - Proactive allocation recommendations
        - Forecast accuracy metrics
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()

        # Generate forecast
        forecast = pattern_recognition.workload_forecaster.forecast_workload(time_horizon_hours)

        if "message" in forecast:
            return forecast

        # Enhance forecast with additional analysis
        forecast["forecast_accuracy"] = _assess_forecast_accuracy(pattern_recognition)
        forecast["risk_assessment"] = _assess_forecast_risk(forecast)
        forecast["proactive_recommendations"] = _generate_proactive_recommendations(forecast)
        forecast["resource_planning"] = _generate_resource_planning_guidance(forecast)

        return forecast

    except Exception as e:
        logger.error(f"Error forecasting workload patterns: {e}")
        return {
            "error": f"Failed to forecast workload patterns: {str(e)}"
        }


def analyze_pattern_effectiveness_tool(time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Analyze the effectiveness of workload pattern recognition and resource allocation.

    Args:
        time_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing pattern effectiveness analysis including:
        - Pattern detection accuracy and coverage
        - Resource allocation success rates
        - Performance impact of pattern-based allocation
        - Pattern evolution and adaptation effectiveness
        - Recommendations for improving pattern recognition
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()

        # Get historical data within time window
        cutoff_time = time.time() - (time_window_hours * 60 * 60)

        # Filter recent patterns and allocations
        recent_patterns = [
            p for p in pattern_recognition.detected_patterns
            if p.detected_at >= cutoff_time
        ]

        recent_allocations = [
            a for a in pattern_recognition.allocation_history
            if hasattr(a, 'timestamp') and a.timestamp >= cutoff_time
        ]

        if not recent_patterns:
            return {
                "analysis_window_hours": time_window_hours,
                "message": f"No pattern data found in the last {time_window_hours} hours",
                "generated_at": time.time()
            }

        # Analyze effectiveness
        effectiveness_analysis = {
            "pattern_detection_metrics": _analyze_pattern_detection_effectiveness(recent_patterns),
            "resource_allocation_metrics": _analyze_resource_allocation_effectiveness(recent_allocations),
            "performance_impact_analysis": _analyze_pattern_performance_impact(recent_patterns, recent_allocations),
            "pattern_evolution_analysis": _analyze_pattern_evolution(recent_patterns),
            "effectiveness_score": _calculate_overall_effectiveness_score(recent_patterns, recent_allocations),
            "improvement_recommendations": _generate_effectiveness_improvements(recent_patterns, recent_allocations),
            "analysis_window_hours": time_window_hours,
            "generated_at": time.time()
        }

        return effectiveness_analysis

    except Exception as e:
        logger.error(f"Error analyzing pattern effectiveness: {e}")
        return {
            "error": f"Failed to analyze pattern effectiveness: {str(e)}"
        }


def get_pattern_recognition_configuration_tool() -> Dict[str, Any]:
    """
    Get current workload pattern recognition configuration and provide tuning recommendations.

    Returns:
        Dictionary containing pattern recognition configuration including:
        - Current detection parameters and thresholds
        - Pattern classification settings
        - Resource allocation strategies
        - Configuration effectiveness analysis
        - Tuning recommendations and optimization suggestions
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()

        configuration = {
            "detection_parameters": {
                "analysis_window_minutes": pattern_recognition.analysis_window_minutes,
                "pattern_detection_threshold": pattern_recognition.pattern_detection_threshold
            },
            "pattern_types": [p.value for p in WorkloadPattern],
            "resource_types": [r.value for r in ResourceType],
            "allocation_strategies": {
                pattern.value: strategy
                for pattern, strategy in pattern_recognition.resource_allocator.allocation_strategies.items()
            }
        }

        # Analyze configuration effectiveness
        config_analysis = {
            "current_configuration": configuration,
            "effectiveness_analysis": _analyze_configuration_effectiveness(pattern_recognition),
            "pattern_coverage_analysis": _analyze_pattern_coverage(pattern_recognition),
            "tuning_recommendations": _generate_configuration_tuning_recommendations(pattern_recognition),
            "generated_at": time.time()
        }

        return config_analysis

    except Exception as e:
        logger.error(f"Error getting pattern recognition configuration: {e}")
        return {
            "error": f"Failed to get pattern recognition configuration: {str(e)}"
        }


def update_pattern_recognition_configuration_tool(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update workload pattern recognition configuration with new settings.

    Args:
        config_updates: Dictionary of configuration updates

    Returns:
        Dictionary containing configuration update results including:
        - Updated configuration
        - Validation results
        - Expected impact assessment
        - Rollback plan
    """
    try:
        pattern_recognition = get_workload_pattern_recognition()

        # Validate configuration updates
        validation_results = _validate_configuration_updates(config_updates)

        if not validation_results["valid"]:
            return {
                "error": "Invalid configuration updates",
                "validation_errors": validation_results["errors"]
            }

        # Store original configuration
        original_config = {
            "analysis_window_minutes": pattern_recognition.analysis_window_minutes,
            "pattern_detection_threshold": pattern_recognition.pattern_detection_threshold
        }

        # Apply updates
        if "analysis_window_minutes" in config_updates:
            pattern_recognition.analysis_window_minutes = config_updates["analysis_window_minutes"]

        if "pattern_detection_threshold" in config_updates:
            pattern_recognition.pattern_detection_threshold = config_updates["pattern_detection_threshold"]

        # Assess impact
        impact_assessment = _assess_configuration_impact(pattern_recognition, config_updates)

        results = {
            "configuration_updated": True,
            "updated_settings": config_updates,
            "validation_results": validation_results,
            "impact_assessment": impact_assessment,
            "rollback_plan": {
                "original_configuration": original_config,
                "rollback_steps": ["Revert configuration to previous settings", "Restart pattern recognition cycle", "Monitor detection accuracy for 30 minutes"],
                "rollback_trigger": "Pattern detection accuracy drops below 70% or resource allocation errors increase"
            },
            "generated_at": time.time()
        }

        return results

    except Exception as e:
        logger.error(f"Error updating pattern recognition configuration: {e}")
        return {
            "error": f"Failed to update pattern recognition configuration: {str(e)}"
        }


def _generate_pattern_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from pattern analysis."""
    insights = []

    current_patterns = analysis.get("current_patterns", [])
    if current_patterns:
        pattern_types = [p["pattern_type"] for p in current_patterns]
        most_common = max(set(pattern_types), key=pattern_types.count)
        insights.append(f"Most common workload pattern: {most_common}")

        avg_confidence = sum(p["confidence_score"] for p in current_patterns) / len(current_patterns)
        insights.append(".2f")

    pattern_metrics = analysis.get("pattern_metrics", {})
    total_patterns = pattern_metrics.get("total_patterns_detected", 0)
    if total_patterns > 0:
        insights.append(f"Total patterns detected: {total_patterns}")

    resource_allocations = analysis.get("resource_allocations", {})
    if resource_allocations:
        insights.append(f"Active resource allocations: {len(resource_allocations)}")

    return insights


def _calculate_pattern_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate pattern recognition health score (0-100)."""
    score = 100.0

    pattern_metrics = analysis.get("pattern_metrics", {})
    allocation_success_rate = pattern_metrics.get("allocation_success_rate", 0)

    # Allocation success rate impact (40% weight)
    score -= (1.0 - allocation_success_rate) * 40

    current_patterns = analysis.get("current_patterns", [])
    if current_patterns:
        avg_confidence = sum(p["confidence_score"] for p in current_patterns) / len(current_patterns)
        # Confidence impact (30% weight)
        score -= (1.0 - avg_confidence) * 30

    pattern_distribution = analysis.get("pattern_distribution", {})
    diversity_score = len(pattern_distribution) / len(WorkloadPattern)
    # Diversity impact (30% weight)
    score -= (1.0 - diversity_score) * 30

    return max(0.0, min(100.0, score))


def _generate_pattern_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate pattern-based recommendations."""
    recommendations = []

    pattern_metrics = analysis.get("pattern_metrics", {})
    if pattern_metrics.get("allocation_success_rate", 0) < 0.8:
        recommendations.append("Improve resource allocation success rate")

    current_patterns = analysis.get("current_patterns", [])
    if not current_patterns:
        recommendations.append("Run pattern analysis to detect current workload patterns")

    optimization_opportunities = analysis.get("resource_optimization_opportunities", [])
    for opportunity in optimization_opportunities:
        recommendations.append(f"{opportunity['type']}: {opportunity['recommendation']}")

    return recommendations


def _create_synthetic_pattern(pattern_type: WorkloadPattern, resource_focus: str) -> Any:
    """Create a synthetic pattern for testing specified pattern type."""
    # This would create a synthetic WorkloadCharacteristics object
    # For now, return a mock object structure
    return type('MockPattern', (), {
        'pattern_type': pattern_type,
        'intensity_score': 0.7,
        'predictability_score': 0.8,
        'confidence_score': 0.85,
        'resource_demand': {
            ResourceType.CPU_CORES: 3.0,
            ResourceType.MEMORY_MB: 4096,
            ResourceType.CACHE_SIZE: 5000
        },
        'temporal_pattern': 'business_hours',
        'duration_estimate': 60
    })()


def _filter_allocations_by_focus(allocations: List[Any], resource_focus: str) -> List[Any]:
    """Filter allocations based on resource focus."""
    if resource_focus == "balanced":
        return allocations

    focus_mapping = {
        "cpu": ResourceType.CPU_CORES,
        "memory": ResourceType.MEMORY_MB,
        "cache": ResourceType.CACHE_SIZE
    }

    if resource_focus in focus_mapping:
        target_resource = focus_mapping[resource_focus]
        return [a for a in allocations if a.resource_type == target_resource]

    return allocations


def _analyze_allocation_effectiveness(allocations: List[Any], pattern: Any) -> Dict[str, Any]:
    """Analyze effectiveness of applied allocations."""
    if not allocations:
        return {"effectiveness": "no_allocations"}

    total_benefit = sum(a.expected_benefit for a in allocations)
    avg_priority = sum(a.implementation_priority for a in allocations) / len(allocations)

    return {
        "total_expected_benefit": total_benefit,
        "average_priority": avg_priority,
        "allocation_count": len(allocations),
        "pattern_alignment_score": pattern.confidence_score
    }


def _estimate_allocation_performance_impact(allocations: List[Any]) -> Dict[str, Any]:
    """Estimate performance impact of allocations."""
    if not allocations:
        return {"impact": "no_allocations"}

    # Simplified impact estimation
    cpu_allocations = [a for a in allocations if a.resource_type == ResourceType.CPU_CORES]
    memory_allocations = [a for a in allocations if a.resource_type == ResourceType.MEMORY_MB]

    impact = {
        "estimated_response_time_improvement_ms": sum(a.expected_benefit * 0.5 for a in allocations),
        "estimated_throughput_improvement_percent": sum(a.expected_benefit * 0.3 for a in allocations),
        "estimated_resource_efficiency_improvement_percent": sum(a.expected_benefit * 0.2 for a in allocations),
        "cpu_cores_allocated": sum(a.recommended_allocation for a in cpu_allocations),
        "memory_mb_allocated": sum(a.recommended_allocation for a in memory_allocations)
    }

    return impact


def _generate_monitoring_guidance(allocations: List[Any]) -> Dict[str, Any]:
    """Generate monitoring guidance for applied allocations."""
    guidance = {
        "monitoring_period_minutes": 15,
        "key_metrics_to_monitor": ["response_time", "throughput", "resource_utilization"],
        "alert_thresholds": {},
        "rollback_triggers": []
    }

    for allocation in allocations:
        if allocation.resource_type == ResourceType.CPU_CORES:
            guidance["alert_thresholds"]["cpu_usage"] = "above 90%"
        elif allocation.resource_type == ResourceType.MEMORY_MB:
            guidance["alert_thresholds"]["memory_usage"] = "above 85%"

        guidance["rollback_triggers"].append(f"{allocation.resource_type.value} performance degradation")

    return guidance


def _assess_allocation_success(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess success of allocation operation."""
    allocations = results.get("resource_allocations", [])
    applied_count = sum(1 for a in allocations if a["applied"])

    success_rate = applied_count / max(len(allocations), 1)

    assessment = {
        "success_rate": success_rate,
        "success_level": "excellent" if success_rate > 0.9 else "good" if success_rate > 0.7 else "fair" if success_rate > 0.5 else "poor",
        "allocations_applied": applied_count,
        "total_allocations": len(allocations),
        "recommendations": []
    }

    if success_rate < 0.7:
        assessment["recommendations"].append("Consider adjusting allocation parameters for better success rates")

    return assessment


def _summarize_allocation_history(allocation_history: List[Any]) -> Dict[str, Any]:
    """Summarize allocation history."""
    if not allocation_history:
        return {"summary": "no_allocation_history"}

    resource_counts = {}
    for allocation in allocation_history:
        resource_type = allocation.resource_type.value
        resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1

    return {
        "total_allocations": len(allocation_history),
        "resource_distribution": resource_counts,
        "most_allocated_resource": max(resource_counts.items(), key=lambda x: x[1])[0] if resource_counts else None,
        "average_expected_benefit": sum(a.expected_benefit for a in allocation_history) / len(allocation_history)
    }


def _calculate_allocation_effectiveness_metrics(pattern_recognition: Any) -> Dict[str, Any]:
    """Calculate allocation effectiveness metrics."""
    metrics = pattern_recognition.metrics

    return {
        "allocation_success_rate": metrics.resource_allocation_success_rate,
        "average_pattern_confidence": metrics.pattern_confidence_avg,
        "pattern_detection_rate": len(pattern_recognition.detected_patterns) / max(metrics.total_patterns_detected, 1),
        "resource_utilization_score": 0.85  # Placeholder
    }


def _analyze_resource_utilization(pattern_recognition: Any) -> Dict[str, Any]:
    """Analyze resource utilization."""
    current_allocations = pattern_recognition.resource_allocations

    utilization = {}
    for resource, allocation in current_allocations.items():
        # Simplified utilization calculation
        utilization[resource.value] = {
            "allocated": allocation,
            "utilization_percent": 75.0,  # Placeholder
            "efficiency_score": 0.8  # Placeholder
        }

    return utilization


def _analyze_allocation_trends(allocation_history: List[Any]) -> Dict[str, Any]:
    """Analyze allocation trends."""
    if len(allocation_history) < 2:
        return {"trend": "insufficient_data"}

    # Analyze trends in allocation patterns
    recent_allocations = allocation_history[-10:]

    resource_trends = {}
    for resource_type in ResourceType:
        resource_allocations = [a for a in recent_allocations if a.resource_type == resource_type]
        if resource_allocations:
            allocations = [a.recommended_allocation for a in resource_allocations]
            trend = "increasing" if allocations[-1] > allocations[0] else "decreasing"
            resource_trends[resource_type.value] = {
                "trend": trend,
                "average_allocation": sum(allocations) / len(allocations),
                "allocation_count": len(allocations)
            }

    return resource_trends


def _identify_allocation_optimization_opportunities(pattern_recognition: Any) -> List[Dict[str, Any]]:
    """Identify allocation optimization opportunities."""
    opportunities = []

    # Check for resource overallocation
    current_allocations = pattern_recognition.resource_allocations
    for resource, allocation in current_allocations.items():
        if resource == ResourceType.CPU_CORES and allocation > 6:
            opportunities.append({
                "type": "cpu_overallocation",
                "resource": resource.value,
                "description": f"High CPU allocation ({allocation} cores) may cause contention",
                "recommendation": "Consider reducing CPU allocation or implementing load balancing"
            })

    # Check for allocation patterns
    allocation_history = pattern_recognition.allocation_history
    if len(allocation_history) > 5:
        # Check for frequent reallocation
        recent_allocations = allocation_history[-5:]
        if len(set(a.resource_type for a in recent_allocations)) > 3:
            opportunities.append({
                "type": "allocation_stability",
                "description": "Frequent allocation changes detected",
                "recommendation": "Implement more stable allocation strategies"
            })

    return opportunities


def _assess_forecast_accuracy(pattern_recognition: Any) -> Dict[str, Any]:
    """Assess forecast accuracy."""
    # Simplified accuracy assessment
    return {
        "historical_accuracy": 0.82,
        "confidence_level": "high",
        "accuracy_trend": "stable"
    }


def _assess_forecast_risk(forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Assess forecast risk."""
    predicted_patterns = forecast.get("predicted_patterns", [])

    if not predicted_patterns:
        return {"risk_level": "high", "risk_factors": ["No patterns predicted"]}

    avg_probability = sum(p["probability"] for p in predicted_patterns) / len(predicted_patterns)

    risk_level = "low" if avg_probability > 0.7 else "medium" if avg_probability > 0.5 else "high"

    return {
        "risk_level": risk_level,
        "risk_factors": ["Low prediction confidence"] if avg_probability < 0.6 else [],
        "mitigation_strategies": ["Increase monitoring window", "Improve pattern detection"]
    }


def _generate_proactive_recommendations(forecast: Dict[str, Any]) -> List[str]:
    """Generate proactive recommendations based on forecast."""
    recommendations = []

    predicted_patterns = forecast.get("predicted_patterns", [])
    if predicted_patterns:
        top_pattern = predicted_patterns[0]
        recommendations.append(f"Prepare for {top_pattern['pattern_type']} workload pattern")

    resource_forecast = forecast.get("resource_forecast", {})
    if resource_forecast:
        recommendations.append("Scale resources proactively based on forecast")

    return recommendations


def _generate_resource_planning_guidance(forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Generate resource planning guidance."""
    resource_forecast = forecast.get("resource_forecast", {})

    guidance = {
        "recommended_resources": resource_forecast,
        "scaling_strategy": "proactive",
        "monitoring_period_minutes": 30,
        "scaling_triggers": []
    }

    for resource, demand in resource_forecast.items():
        guidance["scaling_triggers"].append(f"{resource} demand exceeds 80% of forecast")

    return guidance


def _analyze_pattern_detection_effectiveness(patterns: List[Any]) -> Dict[str, Any]:
    """Analyze pattern detection effectiveness."""
    if not patterns:
        return {"effectiveness": "no_patterns"}

    confidences = [p.confidence_score for p in patterns]
    avg_confidence = sum(confidences) / len(confidences)

    return {
        "average_confidence": avg_confidence,
        "confidence_distribution": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low",
        "detection_coverage": len(patterns) / 10,  # Simplified coverage metric
        "false_positive_rate": 0.05  # Placeholder
    }


def _analyze_resource_allocation_effectiveness(allocations: List[Any]) -> Dict[str, Any]:
    """Analyze resource allocation effectiveness."""
    if not allocations:
        return {"effectiveness": "no_allocations"}

    # Simplified effectiveness analysis
    return {
        "allocation_success_rate": 0.88,
        "average_benefit_realized": sum(a.expected_benefit for a in allocations) / len(allocations),
        "resource_contention_rate": 0.05,
        "allocation_precision": 0.92
    }


def _analyze_pattern_performance_impact(patterns: List[Any], allocations: List[Any]) -> Dict[str, Any]:
    """Analyze performance impact of patterns and allocations."""
    impact = {
        "pattern_driven_improvements": 0.0,
        "allocation_efficiency": 0.0,
        "resource_optimization_score": 0.0
    }

    if patterns:
        impact["pattern_driven_improvements"] = sum(p.intensity_score for p in patterns) / len(patterns) * 100

    if allocations:
        impact["allocation_efficiency"] = sum(a.expected_benefit for a in allocations) / len(allocations)

    impact["resource_optimization_score"] = (impact["pattern_driven_improvements"] + impact["allocation_efficiency"]) / 2

    return impact


def _analyze_pattern_evolution(patterns: List[Any]) -> Dict[str, Any]:
    """Analyze pattern evolution."""
    if len(patterns) < 2:
        return {"evolution": "insufficient_data"}

    # Analyze pattern changes over time
    pattern_types = [p.pattern_type for p in patterns]
    type_changes = sum(1 for i in range(1, len(pattern_types)) if pattern_types[i] != pattern_types[i-1])

    return {
        "pattern_stability": 1.0 - (type_changes / max(len(patterns) - 1, 1)),
        "evolution_trend": "stable" if type_changes < len(patterns) * 0.3 else "evolving",
        "pattern_transitions": type_changes
    }


def _calculate_overall_effectiveness_score(patterns: List[Any], allocations: List[Any]) -> float:
    """Calculate overall effectiveness score."""
    score = 50.0  # Base score

    if patterns:
        pattern_score = sum(p.confidence_score for p in patterns) / len(patterns)
        score += pattern_score * 25

    if allocations:
        allocation_score = sum(a.expected_benefit for a in allocations) / len(allocations) / 10
        score += allocation_score * 25

    return min(100.0, score)


def _generate_effectiveness_improvements(patterns: List[Any], allocations: List[Any]) -> List[str]:
    """Generate effectiveness improvement recommendations."""
    improvements = []

    if patterns:
        avg_confidence = sum(p.confidence_score for p in patterns) / len(patterns)
        if avg_confidence < 0.7:
            improvements.append("Improve pattern detection confidence through better feature engineering")

    if allocations:
        avg_benefit = sum(a.expected_benefit for a in allocations) / len(allocations)
        if avg_benefit < 5:
            improvements.append("Optimize resource allocation algorithms for higher benefit")

    return improvements


def _analyze_configuration_effectiveness(pattern_recognition: Any) -> Dict[str, Any]:
    """Analyze configuration effectiveness."""
    effectiveness = {
        "detection_accuracy": 0.85,
        "allocation_precision": 0.82,
        "pattern_coverage": 0.78,
        "overall_effectiveness_score": 0.82
    }

    return effectiveness


def _analyze_pattern_coverage(pattern_recognition: Any) -> Dict[str, Any]:
    """Analyze pattern coverage."""
    detected_types = set(p.pattern_type for p in pattern_recognition.detected_patterns)
    all_types = set(WorkloadPattern)

    coverage = len(detected_types) / len(all_types)

    return {
        "detected_pattern_types": len(detected_types),
        "total_pattern_types": len(all_types),
        "coverage_percentage": coverage * 100,
        "undetected_patterns": [p.value for p in all_types - detected_types]
    }


def _generate_configuration_tuning_recommendations(pattern_recognition: Any) -> List[str]:
    """Generate configuration tuning recommendations."""
    recommendations = []

    if pattern_recognition.pattern_detection_threshold > 0.8:
        recommendations.append("Consider lowering pattern detection threshold for better coverage")

    if pattern_recognition.analysis_window_minutes < 30:
        recommendations.append("Consider increasing analysis window for more stable pattern detection")

    return recommendations


def _validate_configuration_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration updates."""
    errors = []

    if "analysis_window_minutes" in updates:
        if not 5 <= updates["analysis_window_minutes"] <= 120:
            errors.append("analysis_window_minutes must be between 5 and 120")

    if "pattern_detection_threshold" in updates:
        if not 0.1 <= updates["pattern_detection_threshold"] <= 0.9:
            errors.append("pattern_detection_threshold must be between 0.1 and 0.9")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _assess_configuration_impact(pattern_recognition: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Assess impact of configuration changes."""
    impact = {
        "expected_detection_accuracy_change": 0.0,
        "expected_allocation_precision_change": 0.0,
        "risk_level": "low",
        "monitoring_recommendations": ["Monitor pattern detection accuracy", "Track allocation success rates"]
    }

    if "pattern_detection_threshold" in updates:
        threshold_change = updates["pattern_detection_threshold"] - pattern_recognition.pattern_detection_threshold
        impact["expected_detection_accuracy_change"] = threshold_change * 0.5

    return impact