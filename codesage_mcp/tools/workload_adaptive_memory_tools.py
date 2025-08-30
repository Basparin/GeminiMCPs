"""
Workload-Adaptive Memory Management Tools for CodeSage MCP Server.

This module provides tools for workload-aware memory management that adapts memory allocation
strategies based on current workload patterns, system load, and performance requirements.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.workload_adaptive_memory import (
    get_workload_adaptive_memory_manager,
    WorkloadType,
    MemoryAllocationStrategy
)

logger = logging.getLogger(__name__)


def get_workload_analysis_tool() -> Dict[str, Any]:
    """
    Get comprehensive workload analysis including current patterns, trends, and predictions.

    Returns:
        Dictionary containing workload analysis including:
        - Current workload characteristics
        - Workload trends and patterns
        - Allocation effectiveness
        - Future workload predictions
        - Optimization opportunities
    """
    try:
        manager = get_workload_adaptive_memory_manager()
        analysis = manager.get_workload_analysis()

        if "error" in analysis:
            return analysis

        # Enhance analysis with additional insights
        analysis["insights"] = _generate_workload_insights(analysis)
        analysis["health_score"] = _calculate_workload_health_score(analysis)
        analysis["recommendations"] = _generate_workload_recommendations(analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error getting workload analysis: {e}")
        return {
            "error": f"Failed to get workload analysis: {str(e)}"
        }


def get_memory_allocation_status_tool() -> Dict[str, Any]:
    """
    Get current memory allocation status and effectiveness analysis.

    Returns:
        Dictionary containing memory allocation analysis including:
        - Current allocation plan and breakdown
        - Allocation effectiveness metrics
        - Performance impact analysis
        - Adaptation history and trends
    """
    try:
        manager = get_workload_adaptive_memory_manager()

        status = {
            "current_allocation_plan": None,
            "allocation_effectiveness": manager._analyze_allocation_effectiveness(),
            "allocation_history_summary": _summarize_allocation_history(manager.allocation_history),
            "performance_impact": _analyze_allocation_performance_impact(manager.allocation_history),
            "generated_at": time.time()
        }

        # Add current allocation plan if available
        if manager.current_allocation_plan:
            status["current_allocation_plan"] = {
                "strategy": manager.current_allocation_plan.strategy.value,
                "total_memory_mb": manager.current_allocation_plan.total_memory_mb,
                "allocation_breakdown": manager.current_allocation_plan.allocation_breakdown,
                "expected_performance_impact": manager.current_allocation_plan.expected_performance_impact,
                "confidence_score": manager.current_allocation_plan.confidence_score,
                "adaptation_reason": manager.current_allocation_plan.adaptation_reason,
                "timestamp": manager.current_allocation_plan.timestamp
            }

        # Enhance with additional analysis
        status["allocation_efficiency_score"] = _calculate_allocation_efficiency(status)
        status["optimization_opportunities"] = _identify_allocation_optimization_opportunities(status)

        return status

    except Exception as e:
        logger.error(f"Error getting memory allocation status: {e}")
        return {
            "error": f"Failed to get memory allocation status: {str(e)}"
        }


def trigger_workload_adaptation_tool(strategy: str = "adaptive") -> Dict[str, Any]:
    """
    Trigger manual workload adaptation with specified strategy.

    Args:
        strategy: Memory allocation strategy to use ("conservative", "balanced",
                 "aggressive", "predictive", "adaptive")

    Returns:
        Dictionary containing adaptation results including:
        - Adaptation decision and reasoning
        - New allocation plan
        - Expected performance impact
        - Implementation status
    """
    try:
        # Validate strategy
        try:
            strategy_enum = MemoryAllocationStrategy(strategy.lower())
        except ValueError:
            return {
                "error": f"Invalid strategy: {strategy}",
                "valid_strategies": [s.value for s in MemoryAllocationStrategy]
            }

        manager = get_workload_adaptive_memory_manager()

        # Analyze current workload
        current_workload = manager._analyze_current_workload()

        # Generate allocation plan for specified strategy
        allocation_plan = manager._generate_allocation_plan(current_workload, strategy_enum)

        # Check if adaptation should proceed
        should_adapt = manager._should_adapt_memory(allocation_plan)

        if not should_adapt:
            return {
                "adaptation_status": "skipped",
                "reason": "Adaptation not needed based on current conditions",
                "current_workload": {
                    "type": current_workload.workload_type.value,
                    "intensity": current_workload.intensity_score,
                    "confidence": allocation_plan.confidence_score
                },
                "generated_at": time.time()
            }

        # Apply adaptation
        success = manager._apply_allocation_plan(allocation_plan)

        result = {
            "adaptation_status": "applied" if success else "failed",
            "strategy_used": strategy,
            "allocation_plan": {
                "strategy": allocation_plan.strategy.value,
                "total_memory_mb": allocation_plan.total_memory_mb,
                "allocation_breakdown": allocation_plan.allocation_breakdown,
                "expected_performance_impact": allocation_plan.expected_performance_impact,
                "confidence_score": allocation_plan.confidence_score,
                "adaptation_reason": allocation_plan.adaptation_reason
            },
            "current_workload": {
                "type": current_workload.workload_type.value,
                "intensity": current_workload.intensity_score,
                "cpu_percent": current_workload.cpu_percent,
                "memory_percent": current_workload.memory_percent
            },
            "implementation_details": {
                "success": success,
                "timestamp": time.time(),
                "rollback_available": bool(allocation_plan.rollback_plan)
            }
        }

        if success:
            result["performance_predictions"] = _generate_adaptation_performance_predictions(allocation_plan)
            result["monitoring_guidance"] = _generate_adaptation_monitoring_guidance()

        return result

    except Exception as e:
        logger.error(f"Error triggering workload adaptation: {e}")
        return {
            "error": f"Failed to trigger workload adaptation: {str(e)}"
        }


def get_workload_optimization_recommendations_tool() -> Dict[str, Any]:
    """
    Get workload optimization recommendations based on analysis.

    Returns:
        Dictionary containing workload optimization recommendations including:
        - Workload pattern analysis
        - Memory allocation optimization strategies
        - Performance improvement recommendations
        - Implementation roadmap
    """
    try:
        manager = get_workload_adaptive_memory_manager()
        analysis = manager.get_workload_analysis()

        if "error" in analysis:
            return analysis

        # Generate optimization recommendations
        recommendations = {
            "workload_pattern_optimization": _generate_workload_pattern_recommendations(analysis),
            "memory_allocation_optimization": _generate_memory_allocation_recommendations(analysis),
            "performance_optimization": _generate_performance_optimization_recommendations(analysis),
            "predictive_optimization": _generate_predictive_optimization_recommendations(analysis),
            "implementation_roadmap": _generate_workload_optimization_roadmap(analysis),
            "generated_at": time.time()
        }

        # Calculate overall optimization score
        recommendations["optimization_score"] = _calculate_workload_optimization_score(recommendations)

        return recommendations

    except Exception as e:
        logger.error(f"Error getting workload optimization recommendations: {e}")
        return {
            "error": f"Failed to get workload optimization recommendations: {str(e)}"
        }


def analyze_workload_performance_impact_tool(time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Analyze the performance impact of workload adaptations over time.

    Args:
        time_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing performance impact analysis including:
        - Performance metrics before and after adaptations
        - Effectiveness of different allocation strategies
        - Performance trends and correlations
        - Recommendations for strategy tuning
    """
    try:
        manager = get_workload_adaptive_memory_manager()

        # Get allocation history within time window
        cutoff_time = time.time() - (time_window_hours * 60 * 60)
        recent_allocations = [
            alloc for alloc in manager.allocation_history
            if alloc.timestamp >= cutoff_time
        ]

        if not recent_allocations:
            return {
                "analysis_window_hours": time_window_hours,
                "message": f"No adaptations found in the last {time_window_hours} hours",
                "generated_at": time.time()
            }

        # Analyze performance impact
        performance_impact = {
            "allocation_performance_analysis": _analyze_allocation_performance(recent_allocations),
            "strategy_effectiveness": _analyze_strategy_effectiveness(recent_allocations),
            "performance_trends": _analyze_performance_trends(recent_allocations),
            "correlation_analysis": _analyze_performance_correlations(recent_allocations),
            "strategy_recommendations": _generate_strategy_recommendations(recent_allocations),
            "analysis_window_hours": time_window_hours,
            "allocations_analyzed": len(recent_allocations),
            "generated_at": time.time()
        }

        return performance_impact

    except Exception as e:
        logger.error(f"Error analyzing workload performance impact: {e}")
        return {
            "error": f"Failed to analyze workload performance impact: {str(e)}"
        }


def get_workload_adaptation_rules_tool() -> Dict[str, Any]:
    """
    Get information about workload adaptation rules and their effectiveness.

    Returns:
        Dictionary containing adaptation rules information including:
        - Rule definitions and conditions
        - Effectiveness metrics and success rates
        - Rule performance analysis
        - Recommendations for rule tuning
    """
    try:
        manager = get_workload_adaptive_memory_manager()

        rules_info = []
        for workload_type, strategy_config in manager.allocation_strategies.items():
            rule_info = {
                "workload_type": workload_type.value,
                "strategy": strategy_config["strategy"].value,
                "cache_allocation_percent": strategy_config["cache_allocation_percent"],
                "model_allocation_percent": strategy_config["model_allocation_percent"],
                "buffer_allocation_percent": strategy_config["buffer_allocation_percent"],
                "gc_frequency": strategy_config["gc_frequency"],
                "memory_pressure_threshold": strategy_config["memory_pressure_threshold"],
                "rule_status": "active",
                "usage_count": _count_workload_type_usage(manager, workload_type)
            }
            rules_info.append(rule_info)

        # Sort by usage count
        rules_info.sort(key=lambda x: x["usage_count"], reverse=True)

        # Generate rule analysis
        rule_analysis = {
            "total_rules": len(rules_info),
            "active_rules": len([r for r in rules_info if r["rule_status"] == "active"]),
            "most_used_strategy": max(rules_info, key=lambda x: x["usage_count"])["strategy"],
            "strategy_distribution": _analyze_strategy_distribution(rules_info),
            "rule_effectiveness": _analyze_rule_effectiveness(rules_info),
            "tuning_recommendations": _generate_rule_tuning_recommendations(rules_info)
        }

        return {
            "rules_info": rules_info,
            "rule_analysis": rule_analysis,
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting workload adaptation rules: {e}")
        return {
            "error": f"Failed to get workload adaptation rules: {str(e)}"
        }


def _generate_workload_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from workload analysis."""
    insights = []

    current_workload = analysis.get("current_workload", {})
    trends = analysis.get("workload_trends", {})

    if current_workload:
        workload_type = current_workload.get("workload_type", {}).get("value", "unknown")
        intensity = current_workload.get("intensity_score", 0)

        insights.append(f"Current workload: {workload_type} with intensity {intensity:.2f}")

        if intensity > 0.8:
            insights.append("High workload intensity detected - system under significant load")
        elif intensity < 0.3:
            insights.append("Low workload intensity - system has capacity for additional load")

    if trends:
        intensity_trend = trends.get("intensity_trend", "stable")
        if intensity_trend != "stable":
            insights.append(f"Workload intensity is trending {intensity_trend}")

        dominant_types = trends.get("dominant_workload_types", [])
        if dominant_types:
            top_type = dominant_types[0]
            insights.append(f"Dominant workload type: {top_type['type']} ({top_type['percentage']:.1f}%)")

    optimization_opportunities = analysis.get("optimization_opportunities", [])
    if optimization_opportunities:
        insights.append(f"{len(optimization_opportunities)} workload optimization opportunities identified")

    return insights


def _calculate_workload_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate a workload health score (0-100)."""
    score = 100.0

    current_workload = analysis.get("current_workload", {})
    trends = analysis.get("workload_trends", {})

    # Intensity score (optimal range 0.4-0.8)
    intensity = current_workload.get("intensity_score", 0.5)
    if intensity > 0.8:
        score -= (intensity - 0.8) * 100  # Penalty for too high intensity
    elif intensity < 0.4:
        score -= (0.4 - intensity) * 50  # Smaller penalty for too low intensity

    # Volatility penalty
    volatility = trends.get("workload_volatility", 0)
    score -= min(volatility * 200, 25)  # Max 25 point penalty

    # Optimization opportunities penalty
    opportunities = analysis.get("optimization_opportunities", [])
    score -= len(opportunities) * 5  # 5 points per opportunity

    return max(0.0, min(100.0, score))


def _generate_workload_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate workload optimization recommendations."""
    recommendations = []

    current_workload = analysis.get("current_workload", {})
    trends = analysis.get("workload_trends", {})
    opportunities = analysis.get("optimization_opportunities", [])

    intensity = current_workload.get("intensity_score", 0.5)
    if intensity > 0.8:
        recommendations.append("High workload intensity detected - consider scaling resources")
    elif intensity < 0.3:
        recommendations.append("Low workload intensity - opportunity to optimize resource allocation")

    volatility = trends.get("workload_volatility", 0)
    if volatility > 0.2:
        recommendations.append("High workload volatility - implement workload smoothing strategies")

    for opportunity in opportunities:
        recommendations.append(f"{opportunity['type']}: {opportunity['recommendation']}")

    return recommendations


def _summarize_allocation_history(allocation_history: List) -> Dict[str, Any]:
    """Summarize allocation history."""
    if not allocation_history:
        return {"summary": "No allocation history available"}

    recent_allocations = allocation_history[-10:]  # Last 10 allocations

    strategies_used = {}
    for alloc in recent_allocations:
        strategy = alloc.strategy.value
        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

    return {
        "total_allocations": len(allocation_history),
        "recent_allocations": len(recent_allocations),
        "strategies_used": strategies_used,
        "most_common_strategy": max(strategies_used.items(), key=lambda x: x[1])[0] if strategies_used else None,
        "time_span_hours": (recent_allocations[-1].timestamp - recent_allocations[0].timestamp) / 3600 if len(recent_allocations) > 1 else 0
    }


def _analyze_allocation_performance_impact(allocation_history: List) -> Dict[str, Any]:
    """Analyze performance impact of allocations."""
    if len(allocation_history) < 2:
        return {"impact": "insufficient_data"}

    # Group by strategy
    strategy_performance = {}
    for alloc in allocation_history:
        strategy = alloc.strategy.value
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(alloc.expected_performance_impact)

    # Calculate average impact per strategy
    for strategy, impacts in strategy_performance.items():
        avg_impact = {}
        for key in impacts[0].keys():
            values = [impact[key] for impact in impacts]
            avg_impact[key] = sum(values) / len(values)
        strategy_performance[strategy] = avg_impact

    return strategy_performance


def _calculate_allocation_efficiency(status: Dict[str, Any]) -> float:
    """Calculate allocation efficiency score."""
    effectiveness = status.get("allocation_effectiveness", {})
    avg_effectiveness = effectiveness.get("avg_effectiveness", 0)

    # Factor in strategy diversity and adaptation frequency
    history_summary = status.get("allocation_history_summary", {})
    strategies_used = len(history_summary.get("strategies_used", {}))

    efficiency_score = avg_effectiveness * 0.7 + (strategies_used / 5) * 0.3  # Max 30% for strategy diversity

    return min(1.0, efficiency_score)


def _identify_allocation_optimization_opportunities(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify allocation optimization opportunities."""
    opportunities = []

    effectiveness = status.get("allocation_effectiveness", {})
    avg_effectiveness = effectiveness.get("avg_effectiveness", 0)

    if avg_effectiveness < 0.6:
        opportunities.append({
            "type": "effectiveness_improvement",
            "description": "Allocation effectiveness is below optimal levels",
            "recommendation": "Review and tune allocation strategies"
        })

    history_summary = status.get("allocation_history_summary", {})
    if history_summary.get("total_allocations", 0) > 20:
        opportunities.append({
            "type": "allocation_frequency",
            "description": "High frequency of allocation changes detected",
            "recommendation": "Consider longer adaptation intervals to reduce overhead"
        })

    return opportunities


def _generate_adaptation_performance_predictions(allocation_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate performance predictions for adaptation."""
    expected_impact = allocation_plan.get("expected_performance_impact", {})

    predictions = {
        "response_time_change_ms": expected_impact.get("response_time_change_ms", 0),
        "throughput_change_percent": expected_impact.get("throughput_change_percent", 0),
        "memory_efficiency_change_percent": expected_impact.get("memory_efficiency_change_percent", 0),
        "cache_hit_rate_change_percent": expected_impact.get("cache_hit_rate_change_percent", 0),
        "time_to_effect_hours": 1,
        "confidence_level": "high" if allocation_plan.get("confidence_score", 0) > 0.8 else "medium"
    }

    return predictions


def _generate_adaptation_monitoring_guidance() -> List[str]:
    """Generate monitoring guidance for adaptation."""
    return [
        "Monitor response time for first 30 minutes after adaptation",
        "Track memory usage and cache hit rates",
        "Watch for any performance degradation",
        "Be prepared to rollback if performance drops by >20%",
        "Monitor system stability for 2-4 hours post-adaptation"
    ]


def _generate_workload_pattern_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate workload pattern recommendations."""
    recommendations = []

    trends = analysis.get("workload_trends", {})
    volatility = trends.get("workload_volatility", 0)

    if volatility > 0.2:
        recommendations.append({
            "type": "workload_smoothing",
            "description": "High workload volatility detected",
            "recommendation": "Implement workload smoothing and request queuing",
            "priority": "high"
        })

    dominant_types = trends.get("dominant_workload_types", [])
    if dominant_types:
        top_type = dominant_types[0]
        if top_type["percentage"] > 70:
            recommendations.append({
                "type": "specialization_optimization",
                "description": f"Heavily biased toward {top_type['type']} workload",
                "recommendation": "Optimize system configuration for dominant workload type",
                "priority": "medium"
            })

    return recommendations


def _generate_memory_allocation_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate memory allocation recommendations."""
    recommendations = []

    current_workload = analysis.get("current_workload", {})
    workload_type = current_workload.get("workload_type", {}).get("value", "mixed_load")

    if workload_type == "cpu_intensive":
        recommendations.append({
            "type": "cpu_intensive_allocation",
            "description": "CPU-intensive workload detected",
            "recommendation": "Optimize cache allocation for CPU-bound operations",
            "priority": "high"
        })
    elif workload_type == "memory_intensive":
        recommendations.append({
            "type": "memory_intensive_allocation",
            "description": "Memory-intensive workload detected",
            "recommendation": "Increase memory allocation for cache and buffers",
            "priority": "high"
        })

    return recommendations


def _generate_performance_optimization_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate performance optimization recommendations."""
    recommendations = []

    current_workload = analysis.get("current_workload", {})
    intensity = current_workload.get("intensity_score", 0.5)

    if intensity > 0.8:
        recommendations.append({
            "type": "high_load_optimization",
            "description": "High workload intensity detected",
            "recommendation": "Implement performance optimizations for high-load scenarios",
            "priority": "high"
        })

    return recommendations


def _generate_predictive_optimization_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate predictive optimization recommendations."""
    recommendations = []

    predictions = analysis.get("predictions", [])
    if predictions:
        latest_prediction = predictions[-1]
        predicted_intensity = latest_prediction.get("predicted_intensity", 0.5)

        if predicted_intensity > 0.8:
            recommendations.append({
                "type": "predictive_scaling",
                "description": "High workload intensity predicted",
                "recommendation": "Prepare for increased load with predictive scaling",
                "priority": "medium"
            })

    return recommendations


def _generate_workload_optimization_roadmap(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate workload optimization roadmap."""
    roadmap = {
        "immediate_actions": [],
        "short_term_goals": [],
        "long_term_strategies": []
    }

    recommendations = []
    recommendations.extend(_generate_workload_pattern_recommendations(analysis))
    recommendations.extend(_generate_memory_allocation_recommendations(analysis))
    recommendations.extend(_generate_performance_optimization_recommendations(analysis))
    recommendations.extend(_generate_predictive_optimization_recommendations(analysis))

    for rec in recommendations:
        if rec["priority"] == "high":
            roadmap["immediate_actions"].append(rec)
        elif rec["priority"] == "medium":
            roadmap["short_term_goals"].append(rec)
        else:
            roadmap["long_term_strategies"].append(rec)

    return {
        "roadmap": roadmap,
        "total_recommendations": len(recommendations),
        "immediate_count": len(roadmap["immediate_actions"]),
        "short_term_count": len(roadmap["short_term_goals"]),
        "long_term_count": len(roadmap["long_term_strategies"])
    }


def _calculate_workload_optimization_score(recommendations: Dict[str, Any]) -> float:
    """Calculate overall workload optimization score."""
    roadmap = recommendations.get("implementation_roadmap", {})
    immediate_count = roadmap.get("immediate_count", 0)
    short_term_count = roadmap.get("short_term_count", 0)
    long_term_count = roadmap.get("long_term_count", 0)

    # Higher score for fewer immediate actions needed
    base_score = 100.0
    penalty = (immediate_count * 10) + (short_term_count * 5) + (long_term_count * 2)

    return max(0.0, base_score - penalty)


def _analyze_allocation_performance(allocations: List) -> Dict[str, Any]:
    """Analyze allocation performance."""
    if not allocations:
        return {"performance": "no_data"}

    # Group by strategy
    strategy_performance = {}
    for alloc in allocations:
        strategy = alloc.strategy.value
        if strategy not in strategy_performance:
            strategy_performance[strategy] = {"allocations": [], "impacts": []}
        strategy_performance[strategy]["allocations"].append(alloc)
        strategy_performance[strategy]["impacts"].append(alloc.expected_performance_impact)

    # Calculate performance metrics per strategy
    for strategy, data in strategy_performance.items():
        impacts = data["impacts"]
        avg_impact = {}
        for key in impacts[0].keys():
            values = [impact[key] for impact in impacts]
            avg_impact[key] = sum(values) / len(values)
        strategy_performance[strategy]["avg_impact"] = avg_impact
        strategy_performance[strategy]["allocation_count"] = len(impacts)

    return strategy_performance


def _analyze_strategy_effectiveness(allocations: List) -> Dict[str, Any]:
    """Analyze strategy effectiveness."""
    if not allocations:
        return {"effectiveness": "no_data"}

    strategy_effectiveness = {}
    for alloc in allocations:
        strategy = alloc.strategy.value
        if strategy not in strategy_effectiveness:
            strategy_effectiveness[strategy] = {"success_count": 0, "total_count": 0}

        strategy_effectiveness[strategy]["total_count"] += 1
        # Simplified success criteria
        if alloc.confidence_score > 0.7:
            strategy_effectiveness[strategy]["success_count"] += 1

    # Calculate success rates
    for strategy, data in strategy_effectiveness.items():
        data["success_rate"] = data["success_count"] / data["total_count"] if data["total_count"] > 0 else 0

    return strategy_effectiveness


def _analyze_performance_trends(allocations: List) -> Dict[str, Any]:
    """Analyze performance trends."""
    if len(allocations) < 2:
        return {"trend": "insufficient_data"}

    # Sort by timestamp
    sorted_allocations = sorted(allocations, key=lambda x: x.timestamp)

    # Analyze confidence trend
    confidences = [alloc.confidence_score for alloc in sorted_allocations]
    confidence_trend = "stable"
    if len(confidences) >= 3:
        first_half = confidences[:len(confidences)//2]
        second_half = confidences[len(confidences)//2:]
        if sum(second_half) / len(second_half) > sum(first_half) / len(first_half) + 0.1:
            confidence_trend = "improving"
        elif sum(second_half) / len(second_half) < sum(first_half) / len(first_half) - 0.1:
            confidence_trend = "declining"

    return {
        "confidence_trend": confidence_trend,
        "avg_confidence": sum(confidences) / len(confidences),
        "allocation_frequency_hours": (sorted_allocations[-1].timestamp - sorted_allocations[0].timestamp) / (len(sorted_allocations) * 3600) if len(sorted_allocations) > 1 else 0
    }


def _analyze_performance_correlations(allocations: List) -> Dict[str, Any]:
    """Analyze performance correlations."""
    if len(allocations) < 3:
        return {"correlation": "insufficient_data"}

    # Simple correlation analysis
    confidence_scores = [alloc.confidence_score for alloc in allocations]
    response_time_changes = [alloc.expected_performance_impact.get("response_time_change_ms", 0) for alloc in allocations]

    # Calculate correlation between confidence and performance impact
    correlation = _calculate_correlation(confidence_scores, response_time_changes)

    return {
        "confidence_performance_correlation": correlation,
        "correlation_strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
    }


def _generate_strategy_recommendations(allocations: List) -> List[str]:
    """Generate strategy recommendations."""
    recommendations = []

    effectiveness = _analyze_strategy_effectiveness(allocations)

    if effectiveness:
        best_strategy = max(effectiveness.items(), key=lambda x: x[1]["success_rate"])
        if best_strategy[1]["success_rate"] > 0.8:
            recommendations.append(f"Strategy '{best_strategy[0]}' shows excellent performance - consider using it more frequently")

        worst_strategy = min(effectiveness.items(), key=lambda x: x[1]["success_rate"])
        if worst_strategy[1]["success_rate"] < 0.5:
            recommendations.append(f"Strategy '{worst_strategy[0]}' shows poor performance - consider reviewing or disabling it")

    return recommendations


def _count_workload_type_usage(manager, workload_type) -> int:
    """Count usage of a workload type."""
    count = 0
    for workload in manager.workload_history:
        if workload.workload_type == workload_type:
            count += 1
    return count


def _analyze_strategy_distribution(rules_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze strategy distribution."""
    distribution = {}
    for rule in rules_info:
        strategy = rule["strategy"]
        distribution[strategy] = distribution.get(strategy, 0) + 1

    return distribution


def _analyze_rule_effectiveness(rules_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze rule effectiveness."""
    total_rules = len(rules_info)
    active_rules = sum(1 for r in rules_info if r["rule_status"] == "active")
    high_usage_rules = sum(1 for r in rules_info if r["usage_count"] > 5)

    return {
        "total_rules": total_rules,
        "active_rules": active_rules,
        "high_usage_rules": high_usage_rules,
        "effectiveness_score": (active_rules / total_rules) * (high_usage_rules / max(active_rules, 1))
    }


def _generate_rule_tuning_recommendations(rules_info: List[Dict[str, Any]]) -> List[str]:
    """Generate rule tuning recommendations."""
    recommendations = []

    unused_rules = [r for r in rules_info if r["usage_count"] == 0]
    if unused_rules:
        recommendations.append(f"Consider removing or modifying {len(unused_rules)} unused rules")

    overused_rules = [r for r in rules_info if r["usage_count"] > 20]
    if overused_rules:
        recommendations.append(f"Rules for {len(overused_rules)} workload types may need cooldown adjustment")

    return recommendations


def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

    return numerator / denominator if denominator != 0 else 0.0