"""
Adaptive Cache Management Tools for CodeSage MCP Server.

This module provides tools for dynamic cache sizing and management based on usage patterns,
performance metrics, and system load conditions.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.adaptive_cache_manager import (
    get_adaptive_cache_manager,
    AdaptationStrategy,
    CacheSizingStrategy
)

logger = logging.getLogger(__name__)


def get_adaptive_cache_status_tool() -> Dict[str, Any]:
    """
    Get the current status of adaptive cache management.

    Returns:
        Dictionary containing adaptive cache management status including:
        - Adaptation activity status
        - Recent adaptation decisions
        - Adaptation rules status
        - Performance baselines
        - Effectiveness metrics
    """
    try:
        manager = get_adaptive_cache_manager()
        status = manager.get_adaptation_status()

        # Enhance status with additional insights
        status["insights"] = _generate_adaptation_insights(status)
        status["effectiveness_analysis"] = _analyze_adaptation_effectiveness(status)
        status["recommendations"] = _generate_adaptation_recommendations(status)

        return status

    except Exception as e:
        logger.error(f"Error getting adaptive cache status: {e}")
        return {
            "error": f"Failed to get adaptive cache status: {str(e)}"
        }


def trigger_cache_adaptation_tool(cache_type: str = None, strategy: str = "hybrid") -> Dict[str, Any]:
    """
    Trigger manual cache adaptation for specified cache type or all caches.

    Args:
        cache_type: Optional specific cache type to adapt ("embedding", "search", "file")
        strategy: Adaptation strategy to use ("performance_based", "usage_pattern_based",
                 "load_aware", "predictive", "hybrid")

    Returns:
        Dictionary containing adaptation results including:
        - Adaptation decisions made
        - Expected impacts
        - Implementation status
        - Performance predictions
    """
    try:
        # Validate strategy
        try:
            strategy_enum = AdaptationStrategy(strategy.lower())
        except ValueError:
            return {
                "error": f"Invalid strategy: {strategy}",
                "valid_strategies": [s.value for s in AdaptationStrategy]
            }

        manager = get_adaptive_cache_manager()

        # Collect current metrics
        current_metrics = manager._collect_cache_metrics()

        # Filter by cache type if specified
        if cache_type:
            if cache_type not in current_metrics:
                return {
                    "error": f"Cache type '{cache_type}' not found",
                    "available_cache_types": list(current_metrics.keys())
                }
            current_metrics = {cache_type: current_metrics[cache_type]}

        # Generate adaptation decisions
        applicable_rules = manager._evaluate_adaptation_rules(current_metrics)
        adaptation_decisions = manager._generate_adaptation_decisions(applicable_rules, current_metrics)

        # Apply adaptations
        applied_decisions = manager._apply_adaptation_decisions(adaptation_decisions)

        # Generate results
        results = {
            "strategy_used": strategy,
            "cache_type_targeted": cache_type or "all",
            "total_rules_evaluated": len(manager.adaptation_rules),
            "applicable_rules": len(applicable_rules),
            "adaptation_decisions": [
                {
                    "cache_type": d.cache_type,
                    "decision": d.decision.value,
                    "current_size_mb": d.current_size,
                    "recommended_size_mb": d.recommended_size,
                    "size_change_mb": d.size_change_mb,
                    "confidence": d.confidence,
                    "reason": d.reason,
                    "expected_impact": d.expected_impact
                }
                for d in adaptation_decisions
            ],
            "applied_adaptations": len(applied_decisions),
            "implementation_status": "completed" if applied_decisions else "no_changes_needed",
            "generated_at": time.time()
        }

        # Add performance predictions
        results["performance_predictions"] = _generate_performance_predictions(applied_decisions)

        return results

    except Exception as e:
        logger.error(f"Error triggering cache adaptation: {e}")
        return {
            "error": f"Failed to trigger cache adaptation: {str(e)}"
        }


def get_cache_sizing_recommendations_tool(cache_type: str, strategy: str = "hybrid") -> Dict[str, Any]:
    """
    Get cache sizing recommendations for a specific cache type using different strategies.

    Args:
        cache_type: Cache type to analyze ("embedding", "search", "file")
        strategy: Sizing strategy to use ("performance_based", "usage_pattern_based",
                 "load_aware", "predictive", "hybrid")

    Returns:
        Dictionary containing cache sizing recommendations including:
        - Current cache configuration
        - Recommended sizes by strategy
        - Implementation plan
        - Expected performance impact
    """
    try:
        # Validate strategy
        try:
            strategy_enum = AdaptationStrategy(strategy.lower())
        except ValueError:
            return {
                "error": f"Invalid strategy: {strategy}",
                "valid_strategies": [s.value for s in AdaptationStrategy]
            }

        manager = get_adaptive_cache_manager()
        current_metrics = manager._collect_cache_metrics()

        if cache_type not in current_metrics:
            return {
                "error": f"Cache type '{cache_type}' not found",
                "available_cache_types": list(current_metrics.keys())
            }

        metrics = current_metrics[cache_type]

        # Generate recommendations using different strategies
        recommendations = {}

        for strat in AdaptationStrategy:
            sizing_strategy = CacheSizingStrategy(strat)
            # Mock usage patterns - in real implementation, this would come from actual usage data
            usage_patterns = {
                "temporal_patterns": {"hour_0": [1, 0, 1], "hour_12": [0, 1, 0]},
                "key_frequency": {"key1": 10, "key2": 5, "key3": 3},
                "hit_rate_trend": 0.02
            }

            optimal_size = sizing_strategy.calculate_optimal_size(metrics, usage_patterns)
            recommendations[strat.value] = {
                "recommended_size_mb": optimal_size,
                "size_change_mb": optimal_size - metrics.size,
                "strategy_description": _get_strategy_description(strat)
            }

        # Current configuration
        current_config = {
            "cache_type": cache_type,
            "current_size_mb": metrics.size,
            "hit_rate": metrics.hit_rate,
            "memory_usage_mb": metrics.memory_usage_mb,
            "avg_hit_latency_ms": metrics.avg_hit_latency_ms
        }

        # Implementation plan for primary strategy
        primary_recommendation = recommendations[strategy]
        implementation_plan = _generate_implementation_plan(cache_type, primary_recommendation)

        # Expected impact analysis
        expected_impact = _calculate_expected_sizing_impact(metrics, primary_recommendation)

        return {
            "cache_type": cache_type,
            "current_configuration": current_config,
            "sizing_strategy_used": strategy,
            "recommendations_by_strategy": recommendations,
            "primary_recommendation": primary_recommendation,
            "implementation_plan": implementation_plan,
            "expected_impact": expected_impact,
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting cache sizing recommendations: {e}")
        return {
            "error": f"Failed to get cache sizing recommendations: {str(e)}"
        }


def analyze_cache_adaptation_effectiveness_tool(time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Analyze the effectiveness of cache adaptations over time.

    Args:
        time_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing adaptation effectiveness analysis including:
        - Adaptation success rates
        - Performance impact analysis
        - Trend analysis
        - Recommendations for improvement
    """
    try:
        manager = get_adaptive_cache_manager()

        # Get adaptation history
        adaptation_history = manager.adaptation_history

        # Filter by time window
        cutoff_time = time.time() - (time_window_hours * 60 * 60)
        recent_adaptations = [a for a in adaptation_history if a.timestamp >= cutoff_time]

        if not recent_adaptations:
            return {
                "analysis_window_hours": time_window_hours,
                "message": f"No adaptations found in the last {time_window_hours} hours",
                "generated_at": time.time()
            }

        # Analyze effectiveness
        effectiveness_analysis = _analyze_adaptation_effectiveness(recent_adaptations, manager)

        # Performance impact analysis
        performance_impact = _analyze_performance_impact(recent_adaptations, manager)

        # Trend analysis
        trend_analysis = _analyze_adaptation_trends(recent_adaptations)

        # Generate recommendations
        recommendations = _generate_effectiveness_recommendations(effectiveness_analysis, trend_analysis)

        return {
            "analysis_window_hours": time_window_hours,
            "total_adaptations_analyzed": len(recent_adaptations),
            "effectiveness_analysis": effectiveness_analysis,
            "performance_impact": performance_impact,
            "trend_analysis": trend_analysis,
            "recommendations": recommendations,
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error analyzing cache adaptation effectiveness: {e}")
        return {
            "error": f"Failed to analyze cache adaptation effectiveness: {str(e)}"
        }


def get_cache_adaptation_rules_tool() -> Dict[str, Any]:
    """
    Get information about cache adaptation rules and their performance.

    Returns:
        Dictionary containing adaptation rules information including:
        - Rule definitions and conditions
        - Success rates and performance metrics
        - Rule effectiveness analysis
        - Recommendations for rule tuning
    """
    try:
        manager = get_adaptive_cache_manager()

        rules_info = []
        for rule in manager.adaptation_rules:
            rule_info = {
                "rule_id": rule.rule_id,
                "cache_type": rule.cache_type,
                "condition": rule.condition,
                "action": rule.action,
                "priority": rule.priority,
                "cooldown_minutes": rule.cooldown_minutes,
                "success_rate": rule.success_rate,
                "performance_impact": rule.performance_impact,
                "last_applied_minutes_ago": (time.time() - rule.last_applied) / 60 if rule.last_applied > 0 else None,
                "rule_status": _evaluate_rule_status(rule)
            }
            rules_info.append(rule_info)

        # Sort by priority and success rate
        rules_info.sort(key=lambda x: (x["priority"], x["success_rate"]), reverse=True)

        # Rule effectiveness analysis
        effectiveness_analysis = _analyze_rules_effectiveness(rules_info)

        # Recommendations for rule tuning
        tuning_recommendations = _generate_rule_tuning_recommendations(rules_info, effectiveness_analysis)

        return {
            "total_rules": len(rules_info),
            "rules_info": rules_info,
            "effectiveness_analysis": effectiveness_analysis,
            "tuning_recommendations": tuning_recommendations,
            "rules_by_priority": {
                "high": [r for r in rules_info if r["priority"] >= 8],
                "medium": [r for r in rules_info if 5 <= r["priority"] < 8],
                "low": [r for r in rules_info if r["priority"] < 5]
            },
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting cache adaptation rules: {e}")
        return {
            "error": f"Failed to get cache adaptation rules: {str(e)}"
        }


def _generate_adaptation_insights(status: Dict[str, Any]) -> List[str]:
    """Generate insights from adaptation status."""
    insights = []

    recent_decisions = status.get("recent_decisions", [])
    rules_status = status.get("adaptation_rules_status", [])

    if recent_decisions:
        total_decisions = len(recent_decisions)
        increase_decisions = sum(1 for d in recent_decisions if d["decision"] == "increase")
        decrease_decisions = sum(1 for d in recent_decisions if d["decision"] == "decrease")

        insights.append(f"Recent adaptations: {total_decisions} total, {increase_decisions} increases, {decrease_decisions} decreases")

        avg_confidence = sum(d["confidence"] for d in recent_decisions) / total_decisions
        insights.append(".2f")

    if rules_status:
        high_priority_rules = [r for r in rules_status if r["priority"] >= 8]
        successful_rules = [r for r in rules_status if r["success_rate"] > 0.7]

        insights.append(f"High-priority rules: {len(high_priority_rules)}, Successful rules: {len(successful_rules)}")

    adaptation_active = status.get("adaptation_active", False)
    if adaptation_active:
        insights.append("Adaptive cache management is active and running")
    else:
        insights.append("Adaptive cache management is currently inactive")

    return insights


def _analyze_adaptation_effectiveness(status: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze adaptation effectiveness."""
    recent_decisions = status.get("recent_decisions", [])

    if not recent_decisions:
        return {"message": "No recent adaptations to analyze"}

    total_decisions = len(recent_decisions)
    successful_decisions = sum(1 for d in recent_decisions if d["confidence"] > 0.7)
    success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0

    avg_confidence = sum(d["confidence"] for d in recent_decisions) / total_decisions

    # Group by cache type
    cache_type_effectiveness = {}
    for decision in recent_decisions:
        cache_type = decision["cache_type"]
        if cache_type not in cache_type_effectiveness:
            cache_type_effectiveness[cache_type] = []
        cache_type_effectiveness[cache_type].append(decision["confidence"])

    for cache_type in cache_type_effectiveness:
        confidences = cache_type_effectiveness[cache_type]
        cache_type_effectiveness[cache_type] = {
            "avg_confidence": sum(confidences) / len(confidences),
            "total_decisions": len(confidences),
            "high_confidence_rate": sum(1 for c in confidences if c > 0.8) / len(confidences)
        }

    return {
        "overall_success_rate": success_rate,
        "average_confidence": avg_confidence,
        "total_decisions": total_decisions,
        "cache_type_effectiveness": cache_type_effectiveness,
        "effectiveness_rating": "excellent" if success_rate > 0.8 else "good" if success_rate > 0.6 else "fair" if success_rate > 0.4 else "poor"
    }


def _generate_adaptation_recommendations(status: Dict[str, Any]) -> List[str]:
    """Generate adaptation recommendations."""
    recommendations = []

    effectiveness = status.get("effectiveness_analysis", {})
    success_rate = effectiveness.get("overall_success_rate", 0)

    if success_rate < 0.6:
        recommendations.extend([
            "Review adaptation rules and their conditions",
            "Consider adjusting adaptation confidence thresholds",
            "Monitor cache performance metrics more closely"
        ])

    rules_status = status.get("adaptation_rules_status", [])
    inactive_rules = [r for r in rules_status if r["last_applied_minutes_ago"] is None or r["last_applied_minutes_ago"] > 1440]  # 24 hours

    if inactive_rules:
        recommendations.append(f"Review {len(inactive_rules)} inactive adaptation rules")

    return recommendations


def _generate_performance_predictions(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate performance predictions for applied adaptations."""
    if not decisions:
        return {"message": "No adaptations to predict performance for"}

    predictions = {
        "expected_hit_rate_improvement": 0.0,
        "expected_latency_reduction_ms": 0.0,
        "expected_memory_change_mb": 0.0,
        "confidence_level": "medium",
        "time_to_effect_hours": 2
    }

    for decision in decisions:
        expected_impact = decision.get("expected_impact", {})

        predictions["expected_hit_rate_improvement"] += expected_impact.get("hit_rate_change", 0)
        predictions["expected_latency_reduction_ms"] += expected_impact.get("latency_change_ms", 0)
        predictions["expected_memory_change_mb"] += expected_impact.get("memory_change_mb", 0)

    # Adjust confidence based on number of adaptations
    if len(decisions) > 3:
        predictions["confidence_level"] = "low"
        predictions["time_to_effect_hours"] = 4
    elif len(decisions) == 1:
        predictions["confidence_level"] = "high"
        predictions["time_to_effect_hours"] = 1

    return predictions


def _get_strategy_description(strategy: AdaptationStrategy) -> str:
    """Get description for a sizing strategy."""
    descriptions = {
        AdaptationStrategy.PERFORMANCE_BASED: "Optimizes cache size based on hit rates and latency metrics",
        AdaptationStrategy.USAGE_PATTERN_BASED: "Adjusts cache size based on observed usage patterns and access frequency",
        AdaptationStrategy.LOAD_AWARE: "Considers system load and memory pressure when sizing cache",
        AdaptationStrategy.PREDICTIVE: "Uses predictive analytics to anticipate future cache needs",
        AdaptationStrategy.HYBRID: "Combines multiple strategies for optimal cache sizing"
    }
    return descriptions.get(strategy, "Unknown strategy")


def _generate_implementation_plan(cache_type: str, recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate implementation plan for cache sizing recommendation."""
    size_change = recommendation.get("size_change_mb", 0)

    if abs(size_change) < 50:
        complexity = "Low"
        steps = [
            "Update cache configuration with new size",
            "Monitor performance for 1 hour",
            "Adjust if needed based on observed metrics"
        ]
    elif abs(size_change) < 200:
        complexity = "Medium"
        steps = [
            "Backup current cache configuration",
            "Update cache size gradually (25% increments)",
            "Monitor performance metrics closely",
            "Rollback if performance degrades",
            "Full implementation within 2 hours"
        ]
    else:
        complexity = "High"
        steps = [
            "Perform comprehensive performance baseline",
            "Implement size change during low-traffic period",
            "Monitor extensively for 4+ hours",
            "Have rollback plan ready",
            "Consider A/B testing approach"
        ]

    return {
        "complexity": complexity,
        "estimated_effort_hours": 1 if complexity == "Low" else 2 if complexity == "Medium" else 4,
        "implementation_steps": steps,
        "risk_level": "Low" if abs(size_change) < 100 else "Medium" if abs(size_change) < 300 else "High",
        "rollback_plan": "Revert to previous cache size configuration"
    }


def _calculate_expected_sizing_impact(current_metrics: Dict[str, Any],
                                    recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate expected impact of cache sizing change."""
    size_change_mb = recommendation.get("size_change_mb", 0)
    current_hit_rate = current_metrics.get("hit_rate", 0.8)

    # Estimate hit rate change (rough approximation)
    hit_rate_change = (size_change_mb / 500) * 0.05  # 5% change per 500MB

    # Estimate latency impact
    latency_change = -size_change_mb * 0.01  # 10ms reduction per 1000MB increase

    # Memory efficiency impact
    memory_efficiency_change = size_change_mb * 0.002  # Small efficiency change

    return {
        "hit_rate_change_percent": hit_rate_change * 100,
        "latency_change_ms": latency_change,
        "memory_efficiency_change_percent": memory_efficiency_change * 100,
        "overall_performance_score_change": (hit_rate_change * 50) + (latency_change * 0.5),
        "time_to_realize_benefits_hours": 1 if abs(size_change_mb) < 200 else 2 if abs(size_change_mb) < 500 else 4
    }


def _analyze_adaptation_effectiveness(adaptations: List[Dict[str, Any]],
                                    manager) -> Dict[str, Any]:
    """Analyze effectiveness of adaptations."""
    if not adaptations:
        return {"message": "No adaptations to analyze"}

    # Group by cache type
    cache_effectiveness = {}
    for adaptation in adaptations:
        cache_type = adaptation["cache_type"]
        if cache_type not in cache_effectiveness:
            cache_effectiveness[cache_type] = []
        cache_effectiveness[cache_type].append(adaptation["confidence"])

    # Calculate effectiveness metrics
    effectiveness = {
        "total_adaptations": len(adaptations),
        "avg_confidence": sum(a["confidence"] for a in adaptations) / len(adaptations),
        "high_confidence_rate": sum(1 for a in adaptations if a["confidence"] > 0.8) / len(adaptations),
        "cache_type_breakdown": {}
    }

    for cache_type, confidences in cache_effectiveness.items():
        effectiveness["cache_type_breakdown"][cache_type] = {
            "adaptations_count": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences),
            "high_confidence_rate": sum(1 for c in confidences if c > 0.8) / len(confidences)
        }

    return effectiveness


def _analyze_performance_impact(adaptations: List[Dict[str, Any]], manager) -> Dict[str, Any]:
    """Analyze performance impact of adaptations."""
    if not adaptations:
        return {"message": "No adaptations to analyze"}

    # Calculate average impact
    total_hit_rate_change = sum(a.get("expected_impact", {}).get("hit_rate_change", 0) for a in adaptations)
    total_latency_change = sum(a.get("expected_impact", {}).get("latency_change_ms", 0) for a in adaptations)
    total_memory_change = sum(a.get("expected_impact", {}).get("memory_change_mb", 0) for a in adaptations)

    avg_hit_rate_change = total_hit_rate_change / len(adaptations)
    avg_latency_change = total_latency_change / len(adaptations)
    avg_memory_change = total_memory_change / len(adaptations)

    return {
        "avg_hit_rate_change_percent": avg_hit_rate_change * 100,
        "avg_latency_change_ms": avg_latency_change,
        "avg_memory_change_mb": avg_memory_change,
        "total_expected_impact_score": (avg_hit_rate_change * 50) - (abs(avg_latency_change) * 0.5),
        "impact_distribution": {
            "positive_impact": sum(1 for a in adaptations if a.get("expected_impact", {}).get("performance_score_change", 0) > 0),
            "negative_impact": sum(1 for a in adaptations if a.get("expected_impact", {}).get("performance_score_change", 0) < 0),
            "neutral_impact": sum(1 for a in adaptations if a.get("expected_impact", {}).get("performance_score_change", 0) == 0)
        }
    }


def _analyze_adaptation_trends(adaptations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze adaptation trends over time."""
    if len(adaptations) < 2:
        return {"message": "Insufficient data for trend analysis"}

    # Sort by timestamp
    sorted_adaptations = sorted(adaptations, key=lambda x: x["timestamp"])

    # Analyze frequency
    time_diffs = []
    for i in range(1, len(sorted_adaptations)):
        time_diff = sorted_adaptations[i]["timestamp"] - sorted_adaptations[i-1]["timestamp"]
        time_diffs.append(time_diff)

    avg_frequency_hours = (sum(time_diffs) / len(time_diffs)) / 3600 if time_diffs else 0

    # Analyze decision types
    decision_types = {}
    for adaptation in adaptations:
        decision = adaptation["decision"]
        decision_types[decision] = decision_types.get(decision, 0) + 1

    # Analyze confidence trend
    confidences = [a["confidence"] for a in sorted_adaptations]
    confidence_trend = "stable"
    if len(confidences) >= 3:
        first_half = confidences[:len(confidences)//2]
        second_half = confidences[len(confidences)//2:]
        if sum(second_half) / len(second_half) > sum(first_half) / len(first_half) + 0.1:
            confidence_trend = "improving"
        elif sum(second_half) / len(second_half) < sum(first_half) / len(first_half) - 0.1:
            confidence_trend = "declining"

    return {
        "avg_adaptation_frequency_hours": avg_frequency_hours,
        "decision_type_distribution": decision_types,
        "confidence_trend": confidence_trend,
        "total_time_span_hours": (sorted_adaptations[-1]["timestamp"] - sorted_adaptations[0]["timestamp"]) / 3600
    }


def _generate_effectiveness_recommendations(effectiveness: Dict[str, Any],
                                          trends: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on effectiveness analysis."""
    recommendations = []

    success_rate = effectiveness.get("high_confidence_rate", 0)
    if success_rate < 0.7:
        recommendations.append("Consider adjusting adaptation rule conditions to improve success rate")

    frequency = trends.get("avg_adaptation_frequency_hours", 0)
    if frequency < 1:
        recommendations.append("Adaptation frequency is low - consider reducing cooldown periods")
    elif frequency > 12:
        recommendations.append("Adaptation frequency is high - consider increasing cooldown periods")

    confidence_trend = trends.get("confidence_trend", "stable")
    if confidence_trend == "declining":
        recommendations.append("Adaptation confidence is declining - review recent adaptation performance")

    return recommendations


def _evaluate_rule_status(rule: Dict[str, Any]) -> str:
    """Evaluate the status of an adaptation rule."""
    success_rate = rule.get("success_rate", 0)
    last_applied = rule.get("last_applied_minutes_ago")

    if success_rate > 0.8:
        return "highly_effective"
    elif success_rate > 0.6:
        return "effective"
    elif success_rate > 0.4:
        return "moderately_effective"
    elif last_applied and last_applied > 1440:  # 24 hours
        return "inactive"
    else:
        return "needs_review"


def _analyze_rules_effectiveness(rules_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze effectiveness of adaptation rules."""
    if not rules_info:
        return {"message": "No rules to analyze"}

    total_rules = len(rules_info)
    active_rules = sum(1 for r in rules_info if r["rule_status"] != "inactive")
    effective_rules = sum(1 for r in rules_info if r["rule_status"] in ["highly_effective", "effective"])
    high_priority_rules = sum(1 for r in rules_info if r["priority"] >= 8)

    avg_success_rate = sum(r["success_rate"] for r in rules_info) / total_rules

    return {
        "total_rules": total_rules,
        "active_rules": active_rules,
        "effective_rules": effective_rules,
        "high_priority_rules": high_priority_rules,
        "avg_success_rate": avg_success_rate,
        "rules_effectiveness_rating": "excellent" if avg_success_rate > 0.8 else "good" if avg_success_rate > 0.6 else "fair" if avg_success_rate > 0.4 else "poor"
    }


def _generate_rule_tuning_recommendations(rules_info: List[Dict[str, Any]],
                                        effectiveness: Dict[str, Any]) -> List[str]:
    """Generate recommendations for tuning adaptation rules."""
    recommendations = []

    ineffective_rules = [r for r in rules_info if r["success_rate"] < 0.5 and r["rule_status"] != "inactive"]
    if ineffective_rules:
        recommendations.append(f"Review {len(ineffective_rules)} ineffective rules and consider adjusting conditions or disabling")

    inactive_rules = [r for r in rules_info if r["rule_status"] == "inactive"]
    if inactive_rules:
        recommendations.append(f"Evaluate {len(inactive_rules)} inactive rules - they may be too restrictive")

    high_priority_ineffective = [r for r in rules_info if r["priority"] >= 8 and r["success_rate"] < 0.6]
    if high_priority_ineffective:
        recommendations.append(f"High-priority rules with low success rates need immediate attention")

    avg_success_rate = effectiveness.get("avg_success_rate", 0)
    if avg_success_rate < 0.6:
        recommendations.append("Overall rule effectiveness is low - consider comprehensive rule review")

    return recommendations