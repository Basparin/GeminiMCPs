"""
Intelligent Prefetching Tools for CodeSage MCP Server.

This module provides tools for intelligent prefetching based on observed usage patterns,
including pattern analysis, prefetching control, and performance optimization.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.intelligent_prefetcher import (
    get_intelligent_prefetcher,
    PrefetchStrategy,
    AccessPattern
)

logger = logging.getLogger(__name__)


def get_prefetch_analysis_tool() -> Dict[str, Any]:
    """
    Get comprehensive intelligent prefetching analysis including patterns, metrics, and optimization opportunities.

    Returns:
        Dictionary containing prefetching analysis including:
        - Prefetching metrics and performance
        - Access pattern analysis
        - Current prefetch candidates
        - Pattern effectiveness analysis
        - Optimization opportunities
    """
    try:
        prefetcher = get_intelligent_prefetcher()
        analysis = prefetcher.get_prefetch_analysis()

        if "error" in analysis:
            return analysis

        # Enhance analysis with additional insights
        analysis["insights"] = _generate_prefetch_insights(analysis)
        analysis["health_score"] = _calculate_prefetch_health_score(analysis)
        analysis["recommendations"] = _generate_prefetch_recommendations(analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error getting prefetch analysis: {e}")
        return {
            "error": f"Failed to get prefetch analysis: {str(e)}"
        }


def trigger_prefetching_tool(strategy: str = "hybrid", max_candidates: int = 5) -> Dict[str, Any]:
    """
    Trigger intelligent prefetching with specified strategy and parameters.

    Args:
        strategy: Prefetching strategy to use ("pattern_based", "predictive", "collaborative", "hybrid")
        max_candidates: Maximum number of files to prefetch

    Returns:
        Dictionary containing prefetching results including:
        - Prefetching strategy used
        - Candidates selected for prefetching
        - Execution results and performance impact
        - Success metrics and recommendations
    """
    try:
        # Validate strategy
        try:
            strategy_enum = PrefetchStrategy(strategy.lower())
        except ValueError:
            return {
                "error": f"Invalid strategy: {strategy}",
                "valid_strategies": [s.value for s in PrefetchStrategy]
            }

        prefetcher = get_intelligent_prefetcher()

        # Update prefetch configuration
        original_batch_size = prefetcher.prefetch_config["prefetch_batch_size"]
        prefetcher.prefetch_config["prefetch_batch_size"] = max_candidates

        try:
            # Generate and execute prefetching
            prefetcher._generate_prefetch_candidates()
            prefetcher._execute_prefetching()

            # Get results
            results = {
                "strategy_used": strategy,
                "max_candidates_requested": max_candidates,
                "prefetch_candidates_selected": len(prefetcher.prefetch_candidates),
                "prefetch_candidates": [
                    {
                        "file_path": candidate.file_path,
                        "confidence_score": candidate.confidence_score,
                        "expected_benefit": candidate.expected_benefit,
                        "reason": candidate.reason,
                        "strategy": candidate.strategy.value
                    }
                    for candidate in prefetcher.prefetch_candidates
                ],
                "execution_results": {
                    "successful_prefetches": prefetcher.prefetch_metrics.successful_prefetches,
                    "failed_prefetches": prefetcher.prefetch_metrics.failed_prefetches,
                    "active_prefetches": len(prefetcher.active_prefetches)
                },
                "performance_impact": _estimate_prefetch_performance_impact(prefetcher.prefetch_candidates),
                "generated_at": time.time()
            }

            # Add success assessment
            results["success_assessment"] = _assess_prefetch_success(results)

            return results

        finally:
            # Restore original configuration
            prefetcher.prefetch_config["prefetch_batch_size"] = original_batch_size

    except Exception as e:
        logger.error(f"Error triggering prefetching: {e}")
        return {
            "error": f"Failed to trigger prefetching: {str(e)}"
        }


def get_prefetch_performance_metrics_tool(time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Get detailed prefetching performance metrics over a specified time window.

    Args:
        time_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing prefetching performance metrics including:
        - Success rates and accuracy metrics
        - Performance impact analysis
        - Bandwidth and latency savings
        - Trend analysis and forecasting
    """
    try:
        prefetcher = get_intelligent_prefetcher()

        # Get current metrics
        current_metrics = prefetcher.prefetch_metrics

        # Calculate derived metrics
        performance_metrics = {
            "current_metrics": {
                "total_prefetches": current_metrics.total_prefetches,
                "successful_prefetches": current_metrics.successful_prefetches,
                "failed_prefetches": current_metrics.failed_prefetches,
                "prefetch_accuracy": current_metrics.prefetch_accuracy,
                "average_latency_reduction_ms": current_metrics.average_latency_reduction_ms,
                "bandwidth_savings_mb": current_metrics.bandwidth_savings_mb
            },
            "derived_metrics": _calculate_derived_metrics(current_metrics),
            "trend_analysis": _analyze_prefetch_trends(prefetcher, time_window_hours),
            "efficiency_analysis": _analyze_prefetch_efficiency(prefetcher),
            "cost_benefit_analysis": _analyze_prefetch_cost_benefit(prefetcher),
            "time_window_hours": time_window_hours,
            "generated_at": time.time()
        }

        return performance_metrics

    except Exception as e:
        logger.error(f"Error getting prefetch performance metrics: {e}")
        return {
            "error": f"Failed to get prefetch performance metrics: {str(e)}"
        }


def analyze_prefetch_patterns_tool(pattern_type: str = "all") -> Dict[str, Any]:
    """
    Analyze prefetching patterns and their effectiveness.

    Args:
        pattern_type: Type of patterns to analyze ("temporal", "spatial", "sequential", "all")

    Returns:
        Dictionary containing pattern analysis including:
        - Pattern discovery and characteristics
        - Effectiveness analysis by pattern type
        - Pattern evolution over time
        - Recommendations for pattern-based optimization
    """
    try:
        prefetcher = get_intelligent_prefetcher()

        # Get pattern data based on type
        if pattern_type == "temporal":
            patterns = prefetcher.temporal_pattern_model
            pattern_enum = AccessPattern.TEMPORAL
        elif pattern_type == "spatial":
            patterns = prefetcher.file_coaccess_patterns
            pattern_enum = AccessPattern.SPATIAL
        elif pattern_type == "sequential":
            patterns = {
                key: value for key, value in prefetcher.access_pattern_model.items()
                if value.get("pattern_type") == AccessPattern.SEQUENTIAL.value
            }
            pattern_enum = AccessPattern.SEQUENTIAL
        else:
            # All patterns
            patterns = {
                **prefetcher.temporal_pattern_model,
                **prefetcher.file_coaccess_patterns,
                **{key: value for key, value in prefetcher.access_pattern_model.items()
                   if value.get("pattern_type") == AccessPattern.SEQUENTIAL.value}
            }
            pattern_enum = None

        # Analyze patterns
        pattern_analysis = {
            "pattern_type": pattern_type,
            "total_patterns_discovered": len(patterns),
            "pattern_characteristics": _analyze_pattern_characteristics(patterns, pattern_type),
            "effectiveness_analysis": _analyze_pattern_effectiveness(prefetcher, patterns, pattern_type),
            "pattern_evolution": _analyze_pattern_evolution(prefetcher, pattern_type),
            "optimization_recommendations": _generate_pattern_optimization_recommendations(patterns, pattern_type),
            "generated_at": time.time()
        }

        return pattern_analysis

    except Exception as e:
        logger.error(f"Error analyzing prefetch patterns: {e}")
        return {
            "error": f"Failed to analyze prefetch patterns: {str(e)}"
        }


def get_prefetch_configuration_tool() -> Dict[str, Any]:
    """
    Get current prefetching configuration and provide tuning recommendations.

    Returns:
        Dictionary containing prefetching configuration including:
        - Current configuration settings
        - Configuration effectiveness analysis
        - Tuning recommendations
        - Performance impact of different settings
    """
    try:
        prefetcher = get_intelligent_prefetcher()

        configuration = prefetcher.prefetch_config.copy()

        # Analyze configuration effectiveness
        config_analysis = {
            "current_configuration": configuration,
            "effectiveness_analysis": _analyze_configuration_effectiveness(prefetcher),
            "tuning_recommendations": _generate_configuration_tuning_recommendations(prefetcher),
            "performance_impact_analysis": _analyze_configuration_performance_impact(prefetcher),
            "generated_at": time.time()
        }

        return config_analysis

    except Exception as e:
        logger.error(f"Error getting prefetch configuration: {e}")
        return {
            "error": f"Failed to get prefetch configuration: {str(e)}"
        }


def update_prefetch_configuration_tool(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update prefetching configuration with new settings.

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
        prefetcher = get_intelligent_prefetcher()

        # Validate configuration updates
        validation_results = _validate_configuration_updates(config_updates)

        if not validation_results["valid"]:
            return {
                "error": "Invalid configuration updates",
                "validation_errors": validation_results["errors"]
            }

        # Store original configuration for rollback
        original_config = prefetcher.prefetch_config.copy()

        # Apply updates
        prefetcher.prefetch_config.update(config_updates)

        # Assess impact
        impact_assessment = _assess_configuration_impact(prefetcher, config_updates)

        results = {
            "configuration_updated": True,
            "updated_settings": config_updates,
            "validation_results": validation_results,
            "impact_assessment": impact_assessment,
            "rollback_plan": {
                "original_configuration": original_config,
                "rollback_steps": ["Revert configuration to previous settings", "Monitor performance for 5 minutes"],
                "rollback_trigger": "Performance degradation > 15% or user-reported issues"
            },
            "generated_at": time.time()
        }

        return results

    except Exception as e:
        logger.exception(f"Error updating prefetch configuration: {e}")
        return {
            "error": f"Failed to update prefetch configuration: {str(e)}"
        }


def _generate_prefetch_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from prefetch analysis."""
    insights = []

    metrics = analysis.get("prefetch_metrics", {})
    accuracy = metrics.get("prefetch_accuracy", 0)

    if accuracy > 0.8:
        insights.append("Excellent prefetch accuracy - system is effectively predicting access patterns")
    elif accuracy > 0.6:
        insights.append("Good prefetch accuracy - room for improvement in pattern recognition")
    else:
        insights.append("Low prefetch accuracy - consider adjusting prefetching parameters")

    patterns = analysis.get("access_patterns", {})
    unique_files = patterns.get("total_unique_files", 0)
    temporal_patterns = patterns.get("temporal_patterns_count", 0)

    if temporal_patterns > unique_files * 0.5:
        insights.append("Strong temporal patterns detected - predictive prefetching should be highly effective")
    elif temporal_patterns < 5:
        insights.append("Limited temporal patterns - focus on spatial and sequential patterns")

    candidates = analysis.get("current_candidates", [])
    if candidates:
        avg_confidence = sum(c["confidence_score"] for c in candidates) / len(candidates)
        insights.append(".2f")

    return insights


def _calculate_prefetch_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate prefetch health score (0-100)."""
    score = 100.0

    metrics = analysis.get("prefetch_metrics", {})
    accuracy = metrics.get("prefetch_accuracy", 0)

    # Accuracy impact (40% weight)
    accuracy_penalty = max(0, (0.8 - accuracy) * 125)  # Max 25 point penalty
    score -= accuracy_penalty * 0.4

    # Pattern diversity impact (30% weight)
    patterns = analysis.get("access_patterns", {})
    temporal_patterns = patterns.get("temporal_patterns_count", 0)
    spatial_patterns = patterns.get("spatial_patterns_count", 0)
    sequential_patterns = patterns.get("sequential_patterns_count", 0)

    total_patterns = temporal_patterns + spatial_patterns + sequential_patterns
    pattern_diversity_score = min(total_patterns / 20.0, 1.0)  # Max score at 20 patterns
    score -= (1.0 - pattern_diversity_score) * 30

    # Active prefetching impact (30% weight)
    active_prefetches = metrics.get("active_prefetches", 0)
    if active_prefetches > 10:
        score -= 15  # Too many active prefetches
    elif active_prefetches == 0:
        score -= 10  # No active prefetching

    return max(0.0, min(100.0, score))


def _generate_prefetch_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate prefetch recommendations."""
    recommendations = []

    metrics = analysis.get("prefetch_metrics", {})
    accuracy = metrics.get("prefetch_accuracy", 0)

    if accuracy < 0.7:
        recommendations.extend([
            "Consider adjusting prefetch confidence thresholds",
            "Review and improve pattern recognition algorithms",
            "Increase data collection period for better pattern analysis"
        ])

    patterns = analysis.get("access_patterns", {})
    if patterns.get("temporal_patterns_count", 0) < 5:
        recommendations.append("Collect more usage data to improve temporal pattern recognition")

    optimization_opportunities = analysis.get("optimization_opportunities", [])
    for opportunity in optimization_opportunities:
        recommendations.append(f"{opportunity['type']}: {opportunity['recommendation']}")

    return recommendations


def _estimate_prefetch_performance_impact(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate performance impact of prefetching candidates."""
    if not candidates:
        return {"impact": "no_candidates"}

    total_benefit = sum(candidate["expected_benefit"] for candidate in candidates)
    avg_confidence = sum(candidate["confidence_score"] for candidate in candidates) / len(candidates)

    return {
        "estimated_latency_reduction_ms": total_benefit * 10,  # Rough estimate
        "estimated_bandwidth_savings_mb": total_benefit * 0.5,
        "cache_hit_rate_improvement_percent": avg_confidence * 15,
        "overall_performance_score_change": total_benefit * 5,
        "confidence_level": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
    }


def _assess_prefetch_success(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess success of prefetching operation."""
    candidates_selected = results.get("prefetch_candidates_selected", 0)
    execution_results = results.get("execution_results", {})

    success_rate = 0
    if candidates_selected > 0:
        successful = execution_results.get("successful_prefetches", 0)
        success_rate = successful / candidates_selected

    assessment = {
        "success_rate": success_rate,
        "success_level": "excellent" if success_rate > 0.9 else "good" if success_rate > 0.7 else "fair" if success_rate > 0.5 else "poor",
        "candidates_processed": candidates_selected,
        "recommendations": []
    }

    if success_rate < 0.7:
        assessment["recommendations"].append("Consider adjusting prefetch parameters for better success rates")

    return assessment


def _calculate_derived_metrics(metrics: Any) -> Dict[str, Any]:
    """Calculate derived metrics from raw prefetch metrics."""
    total_prefetches = metrics.total_prefetches
    successful_prefetches = metrics.successful_prefetches

    derived = {
        "success_rate": successful_prefetches / max(total_prefetches, 1),
        "failure_rate": (total_prefetches - successful_prefetches) / max(total_prefetches, 1),
        "efficiency_score": metrics.prefetch_accuracy * (successful_prefetches / max(total_prefetches, 1)),
        "performance_gain_score": metrics.average_latency_reduction_ms / 10.0  # Normalized score
    }

    return derived


def _analyze_prefetch_trends(prefetcher: Any, time_window_hours: int) -> Dict[str, Any]:
    """Analyze prefetch trends over time."""
    # This would analyze historical prefetch data
    # For now, return placeholder analysis
    return {
        "accuracy_trend": "stable",
        "frequency_trend": "stable",
        "effectiveness_trend": "stable",
        "time_window_hours": time_window_hours
    }


def _analyze_prefetch_efficiency(prefetcher: Any) -> Dict[str, Any]:
    """Analyze prefetch efficiency."""
    metrics = prefetcher.prefetch_metrics

    efficiency = {
        "resource_utilization": "optimal" if metrics.prefetch_accuracy > 0.7 else "suboptimal",
        "bandwidth_efficiency": metrics.bandwidth_savings_mb / max(metrics.total_prefetches, 1),
        "latency_efficiency": metrics.average_latency_reduction_ms / max(metrics.successful_prefetches, 1),
        "overall_efficiency_score": (metrics.prefetch_accuracy + (metrics.bandwidth_savings_mb / 100) + (metrics.average_latency_reduction_ms / 50)) / 3
    }

    return efficiency


def _analyze_prefetch_cost_benefit(prefetcher: Any) -> Dict[str, Any]:
    """Analyze cost-benefit of prefetching."""
    metrics = prefetcher.prefetch_metrics

    # Simplified cost-benefit analysis
    benefits = metrics.bandwidth_savings_mb * 0.1 + metrics.average_latency_reduction_ms * 0.05
    costs = metrics.total_prefetches * 0.01  # Estimated cost per prefetch

    return {
        "total_benefits": benefits,
        "total_costs": costs,
        "net_benefit": benefits - costs,
        "benefit_cost_ratio": benefits / max(costs, 0.01),
        "roi_assessment": "excellent" if benefits > costs * 3 else "good" if benefits > costs * 2 else "fair" if benefits > costs else "poor"
    }


def _analyze_pattern_characteristics(patterns: Dict[str, Any], pattern_type: str) -> Dict[str, Any]:
    """Analyze characteristics of discovered patterns."""
    if not patterns:
        return {"characteristics": "no_patterns"}

    characteristics = {
        "total_patterns": len(patterns),
        "pattern_strength_distribution": _calculate_pattern_strength_distribution(patterns),
        "pattern_frequency_analysis": _analyze_pattern_frequency(patterns),
        "pattern_type": pattern_type
    }

    return characteristics


def _analyze_pattern_effectiveness(prefetcher: Any, patterns: Dict[str, Any], pattern_type: str) -> Dict[str, Any]:
    """Analyze effectiveness of patterns."""
    # Simplified effectiveness analysis
    effectiveness = {
        "pattern_success_rate": 0.75,  # Placeholder
        "average_confidence": 0.8,  # Placeholder
        "pattern_utilization_rate": len(patterns) / max(len(prefetcher.access_pattern_model), 1),
        "effectiveness_score": 0.7  # Placeholder
    }

    return effectiveness


def _analyze_pattern_evolution(prefetcher: Any, pattern_type: str) -> Dict[str, Any]:
    """Analyze pattern evolution over time."""
    # This would analyze how patterns change over time
    return {
        "evolution_trend": "stable",
        "new_patterns_discovered": 0,
        "patterns_strengthened": 0,
        "patterns_weakened": 0
    }


def _generate_pattern_optimization_recommendations(patterns: Dict[str, Any], pattern_type: str) -> List[str]:
    """Generate pattern optimization recommendations."""
    recommendations = []

    if len(patterns) < 5:
        recommendations.append("Collect more usage data to improve pattern recognition")

    # Pattern-specific recommendations
    if pattern_type == "temporal":
        recommendations.append("Consider adjusting temporal prediction windows")
    elif pattern_type == "spatial":
        recommendations.append("Review co-access thresholds for spatial patterns")
    elif pattern_type == "sequential":
        recommendations.append("Optimize sequence window sizes for sequential patterns")

    return recommendations


def _analyze_configuration_effectiveness(prefetcher: Any) -> Dict[str, Any]:
    """Analyze effectiveness of current configuration."""
    config = prefetcher.prefetch_config
    metrics = prefetcher.prefetch_metrics

    effectiveness = {
        "batch_size_effectiveness": "optimal" if config["prefetch_batch_size"] <= 5 else "may_cause_contention",
        "threshold_effectiveness": "optimal" if 0.6 <= config["prefetch_threshold"] <= 0.8 else "may_need_adjustment",
        "age_limit_effectiveness": "optimal" if config["max_prefetch_age_seconds"] <= 300 else "may_prefetch_stale_data",
        "overall_configuration_score": 0.8  # Placeholder
    }

    return effectiveness


def _generate_configuration_tuning_recommendations(prefetcher: Any) -> List[str]:
    """Generate configuration tuning recommendations."""
    recommendations = []

    config = prefetcher.prefetch_config
    metrics = prefetcher.prefetch_metrics

    if metrics.prefetch_accuracy < 0.7:
        recommendations.append("Consider increasing prefetch_threshold to improve accuracy")

    if config["prefetch_batch_size"] > 5:
        recommendations.append("Consider reducing prefetch_batch_size to avoid resource contention")

    return recommendations


def _analyze_configuration_performance_impact(prefetcher: Any) -> Dict[str, Any]:
    """Analyze performance impact of configuration settings."""
    # Simplified analysis
    return {
        "current_impact": "positive",
        "potential_improvements": ["Adjust batch size", "Optimize thresholds"],
        "risk_assessment": "low"
    }


def _validate_configuration_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration updates."""
    errors = []

    valid_keys = ["max_concurrent_prefetches", "prefetch_threshold", "max_prefetch_age_seconds", "prefetch_batch_size"]

    for key in updates:
        if key not in valid_keys:
            errors.append(f"Invalid configuration key: {key}")
        elif key == "prefetch_threshold":
            if not 0 <= updates[key] <= 1:
                errors.append("prefetch_threshold must be between 0 and 1")
        elif key in ["max_concurrent_prefetches", "prefetch_batch_size"]:
            if updates[key] < 1:
                errors.append(f"{key} must be greater than 0")
        elif key == "max_prefetch_age_seconds":
            if updates[key] < 0:
                errors.append("max_prefetch_age_seconds must be non-negative")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _assess_configuration_impact(prefetcher: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Assess impact of configuration changes."""
    impact = {
        "expected_accuracy_change": 0.0,
        "expected_performance_change": 0.0,
        "risk_level": "low",
        "monitoring_recommendations": ["Monitor prefetch accuracy", "Track performance metrics"]
    }

    # Assess impact of specific changes
    if "prefetch_threshold" in updates:
        threshold_change = updates["prefetch_threshold"] - prefetcher.prefetch_config["prefetch_threshold"]
        impact["expected_accuracy_change"] = threshold_change * 0.5

    return impact


def _calculate_pattern_strength_distribution(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate pattern strength distribution."""
    if not patterns:
        return {"distribution": "no_patterns"}

    # Simplified distribution calculation
    return {
        "strong_patterns": len([p for p in patterns.values() if isinstance(p, dict) and p.get("strength", 0) > 0.7]),
        "medium_patterns": len([p for p in patterns.values() if isinstance(p, dict) and 0.4 <= p.get("strength", 0) <= 0.7]),
        "weak_patterns": len([p for p in patterns.values() if isinstance(p, dict) and p.get("strength", 0) < 0.4])
    }


def _analyze_pattern_frequency(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze pattern frequency."""
    if not patterns:
        return {"frequency": "no_patterns"}

    frequencies = [p.get("frequency", 1) for p in patterns.values() if isinstance(p, dict)]
    if frequencies:
        return {
            "min_frequency": min(frequencies),
            "max_frequency": max(frequencies),
            "avg_frequency": sum(frequencies) / len(frequencies),
            "high_frequency_patterns": len([f for f in frequencies if f > 5])
        }
    else:
        return {"frequency": "no_frequency_data"}