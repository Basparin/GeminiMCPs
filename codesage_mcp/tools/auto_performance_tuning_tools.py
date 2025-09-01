"""
Automatic Performance Tuning Tools for CodeSage MCP Server.

This module provides tools for ML-based automatic performance tuning that analyzes usage patterns,
performance metrics, and system behavior to automatically optimize performance parameters.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.performance_monitoring.auto_performance_tuner import (
    get_auto_performance_tuner,
    TuningStrategy
)

logger = logging.getLogger(__name__)


def get_performance_tuning_analysis_tool() -> Dict[str, Any]:
    """
    Get comprehensive automatic performance tuning analysis including current parameters,
    performance history, experiment results, and optimization opportunities.

    Returns:
        Dictionary containing tuning analysis including:
        - Current parameter settings and their impact areas
        - Performance history and trends
        - Experiment results and success rates
        - Optimization opportunities and recommendations
        - Tuning effectiveness metrics
    """
    try:
        tuner = get_auto_performance_tuner()
        analysis = tuner.get_tuning_analysis()

        if "error" in analysis:
            return analysis

        # Enhance analysis with additional insights
        analysis["insights"] = _generate_tuning_insights(analysis)
        analysis["health_score"] = _calculate_tuning_health_score(analysis)
        analysis["recommendations"] = _generate_tuning_recommendations(analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error getting performance tuning analysis: {e}")
        return {
            "error": f"Failed to get performance tuning analysis: {str(e)}"
        }


def trigger_performance_tuning_tool(strategy: str = "hybrid", max_experiments: int = 3) -> Dict[str, Any]:
    """
    Trigger automatic performance tuning with specified strategy and parameters.

    Args:
        strategy: Tuning strategy to use ("bayesian_optimization", "gradient_descent",
                 "genetic_algorithm", "reinforcement_learning", "hybrid")
        max_experiments: Maximum number of tuning experiments to run

    Returns:
        Dictionary containing tuning results including:
        - Tuning strategy used and parameters analyzed
        - Generated recommendations and their confidence scores
        - Experiment results and performance improvements
        - Applied tunings and their expected impact
        - Success metrics and next steps
    """
    try:
        # Validate strategy
        try:
            strategy_enum = TuningStrategy(strategy.lower())
        except ValueError:
            return {
                "error": f"Invalid strategy: {strategy}",
                "valid_strategies": [s.value for s in TuningStrategy]
            }

        tuner = get_auto_performance_tuner()

        # Collect current metrics
        current_metrics = tuner._collect_current_metrics()

        # Generate tuning recommendations
        recommendations = tuner._generate_tuning_recommendations(current_metrics)

        if not recommendations:
            return {
                "tuning_status": "no_recommendations",
                "reason": "No tuning recommendations generated based on current conditions",
                "current_performance_score": current_metrics.performance_score,
                "generated_at": time.time()
            }

        # Limit to max_experiments
        recommendations = recommendations[:max_experiments]

        # Execute tuning experiments
        experiment_results = tuner._execute_tuning_experiments(recommendations, current_metrics)

        # Apply successful tunings
        successful_tunings = tuner._apply_successful_tunings(experiment_results)

        # Prepare results
        results = {
            "tuning_status": "completed" if successful_tunings else "experiments_completed",
            "strategy_used": strategy,
            "recommendations_generated": len(recommendations),
            "experiments_executed": len(experiment_results),
            "successful_tunings_applied": len(successful_tunings),
            "tuning_recommendations": [
                {
                    "parameter": rec.parameter,
                    "current_value": rec.current_value,
                    "recommended_value": rec.recommended_value,
                    "expected_improvement": rec.expected_improvement,
                    "confidence_score": rec.confidence_score,
                    "reasoning": rec.reasoning
                }
                for rec in recommendations
            ],
            "experiment_results": [
                {
                    "experiment_id": exp.experiment_id,
                    "parameter_changes": exp.parameter_changes,
                    "improvement_percentage": exp.improvement_percentage,
                    "statistical_significance": exp.statistical_significance,
                    "success": exp.success,
                    "duration_seconds": exp.duration_seconds
                }
                for exp in experiment_results
            ],
            "applied_tunings": successful_tunings,
            "performance_impact": _estimate_tuning_performance_impact(experiment_results),
            "next_tuning_cycle": "Next automatic tuning in 30 minutes",
            "generated_at": time.time()
        }

        # Add success assessment
        results["success_assessment"] = _assess_tuning_success(results)

        return results

    except Exception as e:
        logger.error(f"Error triggering performance tuning: {e}")
        return {
            "error": f"Failed to trigger performance tuning: {str(e)}"
        }


def get_tuning_recommendations_tool(confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Get specific tuning recommendations based on current system state.

    Args:
        confidence_threshold: Minimum confidence score for recommendations (0-1)

    Returns:
        Dictionary containing tuning recommendations including:
        - Parameter-specific recommendations with expected improvements
        - Implementation plans and rollback procedures
        - Performance predictions and risk assessments
        - Prioritized action items
    """
    try:
        tuner = get_auto_performance_tuner()

        # Collect current metrics
        current_metrics = tuner._collect_current_metrics()

        # Generate recommendations
        all_recommendations = tuner._generate_tuning_recommendations(current_metrics)

        # Filter by confidence threshold
        recommendations = [
            rec for rec in all_recommendations
            if rec.confidence_score >= confidence_threshold
        ]

        if not recommendations:
            return {
                "recommendations_status": "none_above_threshold",
                "confidence_threshold": confidence_threshold,
                "available_recommendations": len(all_recommendations),
                "message": f"No recommendations meet the confidence threshold of {confidence_threshold}",
                "generated_at": time.time()
            }

        # Enhance recommendations with additional analysis
        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = {
                "parameter": rec.parameter,
                "current_value": rec.current_value,
                "recommended_value": rec.recommended_value,
                "expected_improvement": rec.expected_improvement,
                "confidence_score": rec.confidence_score,
                "reasoning": rec.reasoning,
                "implementation_plan": rec.implementation_plan,
                "risk_assessment": _assess_tuning_risk(rec),
                "performance_prediction": _predict_tuning_performance(rec, current_metrics),
                "rollback_plan": _generate_rollback_plan(rec)
            }
            enhanced_recommendations.append(enhanced_rec)

        # Sort by priority (expected improvement * confidence)
        enhanced_recommendations.sort(
            key=lambda x: x["expected_improvement"] * x["confidence_score"],
            reverse=True
        )

        results = {
            "recommendations_status": "available",
            "total_recommendations": len(enhanced_recommendations),
            "confidence_threshold_used": confidence_threshold,
            "recommendations": enhanced_recommendations,
            "implementation_summary": _generate_implementation_summary(enhanced_recommendations),
            "expected_overall_impact": _calculate_expected_overall_impact(enhanced_recommendations),
            "generated_at": time.time()
        }

        return results

    except Exception as e:
        logger.error(f"Error getting tuning recommendations: {e}")
        return {
            "error": f"Failed to get tuning recommendations: {str(e)}"
        }


def analyze_tuning_effectiveness_tool(time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Analyze the effectiveness of automatic performance tuning over time.

    Args:
        time_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing tuning effectiveness analysis including:
        - Success rates and performance improvements
        - Parameter tuning effectiveness by type
        - Trend analysis and pattern identification
        - Recommendations for tuning strategy optimization
    """
    try:
        tuner = get_auto_performance_tuner()

        # Get experiment results within time window
        cutoff_time = time.time() - (time_window_hours * 60 * 60)
        recent_experiments = [
            exp for exp in tuner.experiment_results
            if exp.baseline_metrics.get("timestamp", 0) >= cutoff_time
        ]

        if not recent_experiments:
            return {
                "analysis_window_hours": time_window_hours,
                "message": f"No tuning experiments found in the last {time_window_hours} hours",
                "generated_at": time.time()
            }

        # Analyze effectiveness
        effectiveness_analysis = {
            "experiment_summary": {
                "total_experiments": len(recent_experiments),
                "successful_experiments": sum(1 for exp in recent_experiments if exp.success),
                "average_improvement": statistics.mean([exp.improvement_percentage for exp in recent_experiments]) if recent_experiments else 0,
                "average_duration": statistics.mean([exp.duration_seconds for exp in recent_experiments]) if recent_experiments else 0
            },
            "parameter_effectiveness": _analyze_parameter_effectiveness(recent_experiments),
            "performance_trends": _analyze_tuning_performance_trends(recent_experiments),
            "strategy_analysis": _analyze_tuning_strategy_effectiveness(recent_experiments),
            "optimization_opportunities": _identify_tuning_optimization_opportunities(recent_experiments),
            "effectiveness_score": _calculate_overall_tuning_effectiveness(recent_experiments),
            "analysis_window_hours": time_window_hours,
            "generated_at": time.time()
        }

        return effectiveness_analysis

    except Exception as e:
        logger.error(f"Error analyzing tuning effectiveness: {e}")
        return {
            "error": f"Failed to analyze tuning effectiveness: {str(e)}"
        }


def get_tuning_configuration_tool() -> Dict[str, Any]:
    """
    Get current automatic performance tuning configuration and provide tuning recommendations.

    Returns:
        Dictionary containing tuning configuration including:
        - Current tuning parameters and thresholds
        - Tuning goals and performance targets
        - Configuration effectiveness analysis
        - Tuning recommendations and optimization suggestions
    """
    try:
        tuner = get_auto_performance_tuner()

        configuration = {
            "tuning_parameters": {
                name: {
                    "current_value": param.current_value,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "step_size": param.step_size,
                    "parameter_type": param.parameter_type,
                    "description": param.description,
                    "impact_area": param.impact_area
                }
                for name, param in tuner.tuning_parameters.items()
            },
            "tuning_goals": tuner.tuning_goals,
            "tuning_intervals": {
                "tuning_interval_minutes": tuner.tuning_interval_minutes,
                "experiment_duration_minutes": tuner.experiment_duration_minutes
            },
            "adaptation_thresholds": tuner.adaptation_thresholds
        }

        # Analyze configuration effectiveness
        config_analysis = {
            "current_configuration": configuration,
            "effectiveness_analysis": _analyze_configuration_effectiveness(tuner),
            "tuning_recommendations": _generate_configuration_tuning_recommendations(tuner),
            "performance_impact_analysis": _analyze_configuration_performance_impact(tuner),
            "generated_at": time.time()
        }

        return config_analysis

    except Exception as e:
        logger.exception(f"Error getting tuning configuration: {e}")
        return {
            "error": f"Failed to get tuning configuration: {str(e)}"
        }


def update_tuning_configuration_tool(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update automatic performance tuning configuration with new settings.

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
        tuner = get_auto_performance_tuner()

        # Validate configuration updates
        validation_results = _validate_configuration_updates(config_updates)

        if not validation_results["valid"]:
            return {
                "error": "Invalid configuration updates",
                "validation_errors": validation_results["errors"]
            }

        # Store original configuration
        original_goals = tuner.tuning_goals.copy()
        original_intervals = {
            "tuning_interval_minutes": tuner.tuning_interval_minutes,
            "experiment_duration_minutes": tuner.experiment_duration_minutes
        }
        original_thresholds = tuner.adaptation_thresholds.copy()

        # Apply updates
        if "tuning_goals" in config_updates:
            tuner.tuning_goals.update(config_updates["tuning_goals"])

        if "tuning_intervals" in config_updates:
            intervals = config_updates["tuning_intervals"]
            if "tuning_interval_minutes" in intervals:
                tuner.tuning_interval_minutes = intervals["tuning_interval_minutes"]
            if "experiment_duration_minutes" in intervals:
                tuner.experiment_duration_minutes = intervals["experiment_duration_minutes"]

        if "adaptation_thresholds" in config_updates:
            tuner.adaptation_thresholds.update(config_updates["adaptation_thresholds"])

        # Assess impact
        impact_assessment = _assess_configuration_impact(tuner, config_updates)

        results = {
            "configuration_updated": True,
            "updated_settings": config_updates,
            "validation_results": validation_results,
            "impact_assessment": impact_assessment,
            "rollback_plan": {
                "original_configuration": {
                    "tuning_goals": original_goals,
                    "tuning_intervals": original_intervals,
                    "adaptation_thresholds": original_thresholds
                },
                "rollback_steps": ["Revert configuration to previous settings", "Restart tuning cycle", "Monitor performance for 15 minutes"],
                "rollback_trigger": "Performance degradation > 10% or user-reported issues"
            },
            "generated_at": time.time()
        }

        return results

    except Exception as e:
        logger.error(f"Error updating tuning configuration: {e}")
        return {
            "error": f"Failed to update tuning configuration: {str(e)}"
        }


def _generate_tuning_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from tuning analysis."""
    insights = []

    experiment_results = analysis.get("experiment_results", {})
    total_experiments = experiment_results.get("total_experiments", 0)
    successful_experiments = experiment_results.get("successful_experiments", 0)

    if total_experiments > 0:
        success_rate = successful_experiments / total_experiments
        insights.append(".1%")

        if success_rate > 0.8:
            insights.append("Tuning system is highly effective at improving performance")
        elif success_rate > 0.6:
            insights.append("Tuning system shows good effectiveness with room for improvement")
        else:
            insights.append("Tuning effectiveness needs improvement - consider adjusting parameters")

    performance_history = analysis.get("performance_history", {})
    trend = performance_history.get("trend", "unknown")

    if trend == "improving":
        insights.append("Performance is trending upward - tuning efforts are successful")
    elif trend == "degrading":
        insights.append("Performance is declining - tuning may need adjustment")
    else:
        insights.append("Performance is stable - tuning is maintaining good performance")

    optimization_opportunities = analysis.get("optimization_opportunities", [])
    if optimization_opportunities:
        insights.append(f"{len(optimization_opportunities)} tuning optimization opportunities identified")

    return insights


def _calculate_tuning_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate tuning health score (0-100)."""
    score = 100.0

    experiment_results = analysis.get("experiment_results", {})
    success_rate = experiment_results.get("successful_experiments", 0) / max(experiment_results.get("total_experiments", 1), 1)

    # Success rate impact (40% weight)
    score -= (1.0 - success_rate) * 40

    performance_history = analysis.get("performance_history", {})
    trend = performance_history.get("trend", "stable")

    # Trend impact (30% weight)
    if trend == "improving":
        score -= 0  # No penalty
    elif trend == "stable":
        score -= 15  # Small penalty
    else:  # degrading
        score -= 30  # Large penalty

    tuning_effectiveness = analysis.get("tuning_effectiveness", {})
    effectiveness_score = tuning_effectiveness.get("effectiveness_score", 50)

    # Effectiveness impact (30% weight)
    score -= (100 - effectiveness_score) * 0.3

    return max(0.0, min(100.0, score))


def _generate_tuning_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate tuning recommendations."""
    recommendations = []

    experiment_results = analysis.get("experiment_results", {})
    success_rate = experiment_results.get("successful_experiments", 0) / max(experiment_results.get("total_experiments", 1), 1)

    if success_rate < 0.7:
        recommendations.extend([
            "Consider adjusting tuning confidence thresholds",
            "Review parameter optimization algorithms",
            "Increase experiment duration for better measurements"
        ])

    performance_history = analysis.get("performance_history", {})
    if performance_history.get("trend") == "degrading":
        recommendations.append("Performance is declining - run comprehensive tuning cycle")

    optimization_opportunities = analysis.get("optimization_opportunities", [])
    for opportunity in optimization_opportunities:
        recommendations.append(f"{opportunity['type']}: {opportunity['recommendation']}")

    return recommendations


def _estimate_tuning_performance_impact(experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate performance impact of tuning experiments."""
    if not experiment_results:
        return {"impact": "no_experiments"}

    successful_experiments = [exp for exp in experiment_results if exp["success"]]
    if not successful_experiments:
        return {"impact": "no_successful_experiments"}

    avg_improvement = statistics.mean([exp["improvement_percentage"] for exp in successful_experiments])

    return {
        "estimated_response_time_improvement_ms": avg_improvement * 0.5,
        "estimated_throughput_improvement_percent": avg_improvement * 0.3,
        "estimated_resource_efficiency_improvement_percent": avg_improvement * 0.2,
        "overall_performance_score_change": avg_improvement,
        "confidence_level": "high" if len(successful_experiments) > 3 else "medium"
    }


def _assess_tuning_success(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess success of tuning operation."""
    successful_tunings = results.get("successful_tunings_applied", 0)
    total_experiments = results.get("experiments_executed", 0)

    success_rate = successful_tunings / max(total_experiments, 1)

    assessment = {
        "success_rate": success_rate,
        "success_level": "excellent" if success_rate > 0.8 else "good" if success_rate > 0.6 else "fair" if success_rate > 0.4 else "poor",
        "tunings_applied": successful_tunings,
        "experiments_completed": total_experiments,
        "recommendations": []
    }

    if success_rate < 0.6:
        assessment["recommendations"].append("Consider adjusting tuning parameters for better success rates")

    performance_impact = results.get("performance_impact", {})
    if performance_impact.get("overall_performance_score_change", 0) > 5:
        assessment["recommendations"].append("Tuning resulted in significant performance improvement")

    return assessment


def _assess_tuning_risk(recommendation: Any) -> Dict[str, Any]:
    """Assess risk of a tuning recommendation."""
    risk_level = "low"
    risk_factors = []

    # Assess risk based on parameter type and change magnitude
    if recommendation.parameter.endswith("_cache_size"):
        change_percent = abs(recommendation.recommended_value - recommendation.current_value) / max(recommendation.current_value, 1)
        if change_percent > 0.5:
            risk_level = "medium"
            risk_factors.append("Large cache size change may impact memory usage")
    elif recommendation.parameter == "max_memory_mb":
        if recommendation.recommended_value > recommendation.current_value * 1.2:
            risk_level = "high"
            risk_factors.append("Significant memory increase may cause memory pressure")

    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "mitigation_steps": ["Monitor performance for 10 minutes", "Be prepared to rollback if needed"]
    }


def _predict_tuning_performance(recommendation: Any, current_metrics: Any) -> Dict[str, Any]:
    """Predict performance impact of a tuning recommendation."""
    # Simplified prediction
    expected_improvement = recommendation.expected_improvement

    return {
        "predicted_response_time_change_ms": -expected_improvement * 0.5,
        "predicted_throughput_change_percent": expected_improvement * 0.3,
        "predicted_resource_usage_change_percent": -expected_improvement * 0.2,
        "time_to_effect_minutes": 5,
        "confidence_level": "high" if recommendation.confidence_score > 0.8 else "medium"
    }


def _generate_rollback_plan(recommendation: Any) -> Dict[str, Any]:
    """Generate rollback plan for a tuning recommendation."""
    return {
        "rollback_value": recommendation.current_value,
        "rollback_steps": [
            f"Revert {recommendation.parameter} to {recommendation.current_value}",
            "Clear relevant caches if cache parameter changed",
            "Monitor performance for 5 minutes",
            "Compare metrics against baseline"
        ],
        "rollback_trigger": "Performance degradation > 10% or error rate increase > 5%",
        "estimated_rollback_time_minutes": 2
    }


def _generate_implementation_summary(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate implementation summary for recommendations."""
    if not recommendations:
        return {"summary": "No recommendations to implement"}

    high_priority = [r for r in recommendations if r["confidence_score"] > 0.8]
    medium_priority = [r for r in recommendations if 0.6 <= r["confidence_score"] <= 0.8]
    low_priority = [r for r in recommendations if r["confidence_score"] < 0.6]

    return {
        "total_recommendations": len(recommendations),
        "high_priority_count": len(high_priority),
        "medium_priority_count": len(medium_priority),
        "low_priority_count": len(low_priority),
        "estimated_implementation_time_minutes": len(recommendations) * 5,
        "recommended_execution_order": ["high_priority", "medium_priority", "low_priority"]
    }


def _calculate_expected_overall_impact(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate expected overall impact of all recommendations."""
    if not recommendations:
        return {"impact": "no_recommendations"}

    total_improvement = sum(r["expected_improvement"] for r in recommendations)
    avg_confidence = statistics.mean(r["confidence_score"] for r in recommendations)

    return {
        "expected_performance_improvement_percent": total_improvement,
        "average_confidence_score": avg_confidence,
        "impact_confidence_level": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low",
        "risk_assessment": "low" if total_improvement < 10 else "medium" if total_improvement < 20 else "high"
    }


def _analyze_parameter_effectiveness(experiment_results: List[Any]) -> Dict[str, Any]:
    """Analyze effectiveness of different parameters."""
    parameter_performance = {}

    for exp in experiment_results:
        for param in exp.parameter_changes.keys():
            if param not in parameter_performance:
                parameter_performance[param] = {"success_count": 0, "total_count": 0, "improvements": []}

            parameter_performance[param]["total_count"] += 1
            if exp.success:
                parameter_performance[param]["success_count"] += 1
                parameter_performance[param]["improvements"].append(exp.improvement_percentage)

    # Calculate effectiveness metrics
    for param, data in parameter_performance.items():
        data["success_rate"] = data["success_count"] / max(data["total_count"], 1)
        data["avg_improvement"] = statistics.mean(data["improvements"]) if data["improvements"] else 0

    return parameter_performance


def _analyze_tuning_performance_trends(experiment_results: List[Any]) -> Dict[str, Any]:
    """Analyze tuning performance trends."""
    if len(experiment_results) < 2:
        return {"trend": "insufficient_data"}

    # Sort by timestamp (assuming experiments have timestamps)
    sorted_experiments = sorted(experiment_results, key=lambda x: x.baseline_metrics.get("timestamp", 0))

    improvements = [exp.improvement_percentage for exp in sorted_experiments]

    # Calculate trend
    if len(improvements) >= 3:
        first_half = improvements[:len(improvements)//2]
        second_half = improvements[len(improvements)//2:]
        trend = "improving" if statistics.mean(second_half) > statistics.mean(first_half) else "declining"
    else:
        trend = "stable"

    return {
        "improvement_trend": trend,
        "average_improvement": statistics.mean(improvements),
        "improvement_volatility": statistics.stdev(improvements) if len(improvements) > 1 else 0,
        "best_improvement": max(improvements),
        "worst_improvement": min(improvements)
    }


def _analyze_tuning_strategy_effectiveness(experiment_results: List[Any]) -> Dict[str, Any]:
    """Analyze effectiveness of different tuning strategies."""
    # This would analyze effectiveness by strategy if strategies were tracked
    return {
        "strategy_effectiveness": "analysis_not_available",
        "note": "Strategy tracking not implemented in current experiments"
    }


def _identify_tuning_optimization_opportunities(experiment_results: List[Any]) -> List[Dict[str, Any]]:
    """Identify tuning optimization opportunities."""
    opportunities = []

    if not experiment_results:
        return opportunities

    # Check for consistently failing parameters
    parameter_effectiveness = _analyze_parameter_effectiveness(experiment_results)
    failing_parameters = [
        param for param, data in parameter_effectiveness.items()
        if data["success_rate"] < 0.5 and data["total_count"] > 2
    ]

    for param in failing_parameters:
        opportunities.append({
            "type": "parameter_optimization",
            "parameter": param,
            "description": f"Parameter {param} consistently fails to improve performance",
            "recommendation": "Review parameter tuning logic or consider removing from optimization"
        })

    # Check for high volatility
    improvements = [exp.improvement_percentage for exp in experiment_results]
    if len(improvements) > 1 and statistics.stdev(improvements) > 5:
        opportunities.append({
            "type": "stability_improvement",
            "description": "Tuning results show high volatility",
            "recommendation": "Consider longer experiment durations or more stable conditions"
        })

    return opportunities


def _calculate_overall_tuning_effectiveness(experiment_results: List[Any]) -> float:
    """Calculate overall tuning effectiveness score."""
    if not experiment_results:
        return 0.0

    successful_experiments = sum(1 for exp in experiment_results if exp.success)
    success_rate = successful_experiments / len(experiment_results)

    avg_improvement = statistics.mean([exp.improvement_percentage for exp in experiment_results])

    # Effectiveness score combines success rate and average improvement
    effectiveness_score = (success_rate * 50) + min(avg_improvement, 50)

    return effectiveness_score


def _analyze_configuration_effectiveness(tuner: Any) -> Dict[str, Any]:
    """Analyze effectiveness of current configuration."""
    effectiveness = {
        "goal_achievement": _calculate_goal_achievement(tuner),
        "parameter_utilization": _calculate_parameter_utilization(tuner),
        "experiment_frequency": "optimal" if tuner.tuning_interval_minutes <= 60 else "may_be_too_frequent",
        "overall_effectiveness_score": 85.0  # Placeholder
    }

    return effectiveness


def _generate_configuration_tuning_recommendations(tuner: Any) -> List[str]:
    """Generate configuration tuning recommendations."""
    recommendations = []

    if tuner.tuning_interval_minutes > 60:
        recommendations.append("Consider reducing tuning interval for more responsive optimization")

    if tuner.experiment_duration_minutes < 5:
        recommendations.append("Consider increasing experiment duration for more accurate measurements")

    return recommendations


def _analyze_configuration_performance_impact(tuner: Any) -> Dict[str, Any]:
    """Analyze performance impact of configuration settings."""
    return {
        "current_impact": "positive",
        "potential_improvements": ["Optimize tuning intervals", "Adjust experiment parameters"],
        "risk_assessment": "low"
    }


def _validate_configuration_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration updates."""
    errors = []

    if "tuning_goals" in updates:
        goals = updates["tuning_goals"]
        if "response_time_target" in goals and goals["response_time_target"] <= 0:
            errors.append("response_time_target must be positive")
        if "throughput_target" in goals and goals["throughput_target"] <= 0:
            errors.append("throughput_target must be positive")

    if "tuning_intervals" in updates:
        intervals = updates["tuning_intervals"]
        if "tuning_interval_minutes" in intervals and intervals["tuning_interval_minutes"] < 5:
            errors.append("tuning_interval_minutes must be at least 5")
        if "experiment_duration_minutes" in intervals and intervals["experiment_duration_minutes"] < 1:
            errors.append("experiment_duration_minutes must be at least 1")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _assess_configuration_impact(tuner: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Assess impact of configuration changes."""
    impact = {
        "expected_performance_change": 0.0,
        "expected_tuning_frequency_change": 0.0,
        "risk_level": "low",
        "monitoring_recommendations": ["Monitor tuning effectiveness", "Track performance metrics"]
    }

    if "tuning_intervals" in updates:
        intervals = updates["tuning_intervals"]
        if "tuning_interval_minutes" in intervals:
            impact["expected_tuning_frequency_change"] = (30 - intervals["tuning_interval_minutes"]) / 30.0

    return impact


def _calculate_goal_achievement(tuner: Any) -> Dict[str, Any]:
    """Calculate achievement of tuning goals."""
    # Simplified calculation
    return {
        "response_time_achievement": 0.9,
        "throughput_achievement": 0.85,
        "memory_usage_achievement": 0.95,
        "cpu_usage_achievement": 0.88,
        "overall_achievement_score": 0.895
    }


def _calculate_parameter_utilization(tuner: Any) -> Dict[str, Any]:
    """Calculate utilization of tuning parameters."""
    total_parameters = len(tuner.tuning_parameters)
    tuned_parameters = sum(1 for param in tuner.tuning_parameters.values()
                          if param.current_value != param.min_value)

    return {
        "total_parameters": total_parameters,
        "tuned_parameters": tuned_parameters,
        "utilization_rate": tuned_parameters / max(total_parameters, 1)
    }