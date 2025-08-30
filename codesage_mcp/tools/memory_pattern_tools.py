"""
Memory Pattern Monitoring Tools for CodeSage MCP Server.

This module provides tools for monitoring memory usage patterns under varying loads,
including adaptive memory management, load-aware optimization, and predictive memory allocation.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.memory_pattern_monitor import (
    get_memory_pattern_monitor,
    get_adaptive_memory_manager,
    get_load_aware_optimizer,
    LoadLevel
)

logger = logging.getLogger(__name__)


def analyze_memory_patterns_tool(analysis_window_hours: int = 24) -> Dict[str, Any]:
    """
    Analyze memory usage patterns under varying loads.

    Args:
        analysis_window_hours: Number of hours to analyze (default: 24)

    Returns:
        Dictionary containing comprehensive memory pattern analysis including:
        - Memory usage statistics across different load levels
        - Detected patterns and their characteristics
        - Trends and optimization opportunities
        - Load distribution analysis
    """
    try:
        monitor = get_memory_pattern_monitor()
        analysis = monitor.get_memory_analysis(analysis_window_hours)

        if "error" in analysis:
            return analysis

        # Enhance analysis with additional insights
        analysis["insights"] = _generate_memory_insights(analysis)
        analysis["health_score"] = _calculate_memory_health_score(analysis)
        analysis["recommendations"] = _generate_memory_recommendations(analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing memory patterns: {e}")
        return {
            "error": f"Failed to analyze memory patterns: {str(e)}"
        }


def get_adaptive_memory_management_tool() -> Dict[str, Any]:
    """
    Get adaptive memory management recommendations based on current patterns.

    Returns:
        Dictionary containing adaptive memory management analysis including:
        - Current memory pressure assessment
        - Recommended adaptations based on patterns
        - Applied adaptations and their status
        - Memory optimization opportunities
    """
    try:
        manager = get_adaptive_memory_manager()
        adaptation_analysis = manager.adapt_memory_settings()

        # Enhance with additional context
        adaptation_analysis["memory_pressure_analysis"] = _analyze_memory_pressure(adaptation_analysis)
        adaptation_analysis["adaptation_effectiveness"] = _assess_adaptation_effectiveness(adaptation_analysis)

        return adaptation_analysis

    except Exception as e:
        logger.error(f"Error getting adaptive memory management: {e}")
        return {
            "error": f"Failed to get adaptive memory management: {str(e)}"
        }


def optimize_memory_for_load_tool(load_level: str) -> Dict[str, Any]:
    """
    Optimize memory settings for a specific load level.

    Args:
        load_level: Target load level ("idle", "light", "moderate", "heavy", "critical")

    Returns:
        Dictionary containing load-specific memory optimization including:
        - Optimal memory settings for the target load
        - Detailed optimization plan
        - Implementation steps and monitoring guidance
        - Expected performance improvements
    """
    try:
        # Validate load level
        try:
            load_level_enum = LoadLevel(load_level.lower())
        except ValueError:
            return {
                "error": f"Invalid load level: {load_level}",
                "valid_levels": [level.value for level in LoadLevel]
            }

        optimizer = get_load_aware_optimizer()
        optimization = optimizer.optimize_for_load(load_level_enum)

        # Enhance with additional analysis
        optimization["load_analysis"] = _analyze_load_characteristics(load_level_enum)
        optimization["implementation_readiness"] = _assess_implementation_readiness(optimization)

        return optimization

    except Exception as e:
        logger.error(f"Error optimizing memory for load: {e}")
        return {
            "error": f"Failed to optimize memory for load: {str(e)}",
            "load_level": load_level
        }


def get_memory_pressure_analysis_tool() -> Dict[str, Any]:
    """
    Analyze current memory pressure and provide detailed insights.

    Returns:
        Dictionary containing memory pressure analysis including:
        - Current memory pressure level
        - Pressure sources and contributing factors
        - Pressure mitigation strategies
        - Predictive pressure analysis
    """
    try:
        monitor = get_memory_pattern_monitor()
        manager = get_adaptive_memory_manager()

        # Get current memory state
        current_analysis = monitor.get_memory_analysis(1)  # Last hour

        if "error" in current_analysis:
            return current_analysis

        # Assess current pressure
        memory_stats = current_analysis.get("memory_statistics", {})
        current_memory_mb = memory_stats.get("average_mb", 0)

        # Determine pressure level
        pressure_level = _determine_memory_pressure_level(current_memory_mb)

        # Analyze pressure sources
        pressure_sources = _analyze_memory_pressure_sources(current_analysis)

        # Generate mitigation strategies
        mitigation_strategies = _generate_pressure_mitigation_strategies(pressure_level, pressure_sources)

        # Predictive analysis
        predictive_analysis = _analyze_predictive_memory_pressure(monitor)

        return {
            "current_pressure_level": pressure_level,
            "current_memory_mb": current_memory_mb,
            "memory_statistics": memory_stats,
            "pressure_sources": pressure_sources,
            "mitigation_strategies": mitigation_strategies,
            "predictive_analysis": predictive_analysis,
            "analysis_timestamp": current_analysis.get("generated_at")
        }

    except Exception as e:
        logger.error(f"Error analyzing memory pressure: {e}")
        return {
            "error": f"Failed to analyze memory pressure: {str(e)}"
        }


def get_memory_optimization_opportunities_tool() -> Dict[str, Any]:
    """
    Identify memory optimization opportunities based on usage patterns.

    Returns:
        Dictionary containing memory optimization opportunities including:
        - Pattern-based optimization opportunities
        - Load-aware optimization strategies
        - Memory efficiency improvements
        - Implementation prioritization
    """
    try:
        monitor = get_memory_pattern_monitor()
        optimizer = get_load_aware_optimizer()

        # Get pattern-based opportunities
        analysis = monitor.get_memory_analysis()
        if "error" in analysis:
            return analysis

        pattern_opportunities = analysis.get("optimization_opportunities", [])

        # Get load-aware opportunities
        load_opportunities = []
        for load_level in LoadLevel:
            load_opt = optimizer.optimize_for_load(load_level)
            if "optimization_plan" in load_opt:
                load_opportunities.append({
                    "load_level": load_level.value,
                    "optimization_plan": load_opt["optimization_plan"],
                    "priority": _calculate_load_optimization_priority(load_level, load_opt)
                })

        # Combine and prioritize opportunities
        all_opportunities = pattern_opportunities + load_opportunities
        prioritized_opportunities = _prioritize_memory_opportunities(all_opportunities)

        # Generate implementation roadmap
        implementation_roadmap = _generate_memory_optimization_roadmap(prioritized_opportunities)

        return {
            "total_opportunities": len(all_opportunities),
            "pattern_based_opportunities": pattern_opportunities,
            "load_aware_opportunities": load_opportunities,
            "prioritized_opportunities": prioritized_opportunities[:5],  # Top 5
            "implementation_roadmap": implementation_roadmap,
            "generated_at": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting memory optimization opportunities: {e}")
        return {
            "error": f"Failed to get memory optimization opportunities: {str(e)}"
        }


def _generate_memory_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from memory analysis."""
    insights = []

    memory_stats = analysis.get("memory_statistics", {})
    avg_memory = memory_stats.get("average_mb", 0)
    peak_memory = memory_stats.get("peak_mb", 0)
    volatility = memory_stats.get("volatility_mb", 0)

    if avg_memory > 1000:
        insights.append("High average memory usage detected - consider memory optimization")
    elif avg_memory < 200:
        insights.append("Low memory usage - system has significant memory headroom")

    if volatility > 200:
        insights.append("High memory volatility - indicates inconsistent memory usage patterns")
    elif volatility < 50:
        insights.append("Stable memory usage - predictable memory consumption patterns")

    load_distribution = analysis.get("load_distribution", {})
    if load_distribution:
        highest_load = max(load_distribution.values(), key=lambda x: x["avg_memory_mb"])
        insights.append(f"Highest memory usage under {highest_load} load conditions")

    patterns = analysis.get("patterns", [])
    if patterns:
        top_pattern = patterns[0]
        insights.append(f"Most significant pattern: {top_pattern['load_level']} load with "
                       f"{top_pattern['optimization_potential']:.1f} optimization potential")

    return insights


def _calculate_memory_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate a memory health score (0-100)."""
    score = 100.0

    memory_stats = analysis.get("memory_statistics", {})
    avg_memory = memory_stats.get("average_mb", 0)
    volatility = memory_stats.get("volatility_mb", 0)

    # Memory usage score (lower usage is better, up to a point)
    if avg_memory > 1500:
        score -= 30
    elif avg_memory > 1000:
        score -= 15
    elif avg_memory < 100:
        score -= 10  # Too low usage might indicate inefficiency

    # Volatility penalty (lower volatility is better)
    volatility_penalty = min(volatility / 10, 25)
    score -= volatility_penalty

    # Pattern diversity bonus
    patterns = analysis.get("patterns", [])
    if len(patterns) > 3:
        score += 10

    return max(0.0, min(100.0, score))


def _generate_memory_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate memory optimization recommendations."""
    recommendations = []

    memory_stats = analysis.get("memory_statistics", {})
    avg_memory = memory_stats.get("average_mb", 0)
    volatility = memory_stats.get("volatility_mb", 0)

    if avg_memory > 1200:
        recommendations.extend([
            "Consider reducing cache sizes to lower memory footprint",
            "Profile application to identify memory-intensive operations",
            "Implement memory-efficient data structures where possible"
        ])

    if volatility > 150:
        recommendations.extend([
            "Stabilize memory usage patterns",
            "Implement memory usage limits and monitoring",
            "Review garbage collection settings"
        ])

    patterns = analysis.get("patterns", [])
    high_potential_patterns = [p for p in patterns if p["optimization_potential"] > 0.7]
    if high_potential_patterns:
        recommendations.append("High optimization potential patterns detected - implement load-aware memory management")

    return recommendations


def _determine_memory_pressure_level(current_memory_mb: float) -> str:
    """Determine memory pressure level."""
    if current_memory_mb > 1500:
        return "critical"
    elif current_memory_mb > 1200:
        return "high"
    elif current_memory_mb > 800:
        return "moderate"
    elif current_memory_mb > 400:
        return "low"
    else:
        return "minimal"


def _analyze_memory_pressure_sources(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze sources of memory pressure."""
    sources = []

    load_distribution = analysis.get("load_distribution", {})
    for load_level, stats in load_distribution.items():
        if stats["avg_memory_mb"] > 1000:
            sources.append({
                "source": f"high_load_{load_level}",
                "description": f"High memory usage under {load_level} load conditions",
                "memory_contribution_mb": stats["avg_memory_mb"],
                "percentage": stats["percentage"]
            })

    patterns = analysis.get("patterns", [])
    for pattern in patterns:
        if pattern["optimization_potential"] > 0.8:
            sources.append({
                "source": f"pattern_{pattern['pattern_id']}",
                "description": f"Inefficient memory pattern under {pattern['load_level']} load",
                "memory_contribution_mb": pattern["avg_memory_mb"],
                "optimization_potential": pattern["optimization_potential"]
            })

    return sources


def _generate_pressure_mitigation_strategies(pressure_level: str,
                                           pressure_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate memory pressure mitigation strategies."""
    strategies = []

    if pressure_level in ["critical", "high"]:
        strategies.extend([
            {
                "strategy": "immediate_memory_reduction",
                "description": "Implement immediate memory reduction measures",
                "actions": [
                    "Reduce cache sizes by 20-30%",
                    "Trigger garbage collection",
                    "Restart memory-intensive processes if needed"
                ],
                "priority": "high",
                "timeline": "Immediate"
            },
            {
                "strategy": "cache_optimization",
                "description": "Optimize cache configurations",
                "actions": [
                    "Implement intelligent cache eviction policies",
                    "Set appropriate cache size limits",
                    "Enable cache compression if supported"
                ],
                "priority": "high",
                "timeline": "Within 1 hour"
            }
        ])

    if pressure_sources:
        strategies.append({
            "strategy": "source_specific_optimization",
            "description": "Address specific memory pressure sources",
            "actions": [f"Optimize {source['source']} memory usage" for source in pressure_sources[:3]],
            "priority": "medium",
            "timeline": "Within 4 hours"
        })

    return strategies


def _analyze_predictive_memory_pressure(monitor) -> Dict[str, Any]:
    """Analyze predictive memory pressure trends."""
    # Get trend analysis
    analysis = monitor.get_memory_analysis(6)  # Last 6 hours
    trends = analysis.get("trends", {})

    if trends.get("trend") == "increasing":
        slope = trends.get("trend_slope_mb_per_snapshot", 0)
        projected_increase = slope * 60  # 60 snapshots = ~1 hour

        return {
            "pressure_trend": "increasing",
            "projected_increase_mb": projected_increase,
            "time_to_critical_hours": 24 if projected_increase < 500 else 12 if projected_increase < 1000 else 6,
            "recommendation": "Monitor closely and prepare memory optimization measures"
        }
    else:
        return {
            "pressure_trend": "stable_or_decreasing",
            "status": "Memory pressure is under control",
            "recommendation": "Continue normal monitoring"
        }


def _analyze_load_characteristics(load_level: LoadLevel) -> Dict[str, Any]:
    """Analyze characteristics of a specific load level."""
    characteristics = {
        LoadLevel.IDLE: {
            "description": "Minimal system activity",
            "typical_memory_range_mb": "200-400",
            "optimization_focus": "Memory efficiency and cleanup",
            "cache_strategy": "Minimal caching, focus on memory reclamation"
        },
        LoadLevel.LIGHT: {
            "description": "Light system activity with occasional requests",
            "typical_memory_range_mb": "400-600",
            "optimization_focus": "Balanced performance and memory usage",
            "cache_strategy": "Moderate caching with efficient eviction"
        },
        LoadLevel.MODERATE: {
            "description": "Moderate system activity with regular requests",
            "typical_memory_range_mb": "600-900",
            "optimization_focus": "Performance optimization with memory awareness",
            "cache_strategy": "Full caching with intelligent management"
        },
        LoadLevel.HEAVY: {
            "description": "High system activity with frequent requests",
            "typical_memory_range_mb": "900-1300",
            "optimization_focus": "Performance under load with memory constraints",
            "cache_strategy": "Aggressive caching with memory limits"
        },
        LoadLevel.CRITICAL: {
            "description": "Maximum system activity, potential overload conditions",
            "typical_memory_range_mb": "1300+",
            "optimization_focus": "System stability and performance degradation prevention",
            "cache_strategy": "Minimal caching, prioritize essential operations"
        }
    }

    return characteristics.get(load_level, {})


def _assess_implementation_readiness(optimization: Dict[str, Any]) -> Dict[str, Any]:
    """Assess readiness for implementing optimization."""
    # Simple readiness assessment
    readiness_factors = {
        "data_availability": "good" if optimization.get("based_on_pattern") else "limited",
        "pattern_stability": "good" if optimization.get("optimization_plan") else "unknown",
        "risk_assessment": "low" if optimization.get("expected_benefits") else "medium",
        "monitoring_readiness": "good",  # Assume monitoring is in place
        "rollback_plan": "available"  # Assume rollback is possible
    }

    overall_readiness = "ready" if all(factor == "good" for factor in readiness_factors.values()) else "caution"

    return {
        "overall_readiness": overall_readiness,
        "readiness_factors": readiness_factors,
        "recommendations": [
            "Ensure monitoring is in place before implementation",
            "Have rollback plan ready",
            "Test optimization in staging environment first"
        ] if overall_readiness == "caution" else []
    }


def _calculate_load_optimization_priority(load_level: LoadLevel, optimization: Dict[str, Any]) -> str:
    """Calculate priority for load-specific optimization."""
    base_priority = {
        LoadLevel.CRITICAL: "high",
        LoadLevel.HEAVY: "high",
        LoadLevel.MODERATE: "medium",
        LoadLevel.LIGHT: "medium",
        LoadLevel.IDLE: "low"
    }.get(load_level, "medium")

    # Adjust based on optimization potential
    settings = optimization.get("optimal_settings", {})
    if settings.get("memory_pressure_adjustment", 1.0) < 0.9:
        return "high"

    return base_priority


def _prioritize_memory_opportunities(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize memory optimization opportunities."""
    def sort_key(opp):
        # Priority based on expected savings and implementation effort
        expected_savings = opp.get("expected_savings_mb", 0)
        priority_score = {
            "high": 3,
            "medium": 2,
            "low": 1
        }.get(opp.get("priority", "medium"), 2)

        return (priority_score, expected_savings)

    return sorted(opportunities, key=sort_key, reverse=True)


def _generate_memory_optimization_roadmap(opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate implementation roadmap for memory optimizations."""
    roadmap = {
        "immediate": [],  # Within 1 hour
        "short_term": [],  # Within 4 hours
        "medium_term": [],  # Within 24 hours
        "long_term": []  # Within 1 week
    }

    for opp in opportunities:
        priority = opp.get("priority", "medium")

        if priority == "high":
            roadmap["immediate"].append(opp)
        elif priority == "medium":
            roadmap["short_term"].append(opp)
        else:
            roadmap["medium_term"].append(opp)

    return {
        "roadmap": roadmap,
        "immediate_count": len(roadmap["immediate"]),
        "short_term_count": len(roadmap["short_term"]),
        "medium_term_count": len(roadmap["medium_term"]),
        "long_term_count": len(roadmap["long_term"]),
        "total_scheduled": sum(len(phase) for phase in roadmap.values())
    }