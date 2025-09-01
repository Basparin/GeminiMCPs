"""
Cache Analysis Tools for CodeSage MCP Server.

This module provides tools for analyzing cache effectiveness in real-world scenarios,
including performance metrics, optimization recommendations, and cache tuning guidance.
"""

import logging
from typing import Dict, Any, List
from codesage_mcp.features.caching.cache_analysis import (
    get_cache_effectiveness_analyzer,
    get_cache_optimization_recommender
)

logger = logging.getLogger(__name__)


def analyze_cache_effectiveness_tool(cache_type: str = None) -> Dict[str, Any]:
    """
    Analyze cache effectiveness in real-world scenarios.

    Args:
        cache_type: Optional specific cache type to analyze (e.g., 'embedding', 'search', 'file')
                   If not provided, analyzes all cache types.

    Returns:
        Dictionary containing comprehensive cache effectiveness analysis including:
        - Performance metrics (hit rates, latency, memory usage)
        - Effectiveness ratings and trends
        - Access pattern analysis
        - Optimization recommendations
    """
    try:
        analyzer = get_cache_effectiveness_analyzer()
        analysis = analyzer.analyze_cache_effectiveness(cache_type)

        if cache_type:
            # Single cache analysis
            if "error" in analysis:
                return analysis

            # Enhance with additional insights
            analysis["insights"] = _generate_cache_insights(analysis)
            analysis["health_score"] = _calculate_cache_health_score(analysis)

            return analysis
        else:
            # Multi-cache analysis
            if "cross_cache_analysis" in analysis:
                cross_analysis = analysis["cross_cache_analysis"]
                cross_analysis["insights"] = _generate_cross_cache_insights(analysis)
                cross_analysis["optimization_priorities"] = _prioritize_cache_optimizations(analysis)

            return analysis

    except Exception as e:
        logger.error(f"Error analyzing cache effectiveness: {e}")
        return {
            "error": f"Failed to analyze cache effectiveness: {str(e)}"
        }


def get_cache_optimization_recommendations_tool() -> Dict[str, Any]:
    """
    Get cache optimization recommendations based on effectiveness analysis.

    Returns:
        Dictionary containing prioritized cache optimization opportunities with:
        - Detailed analysis of each opportunity
        - Implementation steps and timelines
        - Expected benefits and risks
        - Effort estimates and priorities
    """
    try:
        recommender = get_cache_optimization_recommender()
        opportunities = recommender.generate_optimization_opportunities()

        # Convert opportunities to dictionaries for JSON serialization
        opportunities_data = []
        for opp in opportunities:
            opp_dict = {
                "opportunity_id": opp.opportunity_id,
                "title": opp.title,
                "description": opp.description,
                "cache_type": opp.cache_type,
                "current_impact": opp.current_impact,
                "potential_improvement": opp.potential_improvement,
                "priority": opp.priority,
                "effort_estimate": opp.effort_estimate,
                "expected_benefits": opp.expected_benefits,
                "implementation_steps": opp.implementation_steps,
                "risks": opp.risks,
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

        # Calculate implementation roadmap
        roadmap = _generate_cache_optimization_roadmap(opportunities_data)

        # Calculate expected ROI
        roi_analysis = _calculate_cache_optimization_roi(opportunities_data)

        return {
            "total_opportunities": len(opportunities_data),
            "opportunities_by_priority": priority_groups,
            "top_recommendations": opportunities_data[:5],  # Top 5 opportunities
            "implementation_roadmap": roadmap,
            "roi_analysis": roi_analysis,
            "generated_at": opportunities[0].discovered_at if opportunities else None
        }

    except Exception as e:
        logger.error(f"Error getting cache optimization recommendations: {e}")
        return {
            "error": f"Failed to get cache optimization recommendations: {str(e)}"
        }


def get_cache_performance_metrics_tool(cache_type: str = None) -> Dict[str, Any]:
    """
    Get detailed cache performance metrics for monitoring and analysis.

    Args:
        cache_type: Optional specific cache type to analyze

    Returns:
        Dictionary containing detailed performance metrics including:
        - Hit/miss rates over time
        - Memory usage patterns
        - Latency distributions
        - Access frequency analysis
        - Performance trends
    """
    try:
        analyzer = get_cache_effectiveness_analyzer()

        if cache_type:
            # Get specific cache metrics
            analysis = analyzer.analyze_cache_effectiveness(cache_type)
            if "error" in analysis:
                return analysis

            performance_metrics = analysis.get("performance_metrics", {})
            memory_metrics = analysis.get("memory_metrics", {})
            trend_analysis = analysis.get("trend_analysis", {})

            # Get historical data
            historical_data = list(analyzer.performance_history[cache_type])
            historical_summary = _summarize_historical_performance(historical_data)

            return {
                "cache_type": cache_type,
                "current_metrics": performance_metrics,
                "memory_metrics": memory_metrics,
                "trend_analysis": trend_analysis,
                "historical_summary": historical_summary,
                "performance_score": _calculate_performance_score(performance_metrics),
                "generated_at": time.time()
            }
        else:
            # Get all cache metrics
            all_analysis = analyzer.analyze_cache_effectiveness()
            all_metrics = {}

            for cache_type_key, analysis in all_analysis.items():
                if cache_type_key != "cross_cache_analysis" and isinstance(analysis, dict):
                    if "error" not in analysis:
                        all_metrics[cache_type_key] = {
                            "performance_metrics": analysis.get("performance_metrics", {}),
                            "memory_metrics": analysis.get("memory_metrics", {}),
                            "trend_analysis": analysis.get("trend_analysis", {}),
                            "effectiveness_rating": analysis.get("effectiveness_rating"),
                            "performance_score": _calculate_performance_score(analysis.get("performance_metrics", {}))
                        }

            return {
                "all_cache_metrics": all_metrics,
                "cross_cache_analysis": all_analysis.get("cross_cache_analysis", {}),
                "generated_at": time.time()
            }

    except Exception as e:
        logger.error(f"Error getting cache performance metrics: {e}")
        return {
            "error": f"Failed to get cache performance metrics: {str(e)}"
        }


def get_cache_access_patterns_tool(cache_type: str = None) -> Dict[str, Any]:
    """
    Analyze cache access patterns to identify optimization opportunities.

    Args:
        cache_type: Optional specific cache type to analyze

    Returns:
        Dictionary containing cache access pattern analysis including:
        - Hit/miss patterns by operation type
        - Temporal access patterns (hourly/daily)
        - Key frequency analysis
        - Pattern-based optimization recommendations
    """
    try:
        analyzer = get_cache_effectiveness_analyzer()

        if cache_type:
            # Get specific cache patterns
            analysis = analyzer.analyze_cache_effectiveness(cache_type)
            if "error" in analysis:
                return analysis

            access_patterns = analysis.get("access_patterns", {})

            # Enhance with pattern insights
            pattern_insights = _analyze_access_patterns(access_patterns)

            return {
                "cache_type": cache_type,
                "access_patterns": access_patterns,
                "pattern_insights": pattern_insights,
                "optimization_opportunities": _identify_pattern_based_optimizations(access_patterns),
                "generated_at": time.time()
            }
        else:
            # Get all cache patterns
            all_patterns = {}
            all_analysis = analyzer.analyze_cache_effectiveness()

            for cache_type_key, analysis in all_analysis.items():
                if cache_type_key != "cross_cache_analysis" and isinstance(analysis, dict):
                    if "error" not in analysis:
                        access_patterns = analysis.get("access_patterns", {})
                        all_patterns[cache_type_key] = {
                            "access_patterns": access_patterns,
                            "pattern_insights": _analyze_access_patterns(access_patterns),
                            "optimization_opportunities": _identify_pattern_based_optimizations(access_patterns)
                        }

            return {
                "all_cache_patterns": all_patterns,
                "cross_pattern_analysis": _analyze_cross_cache_patterns(all_patterns),
                "generated_at": time.time()
            }

    except Exception as e:
        logger.error(f"Error getting cache access patterns: {e}")
        return {
            "error": f"Failed to get cache access patterns: {str(e)}"
        }


def get_cache_memory_efficiency_tool() -> Dict[str, Any]:
    """
    Analyze cache memory efficiency and provide optimization recommendations.

    Returns:
        Dictionary containing memory efficiency analysis including:
        - Memory usage by cache type
        - Efficiency scores and benchmarks
        - Memory optimization opportunities
        - Reallocation recommendations
    """
    try:
        analyzer = get_cache_effectiveness_analyzer()
        analysis = analyzer.analyze_cache_effectiveness()

        # Extract memory metrics
        memory_analysis = {}
        total_memory = 0
        total_efficiency_weighted = 0

        for cache_type, cache_analysis in analysis.items():
            if cache_type != "cross_cache_analysis" and isinstance(cache_analysis, dict):
                if "error" not in cache_analysis:
                    memory_metrics = cache_analysis.get("memory_metrics", {})
                    performance_metrics = cache_analysis.get("performance_metrics", {})

                    memory_usage = memory_metrics.get("memory_usage_mb", 0)
                    hit_rate = performance_metrics.get("hit_rate", 0)
                    efficiency_score = memory_metrics.get("memory_efficiency_score", 0)

                    memory_analysis[cache_type] = {
                        "memory_usage_mb": memory_usage,
                        "hit_rate": hit_rate,
                        "efficiency_score": efficiency_score,
                        "efficiency_rating": _rate_memory_efficiency(efficiency_score),
                        "optimization_potential": _calculate_memory_optimization_potential(memory_usage, efficiency_score)
                    }

                    total_memory += memory_usage
                    total_efficiency_weighted += efficiency_score * memory_usage

        # Calculate overall efficiency
        overall_efficiency = total_efficiency_weighted / total_memory if total_memory > 0 else 0

        # Generate memory optimization recommendations
        memory_recommendations = _generate_memory_optimization_recommendations(memory_analysis)

        return {
            "memory_analysis": memory_analysis,
            "total_memory_usage_mb": total_memory,
            "overall_efficiency_score": overall_efficiency,
            "overall_efficiency_rating": _rate_memory_efficiency(overall_efficiency),
            "memory_recommendations": memory_recommendations,
            "memory_distribution": _analyze_memory_distribution(memory_analysis),
            "generated_at": time.time()
        }

    except Exception as e:
        logger.exception(f"Error analyzing cache memory efficiency: {e}")
        return {
            "error": f"Failed to analyze cache memory efficiency: {str(e)}"
        }


def _generate_cache_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from cache analysis."""
    insights = []

    effectiveness = analysis.get("effectiveness_rating", "fair")
    hit_rate = analysis.get("performance_metrics", {}).get("hit_rate", 0)
    trend = analysis.get("trend_analysis", {}).get("trend", "stable")

    if effectiveness == "excellent":
        insights.append("Cache performance is excellent with high hit rates and efficient memory usage")
    elif effectiveness == "good":
        insights.append("Cache performance is good but has room for improvement")
    elif effectiveness == "poor":
        insights.append("Cache performance is poor and requires immediate attention")
    elif effectiveness == "critical":
        insights.append("Cache performance is critical and severely impacting system performance")

    if hit_rate > 0.9:
        insights.append("Exceptionally high hit rate indicates excellent cache configuration")
    elif hit_rate < 0.5:
        insights.append("Low hit rate suggests cache size or strategy needs optimization")

    if trend == "improving":
        insights.append("Cache performance is trending upward")
    elif trend == "degrading":
        insights.append("Cache performance is declining and needs investigation")

    return insights


def _calculate_cache_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate a health score for the cache (0-100)."""
    score = 100.0

    # Hit rate component (40% weight)
    hit_rate = analysis.get("performance_metrics", {}).get("hit_rate", 0)
    hit_rate_score = hit_rate * 100
    score -= (100 - hit_rate_score) * 0.4

    # Memory efficiency component (30% weight)
    memory_efficiency = analysis.get("memory_metrics", {}).get("memory_efficiency_score", 0)
    memory_score = memory_efficiency * 100
    score -= (100 - memory_score) * 0.3

    # Trend component (20% weight)
    trend = analysis.get("trend_analysis", {}).get("trend", "stable")
    trend_score = 100 if trend == "improving" else 50 if trend == "stable" else 25
    score -= (100 - trend_score) * 0.2

    # Volatility penalty (10% weight)
    volatility = analysis.get("trend_analysis", {}).get("volatility", 0)
    volatility_penalty = min(volatility * 200, 50)  # Max 50 point penalty
    score -= volatility_penalty * 0.1

    return max(0.0, min(100.0, score))


def _generate_cross_cache_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate insights from cross-cache analysis."""
    insights = []

    overall_rating = analysis.get("overall_rating", "fair")
    best_cache = analysis.get("best_performing_cache", "unknown")
    worst_cache = analysis.get("worst_performing_cache", "unknown")
    avg_hit_rate = analysis.get("average_hit_rate", 0)

    insights.append(f"Overall cache system rating: {overall_rating}")

    if best_cache != "unknown":
        insights.append(f"Best performing cache: {best_cache}")
    if worst_cache != "unknown":
        insights.append(f"Cache needing attention: {worst_cache}")

    if avg_hit_rate > 0.8:
        insights.append("Excellent overall cache hit rate across all cache types")
    elif avg_hit_rate < 0.6:
        insights.append("Cache hit rates need improvement across multiple cache types")

    return insights


def _prioritize_cache_optimizations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prioritize cache optimizations based on analysis."""
    priorities = []

    for cache_type, cache_analysis in analysis.items():
        if cache_type == "cross_cache_analysis" or not isinstance(cache_analysis, dict):
            continue

        if "error" in cache_analysis:
            continue

        effectiveness = cache_analysis.get("effectiveness_rating", "fair")
        hit_rate = cache_analysis.get("performance_metrics", {}).get("hit_rate", 0)

        priority_score = 0
        reasons = []

        if effectiveness in ["poor", "critical"]:
            priority_score += 3
            reasons.append("Poor effectiveness rating")

        if hit_rate < 0.7:
            priority_score += 2
            reasons.append("Low hit rate")

        if priority_score > 0:
            priorities.append({
                "cache_type": cache_type,
                "priority_score": priority_score,
                "reasons": reasons,
                "effectiveness": effectiveness,
                "hit_rate": hit_rate
            })

    return sorted(priorities, key=lambda x: x["priority_score"], reverse=True)


def _generate_cache_optimization_roadmap(opportunities: List[Dict]) -> Dict[str, Any]:
    """Generate implementation roadmap for cache optimizations."""
    roadmap = {
        "immediate": [],  # 1-2 weeks
        "short_term": [],  # 1-2 months
        "medium_term": [],  # 3-6 months
        "long_term": []  # 6+ months
    }

    for opp in opportunities:
        timeline = opp["effort_estimate"]

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


def _calculate_cache_optimization_roi(opportunities: List[Dict]) -> Dict[str, Any]:
    """Calculate expected ROI for cache optimizations."""
    total_effort_weeks = sum(_estimate_effort_weeks(opp["effort_estimate"]) for opp in opportunities)
    total_benefit_score = sum(opp["potential_improvement"] for opp in opportunities)

    # Assume each improvement point translates to performance benefit
    performance_benefit_multiplier = 5  # 5x performance improvement value
    expected_roi = (total_benefit_score * performance_benefit_multiplier) / max(total_effort_weeks, 1)

    return {
        "total_opportunities": len(opportunities),
        "estimated_effort_weeks": total_effort_weeks,
        "expected_benefit_score": total_benefit_score,
        "expected_roi_multiplier": expected_roi,
        "roi_interpretation": "Excellent" if expected_roi > 10 else "Good" if expected_roi > 5 else "Moderate" if expected_roi > 2 else "Low",
        "payback_period_weeks": total_effort_weeks / max(expected_roi, 0.1)
    }


def _estimate_effort_weeks(effort_estimate: str) -> float:
    """Convert effort estimate to weeks."""
    if "week" in effort_estimate:
        return float(effort_estimate.split("-")[0]) if "-" in effort_estimate else float(effort_estimate.split()[0])
    elif "month" in effort_estimate:
        months = float(effort_estimate.split("-")[0]) if "-" in effort_estimate else float(effort_estimate.split()[0])
        return months * 4  # Assume 4 weeks per month
    else:
        return 2  # Default estimate


def _summarize_historical_performance(historical_data: List) -> Dict[str, Any]:
    """Summarize historical performance data."""
    if not historical_data:
        return {"summary": "No historical data available"}

    hit_rates = [d.hit_rate for d in historical_data if hasattr(d, 'hit_rate')]
    latencies = [d.avg_hit_latency_ms for d in historical_data if hasattr(d, 'avg_hit_latency_ms') and d.avg_hit_latency_ms > 0]

    return {
        "data_points": len(historical_data),
        "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
        "min_hit_rate": min(hit_rates) if hit_rates else 0,
        "max_hit_rate": max(hit_rates) if hit_rates else 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "time_span_hours": (historical_data[-1].timestamp - historical_data[0].timestamp) / 3600 if len(historical_data) > 1 else 0
    }


def _calculate_performance_score(metrics: Dict[str, Any]) -> float:
    """Calculate a performance score for cache metrics."""
    hit_rate = metrics.get("hit_rate", 0)
    latency = metrics.get("avg_hit_latency_ms", 50)

    # Weighted score: 70% hit rate, 30% latency
    hit_rate_score = hit_rate * 100
    latency_score = max(0, (100 - latency) / 100 * 100)  # Lower latency is better

    return (hit_rate_score * 0.7) + (latency_score * 0.3)


def _analyze_access_patterns(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cache access patterns."""
    insights = {}

    # Hit/miss pattern analysis
    hit_patterns = patterns.get("hit_patterns", {})
    miss_patterns = patterns.get("miss_patterns", {})

    if hit_patterns:
        most_common_hit = max(hit_patterns.items(), key=lambda x: x[1])
        insights["most_common_hit_operation"] = most_common_hit[0]
        insights["hit_operation_frequency"] = most_common_hit[1]

    if miss_patterns:
        most_common_miss = max(miss_patterns.items(), key=lambda x: x[1])
        insights["most_common_miss_operation"] = most_common_miss[0]
        insights["miss_operation_frequency"] = most_common_miss[1]

    # Temporal pattern analysis
    temporal_patterns = patterns.get("temporal_patterns", {})
    if temporal_patterns:
        hourly_performance = {}
        for hour, hits in temporal_patterns.items():
            if hits:
                hit_rate = sum(hits) / len(hits)
                hourly_performance[hour] = hit_rate

        if hourly_performance:
            best_hour = max(hourly_performance.items(), key=lambda x: x[1])
            worst_hour = min(hourly_performance.items(), key=lambda x: x[1])
            insights["best_performance_hour"] = best_hour[0]
            insights["worst_performance_hour"] = worst_hour[0]

    return insights


def _identify_pattern_based_optimizations(patterns: Dict[str, Any]) -> List[str]:
    """Identify optimization opportunities based on access patterns."""
    optimizations = []

    # Analyze hit/miss ratios
    hit_patterns = patterns.get("hit_patterns", {})
    miss_patterns = patterns.get("miss_patterns", {})

    total_hits = sum(hit_patterns.values())
    total_misses = sum(miss_patterns.values())

    if total_hits + total_misses > 0:
        hit_rate = total_hits / (total_hits + total_misses)
        if hit_rate < 0.7:
            optimizations.append("Consider increasing cache size or adjusting eviction policy")

    # Analyze temporal patterns
    temporal_patterns = patterns.get("temporal_patterns", {})
    if temporal_patterns:
        # Look for patterns that suggest prefetching opportunities
        high_activity_hours = [hour for hour, hits in temporal_patterns.items()
                             if hits and sum(hits) / len(hits) > 0.8]
        if len(high_activity_hours) > 0:
            optimizations.append(f"Consider prefetching during high-activity hours: {high_activity_hours}")

    return optimizations


def _analyze_cross_cache_patterns(all_patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns across all cache types."""
    analysis = {}

    # Compare access patterns across cache types
    cache_types = list(all_patterns.keys())
    if len(cache_types) > 1:
        # Find most and least active caches
        activity_levels = {}
        for cache_type, patterns in all_patterns.items():
            hit_count = sum(patterns.get("access_patterns", {}).get("hit_patterns", {}).values())
            miss_count = sum(patterns.get("access_patterns", {}).get("miss_patterns", {}).values())
            activity_levels[cache_type] = hit_count + miss_count

        if activity_levels:
            most_active = max(activity_levels.items(), key=lambda x: x[1])
            least_active = min(activity_levels.items(), key=lambda x: x[1])
            analysis["most_active_cache"] = most_active[0]
            analysis["least_active_cache"] = least_active[0]

    return analysis


def _rate_memory_efficiency(efficiency_score: float) -> str:
    """Rate memory efficiency."""
    if efficiency_score >= 0.8:
        return "Excellent"
    elif efficiency_score >= 0.6:
        return "Good"
    elif efficiency_score >= 0.4:
        return "Fair"
    elif efficiency_score >= 0.2:
        return "Poor"
    else:
        return "Critical"


def _calculate_memory_optimization_potential(memory_usage: float, efficiency_score: float) -> float:
    """Calculate memory optimization potential."""
    if efficiency_score < 0.5 and memory_usage > 50:
        return min(memory_usage * 0.3, 100)  # Up to 30% reduction potential
    return 0


def _generate_memory_optimization_recommendations(memory_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate memory optimization recommendations."""
    recommendations = []

    # Identify inefficient caches
    inefficient_caches = [
        cache_type for cache_type, analysis in memory_analysis.items()
        if analysis["efficiency_score"] < 0.5 and analysis["memory_usage_mb"] > 50
    ]

    if inefficient_caches:
        recommendations.append({
            "type": "memory_reduction",
            "priority": "high",
            "description": f"Reduce memory allocation for inefficient caches: {', '.join(inefficient_caches)}",
            "expected_savings_mb": sum(memory_analysis[c]["optimization_potential"] for c in inefficient_caches),
            "implementation_effort": "Medium"
        })

    # Identify high-efficiency caches that could benefit from more memory
    efficient_caches = [
        cache_type for cache_type, analysis in memory_analysis.items()
        if analysis["efficiency_score"] > 0.8
    ]

    if efficient_caches:
        recommendations.append({
            "type": "memory_increase",
            "priority": "medium",
            "description": f"Consider increasing memory for high-efficiency caches: {', '.join(efficient_caches)}",
            "expected_benefit": "Higher hit rates and better performance",
            "implementation_effort": "Low"
        })

    return recommendations


def _analyze_memory_distribution(memory_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze memory distribution across caches."""
    total_memory = sum(analysis["memory_usage_mb"] for analysis in memory_analysis.values())
    distribution = {}

    for cache_type, analysis in memory_analysis.items():
        percentage = (analysis["memory_usage_mb"] / total_memory * 100) if total_memory > 0 else 0
        distribution[cache_type] = {
            "memory_mb": analysis["memory_usage_mb"],
            "percentage": percentage,
            "efficiency_score": analysis["efficiency_score"]
        }

    return {
        "total_memory_mb": total_memory,
        "distribution": distribution,
        "most_memory_intensive": max(distribution.items(), key=lambda x: x[1]["memory_mb"]) if distribution else None,
        "least_memory_intensive": min(distribution.items(), key=lambda x: x[1]["memory_mb"]) if distribution else None
    }