"""
Cache Effectiveness Analysis Module for CodeSage MCP Server.

This module provides comprehensive analysis of cache effectiveness in real-world scenarios,
including cache hit rates, memory efficiency, access patterns, and optimization recommendations.

Classes:
    CacheEffectivenessAnalyzer: Analyzes cache performance and effectiveness
    CacheOptimizationRecommender: Recommends cache optimizations
    MemoryEfficiencyTracker: Tracks memory usage efficiency
    CacheAccessPatternAnalyzer: Analyzes cache access patterns
"""

import logging
import time
import statistics
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class CacheEffectiveness(Enum):
    """Cache effectiveness levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class CachePerformanceMetrics:
    """Cache performance metrics snapshot."""
    timestamp: float
    hit_rate: float
    miss_rate: float
    hit_count: int
    miss_count: int
    total_requests: int
    cache_size: int
    memory_usage_mb: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
    invalidation_count: int
    prefetch_hit_rate: float


@dataclass
class CacheOptimizationOpportunity:
    """Represents a cache optimization opportunity."""
    opportunity_id: str
    title: str
    description: str
    cache_type: str
    current_impact: float
    potential_improvement: float
    priority: str
    effort_estimate: str
    expected_benefits: List[str]
    implementation_steps: List[str]
    risks: List[str]
    discovered_at: float = field(default_factory=time.time)


class CacheEffectivenessAnalyzer:
    """Analyzes cache performance and effectiveness in real-world scenarios."""

    def __init__(self, analysis_window_hours: int = 24):
        self.analysis_window_hours = analysis_window_hours
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.access_patterns: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def record_cache_operation(self, cache_type: str, operation: str,
                              hit: bool, latency_ms: float,
                              key_info: Optional[Dict[str, Any]] = None) -> None:
        """Record a cache operation for analysis."""
        with self._lock:
            timestamp = time.time()

            # Record performance metrics
            metrics = CachePerformanceMetrics(
                timestamp=timestamp,
                hit_rate=0.0,  # Will be calculated
                miss_rate=0.0,  # Will be calculated
                hit_count=1 if hit else 0,
                miss_count=0 if hit else 1,
                total_requests=1,
                cache_size=0,  # Will be updated
                memory_usage_mb=0.0,  # Will be updated
                avg_hit_latency_ms=latency_ms if hit else 0.0,
                avg_miss_latency_ms=0.0 if hit else latency_ms,
                invalidation_count=0,
                prefetch_hit_rate=0.0
            )

            self.performance_history[cache_type].append(metrics)

            # Update access pattern analysis
            self._update_access_patterns(cache_type, operation, hit, key_info)

            # Clean old data
            cutoff_time = timestamp - (self.analysis_window_hours * 60 * 60)
            while self.performance_history[cache_type] and self.performance_history[cache_type][0].timestamp < cutoff_time:
                self.performance_history[cache_type].popleft()

    def record_cache_invalidation(self, cache_type: str, invalidation_count: int) -> None:
        """Record cache invalidation events."""
        with self._lock:
            if self.performance_history[cache_type]:
                # Update the latest metrics with invalidation count
                latest = self.performance_history[cache_type][-1]
                latest.invalidation_count += invalidation_count

    def record_cache_size_update(self, cache_type: str, size: int, memory_mb: float) -> None:
        """Record cache size and memory usage updates."""
        with self._lock:
            if self.performance_history[cache_type]:
                # Update the latest metrics
                latest = self.performance_history[cache_type][-1]
                latest.cache_size = size
                latest.memory_usage_mb = memory_mb

    def analyze_cache_effectiveness(self, cache_type: str = None) -> Dict[str, Any]:
        """Analyze cache effectiveness for specified cache type or all caches."""
        with self._lock:
            if cache_type:
                return self._analyze_single_cache(cache_type)
            else:
                # Analyze all caches
                all_caches = {}
                cache_types = list(self.performance_history.keys())

                for ct in cache_types:
                    all_caches[ct] = self._analyze_single_cache(ct)

                # Generate cross-cache analysis
                all_caches["cross_cache_analysis"] = self._analyze_cross_cache_effectiveness(all_caches)

                return all_caches

    def _analyze_single_cache(self, cache_type: str) -> Dict[str, Any]:
        """Analyze effectiveness of a single cache type."""
        data = list(self.performance_history[cache_type])

        if not data:
            return {"error": f"No data available for cache type: {cache_type}"}

        # Calculate aggregate metrics
        total_hits = sum(d.hit_count for d in data)
        total_misses = sum(d.miss_count for d in data)
        total_requests = total_hits + total_misses

        if total_requests == 0:
            return {"error": f"No requests recorded for cache type: {cache_type}"}

        # Calculate hit rate and effectiveness
        hit_rate = total_hits / total_requests
        miss_rate = total_misses / total_requests

        # Calculate latency metrics
        hit_latencies = [d.avg_hit_latency_ms for d in data if d.avg_hit_latency_ms > 0]
        miss_latencies = [d.avg_miss_latency_ms for d in data if d.avg_miss_latency_ms > 0]

        avg_hit_latency = statistics.mean(hit_latencies) if hit_latencies else 0.0
        avg_miss_latency = statistics.mean(miss_latencies) if miss_latencies else 0.0

        # Calculate memory efficiency
        latest_data = data[-1] if data else None
        memory_efficiency = self._calculate_memory_efficiency(hit_rate, latest_data.memory_usage_mb if latest_data else 0)

        # Determine effectiveness rating
        effectiveness = self._determine_effectiveness(hit_rate, memory_efficiency, avg_hit_latency)

        # Calculate trend analysis
        trend_analysis = self._analyze_cache_trends(data)

        # Generate recommendations
        recommendations = self._generate_cache_recommendations(cache_type, hit_rate, effectiveness, trend_analysis)

        return {
            "cache_type": cache_type,
            "effectiveness_rating": effectiveness.value,
            "performance_metrics": {
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "avg_hit_latency_ms": avg_hit_latency,
                "avg_miss_latency_ms": avg_miss_latency,
                "latency_improvement": avg_miss_latency - avg_hit_latency if avg_miss_latency > 0 else 0
            },
            "memory_metrics": {
                "current_size": latest_data.cache_size if latest_data else 0,
                "memory_usage_mb": latest_data.memory_usage_mb if latest_data else 0,
                "memory_efficiency_score": memory_efficiency
            },
            "trend_analysis": trend_analysis,
            "recommendations": recommendations,
            "access_patterns": self.access_patterns.get(cache_type, {}),
            "analysis_timestamp": time.time()
        }

    def _calculate_memory_efficiency(self, hit_rate: float, memory_usage_mb: float) -> float:
        """Calculate memory efficiency score (0-1)."""
        if memory_usage_mb == 0:
            return 0.0

        # Efficiency is based on hit rate per MB of memory
        efficiency_per_mb = hit_rate / memory_usage_mb

        # Normalize to 0-1 scale (assuming 0.1 is baseline efficiency)
        normalized_efficiency = min(efficiency_per_mb / 0.1, 1.0)

        return normalized_efficiency

    def _determine_effectiveness(self, hit_rate: float, memory_efficiency: float,
                               avg_hit_latency: float) -> CacheEffectiveness:
        """Determine cache effectiveness rating."""
        # Weighted scoring
        hit_rate_score = hit_rate * 0.5
        memory_score = memory_efficiency * 0.3
        latency_score = max(0, (50 - avg_hit_latency) / 50) * 0.2  # Lower latency is better

        overall_score = hit_rate_score + memory_score + latency_score

        if overall_score >= 0.8:
            return CacheEffectiveness.EXCELLENT
        elif overall_score >= 0.6:
            return CacheEffectiveness.GOOD
        elif overall_score >= 0.4:
            return CacheEffectiveness.FAIR
        elif overall_score >= 0.2:
            return CacheEffectiveness.POOR
        else:
            return CacheEffectiveness.CRITICAL

    def _analyze_cache_trends(self, data: List[CachePerformanceMetrics]) -> Dict[str, Any]:
        """Analyze cache performance trends."""
        if len(data) < 2:
            return {"trend": "insufficient_data"}

        # Calculate hit rate trend
        hit_rates = [d.hit_rate for d in data]
        if len(hit_rates) >= 2:
            first_half = hit_rates[:len(hit_rates)//2]
            second_half = hit_rates[len(hit_rates)//2:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg + 0.05:
                trend = "improving"
            elif second_avg < first_avg - 0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Calculate volatility
        if len(hit_rates) > 1:
            volatility = statistics.stdev(hit_rates)
        else:
            volatility = 0.0

        return {
            "trend": trend,
            "volatility": volatility,
            "data_points": len(data),
            "time_span_hours": (data[-1].timestamp - data[0].timestamp) / 3600 if data else 0
        }

    def _generate_cache_recommendations(self, cache_type: str, hit_rate: float,
                                      effectiveness: CacheEffectiveness,
                                      trend_analysis: Dict) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []

        if effectiveness in [CacheEffectiveness.POOR, CacheEffectiveness.CRITICAL]:
            if hit_rate < 0.5:
                recommendations.extend([
                    f"Consider increasing {cache_type} cache size to improve hit rate",
                    f"Review {cache_type} cache eviction policy",
                    f"Analyze {cache_type} access patterns for optimization opportunities"
                ])

        if trend_analysis.get("trend") == "degrading":
            recommendations.append(f"Investigate why {cache_type} hit rate is declining")

        if trend_analysis.get("volatility", 0) > 0.2:
            recommendations.append(f"Address {cache_type} performance volatility")

        if not recommendations:
            recommendations.append(f"{cache_type} cache performance is satisfactory")

        return recommendations

    def _update_access_patterns(self, cache_type: str, operation: str, hit: bool,
                               key_info: Optional[Dict[str, Any]]) -> None:
        """Update access pattern analysis."""
        if cache_type not in self.access_patterns:
            self.access_patterns[cache_type] = {
                "hit_patterns": defaultdict(int),
                "miss_patterns": defaultdict(int),
                "temporal_patterns": defaultdict(list),
                "key_frequency": defaultdict(int),
                "last_updated": time.time()
            }

        patterns = self.access_patterns[cache_type]

        # Update hit/miss patterns
        if hit:
            patterns["hit_patterns"][operation] += 1
        else:
            patterns["miss_patterns"][operation] += 1

        # Update temporal patterns (hourly)
        hour = datetime.fromtimestamp(time.time()).hour
        patterns["temporal_patterns"][hour].append(1 if hit else 0)

        # Keep only recent temporal data (last 10 accesses per hour)
        for hour_key in patterns["temporal_patterns"]:
            if len(patterns["temporal_patterns"][hour_key]) > 10:
                patterns["temporal_patterns"][hour_key] = patterns["temporal_patterns"][hour_key][-10:]

        # Update key frequency if key info is provided
        if key_info and "key" in key_info:
            patterns["key_frequency"][key_info["key"]] += 1

        patterns["last_updated"] = time.time()

    def _analyze_cross_cache_effectiveness(self, cache_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze effectiveness across all cache types."""
        if not cache_analyses:
            return {"analysis": "No cache data available"}

        # Calculate overall cache effectiveness
        effectiveness_scores = {}
        for cache_type, analysis in cache_analyses.items():
            if cache_type != "cross_cache_analysis" and "effectiveness_rating" in analysis:
                effectiveness_map = {
                    "excellent": 5,
                    "good": 4,
                    "fair": 3,
                    "poor": 2,
                    "critical": 1
                }
                effectiveness_scores[cache_type] = effectiveness_map.get(analysis["effectiveness_rating"], 3)

        if effectiveness_scores:
            avg_effectiveness = statistics.mean(effectiveness_scores.values())
            overall_rating = "excellent" if avg_effectiveness >= 4.5 else \
                           "good" if avg_effectiveness >= 3.5 else \
                           "fair" if avg_effectiveness >= 2.5 else "poor"
        else:
            avg_effectiveness = 0
            overall_rating = "unknown"

        # Identify best and worst performing caches
        if effectiveness_scores:
            best_cache = max(effectiveness_scores.items(), key=lambda x: x[1])
            worst_cache = min(effectiveness_scores.items(), key=lambda x: x[1])
        else:
            best_cache = worst_cache = ("unknown", 0)

        # Calculate total memory usage and efficiency
        total_memory = sum(analysis.get("memory_metrics", {}).get("memory_usage_mb", 0)
                          for analysis in cache_analyses.values()
                          if isinstance(analysis, dict) and "memory_metrics" in analysis)

        total_hit_rate = statistics.mean([
            analysis.get("performance_metrics", {}).get("hit_rate", 0)
            for analysis in cache_analyses.values()
            if isinstance(analysis, dict) and "performance_metrics" in analysis
        ]) if cache_analyses else 0

        return {
            "overall_rating": overall_rating,
            "average_effectiveness_score": avg_effectiveness,
            "best_performing_cache": best_cache[0],
            "worst_performing_cache": worst_cache[0],
            "total_memory_usage_mb": total_memory,
            "average_hit_rate": total_hit_rate,
            "cache_count": len([c for c in cache_analyses.keys() if c != "cross_cache_analysis"]),
            "recommendations": self._generate_cross_cache_recommendations(cache_analyses)
        }

    def _generate_cross_cache_recommendations(self, cache_analyses: Dict[str, Dict]) -> List[str]:
        """Generate cross-cache optimization recommendations."""
        recommendations = []

        # Check for memory redistribution opportunities
        memory_usage = {}
        hit_rates = {}

        for cache_type, analysis in cache_analyses.items():
            if cache_type != "cross_cache_analysis" and isinstance(analysis, dict):
                mem_metrics = analysis.get("memory_metrics", {})
                perf_metrics = analysis.get("performance_metrics", {})

                memory_usage[cache_type] = mem_metrics.get("memory_usage_mb", 0)
                hit_rates[cache_type] = perf_metrics.get("hit_rate", 0)

        # Identify caches with high memory but low hit rate
        inefficient_caches = [
            cache_type for cache_type, hit_rate in hit_rates.items()
            if hit_rate < 0.6 and memory_usage.get(cache_type, 0) > 50
        ]

        if inefficient_caches:
            recommendations.append(f"Consider reducing memory allocation for inefficient caches: {', '.join(inefficient_caches)}")

        # Identify caches with high hit rate that could benefit from more memory
        efficient_caches = [
            cache_type for cache_type, hit_rate in hit_rates.items()
            if hit_rate > 0.9
        ]

        if efficient_caches:
            recommendations.append(f"High-performing caches identified: {', '.join(efficient_caches)} - consider increasing their memory allocation")

        if not recommendations:
            recommendations.append("Cache configuration appears balanced across cache types")

        return recommendations


class CacheOptimizationRecommender:
    """Recommends cache optimizations based on effectiveness analysis."""

    def __init__(self, effectiveness_analyzer: CacheEffectivenessAnalyzer):
        self.effectiveness_analyzer = effectiveness_analyzer
        self.opportunities: List[CacheOptimizationOpportunity] = []
        self._lock = threading.RLock()

    def generate_optimization_opportunities(self) -> List[CacheOptimizationOpportunity]:
        """Generate cache optimization opportunities."""
        with self._lock:
            opportunities = []

            # Analyze current cache effectiveness
            cache_analysis = self.effectiveness_analyzer.analyze_cache_effectiveness()

            for cache_type, analysis in cache_analysis.items():
                if cache_type == "cross_cache_analysis":
                    continue

                if not isinstance(analysis, dict) or "error" in analysis:
                    continue

                # Generate opportunities based on analysis
                opportunities.extend(self._generate_single_cache_opportunities(cache_type, analysis))

            # Generate cross-cache opportunities
            if "cross_cache_analysis" in cache_analysis:
                opportunities.extend(self._generate_cross_cache_opportunities(cache_analysis["cross_cache_analysis"]))

            # Sort by priority and potential impact
            opportunities.sort(key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.priority, 1),
                x.potential_improvement
            ), reverse=True)

            self.opportunities = opportunities
            return opportunities

    def _generate_single_cache_opportunities(self, cache_type: str,
                                          analysis: Dict[str, Any]) -> List[CacheOptimizationOpportunity]:
        """Generate optimization opportunities for a single cache."""
        opportunities = []

        effectiveness = analysis.get("effectiveness_rating", "fair")
        hit_rate = analysis.get("performance_metrics", {}).get("hit_rate", 0)
        memory_efficiency = analysis.get("memory_metrics", {}).get("memory_efficiency_score", 0)
        trend = analysis.get("trend_analysis", {}).get("trend", "stable")

        # Low hit rate opportunity
        if hit_rate < 0.7:
            opportunities.append(CacheOptimizationOpportunity(
                opportunity_id=f"cache_opt_{cache_type}_hit_rate_{int(time.time())}",
                title=f"Improve {cache_type} Hit Rate",
                description=f"{cache_type} cache hit rate is {hit_rate:.1%}, below optimal threshold of 70%",
                cache_type=cache_type,
                current_impact=(0.7 - hit_rate) * 100,
                potential_improvement=min(30, (0.7 - hit_rate) * 100),
                priority="high" if hit_rate < 0.5 else "medium",
                effort_estimate="2-4 weeks",
                expected_benefits=[
                    f"Reduce cache misses by up to {(0.7 - hit_rate)*100:.0f}%",
                    "Improve application response times",
                    "Reduce backend load"
                ],
                implementation_steps=[
                    "Analyze access patterns and identify frequently missed keys",
                    "Adjust cache size or eviction policy",
                    "Implement intelligent prefetching if applicable",
                    "Monitor hit rate improvement over time"
                ],
                risks=[
                    "May increase memory usage",
                    "Could affect other cache types if memory is shared"
                ]
            ))

        # Memory inefficiency opportunity
        if memory_efficiency < 0.5 and analysis.get("memory_metrics", {}).get("memory_usage_mb", 0) > 100:
            opportunities.append(CacheOptimizationOpportunity(
                opportunity_id=f"cache_opt_{cache_type}_memory_{int(time.time())}",
                title=f"Optimize {cache_type} Memory Usage",
                description=f"{cache_type} cache has low memory efficiency ({memory_efficiency:.2f}) with high memory usage",
                cache_type=cache_type,
                current_impact=(1 - memory_efficiency) * 50,
                potential_improvement=(1 - memory_efficiency) * 30,
                priority="medium",
                effort_estimate="1-2 weeks",
                expected_benefits=[
                    "Reduce memory footprint by 20-40%",
                    "Improve overall system memory efficiency",
                    "Allow reallocation of memory to other caches"
                ],
                implementation_steps=[
                    "Analyze cache content and identify low-value entries",
                    "Implement more aggressive eviction policies",
                    "Consider cache size reduction",
                    "Monitor memory usage and performance impact"
                ],
                risks=[
                    "May temporarily reduce hit rate",
                    "Could affect performance during transition"
                ]
            ))

        # Degrading trend opportunity
        if trend == "degrading":
            opportunities.append(CacheOptimizationOpportunity(
                opportunity_id=f"cache_opt_{cache_type}_trend_{int(time.time())}",
                title=f"Address {cache_type} Performance Degradation",
                description=f"{cache_type} cache performance is trending downward, requiring investigation and optimization",
                cache_type=cache_type,
                current_impact=20,  # Estimated impact
                potential_improvement=25,
                priority="high",
                effort_estimate="1-3 weeks",
                expected_benefits=[
                    "Stop performance degradation",
                    "Restore optimal cache performance",
                    "Prevent further performance decline"
                ],
                implementation_steps=[
                    "Investigate root cause of performance degradation",
                    "Review recent changes that may have affected cache performance",
                    "Implement corrective measures based on findings",
                    "Establish monitoring to prevent future degradation"
                ],
                risks=[
                    "May require code changes or configuration updates",
                    "Could temporarily affect performance during fixes"
                ]
            ))

        return opportunities

    def _generate_cross_cache_opportunities(self, cross_analysis: Dict[str, Any]) -> List[CacheOptimizationOpportunity]:
        """Generate cross-cache optimization opportunities."""
        opportunities = []

        # Memory reallocation opportunity
        if cross_analysis.get("worst_performing_cache") and cross_analysis.get("best_performing_cache"):
            worst_cache = cross_analysis["worst_performing_cache"]
            best_cache = cross_analysis["best_performing_cache"]

            if worst_cache != best_cache:
                opportunities.append(CacheOptimizationOpportunity(
                    opportunity_id=f"cache_opt_cross_memory_{int(time.time())}",
                    title="Optimize Memory Allocation Across Caches",
                    description=f"Consider reallocating memory from underperforming {worst_cache} cache to high-performing {best_cache} cache",
                    cache_type="cross_cache",
                    current_impact=15,
                    potential_improvement=20,
                    priority="medium",
                    effort_estimate="1 week",
                    expected_benefits=[
                        "Improve overall cache system efficiency",
                        "Better memory utilization across cache types",
                        "Enhanced system-wide performance"
                    ],
                    implementation_steps=[
                        "Analyze current memory allocation across all caches",
                        "Identify optimal memory distribution",
                        "Implement new memory allocation strategy",
                        "Monitor performance impact and adjust as needed"
                    ],
                    risks=[
                        "May temporarily affect performance during reallocation",
                        "Requires careful monitoring and potential rollback"
                    ]
                ))

        return opportunities


# Global instances
_cache_effectiveness_analyzer: Optional[CacheEffectivenessAnalyzer] = None
_cache_optimization_recommender: Optional[CacheOptimizationRecommender] = None


def get_cache_effectiveness_analyzer() -> CacheEffectivenessAnalyzer:
    """Get the global cache effectiveness analyzer instance."""
    global _cache_effectiveness_analyzer
    if _cache_effectiveness_analyzer is None:
        _cache_effectiveness_analyzer = CacheEffectivenessAnalyzer()
    return _cache_effectiveness_analyzer


def get_cache_optimization_recommender() -> CacheOptimizationRecommender:
    """Get the global cache optimization recommender instance."""
    global _cache_optimization_recommender, _cache_effectiveness_analyzer
    if _cache_effectiveness_analyzer is None:
        _cache_effectiveness_analyzer = CacheEffectivenessAnalyzer()
    if _cache_optimization_recommender is None:
        _cache_optimization_recommender = CacheOptimizationRecommender(_cache_effectiveness_analyzer)
    return _cache_optimization_recommender