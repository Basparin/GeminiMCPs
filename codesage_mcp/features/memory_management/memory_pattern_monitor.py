"""
Memory Pattern Monitor Module for CodeSage MCP Server.

This module provides comprehensive memory usage pattern monitoring under varying loads,
including adaptive memory management, load-aware optimization, and predictive memory allocation.

Classes:
    MemoryPatternMonitor: Monitors memory usage patterns under different loads
    AdaptiveMemoryManager: Provides adaptive memory management based on patterns
    LoadAwareMemoryOptimizer: Optimizes memory based on load patterns
    MemoryPredictionEngine: Predicts memory needs based on usage patterns
"""

import logging
import time
import threading
import psutil
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    """System load levels."""
    IDLE = "idle"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CRITICAL = "critical"


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    load_level: LoadLevel
    active_connections: int = 0
    cache_size_mb: float = 0.0
    model_cache_size_mb: float = 0.0
    request_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPattern:
    """Represents a detected memory usage pattern."""
    pattern_id: str
    pattern_type: str
    load_level: LoadLevel
    avg_memory_mb: float
    peak_memory_mb: float
    memory_volatility: float
    duration_minutes: float
    frequency_score: float
    optimization_potential: float
    detected_at: float = field(default_factory=time.time)


@dataclass
class MemoryOptimization:
    """Represents a memory optimization recommendation."""
    optimization_id: str
    title: str
    description: str
    load_condition: LoadLevel
    current_memory_mb: float
    target_memory_mb: float
    expected_savings_mb: float
    implementation_effort: str
    priority: str
    expected_benefits: List[str]
    risks: List[str]
    implementation_steps: List[str]
    created_at: float = field(default_factory=time.time)


class MemoryPatternMonitor:
    """Monitors memory usage patterns under varying loads."""

    def __init__(self, history_window_hours: int = 24, snapshot_interval_seconds: int = 30):
        self.history_window_hours = history_window_hours
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.snapshots: deque = deque(maxlen=10000)
        self.patterns: Dict[str, MemoryPattern] = {}
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Pattern detection parameters
        self.pattern_detection_window_minutes = 60  # Look at 1-hour windows
        self.min_pattern_duration_minutes = 10
        self.pattern_similarity_threshold = 0.8

        # Start monitoring
        self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start the memory monitoring thread."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryPatternMonitor"
        )
        self._monitor_thread.start()
        logger.info("Memory pattern monitoring started")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that captures memory snapshots."""
        while self._monitoring_active:
            try:
                self._capture_snapshot()
                self._analyze_patterns()
                time.sleep(self.snapshot_interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(5.0)

    def _capture_snapshot(self) -> None:
        """Capture a memory usage snapshot."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # Determine load level based on CPU usage and request patterns
            load_level = self._determine_load_level()

            # Get additional context
            active_connections = getattr(process, 'connections', lambda: [])()
            connection_count = len(active_connections) if callable(active_connections) else 0

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=process.memory_percent(),
                load_level=load_level,
                active_connections=connection_count,
                metadata={
                    "cpu_percent": psutil.cpu_percent(interval=1.0),
                    "disk_io": dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
                    "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {}
                }
            )

            with self._lock:
                self.snapshots.append(snapshot)

            # Clean old snapshots
            cutoff_time = time.time() - (self.history_window_hours * 60 * 60)
            while self.snapshots and self.snapshots[0].timestamp < cutoff_time:
                self.snapshots.popleft()

        except Exception as e:
            logger.warning(f"Error capturing memory snapshot: {e}")

    def _determine_load_level(self) -> LoadLevel:
        """Determine current system load level."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)

            if cpu_percent < 20:
                return LoadLevel.IDLE
            elif cpu_percent < 40:
                return LoadLevel.LIGHT
            elif cpu_percent < 65:
                return LoadLevel.MODERATE
            elif cpu_percent < 85:
                return LoadLevel.HEAVY
            else:
                return LoadLevel.CRITICAL
        except Exception:
            return LoadLevel.MODERATE

    def _analyze_patterns(self) -> None:
        """Analyze memory usage patterns from recent snapshots."""
        with self._lock:
            if len(self.snapshots) < 10:  # Need minimum data
                return

            # Get recent snapshots (last hour)
            recent_snapshots = [s for s in self.snapshots
                              if s.timestamp > time.time() - (self.pattern_detection_window_minutes * 60)]

            if len(recent_snapshots) < 5:
                return

            # Group by load level
            load_groups = defaultdict(list)
            for snapshot in recent_snapshots:
                load_groups[snapshot.load_level].append(snapshot)

            # Analyze patterns for each load level
            for load_level, snapshots in load_groups.items():
                if len(snapshots) >= 3:  # Need minimum snapshots per load level
                    self._detect_pattern_for_load_level(load_level, snapshots)

    def _detect_pattern_for_load_level(self, load_level: LoadLevel, snapshots: List[MemorySnapshot]) -> None:
        """Detect memory usage patterns for a specific load level."""
        memory_values = [s.rss_mb for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]

        # Calculate pattern characteristics
        avg_memory = statistics.mean(memory_values)
        peak_memory = max(memory_values)
        memory_volatility = statistics.stdev(memory_values) if len(memory_values) > 1 else 0

        # Calculate pattern duration
        duration_seconds = timestamps[-1] - timestamps[0]
        duration_minutes = duration_seconds / 60

        if duration_minutes < self.min_pattern_duration_minutes:
            return

        # Calculate frequency score (how often this pattern occurs)
        pattern_frequency = self._calculate_pattern_frequency(load_level, avg_memory)

        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(load_level, avg_memory, memory_volatility)

        # Create pattern ID
        pattern_id = f"{load_level.value}_{int(avg_memory)}_{int(time.time())}"

        pattern = MemoryPattern(
            pattern_id=pattern_id,
            pattern_type="load_based_memory_usage",
            load_level=load_level,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            memory_volatility=memory_volatility,
            duration_minutes=duration_minutes,
            frequency_score=pattern_frequency,
            optimization_potential=optimization_potential
        )

        self.patterns[pattern_id] = pattern

        # Keep only recent patterns (last 100)
        if len(self.patterns) > 100:
            oldest_pattern = min(self.patterns.keys(), key=lambda x: self.patterns[x].detected_at)
            del self.patterns[oldest_pattern]

    def _calculate_pattern_frequency(self, load_level: LoadLevel, avg_memory: float) -> float:
        """Calculate how frequently this pattern occurs."""
        # Count similar patterns in history
        similar_patterns = 0
        total_patterns = 0

        for pattern in self.patterns.values():
            total_patterns += 1
            if (pattern.load_level == load_level and
                abs(pattern.avg_memory_mb - avg_memory) / avg_memory < 0.2):  # 20% similarity
                similar_patterns += 1

        return similar_patterns / max(total_patterns, 1)

    def _calculate_optimization_potential(self, load_level: LoadLevel, avg_memory: float,
                                        volatility: float) -> float:
        """Calculate optimization potential for this pattern."""
        # Base potential based on load level
        base_potential = {
            LoadLevel.IDLE: 0.2,
            LoadLevel.LIGHT: 0.4,
            LoadLevel.MODERATE: 0.6,
            LoadLevel.HEAVY: 0.8,
            LoadLevel.CRITICAL: 1.0
        }.get(load_level, 0.5)

        # Adjust based on volatility (higher volatility = higher optimization potential)
        volatility_adjustment = min(volatility / 50, 0.3)  # Max 30% adjustment

        # Adjust based on memory usage level
        memory_adjustment = min(avg_memory / 1000, 0.2)  # Higher memory = higher potential

        return min(base_potential + volatility_adjustment + memory_adjustment, 1.0)

    def get_memory_analysis(self, analysis_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive memory usage analysis."""
        with self._lock:
            if not self.snapshots:
                return {"error": "No memory data available"}

            # Filter snapshots by time window
            cutoff_time = time.time() - (analysis_window_hours * 60 * 60)
            recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

            if not recent_snapshots:
                return {"error": f"No data available for last {analysis_window_hours} hours"}

            # Basic statistics
            memory_values = [s.rss_mb for s in recent_snapshots]
            analysis = {
                "analysis_window_hours": analysis_window_hours,
                "total_snapshots": len(recent_snapshots),
                "memory_statistics": {
                    "average_mb": statistics.mean(memory_values),
                    "peak_mb": max(memory_values),
                    "minimum_mb": min(memory_values),
                    "median_mb": statistics.median(memory_values),
                    "volatility_mb": statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                    "percentiles": {
                        "p95": np.percentile(memory_values, 95),
                        "p99": np.percentile(memory_values, 99)
                    }
                },
                "load_distribution": self._analyze_load_distribution(recent_snapshots),
                "patterns": self._get_top_patterns(),
                "trends": self._analyze_memory_trends(recent_snapshots),
                "optimization_opportunities": self._generate_memory_optimizations(),
                "generated_at": time.time()
            }

            return analysis

    def _analyze_load_distribution(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze memory usage distribution across different load levels."""
        load_groups = defaultdict(list)

        for snapshot in snapshots:
            load_groups[snapshot.load_level].append(snapshot.rss_mb)

        distribution = {}
        for load_level, memory_values in load_groups.items():
            distribution[load_level.value] = {
                "count": len(memory_values),
                "avg_memory_mb": statistics.mean(memory_values),
                "peak_memory_mb": max(memory_values),
                "percentage": (len(memory_values) / len(snapshots)) * 100
            }

        return distribution

    def _get_top_patterns(self) -> List[Dict[str, Any]]:
        """Get top memory usage patterns."""
        # Sort patterns by optimization potential
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.optimization_potential,
            reverse=True
        )

        top_patterns = []
        for pattern in sorted_patterns[:5]:  # Top 5 patterns
            top_patterns.append({
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "load_level": pattern.load_level.value,
                "avg_memory_mb": pattern.avg_memory_mb,
                "peak_memory_mb": pattern.peak_memory_mb,
                "memory_volatility": pattern.memory_volatility,
                "duration_minutes": pattern.duration_minutes,
                "frequency_score": pattern.frequency_score,
                "optimization_potential": pattern.optimization_potential
            })

        return top_patterns

    def _analyze_memory_trends(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        if len(snapshots) < 5:
            return {"trend": "insufficient_data"}

        memory_values = [s.rss_mb for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]

        # Simple linear trend analysis
        x = np.array(range(len(memory_values)))
        y = np.array(memory_values)

        try:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = "increasing" if slope > 1 else "decreasing" if slope < -1 else "stable"

            # Calculate trend strength
            trend_strength = abs(slope) / statistics.mean(memory_values)

            return {
                "trend_direction": trend_direction,
                "trend_slope_mb_per_snapshot": slope,
                "trend_strength": trend_strength,
                "data_points": len(memory_values),
                "time_span_hours": (timestamps[-1] - timestamps[0]) / 3600
            }
        except Exception:
            return {"trend": "analysis_error"}

    def _generate_memory_optimizations(self) -> List[Dict[str, Any]]:
        """Generate memory optimization recommendations."""
        optimizations = []

        # Analyze current patterns for optimization opportunities
        for pattern in self.patterns.values():
            if pattern.optimization_potential > 0.5:  # High optimization potential
                optimization = self._create_optimization_for_pattern(pattern)
                if optimization:
                    optimizations.append(optimization)

        # Sort by expected savings
        optimizations.sort(key=lambda x: x.get("expected_savings_mb", 0), reverse=True)

        return optimizations[:5]  # Top 5 optimizations

    def _create_optimization_for_pattern(self, pattern: MemoryPattern) -> Optional[Dict[str, Any]]:
        """Create optimization recommendation for a pattern."""
        if pattern.load_level == LoadLevel.CRITICAL:
            return {
                "optimization_id": f"mem_opt_{pattern.pattern_id}",
                "title": f"Optimize Memory Usage Under {pattern.load_level.value.title()} Load",
                "description": f"High memory usage detected under {pattern.load_level.value} load conditions "
                             f"(avg: {pattern.avg_memory_mb:.1f}MB, peak: {pattern.peak_memory_mb:.1f}MB)",
                "load_condition": pattern.load_level.value,
                "current_memory_mb": pattern.avg_memory_mb,
                "target_memory_mb": pattern.avg_memory_mb * 0.8,
                "expected_savings_mb": pattern.avg_memory_mb * 0.2,
                "implementation_effort": "Medium",
                "priority": "high",
                "expected_benefits": [
                    f"Reduce memory usage by ~{pattern.avg_memory_mb * 0.2:.0f}MB under {pattern.load_level.value} load",
                    "Improve system stability during peak loads",
                    "Reduce risk of memory-related failures"
                ],
                "risks": [
                    "May temporarily impact performance during optimization",
                    "Requires careful testing under load conditions"
                ],
                "implementation_steps": [
                    "Profile memory usage during high load periods",
                    "Identify memory-intensive operations",
                    "Implement memory-efficient alternatives",
                    "Test optimization under various load conditions"
                ]
            }
        elif pattern.memory_volatility > 100:  # High volatility
            return {
                "optimization_id": f"mem_stab_{pattern.pattern_id}",
                "title": f"Stabilize Memory Usage Under {pattern.load_level.value.title()} Load",
                "description": f"High memory volatility detected ({pattern.memory_volatility:.1f}MB std dev) "
                             f"under {pattern.load_level.value} load conditions",
                "load_condition": pattern.load_level.value,
                "current_memory_mb": pattern.avg_memory_mb,
                "target_memory_mb": pattern.avg_memory_mb,  # Same target, focus on stability
                "expected_savings_mb": pattern.memory_volatility * 0.5,  # Reduce volatility by 50%
                "implementation_effort": "Low to Medium",
                "priority": "medium",
                "expected_benefits": [
                    "More predictable memory usage",
                    "Better capacity planning",
                    "Reduced memory pressure spikes"
                ],
                "risks": [
                    "May require code changes for memory stability",
                    "Could affect performance consistency"
                ],
                "implementation_steps": [
                    "Identify sources of memory volatility",
                    "Implement memory usage stabilization techniques",
                    "Monitor memory stability improvements",
                    "Adjust based on observed patterns"
                ]
            }

        return None

    def stop_monitoring(self) -> None:
        """Stop the memory monitoring thread."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory pattern monitoring stopped")


class AdaptiveMemoryManager:
    """Provides adaptive memory management based on observed patterns."""

    def __init__(self, pattern_monitor: MemoryPatternMonitor):
        self.pattern_monitor = pattern_monitor
        self.adaptation_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def adapt_memory_settings(self) -> Dict[str, Any]:
        """Adapt memory settings based on current patterns."""
        with self._lock:
            current_patterns = self.pattern_monitor.patterns

            if not current_patterns:
                return {"adaptation": "no_patterns_available"}

            # Analyze current memory pressure
            memory_pressure = self._assess_memory_pressure()

            # Generate adaptation recommendations
            adaptations = self._generate_adaptations(current_patterns, memory_pressure)

            # Apply adaptations (in a real implementation, this would modify actual settings)
            applied_adaptations = self._apply_adaptations(adaptations)

            return {
                "memory_pressure": memory_pressure.value,
                "recommended_adaptations": adaptations,
                "applied_adaptations": applied_adaptations,
                "adaptation_timestamp": time.time()
            }

    def _assess_memory_pressure(self) -> MemoryPressure:
        """Assess current memory pressure level."""
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()

            if memory_percent > 90:
                return MemoryPressure.CRITICAL
            elif memory_percent > 80:
                return MemoryPressure.HIGH
            elif memory_percent > 70:
                return MemoryPressure.MODERATE
            else:
                return MemoryPressure.LOW
        except Exception:
            return MemoryPressure.MODERATE

    def _generate_adaptations(self, patterns: Dict[str, MemoryPattern],
                            memory_pressure: MemoryPressure) -> List[Dict[str, Any]]:
        """Generate memory adaptation recommendations."""
        adaptations = []

        # High memory pressure adaptations
        if memory_pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            adaptations.extend([
                {
                    "type": "cache_size_reduction",
                    "description": "Reduce cache sizes to alleviate memory pressure",
                    "target": "all_caches",
                    "reduction_percentage": 20,
                    "priority": "high"
                },
                {
                    "type": "gc_optimization",
                    "description": "Trigger garbage collection to free memory",
                    "target": "system",
                    "priority": "high"
                }
            ])

        # Pattern-based adaptations
        for pattern in patterns.values():
            if pattern.optimization_potential > 0.7:
                adaptations.append({
                    "type": "pattern_specific_optimization",
                    "description": f"Optimize memory for {pattern.load_level.value} load pattern",
                    "target": pattern.load_level.value,
                    "pattern_id": pattern.pattern_id,
                    "priority": "medium"
                })

        return adaptations

    def _apply_adaptations(self, adaptations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply memory adaptations (simulation for now)."""
        applied = []

        for adaptation in adaptations:
            if adaptation["type"] == "gc_optimization":
                # Trigger garbage collection
                import gc
                gc.collect()
                applied.append({
                    **adaptation,
                    "status": "applied",
                    "timestamp": time.time()
                })
            else:
                # For other adaptations, just mark as recommended
                applied.append({
                    **adaptation,
                    "status": "recommended",
                    "timestamp": time.time()
                })

        return applied


class LoadAwareMemoryOptimizer:
    """Optimizes memory based on load patterns."""

    def __init__(self, pattern_monitor: MemoryPatternMonitor):
        self.pattern_monitor = pattern_monitor
        self.optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def optimize_for_load(self, target_load: LoadLevel) -> Dict[str, Any]:
        """Optimize memory settings for a specific load level."""
        with self._lock:
            # Find patterns for target load
            relevant_patterns = [
                pattern for pattern in self.pattern_monitor.patterns.values()
                if pattern.load_level == target_load
            ]

            if not relevant_patterns:
                return {
                    "optimization": "no_patterns_found",
                    "target_load": target_load.value,
                    "message": f"No memory patterns available for {target_load.value} load"
                }

            # Calculate optimal memory settings
            optimal_settings = self._calculate_optimal_settings(relevant_patterns, target_load)

            # Generate optimization plan
            optimization_plan = self._create_optimization_plan(optimal_settings, target_load)

            # Record optimization
            optimization_record = {
                "timestamp": time.time(),
                "target_load": target_load.value,
                "optimal_settings": optimal_settings,
                "optimization_plan": optimization_plan
            }
            self.optimization_history.append(optimization_record)

            return {
                "target_load": target_load.value,
                "optimal_settings": optimal_settings,
                "optimization_plan": optimization_plan,
                "generated_at": time.time()
            }

    def _calculate_optimal_settings(self, patterns: List[MemoryPattern],
                                  target_load: LoadLevel) -> Dict[str, Any]:
        """Calculate optimal memory settings for load level."""
        if not patterns:
            return {"error": "no_patterns"}

        # Use the pattern with highest optimization potential
        best_pattern = max(patterns, key=lambda p: p.optimization_potential)

        # Calculate optimal cache sizes based on pattern
        base_cache_size = 100  # Base cache size in MB
        load_multiplier = {
            LoadLevel.IDLE: 0.5,
            LoadLevel.LIGHT: 0.8,
            LoadLevel.MODERATE: 1.0,
            LoadLevel.HEAVY: 1.2,
            LoadLevel.CRITICAL: 1.5
        }.get(target_load, 1.0)

        optimal_cache_size = base_cache_size * load_multiplier

        # Adjust for memory pressure
        memory_pressure_adjustment = 0.8 if best_pattern.avg_memory_mb > 800 else 1.0

        return {
            "recommended_cache_size_mb": optimal_cache_size * memory_pressure_adjustment,
            "recommended_model_cache_size_mb": optimal_cache_size * 0.3 * memory_pressure_adjustment,
            "recommended_file_cache_size_mb": optimal_cache_size * 0.2 * memory_pressure_adjustment,
            "memory_pressure_adjustment": memory_pressure_adjustment,
            "based_on_pattern": best_pattern.pattern_id
        }

    def _create_optimization_plan(self, optimal_settings: Dict[str, Any],
                                target_load: LoadLevel) -> Dict[str, Any]:
        """Create detailed optimization plan."""
        return {
            "target_load_level": target_load.value,
            "recommended_settings": optimal_settings,
            "implementation_steps": [
                f"Set cache size to {optimal_settings['recommended_cache_size_mb']:.0f}MB",
                f"Configure model cache to {optimal_settings['recommended_model_cache_size_mb']:.0f}MB",
                f"Set file cache size to {optimal_settings['recommended_file_cache_size_mb']:.0f}MB",
                "Monitor memory usage after changes",
                "Adjust settings based on observed performance"
            ],
            "expected_benefits": [
                f"Optimized memory usage for {target_load.value} load conditions",
                "Better cache hit rates",
                "Reduced memory pressure",
                "Improved overall system performance"
            ],
            "monitoring_guidance": [
                "Monitor cache hit rates for at least 1 hour",
                "Track memory usage patterns",
                "Observe response time improvements",
                "Watch for any performance regressions"
            ]
        }


# Global instances
_memory_pattern_monitor: Optional[MemoryPatternMonitor] = None
_adaptive_memory_manager: Optional[AdaptiveMemoryManager] = None
_load_aware_optimizer: Optional[LoadAwareMemoryOptimizer] = None


def get_memory_pattern_monitor() -> MemoryPatternMonitor:
    """Get the global memory pattern monitor instance."""
    global _memory_pattern_monitor
    if _memory_pattern_monitor is None:
        _memory_pattern_monitor = MemoryPatternMonitor()
    return _memory_pattern_monitor


def get_adaptive_memory_manager() -> AdaptiveMemoryManager:
    """Get the global adaptive memory manager instance."""
    global _adaptive_memory_manager, _memory_pattern_monitor
    if _memory_pattern_monitor is None:
        _memory_pattern_monitor = MemoryPatternMonitor()
    if _adaptive_memory_manager is None:
        _adaptive_memory_manager = AdaptiveMemoryManager(_memory_pattern_monitor)
    return _adaptive_memory_manager


def get_load_aware_optimizer() -> LoadAwareMemoryOptimizer:
    """Get the global load-aware memory optimizer instance."""
    global _load_aware_optimizer, _memory_pattern_monitor
    if _memory_pattern_monitor is None:
        _memory_pattern_monitor = MemoryPatternMonitor()
    if _load_aware_optimizer is None:
        _load_aware_optimizer = LoadAwareMemoryOptimizer(_memory_pattern_monitor)
    return _load_aware_optimizer