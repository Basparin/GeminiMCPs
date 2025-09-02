"""
Workload Pattern Recognition Module for CodeSage MCP Server.

This module provides advanced workload pattern recognition capabilities that identify
different workload patterns and enable optimal resource allocation based on pattern analysis.

Classes:
    WorkloadPatternRecognition: Main workload pattern recognition class
    PatternClassifier: ML-based pattern classification system
    ResourceAllocator: Optimal resource allocation based on patterns
    WorkloadForecaster: Predictive workload forecasting
    PatternBasedOptimizer: Pattern-aware optimization strategies
"""

import logging
import time
import threading
import statistics
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class WorkloadPattern(Enum):
    """Types of workload patterns."""
    COMPUTE_INTENSIVE = "compute_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED_LOAD = "mixed_load"
    BURSTY = "bursty"
    STEADY_STATE = "steady_state"
    INTERMITTENT = "intermittent"
    PREDICTABLE = "predictable"
    UNPREDICTABLE = "unpredictable"


class ResourceType(Enum):
    """Types of resources for allocation."""
    CPU_CORES = "cpu_cores"
    MEMORY_MB = "memory_mb"
    DISK_IO = "disk_io"
    NETWORK_BANDWIDTH = "network_bandwidth"
    CACHE_SIZE = "cache_size"
    WORKER_THREADS = "worker_threads"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class OperationType(Enum):
    """Types of operations for timeout configuration."""
    FAST = "fast"              # Cache hits, simple reads (100-500ms)
    MEDIUM = "medium"          # Searches, file operations (1-2s)
    SLOW = "slow"              # LLM analysis, complex computations (5-10s)
    VERY_SLOW = "very_slow"    # Large indexing, batch operations (30-60s)


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a workload pattern."""
    pattern_type: WorkloadPattern
    intensity_score: float  # 0-1 scale
    predictability_score: float  # 0-1 scale
    resource_demand: Dict[ResourceType, float]
    temporal_pattern: str  # hourly, daily, weekly, etc.
    duration_estimate: int  # estimated duration in minutes
    confidence_score: float  # 0-1 scale
    detected_at: float = field(default_factory=time.time)


@dataclass
class ResourceAllocation:
    """Resource allocation recommendation."""
    resource_type: ResourceType
    current_allocation: float
    recommended_allocation: float
    allocation_reason: str
    expected_benefit: float
    implementation_priority: int
    rollback_plan: Dict[str, Any]


@dataclass
class PatternRecognitionMetrics:
    """Metrics for pattern recognition performance."""
    total_patterns_detected: int = 0
    correct_classifications: int = 0
    false_positives: int = 0
    pattern_confidence_avg: float = 0.0
    resource_allocation_success_rate: float = 0.0
    average_pattern_duration: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitBreakerState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.time)


@dataclass
class TimeoutMetrics:
    """Timeout performance metrics."""
    operation_type: OperationType
    requested_timeout: float
    actual_timeout: float
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)


class WorkloadPatternRecognition:
    """Main workload pattern recognition class."""

    def __init__(self, analysis_window_minutes: int = 60, pattern_detection_threshold: float = 0.7):
        self.analysis_window_minutes = analysis_window_minutes
        self.pattern_detection_threshold = pattern_detection_threshold

        # Pattern detection data
        self.workload_history: deque = deque(maxlen=1000)
        self.detected_patterns: List[WorkloadCharacteristics] = []
        self.pattern_transitions: List[Dict[str, Any]] = []

        # Resource allocation tracking
        self.resource_allocations: Dict[ResourceType, float] = {}
        self.allocation_history: List[ResourceAllocation] = []

        # ML models
        self.pattern_classifier = PatternClassifier()
        self.resource_allocator = ResourceAllocator()
        self.workload_forecaster = WorkloadForecaster()

        # Control
        self._recognition_active = False
        self._recognition_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Pattern recognition metrics
        self.metrics = PatternRecognitionMetrics()

        # Circuit breaker for timeout protection
        self.circuit_breaker: Dict[str, CircuitBreakerStats] = defaultdict(
            lambda: CircuitBreakerStats(state=CircuitBreakerState.CLOSED)
        )

        # Adaptive timeout tracking
        self.timeout_metrics: Dict[OperationType, TimeoutMetrics] = {}

        # Start pattern recognition
        self._start_pattern_recognition()

    def _start_pattern_recognition(self) -> None:
        """Start the pattern recognition system."""
        if self._recognition_active:
            return

        # Initialize timeout metrics
        self._initialize_timeout_metrics()

        self._recognition_active = True
        self._recognition_thread = threading.Thread(
            target=self._pattern_recognition_loop,
            daemon=True,
            name="WorkloadPatternRecognition"
        )
        self._recognition_thread.start()
        logger.info("Workload pattern recognition started")

    def _initialize_timeout_metrics(self) -> None:
        """Initialize timeout metrics for all operation types."""
        from codesage_mcp.config.config import (
            FAST_OPERATION_TIMEOUT, MEDIUM_OPERATION_TIMEOUT,
            SLOW_OPERATION_TIMEOUT, VERY_SLOW_OPERATION_TIMEOUT
        )

        self.timeout_metrics = {
            OperationType.FAST: TimeoutMetrics(
                operation_type=OperationType.FAST,
                requested_timeout=FAST_OPERATION_TIMEOUT,
                actual_timeout=FAST_OPERATION_TIMEOUT
            ),
            OperationType.MEDIUM: TimeoutMetrics(
                operation_type=OperationType.MEDIUM,
                requested_timeout=MEDIUM_OPERATION_TIMEOUT,
                actual_timeout=MEDIUM_OPERATION_TIMEOUT
            ),
            OperationType.SLOW: TimeoutMetrics(
                operation_type=OperationType.SLOW,
                requested_timeout=SLOW_OPERATION_TIMEOUT,
                actual_timeout=SLOW_OPERATION_TIMEOUT
            ),
            OperationType.VERY_SLOW: TimeoutMetrics(
                operation_type=OperationType.VERY_SLOW,
                requested_timeout=VERY_SLOW_OPERATION_TIMEOUT,
                actual_timeout=VERY_SLOW_OPERATION_TIMEOUT
            )
        }

    def _pattern_recognition_loop(self) -> None:
        """Main pattern recognition loop."""
        while self._recognition_active:
            try:
                self._perform_pattern_analysis()
                time.sleep(self.analysis_window_minutes * 60)
            except Exception as e:
                logger.error(f"Error in pattern recognition loop: {e}")
                time.sleep(60)

    def _perform_pattern_analysis(self) -> None:
        """Perform comprehensive pattern analysis."""
        with self._lock:
            # Collect current workload data
            current_workload = self._collect_current_workload()

            # Detect patterns
            detected_patterns = self._detect_workload_patterns(current_workload)

            # Classify patterns
            classified_patterns = self._classify_patterns(detected_patterns)

            # Generate resource allocations
            resource_allocations = self._generate_resource_allocations(classified_patterns)

            # Apply allocations
            applied_allocations = self._apply_resource_allocations(resource_allocations)

            # Update metrics
            self._update_recognition_metrics(classified_patterns, applied_allocations)

            logger.info(f"Pattern analysis completed: {len(classified_patterns)} patterns, "
                       f"{len(applied_allocations)} allocations")

    def _collect_current_workload(self) -> Dict[str, Any]:
        """Collect current workload characteristics."""
        # In a real implementation, this would collect actual system metrics
        # For now, return mock data
        current_time = time.time()

        workload_data = {
            "timestamp": current_time,
            "cpu_percent": 65.0 + random.uniform(-20, 20),
            "memory_percent": 70.0 + random.uniform(-15, 15),
            "disk_io_percent": 45.0 + random.uniform(-20, 20),
            "network_io_percent": 30.0 + random.uniform(-15, 15),
            "active_connections": 15 + random.randint(-5, 10),
            "request_rate": 1200 + random.uniform(-300, 400),
            "cache_hit_rate": 85.0 + random.uniform(-10, 10),
            "avg_response_time_ms": 25.0 + random.uniform(-10, 15),
            "error_rate_percent": 0.05 + random.uniform(-0.03, 0.05),
            "workload_intensity": 0.7 + random.uniform(-0.3, 0.3)
        }

        # Store in history
        self.workload_history.append(workload_data)

        return workload_data

    def _detect_workload_patterns(self, workload_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential workload patterns."""
        patterns = []

        # CPU-intensive pattern detection
        if workload_data["cpu_percent"] > 80:
            patterns.append({
                "type": WorkloadPattern.COMPUTE_INTENSIVE,
                "confidence": min(1.0, workload_data["cpu_percent"] / 100.0),
                "characteristics": workload_data
            })

        # Memory-intensive pattern detection
        if workload_data["memory_percent"] > 85:
            patterns.append({
                "type": WorkloadPattern.MEMORY_INTENSIVE,
                "confidence": min(1.0, workload_data["memory_percent"] / 100.0),
                "characteristics": workload_data
            })

        # I/O-intensive pattern detection
        if workload_data["disk_io_percent"] > 70:
            patterns.append({
                "type": WorkloadPattern.IO_INTENSIVE,
                "confidence": min(1.0, workload_data["disk_io_percent"] / 100.0),
                "characteristics": workload_data
            })

        # Bursty pattern detection
        if len(self.workload_history) >= 5:
            recent_intensity = [w["workload_intensity"] for w in list(self.workload_history)[-5:]]
            intensity_variance = statistics.variance(recent_intensity) if len(recent_intensity) > 1 else 0

            if intensity_variance > 0.1:
                patterns.append({
                    "type": WorkloadPattern.BURSTY,
                    "confidence": min(1.0, intensity_variance * 2),
                    "characteristics": workload_data
                })

        # Steady state pattern detection
        if workload_data["workload_intensity"] > 0.6 and workload_data["workload_intensity"] < 0.9:
            patterns.append({
                "type": WorkloadPattern.STEADY_STATE,
                "confidence": 0.8,
                "characteristics": workload_data
            })

        return patterns

    def _classify_patterns(self, detected_patterns: List[Dict[str, Any]]) -> List[WorkloadCharacteristics]:
        """Classify detected patterns using ML-based classification."""
        classified_patterns = []

        for pattern_data in detected_patterns:
            # Use pattern classifier to enhance pattern detection
            enhanced_pattern = self.pattern_classifier.classify_pattern(pattern_data)

            # Create WorkloadCharacteristics object
            characteristics = WorkloadCharacteristics(
                pattern_type=enhanced_pattern["pattern_type"],
                intensity_score=enhanced_pattern["intensity_score"],
                predictability_score=enhanced_pattern["predictability_score"],
                resource_demand=enhanced_pattern["resource_demand"],
                temporal_pattern=enhanced_pattern["temporal_pattern"],
                duration_estimate=enhanced_pattern["duration_estimate"],
                confidence_score=enhanced_pattern["confidence_score"]
            )

            classified_patterns.append(characteristics)

        # Filter by confidence threshold
        classified_patterns = [
            p for p in classified_patterns
            if p.confidence_score >= self.pattern_detection_threshold
        ]

        # Sort by confidence and intensity
        classified_patterns.sort(key=lambda x: (x.confidence_score, x.intensity_score), reverse=True)

        # Store detected patterns
        self.detected_patterns.extend(classified_patterns[-10:])  # Keep last 10
        if len(self.detected_patterns) > 50:
            self.detected_patterns = self.detected_patterns[-50:]

        return classified_patterns

    def _generate_resource_allocations(self, patterns: List[WorkloadCharacteristics]) -> List[ResourceAllocation]:
        """Generate optimal resource allocations based on detected patterns."""
        allocations = []

        for pattern in patterns:
            # Get resource allocation recommendations
            pattern_allocations = self.resource_allocator.generate_allocations(pattern)

            allocations.extend(pattern_allocations)

        # Prioritize allocations
        allocations.sort(key=lambda x: x.implementation_priority, reverse=True)

        # Remove duplicates and conflicting allocations
        final_allocations = self._resolve_allocation_conflicts(allocations)

        return final_allocations

    def _resolve_allocation_conflicts(self, allocations: List[ResourceAllocation]) -> List[ResourceAllocation]:
        """Resolve conflicting resource allocation recommendations."""
        # Group by resource type
        allocations_by_resource = defaultdict(list)
        for allocation in allocations:
            allocations_by_resource[allocation.resource_type].append(allocation)

        resolved_allocations = []

        for resource_type, resource_allocations in allocations_by_resource.items():
            if len(resource_allocations) == 1:
                resolved_allocations.append(resource_allocations[0])
            else:
                # Resolve conflicts by selecting highest priority allocation
                best_allocation = max(resource_allocations, key=lambda x: x.implementation_priority)
                resolved_allocations.append(best_allocation)

        return resolved_allocations

    def _apply_resource_allocations(self, allocations: List[ResourceAllocation]) -> List[ResourceAllocation]:
        """Apply resource allocation recommendations."""
        applied_allocations = []

        for allocation in allocations:
            try:
                # Apply the allocation
                success = self._apply_single_allocation(allocation)

                if success:
                    applied_allocations.append(allocation)
                    self.allocation_history.append(allocation)
                    self.resource_allocations[allocation.resource_type] = allocation.recommended_allocation

                    logger.debug(f"Applied allocation: {allocation.resource_type.value} = {allocation.recommended_allocation}")

            except Exception as e:
                logger.exception(f"Failed to apply allocation {allocation.resource_type.value}: {e}")

        return applied_allocations

    def _apply_single_allocation(self, allocation: ResourceAllocation) -> bool:
        """Apply a single resource allocation."""
        # In a real implementation, this would actually modify system resources
        # For now, simulate successful application

        # Simulate allocation time
        time.sleep(0.01)

        return True

    def _update_recognition_metrics(self, patterns: List[WorkloadCharacteristics],
                                  allocations: List[ResourceAllocation]) -> None:
        """Update pattern recognition metrics."""
        self.metrics.total_patterns_detected += len(patterns)

        # Update confidence average
        if patterns:
            avg_confidence = statistics.mean([p.confidence_score for p in patterns])
            self.metrics.pattern_confidence_avg = (
                self.metrics.pattern_confidence_avg * 0.9 + avg_confidence * 0.1
            )

        # Update allocation success rate
        if allocations:
            self.metrics.resource_allocation_success_rate = (
                self.metrics.resource_allocation_success_rate * 0.9 + 1.0 * 0.1
            )

        # Update average pattern duration (simplified)
        if patterns:
            avg_duration = statistics.mean([p.duration_estimate for p in patterns])
            self.metrics.average_pattern_duration = (
                self.metrics.average_pattern_duration * 0.9 + avg_duration * 0.1
            )

        self.metrics.last_updated = time.time()

    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get comprehensive pattern analysis."""
        with self._lock:
            analysis = {
                "current_patterns": [
                    {
                        "pattern_type": pattern.pattern_type.value,
                        "intensity_score": pattern.intensity_score,
                        "predictability_score": pattern.predictability_score,
                        "confidence_score": pattern.confidence_score,
                        "resource_demand": {k.value: v for k, v in pattern.resource_demand.items()},
                        "temporal_pattern": pattern.temporal_pattern,
                        "duration_estimate": pattern.duration_estimate
                    }
                    for pattern in self.detected_patterns[-5:]  # Last 5 patterns
                ],
                "pattern_metrics": {
                    "total_patterns_detected": self.metrics.total_patterns_detected,
                    "average_confidence": self.metrics.pattern_confidence_avg,
                    "allocation_success_rate": self.metrics.resource_allocation_success_rate,
                    "average_pattern_duration": self.metrics.average_pattern_duration
                },
                "resource_allocations": {
                    resource.value: allocation
                    for resource, allocation in self.resource_allocations.items()
                },
                "pattern_distribution": self._analyze_pattern_distribution(),
                "resource_optimization_opportunities": self._identify_resource_optimization_opportunities(),
                "predictive_insights": self._generate_predictive_insights(),
                "circuit_breaker_status": self.get_circuit_breaker_status(),
                "timeout_metrics": self.get_timeout_metrics(),
                "generated_at": time.time()
            }

            return analysis

    def _analyze_pattern_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of detected patterns."""
        if not self.detected_patterns:
            return {"distribution": "no_patterns"}

        pattern_counts = defaultdict(int)
        for pattern in self.detected_patterns:
            pattern_counts[pattern.pattern_type.value] += 1

        total_patterns = len(self.detected_patterns)
        distribution = {}
        for pattern_type, count in pattern_counts.items():
            distribution[pattern_type] = {
                "count": count,
                "percentage": (count / total_patterns) * 100
            }

        return distribution

    def _identify_resource_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify resource optimization opportunities based on patterns."""
        opportunities = []

        # Analyze resource utilization patterns
        if len(self.allocation_history) >= 5:
            # Check for over-allocation
            resource_usage = defaultdict(list)
            for allocation in self.allocation_history[-10:]:
                resource_usage[allocation.resource_type].append(allocation.recommended_allocation)

            for resource_type, allocations in resource_usage.items():
                if len(allocations) >= 3:
                    avg_allocation = statistics.mean(allocations)
                    max_allocation = max(allocations)
                    min_allocation = min(allocations)

                    # Check for significant variance
                    if max_allocation > avg_allocation * 1.5:
                        opportunities.append({
                            "type": "resource_overallocation",
                            "resource": resource_type.value,
                            "description": f"Significant variance in {resource_type.value} allocation",
                            "recommendation": "Implement more stable allocation strategy",
                            "potential_savings": (max_allocation - avg_allocation) * 0.3
                        })

        # Check for pattern-based optimization opportunities
        pattern_distribution = self._analyze_pattern_distribution()
        if "bursty" in pattern_distribution and pattern_distribution["bursty"]["percentage"] > 50:
            opportunities.append({
                "type": "bursty_workload_optimization",
                "description": "High percentage of bursty workloads detected",
                "recommendation": "Implement burst-aware resource allocation",
                "potential_benefit": "reduced_resource_waste"
            })

        return opportunities

    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights based on pattern analysis."""
        insights = {
            "next_likely_pattern": None,
            "resource_forecast": {},
            "pattern_trends": {},
            "recommendations": []
        }

        if len(self.detected_patterns) >= 3:
            # Predict next pattern based on recent history
            recent_patterns = self.detected_patterns[-3:]
            pattern_types = [p.pattern_type for p in recent_patterns]

            # Simple prediction: most common recent pattern
            most_common = max(set(pattern_types), key=pattern_types.count)
            insights["next_likely_pattern"] = most_common.value

            # Generate resource forecast
            insights["resource_forecast"] = self._forecast_resource_needs(most_common)

        # Analyze pattern trends
        if len(self.detected_patterns) >= 10:
            insights["pattern_trends"] = self._analyze_pattern_trends()

        # Generate recommendations
        insights["recommendations"] = self._generate_pattern_recommendations()

        return insights

    def _forecast_resource_needs(self, pattern_type: WorkloadPattern) -> Dict[str, float]:
        """Forecast resource needs for a pattern type."""
        # Simplified forecasting based on pattern characteristics
        forecasts = {}

        if pattern_type == WorkloadPattern.COMPUTE_INTENSIVE:
            forecasts = {
                "cpu_cores": 4.0,
                "memory_mb": 4096,
                "cache_size": 10000
            }
        elif pattern_type == WorkloadPattern.MEMORY_INTENSIVE:
            forecasts = {
                "cpu_cores": 2.0,
                "memory_mb": 8192,
                "cache_size": 5000
            }
        elif pattern_type == WorkloadPattern.IO_INTENSIVE:
            forecasts = {
                "cpu_cores": 2.0,
                "memory_mb": 2048,
                "disk_io": 100.0
            }

        return forecasts

    def _analyze_pattern_trends(self) -> Dict[str, Any]:
        """Analyze trends in pattern detection."""
        if len(self.detected_patterns) < 10:
            return {"trend": "insufficient_data"}

        # Analyze confidence trend
        confidences = [p.confidence_score for p in self.detected_patterns[-20:]]
        if len(confidences) >= 5:
            first_half = confidences[:len(confidences)//2]
            second_half = confidences[len(confidences)//2:]
            confidence_trend = "improving" if statistics.mean(second_half) > statistics.mean(first_half) else "declining"
        else:
            confidence_trend = "stable"

        return {
            "confidence_trend": confidence_trend,
            "average_confidence": statistics.mean(confidences),
            "pattern_detection_frequency": len(self.detected_patterns) / max(1, (time.time() - self.detected_patterns[0].detected_at) / 3600)  # patterns per hour
        }

    def _generate_pattern_recommendations(self) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []

        # Pattern diversity recommendations
        pattern_distribution = self._analyze_pattern_distribution()
        if len(pattern_distribution) < 3:
            recommendations.append("Low pattern diversity - consider workload diversification")

        # Resource allocation recommendations
        if self.metrics.resource_allocation_success_rate < 0.8:
            recommendations.append("Resource allocation success rate could be improved")

        # Predictive recommendations
        if len(self.detected_patterns) >= 5:
            recommendations.append("Implement predictive resource allocation based on pattern history")

        return recommendations

    def stop_pattern_recognition(self) -> None:
        """Stop the pattern recognition system."""
        self._recognition_active = False
        if self._recognition_thread:
            self._recognition_thread.join(timeout=5.0)
        logger.info("Workload pattern recognition stopped")

    # Circuit Breaker Methods
    def check_circuit_breaker(self, operation_type: str) -> bool:
        """Check if circuit breaker allows the operation."""
        stats = self.circuit_breaker[operation_type]
        current_time = time.time()

        if stats.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            from codesage_mcp.config.config import CIRCUIT_BREAKER_RECOVERY_TIMEOUT
            if (stats.last_failure_time and
                current_time - stats.last_failure_time >= CIRCUIT_BREAKER_RECOVERY_TIMEOUT):
                # Transition to half-open
                stats.state = CircuitBreakerState.HALF_OPEN
                stats.state_changed_at = current_time
                logger.info(f"Circuit breaker for {operation_type} transitioned to HALF_OPEN")
                return True
            return False

        return True

    def record_operation_result(self, operation_type: str, success: bool,
                               response_time: Optional[float] = None) -> None:
        """Record the result of an operation for circuit breaker."""
        stats = self.circuit_breaker[operation_type]
        current_time = time.time()

        if success:
            stats.success_count += 1
            stats.last_success_time = current_time

            # Transition from half-open to closed on success
            if stats.state == CircuitBreakerState.HALF_OPEN:
                from codesage_mcp.config.config import CIRCUIT_BREAKER_SUCCESS_THRESHOLD
                if stats.success_count >= CIRCUIT_BREAKER_SUCCESS_THRESHOLD:
                    stats.state = CircuitBreakerState.CLOSED
                    stats.failure_count = 0  # Reset failure count
                    stats.state_changed_at = current_time
                    logger.info(f"Circuit breaker for {operation_type} transitioned to CLOSED")

            # Update timeout metrics
            if response_time and operation_type in self.timeout_metrics:
                metrics = self.timeout_metrics[OperationType(operation_type)]
                metrics.success_count += 1
                # Update rolling average response time
                if metrics.avg_response_time == 0:
                    metrics.avg_response_time = response_time
                else:
                    metrics.avg_response_time = (metrics.avg_response_time * 0.9 + response_time * 0.1)
                metrics.last_updated = current_time

        else:
            stats.failure_count += 1
            stats.last_failure_time = current_time

            # Check if should open circuit
            from codesage_mcp.config.config import CIRCUIT_BREAKER_FAILURE_THRESHOLD
            if (stats.state == CircuitBreakerState.CLOSED and
                stats.failure_count >= CIRCUIT_BREAKER_FAILURE_THRESHOLD):
                stats.state = CircuitBreakerState.OPEN
                stats.state_changed_at = current_time
                logger.warning(f"Circuit breaker for {operation_type} transitioned to OPEN")

            # Update timeout metrics
            if operation_type in self.timeout_metrics:
                self.timeout_metrics[OperationType(operation_type)].failure_count += 1

    # Adaptive Timeout Methods
    def get_adaptive_timeout(self, operation_type: OperationType,
                           current_load_factor: float = 1.0) -> float:
        """Get adaptive timeout based on operation type and current load."""
        if operation_type not in self.timeout_metrics:
            # Fallback to config defaults
            from codesage_mcp.config.config import (
                FAST_OPERATION_TIMEOUT, MEDIUM_OPERATION_TIMEOUT,
                SLOW_OPERATION_TIMEOUT, VERY_SLOW_OPERATION_TIMEOUT
            )
            defaults = {
                OperationType.FAST: FAST_OPERATION_TIMEOUT,
                OperationType.MEDIUM: MEDIUM_OPERATION_TIMEOUT,
                OperationType.SLOW: SLOW_OPERATION_TIMEOUT,
                OperationType.VERY_SLOW: VERY_SLOW_OPERATION_TIMEOUT
            }
            return defaults.get(operation_type, 10.0)

        metrics = self.timeout_metrics[operation_type]
        base_timeout = metrics.requested_timeout

        # Apply adaptive adjustments
        from codesage_mcp.config.config import ADAPTIVE_TIMEOUT_ENABLED, ADAPTIVE_TIMEOUT_LOAD_FACTOR

        if ADAPTIVE_TIMEOUT_ENABLED:
            # Adjust based on current load
            if current_load_factor > 1.0:
                base_timeout *= ADAPTIVE_TIMEOUT_LOAD_FACTOR

            # Adjust based on recent performance
            if metrics.success_count > 0 and metrics.failure_count > 0:
                success_rate = metrics.success_count / (metrics.success_count + metrics.failure_count)
                if success_rate < 0.8:  # Low success rate
                    base_timeout *= 1.2  # Increase timeout
                elif success_rate > 0.95:  # High success rate
                    base_timeout *= 0.9  # Decrease timeout slightly

            # Adjust based on average response time
            if metrics.avg_response_time > 0:
                response_ratio = metrics.avg_response_time / base_timeout
                if response_ratio > 0.8:  # Close to timeout
                    base_timeout *= 1.1  # Increase timeout

        # Ensure reasonable bounds
        min_timeout = 0.1  # 100ms minimum
        max_timeout = 120.0  # 2 minutes maximum
        adaptive_timeout = max(min_timeout, min(max_timeout, base_timeout))

        # Update actual timeout in metrics
        metrics.actual_timeout = adaptive_timeout
        metrics.last_updated = time.time()

        return adaptive_timeout

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all operation types."""
        status = {}
        for op_type, stats in self.circuit_breaker.items():
            status[op_type] = {
                "state": stats.state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "last_failure_time": stats.last_failure_time,
                "last_success_time": stats.last_success_time,
                "state_changed_at": stats.state_changed_at
            }
        return status

    def get_timeout_metrics(self) -> Dict[str, Any]:
        """Get timeout metrics for all operation types."""
        metrics = {}
        for op_type, timeout_metrics in self.timeout_metrics.items():
            metrics[op_type.value] = {
                "requested_timeout": timeout_metrics.requested_timeout,
                "actual_timeout": timeout_metrics.actual_timeout,
                "success_count": timeout_metrics.success_count,
                "failure_count": timeout_metrics.failure_count,
                "avg_response_time": timeout_metrics.avg_response_time,
                "last_updated": timeout_metrics.last_updated
            }
        return metrics


class PatternClassifier:
    """ML-based pattern classification system."""

    def __init__(self):
        self.classification_model: Dict[str, Any] = {}
        self.training_data: List[Dict[str, Any]] = []

    def classify_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a workload pattern using ML-based approach."""
        # Enhanced pattern classification
        base_pattern = pattern_data["type"]
        characteristics = pattern_data["characteristics"]
        confidence = pattern_data["confidence"]

        # Determine intensity score
        intensity_factors = {
            "cpu_percent": characteristics["cpu_percent"] / 100.0,
            "memory_percent": characteristics["memory_percent"] / 100.0,
            "disk_io_percent": characteristics["disk_io_percent"] / 100.0,
            "network_io_percent": characteristics["network_io_percent"] / 100.0,
            "request_rate": min(1.0, characteristics["request_rate"] / 2000),
            "workload_intensity": characteristics["workload_intensity"]
        }

        intensity_score = statistics.mean(intensity_factors.values())

        # Determine predictability (simplified)
        predictability_score = 0.8  # Placeholder

        # Estimate resource demand
        resource_demand = self._estimate_resource_demand(base_pattern, characteristics)

        # Determine temporal pattern
        temporal_pattern = self._classify_temporal_pattern(characteristics)

        # Estimate duration
        duration_estimate = self._estimate_pattern_duration(base_pattern, intensity_score)

        return {
            "pattern_type": base_pattern,
            "intensity_score": intensity_score,
            "predictability_score": predictability_score,
            "resource_demand": resource_demand,
            "temporal_pattern": temporal_pattern,
            "duration_estimate": duration_estimate,
            "confidence_score": confidence
        }

    def _estimate_resource_demand(self, pattern_type: WorkloadPattern,
                                characteristics: Dict[str, Any]) -> Dict[ResourceType, float]:
        """Estimate resource demand for a pattern."""
        demand = {}

        if pattern_type == WorkloadPattern.COMPUTE_INTENSIVE:
            demand[ResourceType.CPU_CORES] = 4.0
            demand[ResourceType.MEMORY_MB] = 4096
            demand[ResourceType.CACHE_SIZE] = 10000
        elif pattern_type == WorkloadPattern.MEMORY_INTENSIVE:
            demand[ResourceType.CPU_CORES] = 2.0
            demand[ResourceType.MEMORY_MB] = 8192
            demand[ResourceType.CACHE_SIZE] = 5000
        elif pattern_type == WorkloadPattern.IO_INTENSIVE:
            demand[ResourceType.CPU_CORES] = 2.0
            demand[ResourceType.MEMORY_MB] = 2048
            demand[ResourceType.DISK_IO] = 100.0
        else:
            demand[ResourceType.CPU_CORES] = 2.0
            demand[ResourceType.MEMORY_MB] = 4096
            demand[ResourceType.CACHE_SIZE] = 5000

        return demand

    def _classify_temporal_pattern(self, characteristics: Dict[str, Any]) -> str:
        """Classify temporal pattern of workload."""
        # Simplified temporal classification
        current_hour = time.localtime().tm_hour

        if 9 <= current_hour <= 17:
            return "business_hours"
        elif 18 <= current_hour <= 23:
            return "evening"
        else:
            return "off_hours"

    def _estimate_pattern_duration(self, pattern_type: WorkloadPattern, intensity_score: float) -> int:
        """Estimate duration of a pattern in minutes."""
        base_duration = {
            WorkloadPattern.COMPUTE_INTENSIVE: 30,
            WorkloadPattern.MEMORY_INTENSIVE: 45,
            WorkloadPattern.IO_INTENSIVE: 20,
            WorkloadPattern.BURSTY: 10,
            WorkloadPattern.STEADY_STATE: 120
        }

        duration = base_duration.get(pattern_type, 30)
        # Adjust based on intensity
        duration = int(duration * (1 + intensity_score))

        return duration


class ResourceAllocator:
    """Optimal resource allocation based on patterns."""

    def __init__(self):
        self.allocation_strategies: Dict[WorkloadPattern, Dict[str, Any]] = {}
        self._initialize_allocation_strategies()

    def _initialize_allocation_strategies(self) -> None:
        """Initialize resource allocation strategies for different patterns."""
        self.allocation_strategies = {
            WorkloadPattern.COMPUTE_INTENSIVE: {
                "cpu_priority": 0.8,
                "memory_priority": 0.4,
                "cache_priority": 0.6,
                "scaling_factor": 1.5
            },
            WorkloadPattern.MEMORY_INTENSIVE: {
                "cpu_priority": 0.3,
                "memory_priority": 0.9,
                "cache_priority": 0.7,
                "scaling_factor": 2.0
            },
            WorkloadPattern.IO_INTENSIVE: {
                "cpu_priority": 0.4,
                "memory_priority": 0.5,
                "disk_priority": 0.8,
                "scaling_factor": 1.2
            },
            WorkloadPattern.BURSTY: {
                "cpu_priority": 0.9,
                "memory_priority": 0.7,
                "cache_priority": 0.8,
                "scaling_factor": 1.8
            },
            WorkloadPattern.STEADY_STATE: {
                "cpu_priority": 0.6,
                "memory_priority": 0.6,
                "cache_priority": 0.5,
                "scaling_factor": 1.0
            }
        }

    def generate_allocations(self, pattern: WorkloadCharacteristics) -> List[ResourceAllocation]:
        """Generate resource allocations for a workload pattern."""
        allocations = []
        strategy = self.allocation_strategies.get(pattern.pattern_type, {})

        if not strategy:
            return allocations

        # Generate CPU allocation
        if ResourceType.CPU_CORES in pattern.resource_demand:
            cpu_allocation = self._generate_cpu_allocation(pattern, strategy)
            allocations.append(cpu_allocation)

        # Generate memory allocation
        if ResourceType.MEMORY_MB in pattern.resource_demand:
            memory_allocation = self._generate_memory_allocation(pattern, strategy)
            allocations.append(memory_allocation)

        # Generate cache allocation
        if ResourceType.CACHE_SIZE in pattern.resource_demand:
            cache_allocation = self._generate_cache_allocation(pattern, strategy)
            allocations.append(cache_allocation)

        return allocations

    def _generate_cpu_allocation(self, pattern: WorkloadCharacteristics,
                               strategy: Dict[str, Any]) -> ResourceAllocation:
        """Generate CPU allocation recommendation."""
        current_allocation = 2.0  # Default
        recommended_allocation = pattern.resource_demand[ResourceType.CPU_CORES] * strategy["scaling_factor"]

        return ResourceAllocation(
            resource_type=ResourceType.CPU_CORES,
            current_allocation=current_allocation,
            recommended_allocation=recommended_allocation,
            allocation_reason=f"CPU-intensive {pattern.pattern_type.value} workload detected",
            expected_benefit=pattern.intensity_score * 15,  # Estimated performance improvement
            implementation_priority=int(pattern.confidence_score * 10),
            rollback_plan={
                "original_allocation": current_allocation,
                "rollback_steps": ["Reduce CPU cores to original allocation", "Monitor performance for 5 minutes"]
            }
        )

    def _generate_memory_allocation(self, pattern: WorkloadCharacteristics,
                                  strategy: Dict[str, Any]) -> ResourceAllocation:
        """Generate memory allocation recommendation."""
        current_allocation = 4096  # Default MB
        recommended_allocation = pattern.resource_demand[ResourceType.MEMORY_MB] * strategy["scaling_factor"]

        return ResourceAllocation(
            resource_type=ResourceType.MEMORY_MB,
            current_allocation=current_allocation,
            recommended_allocation=recommended_allocation,
            allocation_reason=f"Memory-intensive {pattern.pattern_type.value} workload detected",
            expected_benefit=pattern.intensity_score * 20,
            implementation_priority=int(pattern.confidence_score * 10),
            rollback_plan={
                "original_allocation": current_allocation,
                "rollback_steps": ["Reduce memory to original allocation", "Monitor for memory pressure"]
            }
        )

    def _generate_cache_allocation(self, pattern: WorkloadCharacteristics,
                                strategy: Dict[str, Any]) -> ResourceAllocation:
        """Generate cache allocation recommendation."""
        current_allocation = 5000  # Default entries
        recommended_allocation = pattern.resource_demand[ResourceType.CACHE_SIZE] * strategy["scaling_factor"]

        return ResourceAllocation(
            resource_type=ResourceType.CACHE_SIZE,
            current_allocation=current_allocation,
            recommended_allocation=recommended_allocation,
            allocation_reason=f"Cache optimization for {pattern.pattern_type.value} workload",
            expected_benefit=pattern.intensity_score * 10,
            implementation_priority=int(pattern.confidence_score * 8),
            rollback_plan={
                "original_allocation": current_allocation,
                "rollback_steps": ["Revert cache size to original", "Clear cache to ensure consistency"]
            }
        )


class WorkloadForecaster:
    """Predictive workload forecasting."""

    def __init__(self):
        self.forecast_model: Dict[str, Any] = {}
        self.historical_patterns: List[WorkloadCharacteristics] = []

    def forecast_workload(self, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """Forecast workload patterns for the specified time horizon."""
        forecast = {
            "predicted_patterns": [],
            "resource_forecast": {},
            "confidence_intervals": {},
            "time_horizon_hours": time_horizon_hours
        }

        if len(self.historical_patterns) < 5:
            forecast["message"] = "Insufficient historical data for forecasting"
            return forecast

        # Simple forecasting based on pattern history
        pattern_counts = defaultdict(int)
        for pattern in self.historical_patterns[-20:]:  # Last 20 patterns
            pattern_counts[pattern.pattern_type] += 1

        # Predict most likely patterns
        total_patterns = sum(pattern_counts.values())
        for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            probability = count / total_patterns
            forecast["predicted_patterns"].append({
                "pattern_type": pattern_type.value,
                "probability": probability,
                "confidence_interval": (probability * 0.8, probability * 1.2)
            })

        # Generate resource forecast
        forecast["resource_forecast"] = self._forecast_resources(forecast["predicted_patterns"])

        return forecast

    def _forecast_resources(self, predicted_patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Forecast resource needs based on predicted patterns."""
        resource_forecast = defaultdict(float)

        for pattern in predicted_patterns:
            pattern_type = WorkloadPattern(pattern["pattern_type"])
            probability = pattern["probability"]

            # Get typical resource demands for this pattern
            demands = self._get_pattern_resource_demands(pattern_type)

            for resource, demand in demands.items():
                resource_forecast[resource] += demand * probability

        return dict(resource_forecast)

    def _get_pattern_resource_demands(self, pattern_type: WorkloadPattern) -> Dict[str, float]:
        """Get typical resource demands for a pattern type."""
        demands = {
            WorkloadPattern.COMPUTE_INTENSIVE: {
                "cpu_cores": 4.0,
                "memory_mb": 4096,
                "cache_size": 10000
            },
            WorkloadPattern.MEMORY_INTENSIVE: {
                "cpu_cores": 2.0,
                "memory_mb": 8192,
                "cache_size": 5000
            },
            WorkloadPattern.IO_INTENSIVE: {
                "cpu_cores": 2.0,
                "memory_mb": 2048,
                "disk_io": 100.0
            },
            WorkloadPattern.BURSTY: {
                "cpu_cores": 3.0,
                "memory_mb": 6144,
                "cache_size": 8000
            },
            WorkloadPattern.STEADY_STATE: {
                "cpu_cores": 2.0,
                "memory_mb": 4096,
                "cache_size": 5000
            }
        }

        return demands.get(pattern_type, {})


# Global instances
_workload_pattern_recognition: Optional[WorkloadPatternRecognition] = None


def get_workload_pattern_recognition() -> WorkloadPatternRecognition:
    """Get the global workload pattern recognition instance."""
    global _workload_pattern_recognition
    if _workload_pattern_recognition is None:
        _workload_pattern_recognition = WorkloadPatternRecognition()
    return _workload_pattern_recognition