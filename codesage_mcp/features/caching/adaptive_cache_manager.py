"""
Adaptive Cache Manager Module for CodeSage MCP Server.

This module provides dynamic cache sizing and management based on usage patterns,
performance metrics, and system load conditions. It implements intelligent cache
adaptation strategies to optimize memory usage and performance.

Classes:
    AdaptiveCacheManager: Main adaptive cache management class
    CacheSizingStrategy: Strategies for dynamic cache sizing
    PerformanceBasedAdapter: Performance-driven cache adaptation
    UsagePatternAdapter: Usage pattern-driven cache adaptation
    LoadAwareAdapter: Load-aware cache adaptation
"""

import logging
import time
import threading
import statistics
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Cache adaptation strategies."""
    PERFORMANCE_BASED = "performance_based"
    USAGE_PATTERN_BASED = "usage_pattern_based"
    LOAD_AWARE = "load_aware"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


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


class CacheSizingDecision(Enum):
    """Cache sizing decisions."""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    RESET = "reset"


@dataclass
class CacheMetrics:
    """Cache performance metrics snapshot."""
    cache_type: str
    size: int
    hit_rate: float
    miss_rate: float
    memory_usage_mb: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
    invalidation_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptationDecision:
    """Cache adaptation decision."""
    cache_type: str
    decision: CacheSizingDecision
    current_size: int
    recommended_size: int
    size_change_mb: int
    confidence: float
    reason: str
    expected_impact: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheAdaptationRule:
    """Rule for cache adaptation."""
    rule_id: str
    cache_type: str
    condition: str
    action: str
    priority: int
    cooldown_minutes: int
    last_applied: float = 0
    success_rate: float = 0.0
    performance_impact: float = 0.0


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


class AdaptiveCacheManager:
    """Main adaptive cache management class."""

    def __init__(self, adaptation_interval_minutes: int = 5, max_adaptation_rate: float = 0.2):
        self.adaptation_interval_minutes = adaptation_interval_minutes
        self.max_adaptation_rate = max_adaptation_rate  # Maximum size change rate

        # Cache metrics history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Adaptation decisions history
        self.adaptation_history: List[AdaptationDecision] = []

        # Adaptation rules
        self.adaptation_rules: List[CacheAdaptationRule] = self._initialize_adaptation_rules()

        # Control
        self._adaptation_active = False
        self._adaptation_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.adaptation_effectiveness: Dict[str, List[float]] = defaultdict(list)

        # Circuit breaker for timeout protection
        self.circuit_breaker: Dict[str, CircuitBreakerStats] = defaultdict(
            lambda: CircuitBreakerStats(state=CircuitBreakerState.CLOSED)
        )

        # Adaptive timeout tracking
        self.timeout_metrics: Dict[OperationType, TimeoutMetrics] = {}

        # Start adaptive management
        self._start_adaptive_management()

    def _initialize_adaptation_rules(self) -> List[CacheAdaptationRule]:
        """Initialize default cache adaptation rules."""
        rules = [
            # Performance-based rules
            CacheAdaptationRule(
                rule_id="perf_low_hit_rate",
                cache_type="all",
                condition="hit_rate < 0.7 and memory_usage_mb < 800",
                action="increase_size_by_25_percent",
                priority=8,
                cooldown_minutes=10
            ),
            CacheAdaptationRule(
                rule_id="perf_high_hit_rate",
                cache_type="all",
                condition="hit_rate > 0.95 and memory_usage_mb > 200",
                action="decrease_size_by_20_percent",
                priority=6,
                cooldown_minutes=15
            ),
            CacheAdaptationRule(
                rule_id="perf_high_memory_pressure",
                cache_type="all",
                condition="memory_usage_mb > 1000 and hit_rate < 0.8",
                action="decrease_size_by_30_percent",
                priority=9,
                cooldown_minutes=5
            ),

            # Usage pattern-based rules
            CacheAdaptationRule(
                rule_id="usage_burst_detected",
                cache_type="all",
                condition="request_rate_increased_by_50_percent and hit_rate > 0.8",
                action="increase_size_by_15_percent",
                priority=7,
                cooldown_minutes=5
            ),
            CacheAdaptationRule(
                rule_id="usage_low_activity",
                cache_type="all",
                condition="request_rate_decreased_by_50_percent and memory_usage_mb > 300",
                action="decrease_size_by_25_percent",
                priority=5,
                cooldown_minutes=20
            ),

            # Load-aware rules
            CacheAdaptationRule(
                rule_id="load_high_cpu",
                cache_type="embedding",
                condition="cpu_usage_percent > 85 and hit_rate < 0.9",
                action="decrease_size_by_20_percent",
                priority=8,
                cooldown_minutes=3
            ),
            CacheAdaptationRule(
                rule_id="load_low_memory",
                cache_type="all",
                condition="system_memory_percent > 90",
                action="decrease_size_by_40_percent",
                priority=10,
                cooldown_minutes=2
            ),

            # Predictive rules
            CacheAdaptationRule(
                rule_id="predict_trending_up",
                cache_type="all",
                condition="usage_trend == 'increasing' and hit_rate > 0.85",
                action="increase_size_by_10_percent",
                priority=6,
                cooldown_minutes=30
            ),
            CacheAdaptationRule(
                rule_id="predict_trending_down",
                cache_type="all",
                condition="usage_trend == 'decreasing' and memory_usage_mb > 400",
                action="decrease_size_by_15_percent",
                priority=4,
                cooldown_minutes=45
            )
        ]

        return rules

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

    def _start_adaptive_management(self) -> None:
        """Start the adaptive cache management thread."""
        if self._adaptation_active:
            return

        # Initialize timeout metrics
        self._initialize_timeout_metrics()

        self._adaptation_active = True
        self._adaptation_thread = threading.Thread(
            target=self._adaptive_management_loop,
            daemon=True,
            name="AdaptiveCacheManager"
        )
        self._adaptation_thread.start()
        logger.info("Adaptive cache management started")

    def _adaptive_management_loop(self) -> None:
        """Main adaptive management loop."""
        while self._adaptation_active:
            try:
                self._perform_adaptation_cycle()
                time.sleep(self.adaptation_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in adaptive management loop: {e}")
                time.sleep(60)

    def _perform_adaptation_cycle(self) -> None:
        """Perform one cycle of cache adaptation."""
        with self._lock:
            # Collect current cache metrics
            current_metrics = self._collect_cache_metrics()

            # Evaluate adaptation rules
            applicable_rules = self._evaluate_adaptation_rules(current_metrics)

            # Generate adaptation decisions
            adaptation_decisions = self._generate_adaptation_decisions(applicable_rules, current_metrics)

            # Apply adaptations
            applied_adaptations = self._apply_adaptation_decisions(adaptation_decisions)

            # Track adaptation effectiveness
            self._track_adaptation_effectiveness(applied_adaptations, current_metrics)

            logger.info(f"Adaptive cache management cycle completed: {len(applied_adaptations)} adaptations applied")

    def _collect_cache_metrics(self) -> Dict[str, CacheMetrics]:
        """Collect current cache metrics from all cache types."""
        # This would integrate with the actual cache implementations
        # For now, return mock data structure
        cache_types = ["embedding", "search", "file"]
        metrics = {}

        for cache_type in cache_types:
            # In a real implementation, this would query the actual cache systems
            metrics[cache_type] = CacheMetrics(
                cache_type=cache_type,
                size=1000,  # Mock data
                hit_rate=0.85,
                miss_rate=0.15,
                memory_usage_mb=500,
                avg_hit_latency_ms=5.0,
                avg_miss_latency_ms=50.0,
                invalidation_count=0
            )

        return metrics

    def record_cache_metrics(self, cache_type: str, metrics: Dict[str, Any]) -> None:
        """Record cache metrics for adaptation analysis."""
        with self._lock:
            cache_metrics = CacheMetrics(
                cache_type=cache_type,
                size=metrics.get("size", 0),
                hit_rate=metrics.get("hit_rate", 0.0),
                miss_rate=metrics.get("miss_rate", 1.0 - metrics.get("hit_rate", 0.0)),
                memory_usage_mb=metrics.get("memory_usage_mb", 0),
                avg_hit_latency_ms=metrics.get("avg_hit_latency_ms", 0.0),
                avg_miss_latency_ms=metrics.get("avg_miss_latency_ms", 0.0),
                invalidation_count=metrics.get("invalidation_count", 0)
            )

            self.metrics_history[cache_type].append(cache_metrics)

            # Update performance baseline
            if cache_type not in self.performance_baseline:
                self.performance_baseline[cache_type] = cache_metrics.hit_rate
            else:
                # Exponential moving average
                alpha = 0.1
                self.performance_baseline[cache_type] = (
                    alpha * cache_metrics.hit_rate +
                    (1 - alpha) * self.performance_baseline[cache_type]
                )

    def _evaluate_adaptation_rules(self, current_metrics: Dict[str, CacheMetrics]) -> List[Tuple[CacheAdaptationRule, str]]:
        """Evaluate which adaptation rules are applicable."""
        applicable_rules = []

        for rule in self.adaptation_rules:
            # Check cooldown period
            if time.time() - rule.last_applied < rule.cooldown_minutes * 60:
                continue

            # Evaluate condition for each cache type
            cache_types_to_check = [rule.cache_type] if rule.cache_type != "all" else current_metrics.keys()

            for cache_type in cache_types_to_check:
                if cache_type in current_metrics:
                    metrics = current_metrics[cache_type]

                    if self._evaluate_condition(rule.condition, metrics, cache_type):
                        applicable_rules.append((rule, cache_type))
                        break  # Only apply rule once per cycle

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda x: x[0].priority, reverse=True)

        return applicable_rules

    def _evaluate_condition(self, condition: str, metrics: CacheMetrics, cache_type: str) -> bool:
        """Evaluate a condition expression against cache metrics."""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            if "hit_rate < 0.7" in condition and metrics.hit_rate < 0.7:
                return True
            elif "hit_rate > 0.95" in condition and metrics.hit_rate > 0.95:
                return True
            elif "memory_usage_mb > 1000" in condition and metrics.memory_usage_mb > 1000:
                return True
            elif "memory_usage_mb < 800" in condition and metrics.memory_usage_mb < 800:
                return True
            elif "memory_usage_mb > 200" in condition and metrics.memory_usage_mb > 200:
                return True
            elif "cpu_usage_percent > 85" in condition:
                # This would need to be passed in or retrieved separately
                return False  # Placeholder
            elif "system_memory_percent > 90" in condition:
                # This would need to be retrieved from system metrics
                return False  # Placeholder
            elif "usage_trend == 'increasing'" in condition:
                return self._check_usage_trend(cache_type) == "increasing"
            elif "usage_trend == 'decreasing'" in condition:
                return self._check_usage_trend(cache_type) == "decreasing"

            return False

        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")
            return False

    def _check_usage_trend(self, cache_type: str) -> str:
        """Check usage trend for a cache type."""
        history = list(self.metrics_history[cache_type])
        if len(history) < 5:
            return "stable"

        # Simple trend analysis based on hit rate
        recent_hit_rates = [m.hit_rate for m in history[-5:]]
        first_half = recent_hit_rates[:2]
        second_half = recent_hit_rates[2:]

        first_avg = statistics.mean(first_half) if first_half else 0
        second_avg = statistics.mean(second_half) if second_half else 0

        if second_avg > first_avg + 0.05:
            return "increasing"
        elif second_avg < first_avg - 0.05:
            return "decreasing"
        else:
            return "stable"

    def _generate_adaptation_decisions(self, applicable_rules: List[Tuple[CacheAdaptationRule, str]],
                                     current_metrics: Dict[str, CacheMetrics]) -> List[AdaptationDecision]:
        """Generate adaptation decisions based on applicable rules."""
        decisions = []

        for rule, cache_type in applicable_rules:
            metrics = current_metrics[cache_type]

            # Calculate recommended size change
            size_change_mb = self._calculate_size_change(rule.action, metrics.size)

            # Apply maximum adaptation rate limit
            max_change = int(metrics.size * self.max_adaptation_rate)
            size_change_mb = max(-max_change, min(max_change, size_change_mb))

            recommended_size = max(100, metrics.size + size_change_mb)  # Minimum size of 100MB

            # Determine decision type
            if size_change_mb > 0:
                decision = CacheSizingDecision.INCREASE
            elif size_change_mb < 0:
                decision = CacheSizingDecision.DECREASE
            else:
                decision = CacheSizingDecision.MAINTAIN

            # Calculate confidence and expected impact
            confidence = self._calculate_adaptation_confidence(rule, metrics)
            expected_impact = self._calculate_expected_impact(rule, metrics, size_change_mb)

            adaptation_decision = AdaptationDecision(
                cache_type=cache_type,
                decision=decision,
                current_size=metrics.size,
                recommended_size=recommended_size,
                size_change_mb=size_change_mb,
                confidence=confidence,
                reason=f"Rule '{rule.rule_id}': {rule.condition}",
                expected_impact=expected_impact
            )

            decisions.append(adaptation_decision)

        return decisions

    def _calculate_size_change(self, action: str, current_size: int) -> int:
        """Calculate the size change based on action."""
        if "increase_size_by_25_percent" in action:
            return int(current_size * 0.25)
        elif "increase_size_by_15_percent" in action:
            return int(current_size * 0.15)
        elif "increase_size_by_10_percent" in action:
            return int(current_size * 0.10)
        elif "decrease_size_by_40_percent" in action:
            return int(-current_size * 0.40)
        elif "decrease_size_by_30_percent" in action:
            return int(-current_size * 0.30)
        elif "decrease_size_by_25_percent" in action:
            return int(-current_size * 0.25)
        elif "decrease_size_by_20_percent" in action:
            return int(-current_size * 0.20)
        elif "decrease_size_by_15_percent" in action:
            return int(-current_size * 0.15)
        else:
            return 0

    def _calculate_adaptation_confidence(self, rule: CacheAdaptationRule, metrics: CacheMetrics) -> float:
        """Calculate confidence level for an adaptation decision."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on rule success rate
        confidence += rule.success_rate * 0.3

        # Increase confidence based on metrics clarity
        if metrics.hit_rate > 0.8 or metrics.hit_rate < 0.3:
            confidence += 0.2

        # Increase confidence based on data history
        history_length = len(self.metrics_history[metrics.cache_type])
        confidence += min(0.2, history_length / 50)  # Max 0.2 for 50+ data points

        return min(1.0, confidence)

    def _calculate_expected_impact(self, rule: CacheAdaptationRule, metrics: CacheMetrics,
                                 size_change_mb: int) -> Dict[str, float]:
        """Calculate expected impact of an adaptation."""
        impact = {
            "hit_rate_change": 0.0,
            "latency_change_ms": 0.0,
            "memory_change_mb": size_change_mb,
            "performance_score_change": 0.0
        }

        # Estimate hit rate change based on size change
        size_change_ratio = size_change_mb / metrics.size
        impact["hit_rate_change"] = size_change_ratio * 0.1  # Rough estimate

        # Estimate latency change
        if size_change_mb > 0:
            impact["latency_change_ms"] = -2.0  # Size increase typically reduces latency
        else:
            impact["latency_change_ms"] = 1.0  # Size decrease may increase latency

        # Calculate performance score change
        impact["performance_score_change"] = (
            impact["hit_rate_change"] * 50 -  # Hit rate improvement
            abs(impact["latency_change_ms"]) * 0.5  # Latency penalty
        )

        return impact

    def _apply_adaptation_decisions(self, decisions: List[AdaptationDecision]) -> List[AdaptationDecision]:
        """Apply adaptation decisions to cache systems."""
        applied_decisions = []

        for decision in decisions:
            try:
                # In a real implementation, this would apply the size change to the actual cache
                # For now, we'll just log the decision and mark it as applied

                logger.info(f"Applying cache adaptation: {decision.cache_type} {decision.decision.value} "
                          f"from {decision.current_size}MB to {decision.recommended_size}MB")

                # Update the rule's last applied time
                for rule in self.adaptation_rules:
                    if rule.rule_id in decision.reason:
                        rule.last_applied = time.time()
                        break

                applied_decisions.append(decision)
                self.adaptation_history.append(decision)

            except Exception as e:
                logger.error(f"Error applying adaptation decision for {decision.cache_type}: {e}")

        return applied_decisions

    def _track_adaptation_effectiveness(self, applied_decisions: List[AdaptationDecision],
                                      current_metrics: Dict[str, CacheMetrics]) -> None:
        """Track the effectiveness of applied adaptations."""
        for decision in applied_decisions:
            if decision.cache_type in current_metrics:
                # Calculate effectiveness based on performance change
                # This is a simplified implementation
                effectiveness_score = decision.confidence * 0.8  # Simplified

                self.adaptation_effectiveness[decision.cache_type].append(effectiveness_score)

                # Keep only recent effectiveness scores
                if len(self.adaptation_effectiveness[decision.cache_type]) > 10:
                    self.adaptation_effectiveness[decision.cache_type] = self.adaptation_effectiveness[decision.cache_type][-10:]

                # Update rule success rate
                for rule in self.adaptation_rules:
                    if rule.rule_id in decision.reason:
                        recent_effectiveness = self.adaptation_effectiveness[decision.cache_type][-5:]
                        if recent_effectiveness:
                            rule.success_rate = statistics.mean(recent_effectiveness)
                        break

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and history."""
        with self._lock:
            recent_decisions = self.adaptation_history[-10:]  # Last 10 decisions

            status = {
                "adaptation_active": self._adaptation_active,
                "adaptation_interval_minutes": self.adaptation_interval_minutes,
                "max_adaptation_rate": self.max_adaptation_rate,
                "total_adaptations_applied": len(self.adaptation_history),
                "recent_decisions": [
                    {
                        "cache_type": d.cache_type,
                        "decision": d.decision.value,
                        "size_change_mb": d.size_change_mb,
                        "confidence": d.confidence,
                        "reason": d.reason,
                        "timestamp": d.timestamp
                    }
                    for d in recent_decisions
                ],
                "adaptation_rules_status": [
                    {
                        "rule_id": r.rule_id,
                        "cache_type": r.cache_type,
                        "priority": r.priority,
                        "success_rate": r.success_rate,
                        "last_applied_minutes_ago": (time.time() - r.last_applied) / 60 if r.last_applied > 0 else None
                    }
                    for r in self.adaptation_rules
                ],
                "performance_baselines": self.performance_baseline.copy(),
                "circuit_breaker_status": self.get_circuit_breaker_status(),
                "timeout_metrics": self.get_timeout_metrics(),
                "generated_at": time.time()
            }

            return status

    def stop_adaptive_management(self) -> None:
        """Stop the adaptive cache management thread."""
        self._adaptation_active = False
        if self._adaptation_thread:
            self._adaptation_thread.join(timeout=5.0)
        logger.info("Adaptive cache management stopped")

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


class CacheSizingStrategy:
    """Strategies for dynamic cache sizing."""

    def __init__(self, strategy_type: AdaptationStrategy):
        self.strategy_type = strategy_type
        self.parameters: Dict[str, Any] = {}

    def calculate_optimal_size(self, current_metrics: CacheMetrics,
                             usage_patterns: Dict[str, Any]) -> int:
        """Calculate optimal cache size based on strategy."""
        if self.strategy_type == AdaptationStrategy.PERFORMANCE_BASED:
            return self._performance_based_sizing(current_metrics)
        elif self.strategy_type == AdaptationStrategy.USAGE_PATTERN_BASED:
            return self._usage_pattern_based_sizing(current_metrics, usage_patterns)
        elif self.strategy_type == AdaptationStrategy.LOAD_AWARE:
            return self._load_aware_sizing(current_metrics)
        elif self.strategy_type == AdaptationStrategy.PREDICTIVE:
            return self._predictive_sizing(current_metrics, usage_patterns)
        else:
            return current_metrics.size

    def _performance_based_sizing(self, metrics: CacheMetrics) -> int:
        """Performance-based cache sizing strategy."""
        base_size = 500  # Base size in MB

        # Adjust based on hit rate
        if metrics.hit_rate > 0.9:
            performance_multiplier = 0.8  # Can reduce size
        elif metrics.hit_rate > 0.8:
            performance_multiplier = 1.0  # Optimal size
        elif metrics.hit_rate > 0.6:
            performance_multiplier = 1.2  # Need more size
        else:
            performance_multiplier = 1.5  # Significantly more size needed

        # Adjust based on latency
        if metrics.avg_hit_latency_ms > 20:
            latency_multiplier = 1.3
        elif metrics.avg_hit_latency_ms > 10:
            latency_multiplier = 1.1
        else:
            latency_multiplier = 1.0

        optimal_size = int(base_size * performance_multiplier * latency_multiplier)
        return max(100, min(2000, optimal_size))  # Clamp between 100MB and 2000MB

    def _usage_pattern_based_sizing(self, metrics: CacheMetrics,
                                  usage_patterns: Dict[str, Any]) -> int:
        """Usage pattern-based cache sizing strategy."""
        base_size = 500

        # Analyze access patterns
        temporal_patterns = usage_patterns.get("temporal_patterns", {})
        if temporal_patterns:
            # Calculate peak usage times
            peak_usage_factor = len([hits for hits in temporal_patterns.values()
                                   if hits and sum(hits) > 10]) / len(temporal_patterns)
            pattern_multiplier = 1.0 + (peak_usage_factor * 0.5)
        else:
            pattern_multiplier = 1.0

        # Analyze key frequency
        key_frequency = usage_patterns.get("key_frequency", {})
        if key_frequency:
            unique_keys = len(key_frequency)
            key_diversity_factor = min(unique_keys / 100, 2.0)  # Max 2x multiplier
        else:
            key_diversity_factor = 1.0

        optimal_size = int(base_size * pattern_multiplier * key_diversity_factor)
        return max(100, min(2000, optimal_size))

    def _load_aware_sizing(self, metrics: CacheMetrics) -> int:
        """Load-aware cache sizing strategy."""
        base_size = 500

        # This would integrate with system load metrics
        # For now, use memory usage as a proxy for load
        if metrics.memory_usage_mb > 1000:
            load_multiplier = 0.7  # Reduce cache under high memory pressure
        elif metrics.memory_usage_mb > 700:
            load_multiplier = 0.9  # Slightly reduce cache
        elif metrics.memory_usage_mb < 300:
            load_multiplier = 1.3  # Can increase cache
        else:
            load_multiplier = 1.0  # Normal load

        optimal_size = int(base_size * load_multiplier)
        return max(100, min(2000, optimal_size))

    def _predictive_sizing(self, metrics: CacheMetrics, usage_patterns: Dict[str, Any]) -> int:
        """Predictive cache sizing strategy."""
        # This would implement predictive modeling
        # For now, use a simple approach based on trends

        current_size = metrics.size
        hit_rate_trend = usage_patterns.get("hit_rate_trend", 0)

        if hit_rate_trend > 0.05:  # Hit rate improving
            predictive_multiplier = 1.1
        elif hit_rate_trend < -0.05:  # Hit rate declining
            predictive_multiplier = 0.9
        else:
            predictive_multiplier = 1.0

        optimal_size = int(current_size * predictive_multiplier)
        return max(100, min(2000, optimal_size))


# Global instances
_adaptive_cache_manager: Optional[AdaptiveCacheManager] = None


def get_adaptive_cache_manager() -> AdaptiveCacheManager:
    """Get the global adaptive cache manager instance."""
    global _adaptive_cache_manager
    if _adaptive_cache_manager is None:
        _adaptive_cache_manager = AdaptiveCacheManager()
    return _adaptive_cache_manager