"""
Workload-Adaptive Memory Management Module for CodeSage MCP Server.

This module provides advanced workload-aware memory management that adapts memory allocation
strategies based on current workload patterns, system load, and performance requirements.

Classes:
    WorkloadAdaptiveMemoryManager: Main workload-adaptive memory management class
    WorkloadAnalyzer: Analyzes current workload characteristics
    MemoryAllocationStrategy: Strategies for optimal memory allocation
    WorkloadPredictor: Predicts future workload patterns
    AdaptiveMemoryAllocator: Allocates memory based on workload analysis
"""

import logging
import time
import threading
import psutil
import statistics
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Types of workload patterns."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED_LOAD = "mixed_load"
    BURSTY = "bursty"
    STEADY_STATE = "steady_state"


class MemoryAllocationStrategy(Enum):
    """Memory allocation strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


@dataclass
class WorkloadMetrics:
    """Current workload metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    active_connections: int
    request_rate: float
    cache_hit_rate: float
    avg_response_time_ms: float
    workload_type: WorkloadType
    intensity_score: float  # 0-1 scale


@dataclass
class MemoryAllocationPlan:
    """Memory allocation plan for different components."""
    timestamp: float
    strategy: MemoryAllocationStrategy
    total_memory_mb: int
    allocation_breakdown: Dict[str, int]  # component -> memory_mb
    expected_performance_impact: Dict[str, float]
    confidence_score: float
    adaptation_reason: str
    rollback_plan: Dict[str, Any]


@dataclass
class WorkloadPrediction:
    """Prediction of future workload characteristics."""
    prediction_timestamp: float
    predicted_workload_type: WorkloadType
    predicted_intensity: float
    confidence_interval: Tuple[float, float]
    time_horizon_minutes: int
    influencing_factors: List[str]


class WorkloadAdaptiveMemoryManager:
    """Main workload-adaptive memory management class."""

    def __init__(self, adaptation_interval_minutes: int = 2, prediction_horizon_minutes: int = 30):
        self.adaptation_interval_minutes = adaptation_interval_minutes
        self.prediction_horizon_minutes = prediction_horizon_minutes

        # Workload tracking
        self.workload_history: deque = deque(maxlen=1000)
        self.current_allocation_plan: Optional[MemoryAllocationPlan] = None
        self.allocation_history: List[MemoryAllocationPlan] = []

        # Prediction and analysis
        self.workload_predictions: List[WorkloadPrediction] = []
        self.performance_baselines: Dict[str, float] = {}

        # Control
        self._adaptation_active = False
        self._adaptation_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Strategies and thresholds
        self.allocation_strategies = self._initialize_allocation_strategies()
        self.adaptation_thresholds = {
            "cpu_threshold_high": 80.0,
            "cpu_threshold_low": 30.0,
            "memory_threshold_high": 85.0,
            "memory_threshold_low": 50.0,
            "performance_degradation_threshold": 15.0,  # percent
            "adaptation_confidence_threshold": 0.7
        }

        # Start adaptive management
        self._start_adaptive_management()

    def _initialize_allocation_strategies(self) -> Dict[WorkloadType, Dict[str, Any]]:
        """Initialize memory allocation strategies for different workload types."""
        return {
            WorkloadType.CPU_INTENSIVE: {
                "strategy": MemoryAllocationStrategy.CONSERVATIVE,
                "cache_allocation_percent": 60,
                "model_allocation_percent": 20,
                "buffer_allocation_percent": 20,
                "gc_frequency": "normal",
                "memory_pressure_threshold": 75
            },
            WorkloadType.MEMORY_INTENSIVE: {
                "strategy": MemoryAllocationStrategy.AGGRESSIVE,
                "cache_allocation_percent": 40,
                "model_allocation_percent": 30,
                "buffer_allocation_percent": 30,
                "gc_frequency": "frequent",
                "memory_pressure_threshold": 80
            },
            WorkloadType.IO_INTENSIVE: {
                "strategy": MemoryAllocationStrategy.BALANCED,
                "cache_allocation_percent": 50,
                "model_allocation_percent": 25,
                "buffer_allocation_percent": 25,
                "gc_frequency": "normal",
                "memory_pressure_threshold": 70
            },
            WorkloadType.NETWORK_INTENSIVE: {
                "strategy": MemoryAllocationStrategy.CONSERVATIVE,
                "cache_allocation_percent": 55,
                "model_allocation_percent": 25,
                "buffer_allocation_percent": 20,
                "gc_frequency": "normal",
                "memory_pressure_threshold": 75
            },
            WorkloadType.MIXED_LOAD: {
                "strategy": MemoryAllocationStrategy.BALANCED,
                "cache_allocation_percent": 45,
                "model_allocation_percent": 30,
                "buffer_allocation_percent": 25,
                "gc_frequency": "adaptive",
                "memory_pressure_threshold": 75
            },
            WorkloadType.BURSTY: {
                "strategy": MemoryAllocationStrategy.PREDICTIVE,
                "cache_allocation_percent": 65,
                "model_allocation_percent": 20,
                "buffer_allocation_percent": 15,
                "gc_frequency": "adaptive",
                "memory_pressure_threshold": 70
            },
            WorkloadType.STEADY_STATE: {
                "strategy": MemoryAllocationStrategy.CONSERVATIVE,
                "cache_allocation_percent": 50,
                "model_allocation_percent": 25,
                "buffer_allocation_percent": 25,
                "gc_frequency": "conservative",
                "memory_pressure_threshold": 80
            }
        }

    def _start_adaptive_management(self) -> None:
        """Start the adaptive memory management thread."""
        if self._adaptation_active:
            return

        self._adaptation_active = True
        self._adaptation_thread = threading.Thread(
            target=self._adaptive_management_loop,
            daemon=True,
            name="WorkloadAdaptiveMemoryManager"
        )
        self._adaptation_thread.start()
        logger.info("Workload-adaptive memory management started")

    def _adaptive_management_loop(self) -> None:
        """Main adaptive management loop."""
        while self._adaptation_active:
            try:
                self._perform_workload_adaptation()
                time.sleep(self.adaptation_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in workload adaptive management loop: {e}")
                time.sleep(60)

    def _perform_workload_adaptation(self) -> None:
        """Perform workload-based memory adaptation."""
        with self._lock:
            # Analyze current workload
            current_workload = self._analyze_current_workload()

            # Predict future workload
            workload_prediction = self._predict_future_workload()

            # Determine optimal allocation strategy
            optimal_strategy = self._determine_optimal_strategy(current_workload, workload_prediction)

            # Generate allocation plan
            allocation_plan = self._generate_allocation_plan(current_workload, optimal_strategy)

            # Check if adaptation is needed
            if self._should_adapt_memory(allocation_plan):
                # Apply adaptation
                success = self._apply_allocation_plan(allocation_plan)

                if success:
                    self.current_allocation_plan = allocation_plan
                    self.allocation_history.append(allocation_plan)
                    logger.info(f"Applied workload-adaptive memory allocation: {allocation_plan.strategy.value}")
                else:
                    logger.warning("Failed to apply memory allocation plan")

            # Clean old history
            self._cleanup_history()

    def _analyze_current_workload(self) -> WorkloadMetrics:
        """Analyze current system workload."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=5.0)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            # Get I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_io_percent = (disk_io.read_bytes + disk_io.write_bytes) / max(1, disk_io.read_time + disk_io.write_time) if disk_io else 0

            network_io = psutil.net_io_counters()
            network_io_percent = (network_io.bytes_sent + network_io.bytes_recv) / max(1, 1000000)  # Simplified

            # Get process-specific metrics
            process = psutil.Process()
            connections = process.connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])

            # Calculate workload characteristics
            workload_type = self._classify_workload_type(cpu_percent, memory_percent, disk_io_percent, network_io_percent)
            intensity_score = self._calculate_workload_intensity(cpu_percent, memory_percent, disk_io_percent, network_io_percent)

            # Get performance metrics (would integrate with performance monitor)
            request_rate = 100  # Mock data - would come from performance monitor
            cache_hit_rate = 0.85  # Mock data
            avg_response_time_ms = 25.0  # Mock data

            workload_metrics = WorkloadMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io_percent=disk_io_percent,
                network_io_percent=network_io_percent,
                active_connections=active_connections,
                request_rate=request_rate,
                cache_hit_rate=cache_hit_rate,
                avg_response_time_ms=avg_response_time_ms,
                workload_type=workload_type,
                intensity_score=intensity_score
            )

            # Store in history
            self.workload_history.append(workload_metrics)

            return workload_metrics

        except Exception as e:
            logger.warning(f"Error analyzing current workload: {e}")
            # Return default metrics
            return WorkloadMetrics(
                timestamp=time.time(),
                cpu_percent=50.0,
                memory_percent=60.0,
                disk_io_percent=10.0,
                network_io_percent=5.0,
                active_connections=10,
                request_rate=50,
                cache_hit_rate=0.8,
                avg_response_time_ms=30.0,
                workload_type=WorkloadType.MIXED_LOAD,
                intensity_score=0.5
            )

    def _classify_workload_type(self, cpu_percent: float, memory_percent: float,
                               disk_io_percent: float, network_io_percent: float) -> WorkloadType:
        """Classify the current workload type based on system metrics."""
        # Determine dominant resource usage
        metrics = {
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_io_percent * 100,  # Normalize
            "network": network_io_percent * 100  # Normalize
        }

        max_resource = max(metrics.items(), key=lambda x: x[1])

        # Classify based on dominant resource and patterns
        if max_resource[0] == "cpu" and cpu_percent > 70:
            return WorkloadType.CPU_INTENSIVE
        elif max_resource[0] == "memory" and memory_percent > 75:
            return WorkloadType.MEMORY_INTENSIVE
        elif max_resource[0] == "disk" and disk_io_percent > 50:
            return WorkloadType.IO_INTENSIVE
        elif max_resource[0] == "network" and network_io_percent > 30:
            return WorkloadType.NETWORK_INTENSIVE
        elif self._is_bursty_workload():
            return WorkloadType.BURSTY
        elif self._is_steady_state_workload():
            return WorkloadType.STEADY_STATE
        else:
            return WorkloadType.MIXED_LOAD

    def _calculate_workload_intensity(self, cpu_percent: float, memory_percent: float,
                                    disk_io_percent: float, network_io_percent: float) -> float:
        """Calculate workload intensity score (0-1)."""
        # Weighted combination of resource usages
        intensity = (
            cpu_percent * 0.4 +
            memory_percent * 0.3 +
            disk_io_percent * 100 * 0.2 +
            network_io_percent * 100 * 0.1
        ) / 100.0

        return min(1.0, intensity)

    def _is_bursty_workload(self) -> bool:
        """Determine if current workload is bursty."""
        if len(self.workload_history) < 5:
            return False

        recent_intensities = [w.intensity_score for w in list(self.workload_history)[-5:]]
        intensity_variance = statistics.variance(recent_intensities) if len(recent_intensities) > 1 else 0

        return intensity_variance > 0.1  # High variance indicates bursty workload

    def _is_steady_state_workload(self) -> bool:
        """Determine if current workload is steady state."""
        if len(self.workload_history) < 10:
            return False

        recent_intensities = [w.intensity_score for w in list(self.workload_history)[-10:]]
        intensity_variance = statistics.variance(recent_intensities) if len(recent_intensities) > 1 else 0

        return intensity_variance < 0.05  # Low variance indicates steady state

    def _predict_future_workload(self) -> WorkloadPrediction:
        """Predict future workload characteristics."""
        if len(self.workload_history) < 5:
            # Not enough data for prediction
            return WorkloadPrediction(
                prediction_timestamp=time.time(),
                predicted_workload_type=WorkloadType.MIXED_LOAD,
                predicted_intensity=0.5,
                confidence_interval=(0.4, 0.6),
                time_horizon_minutes=self.prediction_horizon_minutes,
                influencing_factors=["insufficient_data"]
            )

        # Simple prediction based on recent trends
        recent_workloads = list(self.workload_history)[-10:]
        recent_intensities = [w.intensity_score for w in recent_workloads]

        # Linear trend prediction
        if len(recent_intensities) >= 2:
            slope = statistics.linear_regression(range(len(recent_intensities)), recent_intensities)[0]
            predicted_intensity = recent_intensities[-1] + slope * (self.prediction_horizon_minutes / 10)
            predicted_intensity = max(0.0, min(1.0, predicted_intensity))
        else:
            predicted_intensity = statistics.mean(recent_intensities)

        # Predict workload type based on most common recent type
        recent_types = [w.workload_type for w in recent_workloads[-5:]]
        predicted_type = max(set(recent_types), key=recent_types.count)

        # Calculate confidence interval
        intensity_std = statistics.stdev(recent_intensities) if len(recent_intensities) > 1 else 0.1
        confidence_interval = (
            max(0, predicted_intensity - intensity_std),
            min(1, predicted_intensity + intensity_std)
        )

        return WorkloadPrediction(
            prediction_timestamp=time.time(),
            predicted_workload_type=predicted_type,
            predicted_intensity=predicted_intensity,
            confidence_interval=confidence_interval,
            time_horizon_minutes=self.prediction_horizon_minutes,
            influencing_factors=["trend_analysis", "historical_patterns"]
        )

    def _determine_optimal_strategy(self, current_workload: WorkloadMetrics,
                                   prediction: WorkloadPrediction) -> MemoryAllocationStrategy:
        """Determine optimal memory allocation strategy."""
        # Get strategy for current workload type
        current_strategy_config = self.allocation_strategies.get(current_workload.workload_type, {})
        base_strategy = current_strategy_config.get("strategy", MemoryAllocationStrategy.BALANCED)

        # Adjust based on prediction
        if prediction.predicted_intensity > 0.8:
            return MemoryAllocationStrategy.AGGRESSIVE
        elif prediction.predicted_intensity < 0.3:
            return MemoryAllocationStrategy.CONSERVATIVE
        elif prediction.predicted_workload_type != current_workload.workload_type:
            return MemoryAllocationStrategy.ADAPTIVE
        else:
            return base_strategy

    def _generate_allocation_plan(self, workload: WorkloadMetrics,
                                strategy: MemoryAllocationStrategy) -> MemoryAllocationPlan:
        """Generate memory allocation plan."""
        # Get total available memory
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.total // (1024 * 1024)

        # Get strategy configuration
        strategy_config = None
        for workload_type, config in self.allocation_strategies.items():
            if config["strategy"] == strategy:
                strategy_config = config
                break

        if not strategy_config:
            strategy_config = self.allocation_strategies[WorkloadType.MIXED_LOAD]

        # Calculate allocation breakdown
        allocation_breakdown = {
            "cache": int(total_memory_mb * strategy_config["cache_allocation_percent"] / 100),
            "model": int(total_memory_mb * strategy_config["model_allocation_percent"] / 100),
            "buffer": int(total_memory_mb * strategy_config["buffer_allocation_percent"] / 100)
        }

        # Calculate expected performance impact
        expected_impact = self._calculate_expected_impact(workload, allocation_breakdown)

        # Generate adaptation reason
        adaptation_reason = f"Workload type: {workload.workload_type.value}, Strategy: {strategy.value}"

        # Create rollback plan
        rollback_plan = {
            "previous_allocation": self.current_allocation_plan.allocation_breakdown if self.current_allocation_plan else {},
            "rollback_trigger": "performance_degradation > 20% or memory_pressure > 90%",
            "rollback_steps": ["Revert to previous allocation", "Monitor for 5 minutes", "Re-evaluate if needed"]
        }

        return MemoryAllocationPlan(
            timestamp=time.time(),
            strategy=strategy,
            total_memory_mb=total_memory_mb,
            allocation_breakdown=allocation_breakdown,
            expected_performance_impact=expected_impact,
            confidence_score=self._calculate_allocation_confidence(workload, strategy),
            adaptation_reason=adaptation_reason,
            rollback_plan=rollback_plan
        )

    def _calculate_expected_impact(self, workload: WorkloadMetrics,
                                 allocation_breakdown: Dict[str, int]) -> Dict[str, float]:
        """Calculate expected performance impact of allocation plan."""
        impact = {
            "response_time_change_ms": 0.0,
            "throughput_change_percent": 0.0,
            "memory_efficiency_change_percent": 0.0,
            "cache_hit_rate_change_percent": 0.0
        }

        # Calculate impact based on workload type and allocation changes
        if workload.workload_type == WorkloadType.CPU_INTENSIVE:
            impact["response_time_change_ms"] = -5.0  # Faster response
            impact["throughput_change_percent"] = 10.0
        elif workload.workload_type == WorkloadType.MEMORY_INTENSIVE:
            impact["memory_efficiency_change_percent"] = 15.0
            impact["cache_hit_rate_change_percent"] = 8.0
        elif workload.workload_type == WorkloadType.IO_INTENSIVE:
            impact["response_time_change_ms"] = -8.0
            impact["throughput_change_percent"] = 12.0

        return impact

    def _calculate_allocation_confidence(self, workload: WorkloadMetrics,
                                       strategy: MemoryAllocationStrategy) -> float:
        """Calculate confidence score for allocation plan."""
        confidence = 0.7  # Base confidence

        # Increase confidence based on workload clarity
        if workload.intensity_score > 0.7 or workload.intensity_score < 0.3:
            confidence += 0.1

        # Increase confidence based on historical data
        if len(self.workload_history) > 20:
            confidence += 0.1

        # Increase confidence for well-understood strategies
        if strategy in [MemoryAllocationStrategy.BALANCED, MemoryAllocationStrategy.CONSERVATIVE]:
            confidence += 0.1

        return min(1.0, confidence)

    def _should_adapt_memory(self, allocation_plan: MemoryAllocationPlan) -> bool:
        """Determine if memory adaptation should be applied."""
        # Check confidence threshold
        if allocation_plan.confidence_score < self.adaptation_thresholds["adaptation_confidence_threshold"]:
            return False

        # Check if allocation has changed significantly
        if self.current_allocation_plan:
            current_allocation = self.current_allocation_plan.allocation_breakdown
            new_allocation = allocation_plan.allocation_breakdown

            total_change = sum(abs(new_allocation.get(comp, 0) - current_allocation.get(comp, 0))
                             for comp in ["cache", "model", "buffer"])

            if total_change < 100:  # Less than 100MB change
                return False

        return True

    def _apply_allocation_plan(self, allocation_plan: MemoryAllocationPlan) -> bool:
        """Apply memory allocation plan."""
        try:
            # In a real implementation, this would:
            # 1. Update cache size limits
            # 2. Adjust model cache sizes
            # 3. Modify buffer allocations
            # 4. Update memory management policies

            logger.info(f"Applying memory allocation plan: {allocation_plan.strategy.value}")
            logger.info(f"New allocation: {allocation_plan.allocation_breakdown}")

            # For now, just log the plan
            return True

        except Exception as e:
            logger.exception(f"Error applying memory allocation plan: {e}")
            return False

    def _cleanup_history(self) -> None:
        """Clean up old history data."""
        cutoff_time = time.time() - (24 * 60 * 60)  # Keep 24 hours

        # Clean workload history (already handled by deque maxlen)
        # Clean allocation history
        self.allocation_history = [
            plan for plan in self.allocation_history
            if plan.timestamp > cutoff_time
        ]

        # Clean predictions
        self.workload_predictions = [
            pred for pred in self.workload_predictions
            if pred.prediction_timestamp > cutoff_time
        ]

    def get_workload_analysis(self) -> Dict[str, Any]:
        """Get comprehensive workload analysis."""
        with self._lock:
            if not self.workload_history:
                return {"error": "No workload data available"}

            recent_workloads = list(self.workload_history)[-20:]  # Last 20 data points

            analysis = {
                "current_workload": self._analyze_current_workload().__dict__ if self.workload_history else None,
                "workload_trends": self._analyze_workload_trends(recent_workloads),
                "allocation_effectiveness": self._analyze_allocation_effectiveness(),
                "predictions": self._get_workload_predictions(),
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "generated_at": time.time()
            }

            return analysis

    def _analyze_workload_trends(self, workloads: List[WorkloadMetrics]) -> Dict[str, Any]:
        """Analyze workload trends."""
        if len(workloads) < 2:
            return {"trend": "insufficient_data"}

        intensities = [w.intensity_score for w in workloads]
        cpu_usage = [w.cpu_percent for w in workloads]
        memory_usage = [w.memory_percent for w in workloads]

        # Calculate trends
        intensity_trend = "stable"
        if len(intensities) >= 3:
            first_half = intensities[:len(intensities)//2]
            second_half = intensities[len(intensities)//2:]
            if statistics.mean(second_half) > statistics.mean(first_half) + 0.1:
                intensity_trend = "increasing"
            elif statistics.mean(second_half) < statistics.mean(first_half) - 0.1:
                intensity_trend = "decreasing"

        return {
            "intensity_trend": intensity_trend,
            "avg_intensity": statistics.mean(intensities),
            "avg_cpu_usage": statistics.mean(cpu_usage),
            "avg_memory_usage": statistics.mean(memory_usage),
            "workload_volatility": statistics.stdev(intensities) if len(intensities) > 1 else 0,
            "dominant_workload_types": self._get_dominant_workload_types(workloads)
        }

    def _get_dominant_workload_types(self, workloads: List[WorkloadMetrics]) -> List[Dict[str, Any]]:
        """Get dominant workload types."""
        type_counts = defaultdict(int)
        for workload in workloads:
            type_counts[workload.workload_type.value] += 1

        total = len(workloads)
        dominant_types = []
        for workload_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            dominant_types.append({
                "type": workload_type,
                "count": count,
                "percentage": (count / total) * 100
            })

        return dominant_types[:3]  # Top 3

    def _analyze_allocation_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of memory allocations."""
        if not self.allocation_history:
            return {"effectiveness": "no_allocations"}

        recent_allocations = self.allocation_history[-5:]  # Last 5 allocations

        effectiveness_scores = []
        for allocation in recent_allocations:
            # Simplified effectiveness calculation
            effectiveness = allocation.confidence_score * 0.8  # Mock calculation
            effectiveness_scores.append(effectiveness)

        return {
            "avg_effectiveness": statistics.mean(effectiveness_scores) if effectiveness_scores else 0,
            "allocation_count": len(recent_allocations),
            "best_strategy": max(recent_allocations, key=lambda x: x.confidence_score).strategy.value,
            "effectiveness_trend": "stable"  # Would analyze trend in real implementation
        }

    def _get_workload_predictions(self) -> List[Dict[str, Any]]:
        """Get workload predictions."""
        predictions = []
        for pred in self.workload_predictions[-3:]:  # Last 3 predictions
            predictions.append({
                "predicted_type": pred.predicted_workload_type.value,
                "predicted_intensity": pred.predicted_intensity,
                "confidence_interval": pred.confidence_interval,
                "time_horizon_minutes": pred.time_horizon_minutes,
                "influencing_factors": pred.influencing_factors
            })

        return predictions

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify workload optimization opportunities."""
        opportunities = []

        if len(self.workload_history) < 10:
            return opportunities

        recent_workloads = list(self.workload_history)[-10:]

        # Check for memory pressure issues
        high_memory_workloads = [w for w in recent_workloads if w.memory_percent > 80]
        if len(high_memory_workloads) > 5:
            opportunities.append({
                "type": "memory_pressure_optimization",
                "description": "High memory pressure detected during workload peaks",
                "impact": "high",
                "recommendation": "Implement predictive memory scaling"
            })

        # Check for CPU bottlenecks
        high_cpu_workloads = [w for w in recent_workloads if w.cpu_percent > 85]
        if len(high_cpu_workloads) > 3:
            opportunities.append({
                "type": "cpu_optimization",
                "description": "CPU bottlenecks detected during intensive workloads",
                "impact": "medium",
                "recommendation": "Optimize CPU-intensive operations"
            })

        # Check for bursty patterns
        intensity_values = [w.intensity_score for w in recent_workloads]
        if len(intensity_values) > 1 and statistics.stdev(intensity_values) > 0.2:
            opportunities.append({
                "type": "workload_stabilization",
                "description": "Bursty workload patterns detected",
                "impact": "medium",
                "recommendation": "Implement workload smoothing strategies"
            })

        return opportunities

    def stop_adaptive_management(self) -> None:
        """Stop the workload-adaptive memory management thread."""
        self._adaptation_active = False
        if self._adaptation_thread:
            self._adaptation_thread.join(timeout=5.0)
        logger.info("Workload-adaptive memory management stopped")


# Global instances
_workload_adaptive_manager: Optional[WorkloadAdaptiveMemoryManager] = None


def get_workload_adaptive_memory_manager() -> WorkloadAdaptiveMemoryManager:
    """Get the global workload-adaptive memory manager instance."""
    global _workload_adaptive_manager
    if _workload_adaptive_manager is None:
        _workload_adaptive_manager = WorkloadAdaptiveMemoryManager()
    return _workload_adaptive_manager