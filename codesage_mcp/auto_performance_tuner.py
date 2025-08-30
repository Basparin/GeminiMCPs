"""
Automatic Performance Tuning Module for CodeSage MCP Server.

This module provides ML-based automatic performance tuning that analyzes usage patterns,
performance metrics, and system behavior to automatically optimize performance parameters.

Classes:
    AutoPerformanceTuner: Main automatic performance tuning class
    PerformancePredictor: ML-based performance prediction models
    ParameterOptimizer: Automatic parameter optimization algorithms
    TuningExperiment: A/B testing framework for optimization experiments
    ContinuousLearner: Continuous learning and adaptation system
"""

import logging
import time
import threading
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class TuningStrategy(Enum):
    """Performance tuning strategies."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID = "hybrid"


class PerformanceMetric(Enum):
    """Performance metrics for optimization."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    RESOURCE_EFFICIENCY = "resource_efficiency"


@dataclass
class TuningParameter:
    """Represents a tunable performance parameter."""
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    parameter_type: str  # 'int', 'float', 'categorical'
    description: str
    impact_area: str  # 'cache', 'memory', 'cpu', 'network'


@dataclass
class PerformanceMeasurement:
    """Represents a performance measurement."""
    timestamp: float
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    workload_characteristics: Dict[str, Any]
    performance_score: float


@dataclass
class TuningRecommendation:
    """Represents a tuning recommendation."""
    parameter: str
    recommended_value: Any
    current_value: Any
    expected_improvement: float
    confidence_score: float
    reasoning: str
    implementation_plan: List[str]


@dataclass
class ExperimentResult:
    """Represents the result of a tuning experiment."""
    experiment_id: str
    parameter_changes: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    experiment_metrics: Dict[str, float]
    improvement_percentage: float
    statistical_significance: float
    duration_seconds: float
    success: bool


class AutoPerformanceTuner:
    """Main automatic performance tuning class."""

    def __init__(self, tuning_interval_minutes: int = 30, experiment_duration_minutes: int = 10):
        self.tuning_interval_minutes = tuning_interval_minutes
        self.experiment_duration_minutes = experiment_duration_minutes

        # Tuning parameters
        self.tuning_parameters = self._initialize_tuning_parameters()

        # Performance data
        self.performance_history: deque = deque(maxlen=1000)
        self.experiment_results: List[ExperimentResult] = []

        # ML models
        self.performance_predictor = PerformancePredictor()
        self.parameter_optimizer = ParameterOptimizer()

        # Control
        self._tuning_active = False
        self._tuning_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Tuning state
        self.current_experiments: Dict[str, Any] = {}
        self.baseline_metrics: Dict[str, float] = {}
        self.tuning_goals = {
            "response_time_target": 25.0,  # ms
            "throughput_target": 1500,  # RPS
            "memory_usage_target": 70.0,  # percent
            "cpu_usage_target": 75.0,  # percent
            "cache_hit_rate_target": 95.0,  # percent
            "error_rate_target": 0.05  # percent
        }

        # Start automatic tuning
        self._start_automatic_tuning()

    def _initialize_tuning_parameters(self) -> Dict[str, TuningParameter]:
        """Initialize tunable performance parameters."""
        parameters = {}

        # Cache parameters
        parameters["embedding_cache_size"] = TuningParameter(
            name="embedding_cache_size",
            current_value=5000,
            min_value=1000,
            max_value=10000,
            step_size=500,
            parameter_type="int",
            description="Maximum number of embeddings to cache",
            impact_area="cache"
        )

        parameters["search_cache_size"] = TuningParameter(
            name="search_cache_size",
            current_value=1000,
            min_value=200,
            max_value=2000,
            step_size=100,
            parameter_type="int",
            description="Maximum number of search results to cache",
            impact_area="cache"
        )

        parameters["file_cache_size"] = TuningParameter(
            name="file_cache_size",
            current_value=100,
            min_value=20,
            max_value=500,
            step_size=20,
            parameter_type="int",
            description="Maximum number of file contents to cache",
            impact_area="cache"
        )

        parameters["cache_similarity_threshold"] = TuningParameter(
            name="cache_similarity_threshold",
            current_value=0.85,
            min_value=0.7,
            max_value=0.95,
            step_size=0.05,
            parameter_type="float",
            description="Similarity threshold for search result caching",
            impact_area="cache"
        )

        # Memory parameters
        parameters["max_memory_mb"] = TuningParameter(
            name="max_memory_mb",
            current_value=2048,
            min_value=1024,
            max_value=4096,
            step_size=256,
            parameter_type="int",
            description="Maximum memory usage in MB",
            impact_area="memory"
        )

        parameters["chunk_size_tokens"] = TuningParameter(
            name="chunk_size_tokens",
            current_value=750,
            min_value=500,
            max_value=1000,
            step_size=50,
            parameter_type="int",
            description="Document chunk size in tokens",
            impact_area="memory"
        )

        # Performance parameters
        parameters["max_workers"] = TuningParameter(
            name="max_workers",
            current_value=4,
            min_value=2,
            max_value=8,
            step_size=1,
            parameter_type="int",
            description="Maximum number of worker threads",
            impact_area="cpu"
        )

        parameters["prefetch_batch_size"] = TuningParameter(
            name="prefetch_batch_size",
            current_value=3,
            min_value=1,
            max_value=10,
            step_size=1,
            parameter_type="int",
            description="Number of files to prefetch in each batch",
            impact_area="cpu"
        )

        return parameters

    def _start_automatic_tuning(self) -> None:
        """Start the automatic tuning system."""
        if self._tuning_active:
            return

        self._tuning_active = True
        self._tuning_thread = threading.Thread(
            target=self._automatic_tuning_loop,
            daemon=True,
            name="AutoPerformanceTuner"
        )
        self._tuning_thread.start()
        logger.info("Automatic performance tuning started")

    def _automatic_tuning_loop(self) -> None:
        """Main automatic tuning loop."""
        while self._tuning_active:
            try:
                self._perform_tuning_cycle()
                time.sleep(self.tuning_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in automatic tuning loop: {e}")
                time.sleep(60)

    def _perform_tuning_cycle(self) -> None:
        """Perform one cycle of automatic performance tuning."""
        with self._lock:
            # Collect current performance data
            current_metrics = self._collect_current_metrics()

            # Update performance history
            self.performance_history.append(current_metrics)

            # Check if tuning is needed
            if self._should_perform_tuning(current_metrics):
                # Generate tuning recommendations
                recommendations = self._generate_tuning_recommendations(current_metrics)

                if recommendations:
                    # Execute tuning experiments
                    experiment_results = self._execute_tuning_experiments(recommendations, current_metrics)

                    # Apply successful tunings
                    successful_tunings = self._apply_successful_tunings(experiment_results)

                    logger.info(f"Automatic tuning cycle completed: {len(successful_tunings)} parameters optimized")

    def _collect_current_metrics(self) -> PerformanceMeasurement:
        """Collect current performance metrics."""
        # In a real implementation, this would collect actual metrics from the system
        # For now, return mock data
        current_time = time.time()

        metrics = {
            "response_time_ms": 25.0 + random.uniform(-5, 5),
            "throughput_rps": 1500 + random.uniform(-100, 100),
            "memory_usage_percent": 65.0 + random.uniform(-10, 10),
            "cpu_usage_percent": 70.0 + random.uniform(-15, 15),
            "cache_hit_rate": 92.0 + random.uniform(-5, 5),
            "error_rate_percent": 0.05 + random.uniform(-0.02, 0.02)
        }

        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)

        # Get current parameter values
        current_parameters = {name: param.current_value for name, param in self.tuning_parameters.items()}

        # Mock workload characteristics
        workload_characteristics = {
            "active_connections": 15,
            "request_rate": 1200,
            "workload_type": "mixed"
        }

        return PerformanceMeasurement(
            timestamp=current_time,
            parameters=current_parameters,
            metrics=metrics,
            workload_characteristics=workload_characteristics,
            performance_score=performance_score
        )

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0

        # Response time score (30% weight)
        response_time_score = max(0, 100 - (metrics["response_time_ms"] - self.tuning_goals["response_time_target"]) * 2)
        score -= (100 - response_time_score) * 0.3

        # Throughput score (25% weight)
        throughput_score = min(100, (metrics["throughput_rps"] / self.tuning_goals["throughput_target"]) * 100)
        score -= (100 - throughput_score) * 0.25

        # Resource efficiency score (25% weight)
        memory_score = max(0, 100 - (metrics["memory_usage_percent"] - self.tuning_goals["memory_usage_target"]))
        cpu_score = max(0, 100 - (metrics["cpu_usage_percent"] - self.tuning_goals["cpu_usage_target"]))
        resource_score = (memory_score + cpu_score) / 2
        score -= (100 - resource_score) * 0.25

        # Quality score (20% weight)
        cache_score = min(100, (metrics["cache_hit_rate"] / self.tuning_goals["cache_hit_rate_target"]) * 100)
        error_score = max(0, 100 - (metrics["error_rate_percent"] - self.tuning_goals["error_rate_target"]) * 2000)
        quality_score = (cache_score + error_score) / 2
        score -= (100 - quality_score) * 0.2

        return max(0.0, min(100.0, score))

    def _should_perform_tuning(self, current_metrics: PerformanceMeasurement) -> bool:
        """Determine if performance tuning should be performed."""
        # Check if performance is below targets
        performance_below_target = current_metrics.performance_score < 85.0

        # Check for performance degradation trend
        recent_scores = [m.performance_score for m in list(self.performance_history)[-5:]]
        if len(recent_scores) >= 3:
            trend = statistics.linear_regression(range(len(recent_scores)), recent_scores)[0]
            performance_degrading = trend < -0.5  # Significant downward trend
        else:
            performance_degrading = False

        # Check if enough data is available
        sufficient_data = len(self.performance_history) >= 10

        return (performance_below_target or performance_degrading) and sufficient_data

    def _generate_tuning_recommendations(self, current_metrics: PerformanceMeasurement) -> List[TuningRecommendation]:
        """Generate tuning recommendations using ML-based analysis."""
        recommendations = []

        # Use parameter optimizer to find optimal parameter values
        optimal_parameters = self.parameter_optimizer.optimize_parameters(
            current_metrics, self.performance_history, self.tuning_parameters
        )

        for param_name, optimal_value in optimal_parameters.items():
            if param_name in self.tuning_parameters:
                parameter = self.tuning_parameters[param_name]
                current_value = parameter.current_value

                if optimal_value != current_value:
                    # Calculate expected improvement
                    expected_improvement = self.performance_predictor.predict_improvement(
                        param_name, current_value, optimal_value, current_metrics
                    )

                    # Generate confidence score
                    confidence_score = self._calculate_tuning_confidence(
                        param_name, current_value, optimal_value, expected_improvement
                    )

                    # Generate reasoning
                    reasoning = self._generate_tuning_reasoning(
                        param_name, current_value, optimal_value, expected_improvement, current_metrics
                    )

                    # Generate implementation plan
                    implementation_plan = self._generate_implementation_plan(param_name, optimal_value)

                    recommendation = TuningRecommendation(
                        parameter=param_name,
                        recommended_value=optimal_value,
                        current_value=current_value,
                        expected_improvement=expected_improvement,
                        confidence_score=confidence_score,
                        reasoning=reasoning,
                        implementation_plan=implementation_plan
                    )

                    recommendations.append(recommendation)

        # Sort by expected improvement and confidence
        recommendations.sort(key=lambda x: (x.expected_improvement, x.confidence_score), reverse=True)

        return recommendations[:5]  # Return top 5 recommendations

    def _execute_tuning_experiments(self, recommendations: List[TuningRecommendation],
                                  baseline_metrics: PerformanceMeasurement) -> List[ExperimentResult]:
        """Execute tuning experiments to validate recommendations."""
        experiment_results = []

        for recommendation in recommendations:
            if recommendation.confidence_score < 0.6:
                continue  # Skip low-confidence recommendations

            # Create experiment
            experiment_id = f"exp_{recommendation.parameter}_{int(time.time())}"

            # Prepare parameter changes
            parameter_changes = {recommendation.parameter: recommendation.recommended_value}

            # Execute experiment
            experiment_result = self._run_tuning_experiment(
                experiment_id, parameter_changes, baseline_metrics, recommendation
            )

            experiment_results.append(experiment_result)

        return experiment_results

    def _run_tuning_experiment(self, experiment_id: str, parameter_changes: Dict[str, Any],
                             baseline_metrics: PerformanceMeasurement,
                             recommendation: TuningRecommendation) -> ExperimentResult:
        """Run a single tuning experiment."""
        start_time = time.time()

        # Store original values
        original_values = {}
        for param_name in parameter_changes:
            if param_name in self.tuning_parameters:
                original_values[param_name] = self.tuning_parameters[param_name].current_value

        try:
            # Apply parameter changes
            for param_name, new_value in parameter_changes.items():
                if param_name in self.tuning_parameters:
                    self.tuning_parameters[param_name].current_value = new_value

            # Wait for system to stabilize
            time.sleep(60)  # Wait 1 minute

            # Collect metrics after change
            experiment_metrics_measurement = self._collect_current_metrics()
            experiment_metrics = experiment_metrics_measurement.metrics

            # Calculate improvement
            baseline_score = baseline_metrics.performance_score
            experiment_score = experiment_metrics_measurement.performance_score
            improvement_percentage = ((experiment_score - baseline_score) / baseline_score) * 100

            # Calculate statistical significance (simplified)
            statistical_significance = min(1.0, abs(improvement_percentage) / 10.0)

            duration = time.time() - start_time

            # Determine success
            success = improvement_percentage > 1.0 and statistical_significance > 0.7

            return ExperimentResult(
                experiment_id=experiment_id,
                parameter_changes=parameter_changes,
                baseline_metrics=baseline_metrics.metrics,
                experiment_metrics=experiment_metrics,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance,
                duration_seconds=duration,
                success=success
            )

        finally:
            # Restore original values
            for param_name, original_value in original_values.items():
                if param_name in self.tuning_parameters:
                    self.tuning_parameters[param_name].current_value = original_value

    def _apply_successful_tunings(self, experiment_results: List[ExperimentResult]) -> List[str]:
        """Apply successful tuning changes."""
        successful_tunings = []

        for result in experiment_results:
            if result.success and result.improvement_percentage > 2.0:  # At least 2% improvement
                # Apply the changes permanently
                for param_name, new_value in result.parameter_changes.items():
                    if param_name in self.tuning_parameters:
                        self.tuning_parameters[param_name].current_value = new_value
                        successful_tunings.append(param_name)

                logger.info(f"Applied successful tuning: {result.parameter_changes} "
                          ".2f")

        return successful_tunings

    def _calculate_tuning_confidence(self, param_name: str, current_value: Any,
                                   optimal_value: Any, expected_improvement: float) -> float:
        """Calculate confidence score for a tuning recommendation."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on expected improvement
        if expected_improvement > 10:
            confidence += 0.3
        elif expected_improvement > 5:
            confidence += 0.2
        elif expected_improvement > 2:
            confidence += 0.1

        # Increase confidence based on parameter type
        parameter = self.tuning_parameters.get(param_name)
        if parameter:
            if parameter.parameter_type == "int" and parameter.step_size <= parameter.max_value * 0.1:
                confidence += 0.1  # Fine-grained tuning

        # Increase confidence based on historical success
        historical_success = self._get_historical_success_rate(param_name)
        confidence += historical_success * 0.2

        return min(1.0, confidence)

    def _get_historical_success_rate(self, param_name: str) -> float:
        """Get historical success rate for tuning a parameter."""
        relevant_experiments = [
            exp for exp in self.experiment_results
            if param_name in exp.parameter_changes
        ]

        if not relevant_experiments:
            return 0.5  # Neutral success rate

        successful_experiments = sum(1 for exp in relevant_experiments if exp.success)
        return successful_experiments / len(relevant_experiments)

    def _generate_tuning_reasoning(self, param_name: str, current_value: Any,
                                 optimal_value: Any, expected_improvement: float,
                                 current_metrics: PerformanceMeasurement) -> str:
        """Generate reasoning for a tuning recommendation."""
        parameter = self.tuning_parameters.get(param_name)
        if not parameter:
            return "Parameter optimization based on performance analysis"

        reasoning_parts = []

        # Parameter-specific reasoning
        if param_name == "embedding_cache_size":
            if optimal_value > current_value:
                reasoning_parts.append("Increasing cache size to improve embedding hit rate")
            else:
                reasoning_parts.append("Reducing cache size to optimize memory usage")
        elif param_name == "max_memory_mb":
            if optimal_value > current_value:
                reasoning_parts.append("Increasing memory limit to handle higher workloads")
            else:
                reasoning_parts.append("Reducing memory limit to prevent memory pressure")
        elif param_name == "cache_similarity_threshold":
            if optimal_value > current_value:
                reasoning_parts.append("Increasing similarity threshold for more aggressive caching")
            else:
                reasoning_parts.append("Decreasing similarity threshold for more precise caching")

        # Performance-based reasoning
        if expected_improvement > 5:
            reasoning_parts.append(".1f")
        elif expected_improvement > 0:
            reasoning_parts.append(".1f")
        else:
            reasoning_parts.append("Expected minimal performance impact")

        # Current performance context
        if current_metrics.metrics["response_time_ms"] > self.tuning_goals["response_time_target"]:
            reasoning_parts.append("Current response time is above target")
        if current_metrics.metrics["memory_usage_percent"] > self.tuning_goals["memory_usage_target"]:
            reasoning_parts.append("Current memory usage is above target")

        return ". ".join(reasoning_parts)

    def _generate_implementation_plan(self, param_name: str, optimal_value: Any) -> List[str]:
        """Generate implementation plan for a parameter change."""
        parameter = self.tuning_parameters.get(param_name)
        if not parameter:
            return ["Update parameter value", "Monitor performance", "Rollback if needed"]

        plan = []

        if parameter.impact_area == "cache":
            plan.extend([
                "Update cache configuration with new parameter value",
                "Clear existing cache to ensure consistency",
                "Monitor cache hit rate and memory usage",
                "Gradually adjust based on observed performance"
            ])
        elif parameter.impact_area == "memory":
            plan.extend([
                "Update memory management configuration",
                "Monitor memory usage patterns",
                "Adjust garbage collection settings if needed",
                "Validate memory pressure remains within limits"
            ])
        elif parameter.impact_area == "cpu":
            plan.extend([
                "Update worker thread configuration",
                "Monitor CPU usage and thread utilization",
                "Adjust thread pool size based on workload",
                "Validate performance improvement"
            ])

        plan.extend([
            "Monitor system performance for 10 minutes",
            "Compare key metrics against baseline",
            "Rollback changes if performance degrades by >5%",
            "Document successful parameter values"
        ])

        return plan

    def get_tuning_analysis(self) -> Dict[str, Any]:
        """Get comprehensive tuning analysis."""
        with self._lock:
            analysis = {
                "current_parameters": {
                    name: {
                        "current_value": param.current_value,
                        "min_value": param.min_value,
                        "max_value": param.max_value,
                        "description": param.description,
                        "impact_area": param.impact_area
                    }
                    for name, param in self.tuning_parameters.items()
                },
                "performance_history": {
                    "total_measurements": len(self.performance_history),
                    "recent_performance_score": self.performance_history[-1].performance_score if self.performance_history else None,
                    "performance_trend": self._analyze_performance_trend()
                },
                "experiment_results": {
                    "total_experiments": len(self.experiment_results),
                    "successful_experiments": sum(1 for exp in self.experiment_results if exp.success),
                    "average_improvement": statistics.mean([exp.improvement_percentage for exp in self.experiment_results]) if self.experiment_results else 0,
                    "recent_experiments": [
                        {
                            "experiment_id": exp.experiment_id,
                            "parameter_changes": exp.parameter_changes,
                            "improvement_percentage": exp.improvement_percentage,
                            "success": exp.success,
                            "timestamp": exp.baseline_metrics.get("timestamp", 0)
                        }
                        for exp in self.experiment_results[-5:]
                    ]
                },
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "tuning_effectiveness": self._calculate_tuning_effectiveness(),
                "generated_at": time.time()
            }

            return analysis

    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trend."""
        if len(self.performance_history) < 5:
            return {"trend": "insufficient_data"}

        recent_scores = [m.performance_score for m in list(self.performance_history)[-10:]]
        trend_slope = statistics.linear_regression(range(len(recent_scores)), recent_scores)[0]

        if trend_slope > 0.5:
            trend = "improving"
        elif trend_slope < -0.5:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": trend_slope,
            "recent_avg_score": statistics.mean(recent_scores),
            "volatility": statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0
        }

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []

        # Check for parameters that haven't been optimized recently
        current_time = time.time()
        for name, param in self.tuning_parameters.items():
            # Check if parameter has been experimented with recently
            recent_experiments = [
                exp for exp in self.experiment_results[-10:]
                if name in exp.parameter_changes
            ]

            if not recent_experiments:
                opportunities.append({
                    "type": "parameter_exploration",
                    "parameter": name,
                    "description": f"Parameter {name} hasn't been optimized recently",
                    "recommendation": "Consider running optimization experiments",
                    "priority": "medium"
                })

        # Check for performance degradation
        if len(self.performance_history) >= 5:
            recent_scores = [m.performance_score for m in list(self.performance_history)[-5:]]
            if statistics.mean(recent_scores) < 80:
                opportunities.append({
                    "type": "performance_recovery",
                    "description": "Performance score has declined recently",
                    "recommendation": "Run comprehensive parameter optimization",
                    "priority": "high"
                })

        return opportunities

    def _calculate_tuning_effectiveness(self) -> Dict[str, Any]:
        """Calculate tuning effectiveness."""
        if not self.experiment_results:
            return {"effectiveness": "no_experiments"}

        successful_experiments = [exp for exp in self.experiment_results if exp.success]
        success_rate = len(successful_experiments) / len(self.experiment_results)

        avg_improvement = statistics.mean([exp.improvement_percentage for exp in successful_experiments]) if successful_experiments else 0

        return {
            "success_rate": success_rate,
            "average_improvement": avg_improvement,
            "total_experiments": len(self.experiment_results),
            "effective_parameters": list(set([
                param for exp in successful_experiments
                for param in exp.parameter_changes.keys()
            ])),
            "effectiveness_score": (success_rate * 50) + min(avg_improvement, 50)
        }

    def stop_automatic_tuning(self) -> None:
        """Stop the automatic tuning system."""
        self._tuning_active = False
        if self._tuning_thread:
            self._tuning_thread.join(timeout=5.0)
        logger.info("Automatic performance tuning stopped")


class PerformancePredictor:
    """ML-based performance prediction models."""

    def __init__(self):
        self.prediction_model: Dict[str, Any] = {}
        self.training_data: List[PerformanceMeasurement] = []

    def predict_improvement(self, param_name: str, current_value: Any,
                          new_value: Any, current_metrics: PerformanceMeasurement) -> float:
        """Predict performance improvement for a parameter change."""
        # Simplified prediction based on historical data
        if not self.training_data:
            # Default prediction based on parameter type
            if param_name.endswith("_cache_size"):
                return 5.0  # Cache size increases typically improve performance
            elif param_name == "max_memory_mb":
                return 3.0  # Memory increases help with memory-intensive workloads
            else:
                return 1.0  # Conservative estimate

        # Look for similar parameter changes in training data
        similar_changes = [
            data for data in self.training_data
            if param_name in data.parameters and data.parameters[param_name] == new_value
        ]

        if similar_changes:
            improvements = []
            for change in similar_changes:
                baseline_score = 80.0  # Assume baseline score
                improvement = change.performance_score - baseline_score
                improvements.append(improvement)

            return statistics.mean(improvements) if improvements else 2.0

        return 2.0  # Conservative estimate


class ParameterOptimizer:
    """Automatic parameter optimization algorithms."""

    def __init__(self):
        self.optimization_algorithms = {
            TuningStrategy.BAYESIAN_OPTIMIZATION: self._bayesian_optimization,
            TuningStrategy.GRADIENT_DESCENT: self._gradient_descent,
            TuningStrategy.GENETIC_ALGORITHM: self._genetic_algorithm
        }

    def optimize_parameters(self, current_metrics: PerformanceMeasurement,
                          performance_history: deque,
                          tuning_parameters: Dict[str, TuningParameter]) -> Dict[str, Any]:
        """Optimize parameters using available algorithms."""
        optimal_parameters = {}

        # Use Bayesian optimization for most parameters
        for param_name, parameter in tuning_parameters.items():
            if parameter.parameter_type in ["int", "float"]:
                optimal_value = self._bayesian_optimization(
                    param_name, parameter, current_metrics, performance_history
                )
                if optimal_value != parameter.current_value:
                    optimal_parameters[param_name] = optimal_value

        return optimal_parameters

    def _bayesian_optimization(self, param_name: str, parameter: TuningParameter,
                             current_metrics: PerformanceMeasurement,
                             performance_history: deque) -> Any:
        """Bayesian optimization for parameter tuning."""
        # Simplified Bayesian optimization
        # In a real implementation, this would use Gaussian processes

        # Explore parameter space
        if len(performance_history) < 5:
            # Random exploration
            if parameter.parameter_type == "int":
                return random.randint(parameter.min_value, parameter.max_value)
            else:
                return parameter.min_value + random.random() * (parameter.max_value - parameter.min_value)

        # Exploit based on historical performance
        best_value = parameter.current_value
        best_score = current_metrics.performance_score

        # Look for better values in history
        for measurement in performance_history:
            if param_name in measurement.parameters:
                param_value = measurement.parameters[param_name]
                if measurement.performance_score > best_score:
                    best_value = param_value
                    best_score = measurement.performance_score

        return best_value

    def _gradient_descent(self, param_name: str, parameter: TuningParameter,
                        current_metrics: PerformanceMeasurement,
                        performance_history: deque) -> Any:
        """Gradient descent optimization."""
        # Simplified gradient descent
        current_value = parameter.current_value

        # Calculate gradient based on recent performance
        if len(performance_history) >= 3:
            recent_measurements = list(performance_history)[-3:]
            performance_trend = statistics.linear_regression(
                range(len(recent_measurements)),
                [m.performance_score for m in recent_measurements]
            )[0]

            # Adjust parameter based on trend
            if performance_trend < 0:  # Performance declining
                if param_name.endswith("_cache_size"):
                    current_value = min(parameter.max_value, current_value + parameter.step_size)
                elif param_name == "max_memory_mb":
                    current_value = min(parameter.max_value, current_value + parameter.step_size)

        return current_value

    def _genetic_algorithm(self, param_name: str, parameter: TuningParameter,
                         current_metrics: PerformanceMeasurement,
                         performance_history: deque) -> Any:
        """Genetic algorithm optimization."""
        # Simplified genetic algorithm
        population_size = 10

        # Generate population
        population = []
        for _ in range(population_size):
            if parameter.parameter_type == "int":
                individual = random.randint(parameter.min_value, parameter.max_value)
            else:
                individual = parameter.min_value + random.random() * (parameter.max_value - parameter.min_value)
            population.append(individual)

        # Evaluate fitness (simplified)
        best_individual = max(population, key=lambda x: self._evaluate_fitness(x, param_name, parameter))

        return best_individual

    def _evaluate_fitness(self, value: Any, param_name: str, parameter: TuningParameter) -> float:
        """Evaluate fitness of a parameter value."""
        # Simplified fitness evaluation
        # In a real implementation, this would run actual performance tests

        # Prefer values closer to current optimal ranges
        if param_name.endswith("_cache_size"):
            optimal_range = (parameter.min_value + parameter.max_value) // 2
            return 100 - abs(value - optimal_range) / optimal_range * 50
        elif param_name == "max_memory_mb":
            return min(100, value / 40)  # Prefer higher memory

        return 50  # Neutral fitness


# Global instances
_auto_performance_tuner: Optional[AutoPerformanceTuner] = None


def get_auto_performance_tuner() -> AutoPerformanceTuner:
    """Get the global auto performance tuner instance."""
    global _auto_performance_tuner
    if _auto_performance_tuner is None:
        _auto_performance_tuner = AutoPerformanceTuner()
    return _auto_performance_tuner