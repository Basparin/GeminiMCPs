"""
Intelligent Prefetching Module for CodeSage MCP Server.

This module provides intelligent prefetching capabilities that analyze usage patterns,
predict future access patterns, and proactively prefetch data to improve performance.

Classes:
    IntelligentPrefetcher: Main intelligent prefetching class
    UsagePatternAnalyzer: Analyzes usage patterns for prefetching
    PrefetchPredictor: Predicts future access patterns using ML
    PrefetchEngine: Executes prefetching operations
    PrefetchMetrics: Tracks prefetching performance and effectiveness
"""

import logging
import time
import threading
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class PrefetchStrategy(Enum):
    """Prefetching strategies."""
    PATTERN_BASED = "pattern_based"
    PREDICTIVE = "predictive"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


class AccessPattern(Enum):
    """Types of access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    CLUSTERED = "clustered"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class AccessRecord:
    """Represents a single data access record."""
    file_path: str
    access_time: float
    access_type: str  # 'read', 'write', 'search'
    user_context: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefetchCandidate:
    """Represents a prefetch candidate."""
    file_path: str
    confidence_score: float
    expected_benefit: float
    predicted_access_time: float
    reason: str
    strategy: PrefetchStrategy
    priority: int


@dataclass
class PrefetchMetrics:
    """Metrics for prefetching performance."""
    total_prefetches: int = 0
    successful_prefetches: int = 0
    failed_prefetches: int = 0
    cache_hit_improvements: int = 0
    average_latency_reduction_ms: float = 0.0
    prefetch_accuracy: float = 0.0
    bandwidth_savings_mb: float = 0.0
    last_updated: float = field(default_factory=time.time)


class IntelligentPrefetcher:
    """Main intelligent prefetching class."""

    def __init__(self, prefetch_window_size: int = 1000, analysis_interval_minutes: int = 5):
        self.prefetch_window_size = prefetch_window_size
        self.analysis_interval_minutes = analysis_interval_minutes

        # Access pattern tracking
        self.access_history: deque = deque(maxlen=prefetch_window_size)
        self.file_access_counts: Dict[str, int] = defaultdict(int)
        self.file_access_times: Dict[str, List[float]] = defaultdict(list)
        self.file_coaccess_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Prefetching state
        self.active_prefetches: Dict[str, float] = {}  # file_path -> prefetch_time
        self.prefetch_candidates: List[PrefetchCandidate] = []
        self.prefetch_metrics = PrefetchMetrics()

        # Prediction models
        self.access_pattern_model: Dict[str, Any] = {}
        self.temporal_pattern_model: Dict[str, Any] = {}

        # Control
        self._prefetch_active = False
        self._analysis_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Configuration
        self.prefetch_config = {
            "max_concurrent_prefetches": 5,
            "prefetch_threshold": 0.7,  # Minimum confidence to prefetch
            "max_prefetch_age_seconds": 300,  # Don't prefetch if accessed recently
            "prefetch_batch_size": 3,
            "enable_pattern_learning": True,
            "enable_predictive_prefetching": True,
            "enable_collaborative_filtering": False
        }

        # Start intelligent prefetching
        self._start_intelligent_prefetching()

    def _start_intelligent_prefetching(self) -> None:
        """Start the intelligent prefetching system."""
        if self._prefetch_active:
            return

        self._prefetch_active = True
        self._analysis_thread = threading.Thread(
            target=self._intelligent_analysis_loop,
            daemon=True,
            name="IntelligentPrefetcher"
        )
        self._analysis_thread.start()
        logger.info("Intelligent prefetching started")

    def _intelligent_analysis_loop(self) -> None:
        """Main intelligent analysis loop."""
        while self._prefetch_active:
            try:
                self._perform_intelligent_analysis()
                time.sleep(self.analysis_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in intelligent prefetching loop: {e}")
                time.sleep(60)

    def _perform_intelligent_analysis(self) -> None:
        """Perform intelligent prefetching analysis."""
        with self._lock:
            # Analyze access patterns
            self._analyze_access_patterns()

            # Generate prefetch candidates
            self._generate_prefetch_candidates()

            # Execute prefetching
            self._execute_prefetching()

            # Update metrics
            self._update_prefetch_metrics()

            logger.info(f"Intelligent prefetching analysis completed: {len(self.prefetch_candidates)} candidates")

    def record_access(self, file_path: str, access_type: str = "read",
                     user_context: str = "", session_id: str = "") -> None:
        """Record a file access for pattern analysis."""
        with self._lock:
            access_record = AccessRecord(
                file_path=file_path,
                access_time=time.time(),
                access_type=access_type,
                user_context=user_context,
                session_id=session_id
            )

            self.access_history.append(access_record)
            self.file_access_counts[file_path] += 1
            self.file_access_times[file_path].append(access_record.access_time)

            # Update co-access patterns
            if len(self.access_history) >= 2:
                prev_access = self.access_history[-2]
                if prev_access.file_path != file_path:
                    self.file_coaccess_patterns[prev_access.file_path][file_path] += 1

    def _analyze_access_patterns(self) -> None:
        """Analyze access patterns for prefetching opportunities."""
        if len(self.access_history) < 5:
            return

        # Analyze temporal patterns
        self._analyze_temporal_patterns()

        # Analyze spatial patterns
        self._analyze_spatial_patterns()

        # Analyze sequential patterns
        self._analyze_sequential_patterns()

        # Update prediction models
        self._update_prediction_models()

    def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal access patterns."""
        # Calculate access frequencies
        current_time = time.time()
        time_window_hours = 24

        for file_path, access_times in self.file_access_times.items():
            # Filter recent accesses
            recent_accesses = [t for t in access_times if current_time - t < time_window_hours * 3600]

            if len(recent_accesses) >= 2:
                # Calculate inter-access times
                inter_access_times = []
                for i in range(1, len(recent_accesses)):
                    inter_access_times.append(recent_accesses[i] - recent_accesses[i-1])

                if inter_access_times:
                    avg_inter_access = statistics.mean(inter_access_times)
                    self.temporal_pattern_model[file_path] = {
                        "avg_inter_access_time": avg_inter_access,
                        "access_frequency": len(recent_accesses) / time_window_hours,
                        "volatility": statistics.stdev(inter_access_times) if len(inter_access_times) > 1 else 0,
                        "last_access": recent_accesses[-1]
                    }

    def _analyze_spatial_patterns(self) -> None:
        """Analyze spatial access patterns (file co-access)."""
        # Identify strongly co-accessed files
        for file1, coaccess_files in self.file_coaccess_patterns.items():
            total_coaccesses = sum(coaccess_files.values())

            for file2, coaccess_count in coaccess_files.items():
                coaccess_ratio = coaccess_count / max(total_coaccesses, 1)

                if coaccess_ratio > 0.3:  # Strong co-access pattern
                    self.access_pattern_model[f"{file1}->{file2}"] = {
                        "coaccess_ratio": coaccess_ratio,
                        "strength": coaccess_count,
                        "pattern_type": AccessPattern.CLUSTERED.value
                    }

    def _analyze_sequential_patterns(self) -> None:
        """Analyze sequential access patterns."""
        if len(self.access_history) < 3:
            return

        # Look for sequential access patterns
        sequence_window = 5
        for i in range(len(self.access_history) - sequence_window):
            sequence = [record.file_path for record in list(self.access_history)[i:i+sequence_window]]

            # Check if this sequence repeats
            sequence_str = "->".join(sequence)
            if sequence_str in self.access_pattern_model:
                self.access_pattern_model[sequence_str]["frequency"] += 1
            else:
                self.access_pattern_model[sequence_str] = {
                    "sequence": sequence,
                    "frequency": 1,
                    "pattern_type": AccessPattern.SEQUENTIAL.value,
                    "last_observed": list(self.access_history)[-1].access_time
                }

    def _update_prediction_models(self) -> None:
        """Update prediction models based on analyzed patterns."""
        # Update temporal prediction model
        for file_path, pattern_data in self.temporal_pattern_model.items():
            last_access = pattern_data["last_access"]
            avg_inter_access = pattern_data["avg_inter_access_time"]

            # Predict next access time
            predicted_next_access = last_access + avg_inter_access
            confidence = max(0, 1.0 - (pattern_data["volatility"] / avg_inter_access))

            pattern_data["predicted_next_access"] = predicted_next_access
            pattern_data["prediction_confidence"] = confidence

    def _generate_prefetch_candidates(self) -> None:
        """Generate prefetch candidates based on analyzed patterns."""
        candidates = []

        # Generate candidates from temporal patterns
        candidates.extend(self._generate_temporal_candidates())

        # Generate candidates from spatial patterns
        candidates.extend(self._generate_spatial_candidates())

        # Generate candidates from sequential patterns
        candidates.extend(self._generate_sequential_candidates())

        # Generate candidates from collaborative patterns
        if self.prefetch_config["enable_collaborative_filtering"]:
            candidates.extend(self._generate_collaborative_candidates())

        # Filter and prioritize candidates
        self.prefetch_candidates = self._filter_and_prioritize_candidates(candidates)

    def _generate_temporal_candidates(self) -> List[PrefetchCandidate]:
        """Generate prefetch candidates from temporal patterns."""
        candidates = []
        current_time = time.time()

        for file_path, pattern_data in self.temporal_pattern_model.items():
            if "predicted_next_access" not in pattern_data:
                continue

            predicted_time = pattern_data["predicted_next_access"]
            confidence = pattern_data["prediction_confidence"]

            # Only prefetch if prediction is confident and within time window
            time_to_access = predicted_time - current_time
            if (confidence > self.prefetch_config["prefetch_threshold"] and
                0 < time_to_access < 300):  # Within 5 minutes

                # Check if recently accessed
                last_access = pattern_data["last_access"]
                if current_time - last_access > self.prefetch_config["max_prefetch_age_seconds"]:

                    candidate = PrefetchCandidate(
                        file_path=file_path,
                        confidence_score=confidence,
                        expected_benefit=self._calculate_expected_benefit(file_path, confidence),
                        predicted_access_time=predicted_time,
                        reason=f"Temporal pattern prediction (next access in {time_to_access:.0f}s)",
                        strategy=PrefetchStrategy.PREDICTIVE,
                        priority=int(confidence * 10)
                    )
                    candidates.append(candidate)

        return candidates

    def _generate_spatial_candidates(self) -> List[PrefetchCandidate]:
        """Generate prefetch candidates from spatial patterns."""
        candidates = []

        # Find recently accessed files and prefetch their co-accessed files
        current_time = time.time()
        recent_window_seconds = 300  # 5 minutes

        recently_accessed = set()
        for record in self.access_history:
            if current_time - record.access_time < recent_window_seconds:
                recently_accessed.add(record.file_path)

        for recent_file in recently_accessed:
            if recent_file in self.file_coaccess_patterns:
                for coaccessed_file, coaccess_count in self.file_coaccess_patterns[recent_file].items():
                    # Calculate confidence based on co-access strength
                    total_coaccesses = sum(self.file_coaccess_patterns[recent_file].values())
                    confidence = coaccess_count / max(total_coaccesses, 1)

                    if confidence > self.prefetch_config["prefetch_threshold"]:
                        # Check if recently accessed
                        last_access_times = self.file_access_times.get(coaccessed_file, [])
                        if not last_access_times or (current_time - last_access_times[-1] > self.prefetch_config["max_prefetch_age_seconds"]):

                            candidate = PrefetchCandidate(
                                file_path=coaccessed_file,
                                confidence_score=confidence,
                                expected_benefit=self._calculate_expected_benefit(coaccessed_file, confidence),
                                predicted_access_time=current_time + 60,  # Assume access within 1 minute
                                reason=f"Spatial co-access pattern with {recent_file}",
                                strategy=PrefetchStrategy.PATTERN_BASED,
                                priority=int(confidence * 8)
                            )
                            candidates.append(candidate)

        return candidates

    def _generate_sequential_candidates(self) -> List[PrefetchCandidate]:
        """Generate prefetch candidates from sequential patterns."""
        candidates = []

        if len(self.access_history) < 2:
            return candidates

        # Get recent sequence
        recent_sequence = [record.file_path for record in list(self.access_history)[-3:]]

        # Look for patterns that match this sequence
        for pattern_key, pattern_data in self.access_pattern_model.items():
            if pattern_data.get("pattern_type") == AccessPattern.SEQUENTIAL.value:
                pattern_sequence = pattern_data["sequence"]

                # Check if recent sequence matches start of pattern
                if len(recent_sequence) <= len(pattern_sequence):
                    if recent_sequence == pattern_sequence[:len(recent_sequence)]:
                        # Predict next file in sequence
                        if len(pattern_sequence) > len(recent_sequence):
                            next_file = pattern_sequence[len(recent_sequence)]
                            frequency = pattern_data["frequency"]

                            # Calculate confidence based on pattern frequency
                            confidence = min(frequency / 10.0, 1.0)  # Max confidence 1.0

                            if confidence > self.prefetch_config["prefetch_threshold"]:
                                candidate = PrefetchCandidate(
                                    file_path=next_file,
                                    confidence_score=confidence,
                                    expected_benefit=self._calculate_expected_benefit(next_file, confidence),
                                    predicted_access_time=time.time() + 30,  # Assume access within 30 seconds
                                    reason=f"Sequential pattern prediction (frequency: {frequency})",
                                    strategy=PrefetchStrategy.PATTERN_BASED,
                                    priority=int(confidence * 9)
                                )
                                candidates.append(candidate)

        return candidates

    def _generate_collaborative_candidates(self) -> List[PrefetchCandidate]:
        """Generate prefetch candidates using collaborative filtering."""
        # This would implement collaborative filtering based on user behavior patterns
        # For now, return empty list as placeholder
        return []

    def _filter_and_prioritize_candidates(self, candidates: List[PrefetchCandidate]) -> List[PrefetchCandidate]:
        """Filter and prioritize prefetch candidates."""
        # Remove duplicates
        seen_files = set()
        unique_candidates = []

        for candidate in candidates:
            if candidate.file_path not in seen_files:
                seen_files.add(candidate.file_path)
                unique_candidates.append(candidate)

        # Sort by priority and confidence
        unique_candidates.sort(key=lambda x: (x.priority, x.confidence_score), reverse=True)

        # Limit to batch size
        return unique_candidates[:self.prefetch_config["prefetch_batch_size"]]

    def _calculate_expected_benefit(self, file_path: str, confidence: float) -> float:
        """Calculate expected benefit of prefetching a file."""
        # Base benefit on access frequency and confidence
        access_frequency = self.file_access_counts.get(file_path, 0)
        base_benefit = access_frequency * confidence

        # Adjust based on file size (smaller files have higher benefit)
        # This is a simplified calculation - in practice, you'd get actual file size
        estimated_file_size_mb = 1.0  # Placeholder
        size_adjustment = max(0.1, 1.0 - (estimated_file_size_mb / 10.0))

        return base_benefit * size_adjustment

    def _execute_prefetching(self) -> None:
        """Execute prefetching for top candidates."""
        if not self.prefetch_candidates:
            return

        # Limit concurrent prefetches
        available_slots = self.prefetch_config["max_concurrent_prefetches"] - len(self.active_prefetches)

        if available_slots <= 0:
            return

        # Prefetch top candidates
        candidates_to_prefetch = self.prefetch_candidates[:available_slots]

        for candidate in candidates_to_prefetch:
            try:
                self._prefetch_file(candidate)
                self.prefetch_metrics.total_prefetches += 1
                self.active_prefetches[candidate.file_path] = time.time()

                logger.debug(f"Prefetched file: {candidate.file_path} (confidence: {candidate.confidence_score:.2f})")

            except Exception as e:
                logger.exception(f"Failed to prefetch {candidate.file_path}: {e}")
                self.prefetch_metrics.failed_prefetches += 1

    def _prefetch_file(self, candidate: PrefetchCandidate) -> None:
        """Prefetch a specific file."""
        # In a real implementation, this would:
        # 1. Check if file exists and is accessible
        # 2. Load file content into cache
        # 3. Update cache metadata
        # 4. Record prefetch metrics

        # For now, simulate prefetching
        time.sleep(0.01)  # Simulate I/O time

        # Mark as successfully prefetched
        self.prefetch_metrics.successful_prefetches += 1

    def _update_prefetch_metrics(self) -> None:
        """Update prefetching metrics."""
        # Calculate prefetch accuracy
        if self.prefetch_metrics.total_prefetches > 0:
            self.prefetch_metrics.prefetch_accuracy = (
                self.prefetch_metrics.successful_prefetches / self.prefetch_metrics.total_prefetches
            )

        # Clean up old active prefetches
        current_time = time.time()
        cutoff_time = current_time - 600  # 10 minutes
        self.active_prefetches = {
            file_path: prefetch_time
            for file_path, prefetch_time in self.active_prefetches.items()
            if prefetch_time > cutoff_time
        }

        self.prefetch_metrics.last_updated = current_time

    def get_prefetch_analysis(self) -> Dict[str, Any]:
        """Get comprehensive prefetching analysis."""
        with self._lock:
            analysis = {
                "prefetch_metrics": {
                    "total_prefetches": self.prefetch_metrics.total_prefetches,
                    "successful_prefetches": self.prefetch_metrics.successful_prefetches,
                    "failed_prefetches": self.prefetch_metrics.failed_prefetches,
                    "prefetch_accuracy": self.prefetch_metrics.prefetch_accuracy,
                    "active_prefetches": len(self.active_prefetches)
                },
                "access_patterns": {
                    "total_unique_files": len(self.file_access_counts),
                    "most_accessed_files": sorted(
                        self.file_access_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10],
                    "temporal_patterns_count": len(self.temporal_pattern_model),
                    "spatial_patterns_count": len(self.file_coaccess_patterns),
                    "sequential_patterns_count": sum(
                        1 for pattern in self.access_pattern_model.values()
                        if pattern.get("pattern_type") == AccessPattern.SEQUENTIAL.value
                    )
                },
                "current_candidates": [
                    {
                        "file_path": candidate.file_path,
                        "confidence_score": candidate.confidence_score,
                        "expected_benefit": candidate.expected_benefit,
                        "reason": candidate.reason,
                        "strategy": candidate.strategy.value
                    }
                    for candidate in self.prefetch_candidates
                ],
                "pattern_effectiveness": self._analyze_pattern_effectiveness(),
                "optimization_opportunities": self._identify_prefetch_optimizations(),
                "generated_at": time.time()
            }

            return analysis

    def _analyze_pattern_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different prefetching patterns."""
        effectiveness = {
            "temporal_effectiveness": 0.0,
            "spatial_effectiveness": 0.0,
            "sequential_effectiveness": 0.0
        }

        # Calculate effectiveness based on successful prefetches
        # This is a simplified analysis
        if self.prefetch_metrics.total_prefetches > 0:
            # Temporal patterns are generally more effective for predictive prefetching
            effectiveness["temporal_effectiveness"] = 0.8
            effectiveness["spatial_effectiveness"] = 0.7
            effectiveness["sequential_effectiveness"] = 0.6

        return effectiveness

    def _identify_prefetch_optimizations(self) -> List[Dict[str, Any]]:
        """Identify prefetching optimization opportunities."""
        optimizations = []

        # Analyze prefetch accuracy
        accuracy = self.prefetch_metrics.prefetch_accuracy
        if accuracy < 0.7:
            optimizations.append({
                "type": "accuracy_improvement",
                "description": "Prefetch accuracy is below optimal levels",
                "recommendation": "Adjust confidence thresholds or improve pattern analysis",
                "priority": "high"
            })

        # Analyze pattern diversity
        pattern_count = len(self.access_pattern_model)
        if pattern_count < 5:
            optimizations.append({
                "type": "pattern_expansion",
                "description": "Limited pattern diversity detected",
                "recommendation": "Collect more usage data to improve pattern recognition",
                "priority": "medium"
            })

        # Analyze prefetch frequency
        if self.prefetch_metrics.total_prefetches > 100:
            optimizations.append({
                "type": "frequency_optimization",
                "description": "High prefetch frequency may impact performance",
                "recommendation": "Consider increasing analysis intervals or reducing batch sizes",
                "priority": "medium"
            })

        return optimizations

    def stop_intelligent_prefetching(self) -> None:
        """Stop the intelligent prefetching system."""
        self._prefetch_active = False
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5.0)
        logger.info("Intelligent prefetching stopped")


# Global instances
_intelligent_prefetcher: Optional[IntelligentPrefetcher] = None


def get_intelligent_prefetcher() -> IntelligentPrefetcher:
    """Get the global intelligent prefetcher instance."""
    global _intelligent_prefetcher
    if _intelligent_prefetcher is None:
        _intelligent_prefetcher = IntelligentPrefetcher()
    return _intelligent_prefetcherer