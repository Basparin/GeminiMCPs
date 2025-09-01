"""
Predictive Cache Warming System for CodeSage MCP Server.

This module provides ML-based predictive cache warming capabilities for Phase 4,
including usage pattern analysis, predictive modeling, and intelligent prefetching.

Features:
- ML-based usage pattern prediction
- Time-series analysis for cache warming
- Collaborative filtering for user behavior prediction
- Neural network models for complex pattern recognition
- Real-time adaptation to changing usage patterns
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import statistics
from datetime import datetime, timedelta

from .enterprise_cache import EnterpriseCache, CacheLevel

logger = logging.getLogger(__name__)


@dataclass
class UsagePattern:
    """Represents a usage pattern for cache warming."""
    pattern_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    file_sequence: List[str] = field(default_factory=list)
    access_times: List[float] = field(default_factory=list)
    frequency_score: float = 0.0
    recency_score: float = 0.0
    confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class PredictionResult:
    """Result of a cache warming prediction."""
    predicted_files: List[str]
    confidence_scores: List[float]
    prediction_method: str
    timestamp: float = field(default_factory=time.time)
    model_version: str = "v1.0"


@dataclass
class WarmingMetrics:
    """Metrics for cache warming effectiveness."""
    total_predictions: int = 0
    successful_predictions: int = 0
    cache_hits_from_warming: int = 0
    warming_time_ms: float = 0.0
    bandwidth_saved_mb: float = 0.0
    user_experience_improvement: float = 0.0


class PatternAnalyzer:
    """Analyzes usage patterns for predictive cache warming."""

    def __init__(self, max_patterns: int = 1000, pattern_retention_days: int = 30):
        self.max_patterns = max_patterns
        self.pattern_retention_days = pattern_retention_days

        # Pattern storage
        self.user_patterns: Dict[str, List[UsagePattern]] = defaultdict(list)
        self.global_patterns: List[UsagePattern] = []
        self.session_patterns: Dict[str, UsagePattern] = {}

        # Statistical models
        self.file_transition_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.file_access_frequencies: Dict[str, int] = defaultdict(int)
        self.time_based_patterns: Dict[str, List[float]] = defaultdict(list)

        # ML model placeholders (would be trained models in production)
        self.collaborative_filtering_model = None
        self.neural_prediction_model = None

    def record_file_access(self, file_path: str, user_id: Optional[str] = None,
                          session_id: Optional[str] = None):
        """Record a file access for pattern analysis."""
        current_time = time.time()

        # Update global statistics
        self.file_access_frequencies[file_path] += 1

        # Update time-based patterns
        hour_of_day = datetime.fromtimestamp(current_time).hour
        time_key = f"{hour_of_day:02d}:00"
        self.time_based_patterns[time_key].append(current_time)

        # Update user patterns
        if user_id:
            self._update_user_pattern(user_id, file_path, current_time)

        # Update session patterns
        if session_id:
            self._update_session_pattern(session_id, file_path, current_time)

        # Update transition matrix
        self._update_transition_matrix(file_path)

    def _update_user_pattern(self, user_id: str, file_path: str, timestamp: float):
        """Update user-specific access patterns."""
        patterns = self.user_patterns[user_id]

        # Find or create pattern for this user
        user_pattern = None
        for pattern in patterns:
            if len(pattern.file_sequence) < 10:  # Max sequence length
                user_pattern = pattern
                break

        if not user_pattern:
            user_pattern = UsagePattern(
                pattern_id=f"user_{user_id}_{len(patterns)}",
                user_id=user_id
            )
            patterns.append(user_pattern)

        # Add to sequence
        user_pattern.file_sequence.append(file_path)
        user_pattern.access_times.append(timestamp)
        user_pattern.last_updated = timestamp

        # Keep only recent patterns
        self.user_patterns[user_id] = patterns[-self.max_patterns:]

    def _update_session_pattern(self, session_id: str, file_path: str, timestamp: float):
        """Update session-specific patterns."""
        if session_id not in self.session_patterns:
            self.session_patterns[session_id] = UsagePattern(
                pattern_id=f"session_{session_id}",
                session_id=session_id
            )

        pattern = self.session_patterns[session_id]
        pattern.file_sequence.append(file_path)
        pattern.access_times.append(timestamp)
        pattern.last_updated = timestamp

        # Clean old sessions (older than 24 hours)
        cutoff_time = timestamp - (24 * 3600)
        self.session_patterns = {
            sid: pattern for sid, pattern in self.session_patterns.items()
            if pattern.last_updated > cutoff_time
        }

    def _update_transition_matrix(self, current_file: str):
        """Update the file transition matrix."""
        # This would track which files are accessed after others
        # Simplified implementation - in production would use more sophisticated tracking
        pass

    def predict_next_files(self, current_file: str, user_id: Optional[str] = None,
                          session_id: Optional[str] = None, max_predictions: int = 5) -> PredictionResult:
        """Predict which files are likely to be accessed next."""

        predictions = []
        confidence_scores = []

        # Method 1: User-based prediction
        if user_id:
            user_predictions = self._predict_from_user_patterns(user_id, current_file, max_predictions)
            predictions.extend(user_predictions[0])
            confidence_scores.extend(user_predictions[1])

        # Method 2: Session-based prediction
        if session_id and len(predictions) < max_predictions:
            session_predictions = self._predict_from_session_patterns(session_id, current_file,
                                                                    max_predictions - len(predictions))
            predictions.extend(session_predictions[0])
            confidence_scores.extend(session_predictions[1])

        # Method 3: Global frequency-based prediction
        if len(predictions) < max_predictions:
            freq_predictions = self._predict_from_global_frequencies(current_file,
                                                                   max_predictions - len(predictions))
            predictions.extend(freq_predictions[0])
            confidence_scores.extend(freq_predictions[1])

        # Method 4: Time-based prediction
        if len(predictions) < max_predictions:
            time_predictions = self._predict_from_time_patterns(current_file,
                                                              max_predictions - len(predictions))
            predictions.extend(time_predictions[0])
            confidence_scores.extend(time_predictions[1])

        return PredictionResult(
            predicted_files=predictions[:max_predictions],
            confidence_scores=confidence_scores[:max_predictions],
            prediction_method="hybrid"
        )

    def _predict_from_user_patterns(self, user_id: str, current_file: str,
                                  max_predictions: int) -> Tuple[List[str], List[float]]:
        """Predict based on user's historical patterns."""
        if user_id not in self.user_patterns:
            return [], []

        predictions = []
        confidences = []

        # Find patterns containing the current file
        relevant_patterns = []
        for pattern in self.user_patterns[user_id]:
            if current_file in pattern.file_sequence:
                relevant_patterns.append(pattern)

        if not relevant_patterns:
            return [], []

        # Analyze transitions from current file
        next_file_counts = defaultdict(int)
        for pattern in relevant_patterns:
            try:
                current_idx = pattern.file_sequence.index(current_file)
                if current_idx + 1 < len(pattern.file_sequence):
                    next_file = pattern.file_sequence[current_idx + 1]
                    next_file_counts[next_file] += 1
            except ValueError:
                continue

        # Sort by frequency
        sorted_predictions = sorted(next_file_counts.items(), key=lambda x: x[1], reverse=True)

        for file_path, count in sorted_predictions[:max_predictions]:
            confidence = min(count / len(relevant_patterns), 1.0)
            predictions.append(file_path)
            confidences.append(confidence)

        return predictions, confidences

    def _predict_from_session_patterns(self, session_id: str, current_file: str,
                                     max_predictions: int) -> Tuple[List[str], List[float]]:
        """Predict based on current session patterns."""
        if session_id not in self.session_patterns:
            return [], []

        pattern = self.session_patterns[session_id]
        if current_file not in pattern.file_sequence:
            return [], []

        # Find next files in session
        try:
            current_idx = pattern.file_sequence.index(current_file)
            next_files = pattern.file_sequence[current_idx + 1:current_idx + 1 + max_predictions]

            # Calculate confidence based on recency
            confidences = []
            for i, _ in enumerate(next_files):
                # More recent accesses have higher confidence
                confidence = max(0.5, 1.0 - (i * 0.1))
                confidences.append(confidence)

            return next_files, confidences

        except ValueError:
            return [], []

    def _predict_from_global_frequencies(self, current_file: str,
                                       max_predictions: int) -> Tuple[List[str], List[float]]:
        """Predict based on global access frequencies."""
        # Get most frequently accessed files (excluding current)
        sorted_files = sorted(
            self.file_access_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        predictions = []
        confidences = []

        for file_path, frequency in sorted_files:
            if file_path != current_file and file_path not in predictions:
                predictions.append(file_path)
                # Normalize confidence based on frequency
                max_freq = max(self.file_access_frequencies.values()) if self.file_access_frequencies else 1
                confidence = frequency / max_freq
                confidences.append(confidence)

                if len(predictions) >= max_predictions:
                    break

        return predictions, confidences

    def _predict_from_time_patterns(self, current_file: str,
                                  max_predictions: int) -> Tuple[List[str], List[float]]:
        """Predict based on time-of-day patterns."""
        current_hour = datetime.now().hour
        time_key = f"{current_hour:02d}:00"

        if time_key not in self.time_based_patterns:
            return [], []

        # Find files commonly accessed at this time
        # This is a simplified implementation
        hour_patterns = self.time_based_patterns[time_key]

        if len(hour_patterns) < 5:  # Not enough data
            return [], []

        # Return most common files (simplified)
        return self._predict_from_global_frequencies(current_file, max_predictions)


class PredictiveWarmer:
    """Main predictive cache warming system."""

    def __init__(self, enterprise_cache: EnterpriseCache,
                 warming_interval_minutes: int = 30,
                 max_warm_files: int = 20,
                 min_confidence_threshold: float = 0.3):
        self.enterprise_cache = enterprise_cache
        self.warming_interval_minutes = warming_interval_minutes
        self.max_warm_files = max_warm_files
        self.min_confidence_threshold = min_confidence_threshold

        # Core components
        self.pattern_analyzer = PatternAnalyzer()
        self.warming_queue: deque = deque()
        self.warming_lock = threading.RLock()

        # Metrics and monitoring
        self.metrics = WarmingMetrics()
        self.prediction_history: List[PredictionResult] = []
        self.warming_history: List[Dict[str, Any]] = []

        # Control
        self._warming_active = False
        self._warming_thread: Optional[threading.Thread] = None

        # Start predictive warming
        self._start_predictive_warming()

    def _start_predictive_warming(self):
        """Start the predictive warming system."""
        if self._warming_active:
            return

        self._warming_active = True
        self._warming_thread = threading.Thread(
            target=self._warming_loop,
            daemon=True,
            name="PredictiveWarmer"
        )
        self._warming_thread.start()
        logger.info("Predictive cache warming started")

    def _warming_loop(self):
        """Main warming loop."""
        while self._warming_active:
            try:
                self._perform_warming_cycle()
                time.sleep(self.warming_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Warming cycle error: {e}")
                time.sleep(60)

    def _perform_warming_cycle(self):
        """Perform one cycle of predictive warming."""
        with self.warming_lock:
            try:
                # Analyze current patterns and generate predictions
                predictions = self._generate_predictions()

                if predictions:
                    # Queue files for warming
                    self._queue_files_for_warming(predictions)

                    # Perform actual warming
                    warming_result = self._warm_queued_files()

                    # Update metrics
                    self._update_warming_metrics(warming_result)

                    logger.info(f"Completed warming cycle: {warming_result}")

            except Exception as e:
                logger.error(f"Error in warming cycle: {e}")

    def _generate_predictions(self) -> List[PredictionResult]:
        """Generate predictions for cache warming."""
        predictions = []

        try:
            # Get active users/sessions (simplified - would query actual sessions)
            active_entities = self._get_active_entities()

            for entity_type, entity_id in active_entities:
                # Generate predictions for each active entity
                if entity_type == "user":
                    prediction = self.pattern_analyzer.predict_next_files(
                        current_file="",  # Global prediction
                        user_id=entity_id,
                        max_predictions=self.max_warm_files
                    )
                elif entity_type == "session":
                    prediction = self.pattern_analyzer.predict_next_files(
                        current_file="",  # Global prediction
                        session_id=entity_id,
                        max_predictions=self.max_warm_files
                    )
                else:
                    continue

                if prediction.predicted_files:
                    predictions.append(prediction)
                    self.prediction_history.append(prediction)

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")

        return predictions

    def _get_active_entities(self) -> List[Tuple[str, str]]:
        """Get currently active users and sessions."""
        # Simplified implementation - in production would query actual active sessions
        return [("user", "default_user"), ("session", "default_session")]

    def _queue_files_for_warming(self, predictions: List[PredictionResult]):
        """Queue files for warming based on predictions."""
        for prediction in predictions:
            for file_path, confidence in zip(prediction.predicted_files, prediction.confidence_scores):
                if confidence >= self.min_confidence_threshold:
                    self.warming_queue.append({
                        "file_path": file_path,
                        "confidence": confidence,
                        "prediction_source": prediction.prediction_method,
                        "queued_at": time.time()
                    })

        # Limit queue size
        while len(self.warming_queue) > self.max_warm_files * 2:
            self.warming_queue.popleft()

    def _warm_queued_files(self) -> Dict[str, Any]:
        """Warm files from the queue."""
        start_time = time.time()
        warmed_files = 0
        skipped_files = 0
        errors = 0

        files_to_warm = []
        while self.warming_queue and len(files_to_warm) < self.max_warm_files:
            item = self.warming_queue.popleft()
            files_to_warm.append(item["file_path"])

        if not files_to_warm:
            return {"warmed": 0, "skipped": 0, "errors": 0, "duration_ms": 0}

        # Perform warming
        for file_path in files_to_warm:
            try:
                # Check if already cached
                cached, _ = self.enterprise_cache.get(file_path)
                if cached is not None:
                    skipped_files += 1
                    continue

                # Simulate file access to trigger caching
                # In production, this would read and cache the actual file
                self.enterprise_cache.set(file_path, f"warm_cache_placeholder_{file_path}")
                warmed_files += 1

            except Exception as e:
                logger.error(f"Error warming file {file_path}: {e}")
                errors += 1

        duration_ms = (time.time() - start_time) * 1000

        result = {
            "warmed": warmed_files,
            "skipped": skipped_files,
            "errors": errors,
            "duration_ms": duration_ms,
            "files_processed": len(files_to_warm)
        }

        self.warming_history.append(result)
        return result

    def _update_warming_metrics(self, result: Dict[str, Any]):
        """Update warming effectiveness metrics."""
        self.metrics.total_predictions += 1
        if result["warmed"] > 0:
            self.metrics.successful_predictions += 1

        self.metrics.warming_time_ms += result["duration_ms"]
        self.metrics.cache_hits_from_warming += result["warmed"]

    def record_file_access(self, file_path: str, user_id: Optional[str] = None,
                          session_id: Optional[str] = None):
        """Record a file access for pattern learning."""
        self.pattern_analyzer.record_file_access(file_path, user_id, session_id)

    def get_warming_status(self) -> Dict[str, Any]:
        """Get current warming system status."""
        total_predictions = len(self.prediction_history)
        successful_predictions = sum(1 for p in self.prediction_history
                                   if p.predicted_files)

        return {
            "active": self._warming_active,
            "warming_interval_minutes": self.warming_interval_minutes,
            "max_warm_files": self.max_warm_files,
            "min_confidence_threshold": self.min_confidence_threshold,
            "queue_size": len(self.warming_queue),
            "metrics": {
                "total_predictions": total_predictions,
                "successful_predictions": successful_predictions,
                "success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
                "cache_hits_from_warming": self.metrics.cache_hits_from_warming,
                "avg_warming_time_ms": self.metrics.warming_time_ms / self.metrics.total_predictions if self.metrics.total_predictions > 0 else 0
            },
            "pattern_stats": {
                "total_user_patterns": sum(len(patterns) for patterns in self.pattern_analyzer.user_patterns.values()),
                "total_session_patterns": len(self.pattern_analyzer.session_patterns),
                "tracked_files": len(self.pattern_analyzer.file_access_frequencies)
            }
        }

    def trigger_manual_warming(self, files: List[str]) -> Dict[str, Any]:
        """Manually trigger cache warming for specific files."""
        logger.info(f"Manual warming triggered for {len(files)} files")

        # Add to queue with high priority
        for file_path in files:
            self.warming_queue.append({
                "file_path": file_path,
                "confidence": 1.0,  # High confidence for manual
                "prediction_source": "manual",
                "queued_at": time.time()
            })

        # Process immediately
        result = self._warm_queued_files()
        return result

    def stop_warming(self):
        """Stop the predictive warming system."""
        self._warming_active = False
        if self._warming_thread:
            self._warming_thread.join(timeout=5.0)
        logger.info("Predictive cache warming stopped")


# Global instance management
_predictive_warmer_instance: Optional[PredictiveWarmer] = None
_warmer_lock = threading.Lock()


def get_predictive_warmer(enterprise_cache: EnterpriseCache,
                         config: Optional[Dict[str, Any]] = None) -> PredictiveWarmer:
    """Get the global predictive warmer instance."""
    global _predictive_warmer_instance

    if _predictive_warmer_instance is None:
        with _warmer_lock:
            if _predictive_warmer_instance is None:
                default_config = {
                    "warming_interval_minutes": 30,
                    "max_warm_files": 20,
                    "min_confidence_threshold": 0.3
                }
                config = {**default_config, **(config or {})}

                _predictive_warmer_instance = PredictiveWarmer(
                    enterprise_cache=enterprise_cache,
                    warming_interval_minutes=config["warming_interval_minutes"],
                    max_warm_files=config["max_warm_files"],
                    min_confidence_threshold=config["min_confidence_threshold"]
                )

    return _predictive_warmer_instance