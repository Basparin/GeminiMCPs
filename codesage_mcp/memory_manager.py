"""
Memory Management Module for CodeSage MCP Server.

This module provides memory optimization features including:
- Memory-mapped FAISS indexes
- Model caching with TTL
- Memory monitoring with psutil
- Configurable memory limits
- Automatic cleanup procedures

Classes:
    MemoryManager: Manages memory usage and optimization.
    ModelCache: Handles model loading and caching.
"""

import psutil
import threading
import time
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)

from .config import (
    ENABLE_MEMORY_MAPPED_INDEXES,
    INDEX_TYPE,
    MAX_MEMORY_MB,
    MODEL_CACHE_TTL_MINUTES,
    ENABLE_MODEL_QUANTIZATION,
    ENABLE_MEMORY_MONITORING,
)

# Import custom exceptions
from .exceptions import BaseMCPError


class ModelCache:
    """Handles model loading and caching with TTL support."""

    def __init__(self, ttl_minutes: int = 60):
        self.ttl_minutes = ttl_minutes
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_model(self, model_name: str) -> Tuple[Optional[SentenceTransformer], bool]:
        """Get a model from cache or return None if not available/expired."""
        with self._lock:
            if model_name in self._cache:
                entry = self._cache[model_name]
                if datetime.now() < entry["expires_at"]:
                    return entry["model"], True
                else:
                    # Remove expired model
                    del self._cache[model_name]
            return None, False

    def store_model(self, model_name: str, model: SentenceTransformer) -> None:
        """Store a model in cache with TTL."""
        with self._lock:
            expires_at = datetime.now() + timedelta(minutes=self.ttl_minutes)
            self._cache[model_name] = {
                "model": model,
                "expires_at": expires_at,
                "created_at": datetime.now(),
            }

    def clear_expired(self) -> int:
        """Clear expired models and return count of cleared models."""
        with self._lock:
            expired = []
            for model_name, entry in self._cache.items():
                if datetime.now() >= entry["expires_at"]:
                    expired.append(model_name)

            for model_name in expired:
                del self._cache[model_name]

            return len(expired)

    def clear_all(self) -> int:
        """Clear all cached models and return count."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_models": len(self._cache),
                "model_names": list(self._cache.keys()),
                "ttl_minutes": self.ttl_minutes,
            }


class MemoryManager:
    """Manages memory usage and optimization for FAISS indexes and models."""

    def __init__(self):
        self.process = psutil.Process()
        self.model_cache = ModelCache(MODEL_CACHE_TTL_MINUTES)
        self.memory_mapped_indexes: Dict[str, faiss.Index] = {}
        self._lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        if ENABLE_MEMORY_MONITORING:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory, daemon=True
        )
        self._monitoring_thread.start()

    def _monitor_memory(self) -> None:
        """Background thread to monitor memory usage."""
        while not self._stop_monitoring.is_set():
            try:
                memory_mb = self.get_memory_usage_mb()
                if memory_mb > MAX_MEMORY_MB:
                    logger.warning(
                        f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({MAX_MEMORY_MB}MB)"
                    )
                    self._cleanup_memory()

                # Clear expired models periodically
                cleared = self.model_cache.clear_expired()
                if cleared > 0:
                    logger.info(f"Cleared {cleared} expired models from cache")

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": memory_percent,
            "limit_mb": MAX_MEMORY_MB,
            "available_mb": MAX_MEMORY_MB - (memory_info.rss / 1024 / 1024),
            "memory_mapped_indexes": len(self.memory_mapped_indexes),
            "model_cache_stats": self.model_cache.get_stats(),
        }

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup when usage is high."""
        logger.info("Performing memory cleanup...")

        # Clear expired models
        self.model_cache.clear_expired()

        # If still high memory, clear all model cache
        if self.get_memory_usage_mb() > MAX_MEMORY_MB * 0.9:
            cleared_all = self.model_cache.clear_all()
            logger.warning(
                f"Cleared all {cleared_all} models from cache due to high memory usage"
            )

        # Force garbage collection
        import gc

        gc.collect()

    def load_model(self, model_name: str) -> SentenceTransformer:
        """Load a model with caching support."""
        # Try to get from cache first
        cached_model, cache_hit = self.model_cache.get_model(model_name)
        if cache_hit and cached_model:
            logger.debug(f"Loaded model '{model_name}' from cache")
            return cached_model

        logger.info(f"Loading model '{model_name}'...")
        try:
            model = SentenceTransformer(model_name)

            # Apply quantization if enabled
            if ENABLE_MODEL_QUANTIZATION:
                try:
                    model = self._quantize_model(model)
                    logger.info(f"Applied quantization to model '{model_name}'")
                except Exception as e:
                    logger.warning(f"Could not quantize model '{model_name}': {e}")

            # Cache the model
            self.model_cache.store_model(model_name, model)
            logger.debug(f"Cached model '{model_name}'")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    def _quantize_model(self, model: SentenceTransformer) -> SentenceTransformer:
        """Apply quantization to reduce model memory usage."""
        # This is a placeholder for model quantization
        # In a real implementation, you would use libraries like:
        # - torch.quantization for PyTorch models
        # - transformers quantization methods
        # - ONNX quantization

        # For now, we'll just return the original model
        # TODO: Implement actual quantization when needed
        return model

    def load_faiss_index(
        self, index_path: str, memory_mapped: bool = None
    ) -> faiss.Index:
        """Load FAISS index with optional memory mapping."""
        if memory_mapped is None:
            memory_mapped = ENABLE_MEMORY_MAPPED_INDEXES

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        try:
            if memory_mapped:
                logger.info(f"Loading memory-mapped FAISS index: {index_path}")
                index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
                with self._lock:
                    self.memory_mapped_indexes[str(index_path)] = index
            else:
                logger.info(f"Loading FAISS index into memory: {index_path}")
                index = faiss.read_index(str(index_path))

            return index

        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index '{index_path}': {e}")

    def create_optimized_index(
        self, embeddings: np.ndarray, index_type: str = None
    ) -> faiss.Index:
        """Create an optimized FAISS index based on dataset size and configuration."""
        if index_type is None:
            index_type = INDEX_TYPE

        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        logger.info(
            f"Creating {index_type} index for {n_vectors} vectors of dimension {dimension}"
        )

        if index_type == "auto":
            # Auto-select index type based on dataset size
            if n_vectors < 1000:
                index_type = "flat"
            elif n_vectors < 10000:
                index_type = "ivf_small"
            else:
                index_type = "ivf"

        if index_type == "flat":
            return faiss.IndexFlatL2(dimension)

        elif index_type in ["ivf", "ivf_small"]:
            # IVF indexes require training
            nlist = min(
                100, max(4, n_vectors // 39)
            )  # Rule of thumb: nlist = 4*sqrt(n)

            if index_type == "ivf_small":
                nlist = min(nlist, 50)  # Smaller IVF for smaller datasets

            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters...")
            if n_vectors >= nlist:
                index.train(embeddings.astype(np.float32))
            else:
                logger.warning("Not enough vectors for training, using Flat index")
                return faiss.IndexFlatL2(dimension)

            return index

        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def unload_memory_mapped_index(self, index_path: str) -> None:
        """Unload a memory-mapped index to free memory."""
        with self._lock:
            if index_path in self.memory_mapped_indexes:
                del self.memory_mapped_indexes[index_path]
                logger.info(f"Unloaded memory-mapped index: {index_path}")

    def cleanup(self) -> None:
        """Perform full cleanup of memory resources."""
        logger.info("Performing full memory cleanup...")

        # Stop monitoring thread
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)

        # Clear model cache
        cleared_models = self.model_cache.clear_all()

        # Clear memory-mapped indexes
        with self._lock:
            cleared_indexes = len(self.memory_mapped_indexes)
            self.memory_mapped_indexes.clear()

        # Force garbage collection
        import gc

        gc.collect()

        logger.info(
            f"Memory cleanup completed: {cleared_models} models, {cleared_indexes} indexes cleared"
        )

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup memory manager on destruction: {cleanup_error}")


# Global memory manager instance
_memory_manager_instance: Optional[MemoryManager] = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager_instance

    if _memory_manager_instance is None:
        with _memory_manager_lock:
            if _memory_manager_instance is None:
                _memory_manager_instance = MemoryManager()

    return _memory_manager_instance
