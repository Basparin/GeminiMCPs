"""
Comprehensive Unit Tests for Memory Manager Module.

This module contains unit tests for the MemoryManager class and related components,
focusing on memory monitoring, model caching, memory-mapped indexes, and cleanup procedures.
"""

import pytest
import time
import threading
import psutil
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import tempfile
import os
import numpy as np
import faiss
from datetime import datetime, timedelta

from codesage_mcp.memory_manager import (
    MemoryManager,
    ModelCache,
    get_memory_manager,
    _memory_manager_instance,
    _memory_manager_lock
)
from codesage_mcp.config import (
    ENABLE_MEMORY_MAPPED_INDEXES,
    INDEX_TYPE,
    MAX_MEMORY_MB,
    MODEL_CACHE_TTL_MINUTES,
    ENABLE_MODEL_QUANTIZATION,
    ENABLE_MEMORY_MONITORING,
)


class TestModelCache:
    """Test cases for ModelCache class."""

    def test_initialization(self):
        """Test ModelCache initialization."""
        cache = ModelCache(ttl_minutes=30)
        assert cache.ttl_minutes == 30
        assert cache._cache == {}
        assert hasattr(cache._lock, 'acquire')  # Check it's a lock-like object

    def test_get_model_cache_hit(self):
        """Test getting a model that exists and is not expired."""
        cache = ModelCache(ttl_minutes=60)
        mock_model = MagicMock()

        # Store a model
        cache.store_model("test_model", mock_model)

        # Retrieve it
        retrieved_model, cache_hit = cache.get_model("test_model")

        assert retrieved_model is mock_model
        assert cache_hit is True

    def test_get_model_cache_miss_expired(self):
        """Test getting an expired model returns None."""
        cache = ModelCache(ttl_minutes=0)  # 0 minutes TTL for immediate expiry
        mock_model = MagicMock()

        # Store a model
        cache.store_model("test_model", mock_model)

        # Wait a bit to ensure expiry
        time.sleep(0.01)

        # Try to retrieve it
        retrieved_model, cache_hit = cache.get_model("test_model")

        assert retrieved_model is None
        assert cache_hit is False
        assert "test_model" not in cache._cache

    def test_get_model_nonexistent(self):
        """Test getting a model that doesn't exist."""
        cache = ModelCache(ttl_minutes=60)

        retrieved_model, cache_hit = cache.get_model("nonexistent_model")

        assert retrieved_model is None
        assert cache_hit is False

    def test_store_model(self):
        """Test storing a model in cache."""
        cache = ModelCache(ttl_minutes=45)
        mock_model = MagicMock()

        cache.store_model("test_model", mock_model)

        assert "test_model" in cache._cache
        entry = cache._cache["test_model"]
        assert entry["model"] is mock_model
        assert isinstance(entry["expires_at"], datetime)
        assert isinstance(entry["created_at"], datetime)

        # Check TTL calculation (allow for small timing differences)
        expected_expiry = entry["created_at"] + timedelta(minutes=45)
        time_diff = abs((entry["expires_at"] - expected_expiry).total_seconds())
        assert time_diff < 0.01  # Less than 10ms difference

    def test_clear_expired(self):
        """Test clearing expired models."""
        cache = ModelCache(ttl_minutes=0)  # Immediate expiry
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()

        # Store two models
        cache.store_model("model1", mock_model1)
        cache.store_model("model2", mock_model2)

        # Wait for expiry
        time.sleep(0.01)

        # Clear expired models
        cleared_count = cache.clear_expired()

        assert cleared_count == 2
        assert len(cache._cache) == 0

    def test_clear_all(self):
        """Test clearing all cached models."""
        cache = ModelCache(ttl_minutes=60)
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()

        # Store two models
        cache.store_model("model1", mock_model1)
        cache.store_model("model2", mock_model2)

        # Clear all
        cleared_count = cache.clear_all()

        assert cleared_count == 2
        assert len(cache._cache) == 0

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ModelCache(ttl_minutes=30)
        mock_model = MagicMock()

        cache.store_model("test_model", mock_model)

        stats = cache.get_stats()

        assert stats["cached_models"] == 1
        assert "test_model" in stats["model_names"]
        assert stats["ttl_minutes"] == 30


class TestMemoryManager:
    """Test cases for MemoryManager class."""

    def test_initialization(self):
        """Test MemoryManager initialization."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            manager = MemoryManager()

            assert isinstance(manager.process, psutil.Process)
            assert isinstance(manager.model_cache, ModelCache)
            assert manager.memory_mapped_indexes == {}
            assert hasattr(manager._lock, 'acquire')  # Check it's a lock-like object

            # Check if monitoring thread was started
            assert manager._monitoring_thread is not None
            assert manager._monitoring_thread.is_alive()

            # Cleanup
            manager.cleanup()

    def test_initialization_no_monitoring(self):
        """Test MemoryManager initialization with monitoring disabled."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', False):
            manager = MemoryManager()

            assert manager._monitoring_thread is None

    def test_get_memory_usage_mb(self):
        """Test getting current memory usage."""
        manager = MemoryManager()

        usage = manager.get_memory_usage_mb()

        assert isinstance(usage, float)
        assert usage >= 0

    def test_get_memory_stats(self):
        """Test getting detailed memory statistics."""
        manager = MemoryManager()

        stats = manager.get_memory_stats()

        required_keys = [
            'rss_mb', 'vms_mb', 'percent', 'limit_mb',
            'available_mb', 'memory_mapped_indexes', 'model_cache_stats'
        ]

        for key in required_keys:
            assert key in stats

        assert isinstance(stats['model_cache_stats'], dict)

    def test_cleanup_memory_high_usage(self):
        """Test memory cleanup when usage is high."""
        with patch('codesage_mcp.memory_manager.MAX_MEMORY_MB', 100):
            manager = MemoryManager()

            # Mock high memory usage
            with patch.object(manager, 'get_memory_usage_mb', return_value=95):
                # Add some expired models to cache
                manager.model_cache._cache["expired_model"] = {
                    "model": MagicMock(),
                    "expires_at": datetime.now() - timedelta(minutes=1),
                    "created_at": datetime.now() - timedelta(minutes=2)
                }

                manager._cleanup_memory()

                # Check that expired models were cleared
                assert len(manager.model_cache._cache) == 0

    def test_load_model_from_cache(self):
        """Test loading a model that's already in cache."""
        manager = MemoryManager()
        mock_model = MagicMock()

        # Pre-populate cache
        manager.model_cache.store_model("test_model", mock_model)

        with patch('codesage_mcp.memory_manager.SentenceTransformer') as mock_st:
            loaded_model = manager.load_model("test_model")

            assert loaded_model is mock_model
            mock_st.assert_not_called()  # Should not create new model

    def test_load_model_new(self):
        """Test loading a new model not in cache."""
        manager = MemoryManager()
        mock_model = MagicMock()

        with patch('codesage_mcp.memory_manager.SentenceTransformer', return_value=mock_model) as mock_st:
            with patch('codesage_mcp.memory_manager.ENABLE_MODEL_QUANTIZATION', False):
                loaded_model = manager.load_model("new_model")

                assert loaded_model is mock_model
                mock_st.assert_called_once_with("new_model")

                # Check it was cached
                cached_model, cache_hit = manager.model_cache.get_model("new_model")
                assert cache_hit is True
                assert cached_model is mock_model

    def test_load_model_with_quantization(self):
        """Test loading a model with quantization enabled."""
        manager = MemoryManager()
        mock_model = MagicMock()
        quantized_model = MagicMock()

        with patch('codesage_mcp.memory_manager.SentenceTransformer', return_value=mock_model) as mock_st:
            with patch('codesage_mcp.memory_manager.ENABLE_MODEL_QUANTIZATION', True):
                with patch.object(manager, '_quantize_model', return_value=quantized_model) as mock_quantize:
                    loaded_model = manager.load_model("quantized_model")

                    assert loaded_model is quantized_model
                    mock_quantize.assert_called_once_with(mock_model)

    def test_quantize_model_placeholder(self):
        """Test the placeholder quantization method."""
        manager = MemoryManager()
        mock_model = MagicMock()

        # The current implementation just returns the original model
        result = manager._quantize_model(mock_model)

        assert result is mock_model

    def test_load_faiss_index_memory_mapped(self):
        """Test loading FAISS index with memory mapping."""
        manager = MemoryManager()

        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create a simple FAISS index for testing
            dimension = 128
            index = faiss.IndexFlatL2(dimension)
            faiss.write_index(index, tmp_path)

            with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MAPPED_INDEXES', True):
                loaded_index = manager.load_faiss_index(tmp_path)

                assert isinstance(loaded_index, faiss.Index)
                assert tmp_path in manager.memory_mapped_indexes

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_faiss_index_regular(self):
        """Test loading FAISS index without memory mapping."""
        manager = MemoryManager()

        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create a simple FAISS index for testing
            dimension = 128
            index = faiss.IndexFlatL2(dimension)
            faiss.write_index(index, tmp_path)

            with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MAPPED_INDEXES', False):
                loaded_index = manager.load_faiss_index(tmp_path)

                assert isinstance(loaded_index, faiss.Index)
                assert tmp_path not in manager.memory_mapped_indexes

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_faiss_index_nonexistent(self):
        """Test loading non-existent FAISS index."""
        manager = MemoryManager()

        with pytest.raises(FileNotFoundError):
            manager.load_faiss_index("/nonexistent/path/index.faiss")

    def test_create_optimized_index_flat(self):
        """Test creating optimized flat index."""
        manager = MemoryManager()

        # Create test embeddings
        embeddings = np.random.rand(50, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.INDEX_TYPE', 'flat'):
            index = manager.create_optimized_index(embeddings)

            assert isinstance(index, faiss.IndexFlatL2)
            assert index.d == 128

    def test_create_optimized_index_ivf(self):
        """Test creating optimized IVF index."""
        manager = MemoryManager()

        # Create test embeddings (enough for IVF training)
        embeddings = np.random.rand(200, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.INDEX_TYPE', 'ivf'):
            index = manager.create_optimized_index(embeddings)

            assert isinstance(index, faiss.IndexIVFFlat)
            assert index.d == 128

    def test_create_optimized_index_auto_flat(self):
        """Test auto index type selection for small datasets."""
        manager = MemoryManager()

        # Small dataset should use flat index
        embeddings = np.random.rand(50, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.INDEX_TYPE', 'auto'):
            index = manager.create_optimized_index(embeddings)

            assert isinstance(index, faiss.IndexFlatL2)

    def test_create_optimized_index_auto_ivf(self):
        """Test auto index type selection for large datasets."""
        manager = MemoryManager()

        # Large dataset should use IVF index
        embeddings = np.random.rand(15000, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.INDEX_TYPE', 'auto'):
            index = manager.create_optimized_index(embeddings)

            assert isinstance(index, faiss.IndexIVFFlat)

    def test_unload_memory_mapped_index(self):
        """Test unloading a memory-mapped index."""
        manager = MemoryManager()

        # Add a mock index to the memory-mapped indexes
        mock_index = MagicMock()
        test_path = "/test/path/index.faiss"
        manager.memory_mapped_indexes[test_path] = mock_index

        manager.unload_memory_mapped_index(test_path)

        assert test_path not in manager.memory_mapped_indexes

    def test_cleanup(self):
        """Test full cleanup of memory resources."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            manager = MemoryManager()

            # Add some test data
            manager.memory_mapped_indexes["/test/path"] = MagicMock()
            manager.model_cache.store_model("test_model", MagicMock())

            # Wait a moment for monitoring thread to start
            time.sleep(0.1)

            # Check that stop event is initially clear
            assert not manager._stop_monitoring.is_set()

            # Perform cleanup
            manager.cleanup()

            # Check cleanup results
            assert len(manager.memory_mapped_indexes) == 0
            assert len(manager.model_cache._cache) == 0

            # Check that stop event is set (thread should be stopping)
            assert manager._stop_monitoring.is_set()

            # Thread might still be alive due to timeout, but it should be stopping
            # The important thing is that the cleanup was initiated properly


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager with other components."""

    def test_memory_monitoring_thread(self):
        """Test that memory monitoring thread works correctly."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            with patch('codesage_mcp.memory_manager.MAX_MEMORY_MB', 1000):  # High limit
                manager = MemoryManager()

                # Let monitoring thread run for a short time
                time.sleep(0.1)

                # Thread should still be alive
                assert manager._monitoring_thread.is_alive()

                # Cleanup
                manager.cleanup()

    def test_get_memory_manager_singleton(self):
        """Test that get_memory_manager returns a singleton instance."""
        # Reset the global instance
        import codesage_mcp.memory_manager
        codesage_mcp.memory_manager._memory_manager_instance = None

        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        assert manager1 is manager2
        assert isinstance(manager1, MemoryManager)

    def test_memory_threshold_handling(self):
        """Test memory threshold handling in monitoring."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            with patch('codesage_mcp.memory_manager.MAX_MEMORY_MB', 10):  # Very low limit
                manager = MemoryManager()

                # Mock high memory usage to trigger cleanup
                with patch.object(manager, 'get_memory_usage_mb', return_value=15):
                    with patch.object(manager, '_cleanup_memory') as mock_cleanup:
                        # Let monitoring thread detect high memory
                        time.sleep(0.1)

                        # Cleanup should have been called
                        mock_cleanup.assert_called()

                # Cleanup
                manager.cleanup()


class TestMemoryManagerErrorHandling:
    """Test error handling in MemoryManager."""

    def test_load_model_exception_handling(self):
        """Test exception handling when loading a model fails."""
        manager = MemoryManager()

        with patch('codesage_mcp.memory_manager.SentenceTransformer', side_effect=Exception("Model loading failed")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                manager.load_model("failing_model")

    def test_load_faiss_index_exception_handling(self):
        """Test exception handling when loading FAISS index fails."""
        manager = MemoryManager()

        with patch('pathlib.Path.exists', return_value=True), \
             patch('faiss.read_index', side_effect=Exception("Index loading failed")):
            with pytest.raises(RuntimeError, match="Failed to load FAISS index"):
                manager.load_faiss_index("/test/path/index.faiss")


class TestMemoryManagerPerformance:
    """Performance tests for MemoryManager."""

    def test_model_cache_performance(self):
        """Test model cache performance with multiple operations."""
        cache = ModelCache(ttl_minutes=60)

        # Test storing and retrieving multiple models
        models = {}
        for i in range(100):
            model_name = f"model_{i}"
            mock_model = MagicMock()
            models[model_name] = mock_model
            cache.store_model(model_name, mock_model)

        # Test retrieval performance
        import time
        start_time = time.time()

        for model_name, expected_model in models.items():
            retrieved_model, cache_hit = cache.get_model(model_name)
            assert cache_hit is True
            assert retrieved_model is expected_model

        end_time = time.time()
        retrieval_time = end_time - start_time

        # Should be very fast (< 0.1 seconds for 100 models)
        assert retrieval_time < 0.1

    def test_memory_stats_performance(self):
        """Test performance of memory statistics collection."""
        manager = MemoryManager()

        import time
        start_time = time.time()

        # Collect stats multiple times
        for _ in range(100):
            stats = manager.get_memory_stats()
            assert isinstance(stats, dict)

        end_time = time.time()
        stats_time = end_time - start_time

        # Should be reasonably fast (< 1 second for 100 calls)
        assert stats_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])