"""
Comprehensive Unit Tests for Memory Manager Module.

This module contains unit tests for the MemoryManager class and related components,
focusing on memory monitoring, model caching, memory-mapped indexes, and cleanup procedures.
"""

import pytest
import time
import psutil
from unittest.mock import MagicMock, patch
import numpy as np
import faiss
from datetime import datetime, timedelta

from codesage_mcp.features.memory_management.memory_manager import (
    MemoryManager,
    ModelCache,
    get_memory_manager
)


class TestModelCache:
    """Test cases for ModelCache class."""

    def test_initialization(self, model_cache):
        """Test ModelCache initialization with proper attributes."""
        assert model_cache.ttl_minutes == 30
        assert model_cache._cache == {}
        assert hasattr(model_cache._lock, 'acquire'), "Cache should have a lock for thread safety"

    def test_get_model_cache_hit(self, model_cache, mock_model):
        """Test retrieving a cached model that exists and is not expired."""
        # Store a model
        model_cache.store_model("test_model", mock_model)

        # Retrieve it
        retrieved_model, cache_hit = model_cache.get_model("test_model")

        assert retrieved_model is mock_model, "Should return the cached model"
        assert cache_hit is True, "Should indicate cache hit"

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

    def test_get_memory_usage_mb(self, memory_manager):
        """Test getting current memory usage in MB."""
        usage = memory_manager.get_memory_usage_mb()

        assert isinstance(usage, float), "Memory usage should be a float"
        assert usage >= 0, "Memory usage should be non-negative"

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

    def test_load_model_from_cache(self, memory_manager, mock_model):
        """Test loading a model that exists in cache."""
        # Pre-populate cache
        memory_manager.model_cache.store_model("test_model", mock_model)

        with patch('codesage_mcp.memory_manager.SentenceTransformer') as mock_st:
            loaded_model = memory_manager.load_model("test_model")

            assert loaded_model is mock_model, "Should return cached model"
            mock_st.assert_not_called(), "Should not create new model when cached"

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

    def test_load_faiss_index_memory_mapped(self, memory_manager, temp_faiss_index):
        """Test loading FAISS index with memory mapping enabled."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MAPPED_INDEXES', True):
            loaded_index = memory_manager.load_faiss_index(temp_faiss_index)

            assert isinstance(loaded_index, faiss.Index), "Should load FAISS index"
            assert temp_faiss_index in memory_manager.memory_mapped_indexes, "Should track memory-mapped index"

    def test_load_faiss_index_regular(self, memory_manager, temp_faiss_index):
        """Test loading FAISS index without memory mapping."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MAPPED_INDEXES', False):
            loaded_index = memory_manager.load_faiss_index(temp_faiss_index)

            assert isinstance(loaded_index, faiss.Index), "Should load FAISS index"
            assert temp_faiss_index not in memory_manager.memory_mapped_indexes, "Should not track as memory-mapped"

    def test_load_faiss_index_nonexistent(self):
        """Test loading non-existent FAISS index."""
        manager = MemoryManager()

        with pytest.raises(FileNotFoundError):
            manager.load_faiss_index("/nonexistent/path/index.faiss")

    def test_create_optimized_index_flat(self, memory_manager, sample_embeddings):
        """Test creating optimized flat index for small datasets."""
        with patch('codesage_mcp.memory_manager.INDEX_TYPE', 'flat'):
            index = memory_manager.create_optimized_index(sample_embeddings)

            assert isinstance(index, faiss.IndexFlatL2), "Should create Flat index"
            assert index.d == 128, "Should have correct dimension"

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

    def test_unload_memory_mapped_index(self, memory_manager):
        """Test unloading a memory-mapped index from tracking."""
        # Add a mock index to the memory-mapped indexes
        mock_index = MagicMock()
        test_path = "/test/path/index.faiss"
        memory_manager.memory_mapped_indexes[test_path] = mock_index

        memory_manager.unload_memory_mapped_index(test_path)

        assert test_path not in memory_manager.memory_mapped_indexes, "Should remove index from tracking"

    def test_cleanup(self, memory_manager):
        """Test full cleanup of memory resources."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            # Add some test data
            memory_manager.memory_mapped_indexes["/test/path"] = MagicMock()
            memory_manager.model_cache.store_model("test_model", MagicMock())

            # Wait a moment for monitoring thread to start
            time.sleep(0.1)

            # Check that stop event is initially clear
            assert not memory_manager._stop_monitoring.is_set(), "Stop event should be clear initially"

            # Perform cleanup
            memory_manager.cleanup()

            # Check cleanup results
            assert len(memory_manager.memory_mapped_indexes) == 0, "Should clear memory-mapped indexes"
            assert len(memory_manager.model_cache._cache) == 0, "Should clear model cache"

            # Check that stop event is set (thread should be stopping)
            assert memory_manager._stop_monitoring.is_set(), "Stop event should be set after cleanup"


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
        import codesage_mcp.features.memory_management.memory_manager
        codesage_mcp.memory_manager._memory_manager_instance = None

        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        assert manager1 is manager2, "Should return same instance (singleton)"
        assert isinstance(manager1, MemoryManager), "Should return MemoryManager instance"

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


    def test_create_optimized_index_unknown_type(self):
        """Test creating index with unknown type raises ValueError."""
        manager = MemoryManager()

        embeddings = np.random.rand(50, 128).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown index type"):
            manager.create_optimized_index(embeddings, index_type="unknown")

    def test_create_optimized_index_ivf_training_log(self):
        """Test IVF index creation with enough vectors to trigger training log."""
        manager = MemoryManager()

        # Create enough vectors for IVF training
        embeddings = np.random.rand(200, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.logger') as mock_logger:
            index = manager.create_optimized_index(embeddings, index_type="ivf")

            # Check that training log was called
            mock_logger.info.assert_any_call("Training IVF index with 5 clusters...")

    def test_create_optimized_index_insufficient_vectors(self):
        """Test IVF index creation with insufficient vectors for training."""
        manager = MemoryManager()

        # Create very few vectors (less than nlist minimum)
        embeddings = np.random.rand(2, 128).astype(np.float32)

        with patch('codesage_mcp.memory_manager.logger') as mock_logger:
            index = manager.create_optimized_index(embeddings, index_type="ivf")

            # Should fall back to Flat index
            assert isinstance(index, faiss.IndexFlatL2)
            mock_logger.warning.assert_called_with("Not enough vectors for training, using Flat index")

    def test_load_model_quantization_exception(self):
        """Test exception handling during model quantization."""
        manager = MemoryManager()
        mock_model = MagicMock()

        with patch('codesage_mcp.memory_manager.SentenceTransformer', return_value=mock_model) as mock_st:
            with patch('codesage_mcp.memory_manager.ENABLE_MODEL_QUANTIZATION', True):
                with patch.object(manager, '_quantize_model', side_effect=Exception("Quantization failed")) as mock_quantize:
                    with patch('codesage_mcp.memory_manager.logger') as mock_logger:
                        loaded_model = manager.load_model("test_model")

                        # Should still return the model despite quantization failure
                        assert loaded_model is mock_model
                        mock_logger.warning.assert_called_with("Could not quantize model 'test_model': Quantization failed")

    def test_monitor_memory_exception_handling(self):
        """Test exception handling in memory monitoring thread."""
        with patch('codesage_mcp.memory_manager.ENABLE_MEMORY_MONITORING', True):
            manager = MemoryManager()

            # Mock get_memory_usage_mb to raise exception
            with patch.object(manager, 'get_memory_usage_mb', side_effect=Exception("Memory check failed")):
                with patch('codesage_mcp.memory_manager.logger') as mock_logger:
                    # Call _monitor_memory directly (since it's hard to trigger in thread)
                    manager._monitor_memory()

                    # Should log the error
                    mock_logger.error.assert_called_with("Memory monitoring error: Memory check failed")

            # Cleanup
            manager.cleanup()

    def test_memory_manager_del_exception_handling(self):
        """Test exception handling in __del__ method."""
        manager = MemoryManager()

        # Mock cleanup to raise exception
        with patch.object(manager, 'cleanup', side_effect=Exception("Cleanup failed")):
            with patch('codesage_mcp.memory_manager.logger') as mock_logger:
                # Call __del__ directly
                manager.__del__()

                # Should log the warning
                mock_logger.warning.assert_called_with("Failed to cleanup memory manager on destruction: Cleanup failed")
class TestMemoryManagerPerformance:
    """Performance tests for MemoryManager."""

    def test_model_cache_performance(self, model_cache):
        """Test model cache performance with multiple operations."""
        # Test storing and retrieving multiple models
        models = {}
        for i in range(100):
            model_name = f"model_{i}"
            mock_model = MagicMock()
            models[model_name] = mock_model
            model_cache.store_model(model_name, mock_model)

        # Test retrieval performance
        start_time = time.time()

        for model_name, expected_model in models.items():
            retrieved_model, cache_hit = model_cache.get_model(model_name)
            assert cache_hit is True, f"Should have cache hit for {model_name}"
            assert retrieved_model is expected_model, f"Should return correct model for {model_name}"

        end_time = time.time()
        retrieval_time = end_time - start_time

        # Should be very fast (< 0.1 seconds for 100 models)
        assert retrieval_time < 0.1, f"Retrieval should be fast, took {retrieval_time:.3f}s"

    def test_memory_stats_performance(self, memory_manager):
        """Test performance of memory statistics collection."""
        start_time = time.time()

        # Collect stats multiple times
        for _ in range(100):
            stats = memory_manager.get_memory_stats()
            assert isinstance(stats, dict), "Should return dictionary of stats"

        end_time = time.time()
        stats_time = end_time - start_time

        # Should be reasonably fast (< 1 second for 100 calls)
        assert stats_time < 1.0, f"Stats collection should be fast, took {stats_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__])