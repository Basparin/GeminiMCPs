"""
Comprehensive Unit Tests for Cache System Module.

This module contains unit tests for the IntelligentCache class and all related cache components,
focusing on LRU eviction, embedding caching, search result caching, file content caching,
and intelligent cache management features.
"""

import pytest
import time
import json
import hashlib
import threading
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import tempfile
import os
import shutil
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict

from codesage_mcp.cache import (
    LRUCache,
    EmbeddingCache,
    SearchResultCache,
    FileContentCache,
    IntelligentCache,
    get_cache_instance,
    reset_cache_instance,
    _cache_instance,
    _cache_lock
)
from codesage_mcp.config import (
    ENABLE_CACHING,
    EMBEDDING_CACHE_SIZE,
    SEARCH_CACHE_SIZE,
    FILE_CACHE_SIZE,
    CACHE_SIMILARITY_THRESHOLD,
    MAX_FILE_SIZE_MB,
    ENABLE_CACHE_PERSISTENCE,
    ENABLE_CACHE_WARMING,
)


class TestLRUCache:
    """Test cases for LRUCache class."""

    def test_initialization(self):
        """Test LRUCache initialization."""
        cache = LRUCache(max_size=100)
        assert cache.max_size == 100
        assert cache.cache == OrderedDict()
        assert hasattr(cache.lock, 'acquire')  # Check it's a lock-like object

    def test_get_existing_item(self):
        """Test getting an existing item moves it to end (most recently used)."""
        cache = LRUCache(max_size=10)

        # Add items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Get key2 (should move to end)
        result = cache.get("key2")

        assert result == "value2"
        # Check order: key1, key3, key2 (key2 moved to end)
        assert list(cache.cache.keys()) == ["key1", "key3", "key2"]

    def test_get_nonexistent_item(self):
        """Test getting a non-existent item returns None."""
        cache = LRUCache(max_size=10)

        result = cache.get("nonexistent")

        assert result is None

    def test_put_new_item(self):
        """Test putting a new item in cache."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")

        assert cache.cache["key1"] == "value1"
        assert len(cache.cache) == 1

    def test_put_existing_item(self):
        """Test putting an existing item updates it and moves to end."""
        cache = LRUCache(max_size=10)

        # Add initial items
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Update key1
        cache.put("key1", "new_value1")

        assert cache.cache["key1"] == "new_value1"
        # Check order: key2, key1 (key1 moved to end)
        assert list(cache.cache.keys()) == ["key2", "key1"]

    def test_put_eviction(self):
        """Test that putting items beyond max_size evicts least recently used."""
        cache = LRUCache(max_size=2)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1

        assert len(cache.cache) == 2
        assert "key1" not in cache.cache
        assert "key2" in cache.cache
        assert "key3" in cache.cache

    def test_delete_existing_item(self):
        """Test deleting an existing item."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        result = cache.delete("key1")

        assert result is True
        assert "key1" not in cache.cache

    def test_delete_nonexistent_item(self):
        """Test deleting a non-existent item."""
        cache = LRUCache(max_size=10)

        result = cache.delete("nonexistent")

        assert result is False

    def test_clear(self):
        """Test clearing all items from cache."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert len(cache.cache) == 0

    def test_size(self):
        """Test getting current cache size."""
        cache = LRUCache(max_size=10)

        assert cache.size() == 0

        cache.put("key1", "value1")
        assert cache.size() == 1

        cache.put("key2", "value2")
        assert cache.size() == 2

    def test_stats(self):
        """Test getting cache statistics."""
        cache = LRUCache(max_size=100)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["utilization"] == 0.02


class TestEmbeddingCache:
    """Test cases for EmbeddingCache class."""

    def test_initialization(self):
        """Test EmbeddingCache initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(max_size=100, cache_dir=temp_dir)

            assert cache.max_size == 100
            assert cache.cache_dir == Path(temp_dir)
            # Cache files are created lazily, so check they can be created
            assert cache.cache_file.parent.exists()
            # embeddings_file is also created lazily
            assert cache.embeddings_file.parent.exists()

    def test_get_cache_key(self):
        """Test cache key generation."""
        cache = EmbeddingCache()

        key = cache._get_cache_key("/path/to/file.py", "abc123")

        assert key == "/path/to/file.py:abc123"

    def test_hash_content(self):
        """Test content hashing."""
        cache = EmbeddingCache()

        hash1 = cache._hash_content("test content")
        hash2 = cache._hash_content("test content")
        hash3 = cache._hash_content("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hex length

    def test_get_embedding_hit(self):
        """Test getting cached embedding."""
        cache = EmbeddingCache()
        test_embedding = np.array([0.1, 0.2, 0.3])

        # Store embedding
        cache.store_embedding("/test/file.py", "test content", test_embedding)

        # Retrieve it
        retrieved = cache.get_embedding("/test/file.py", "test content")

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, test_embedding)

    def test_get_embedding_miss(self):
        """Test getting non-existent embedding."""
        cache = EmbeddingCache()

        retrieved = cache.get_embedding("/test/file.py", "test content")

        assert retrieved is None

    def test_get_embedding_content_changed(self):
        """Test getting embedding when content has changed."""
        cache = EmbeddingCache()
        old_embedding = np.array([0.1, 0.2, 0.3])
        new_embedding = np.array([0.4, 0.5, 0.6])

        # Store with old content
        cache.store_embedding("/test/file.py", "old content", old_embedding)

        # Try to get with new content
        retrieved = cache.get_embedding("/test/file.py", "new content")

        assert retrieved is None

    def test_store_embedding(self):
        """Test storing embedding in cache."""
        cache = EmbeddingCache()
        embedding = np.array([0.1, 0.2, 0.3, 0.4])

        cache.store_embedding("/test/file.py", "test content", embedding)

        # Check it was stored
        retrieved = cache.get_embedding("/test/file.py", "test content")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

        # Check file-to-keys mapping
        assert "/test/file.py" in cache.file_to_keys
        assert len(cache.file_to_keys["/test/file.py"]) == 1

    def test_invalidate_file(self):
        """Test invalidating all embeddings for a file."""
        cache = EmbeddingCache()

        # Store multiple embeddings for same file
        cache.store_embedding("/test/file.py", "content1", np.array([0.1, 0.2]))
        cache.store_embedding("/test/file.py", "content2", np.array([0.3, 0.4]))

        # Invalidate file
        invalidated_count = cache.invalidate_file("/test/file.py")

        assert invalidated_count == 2
        assert "/test/file.py" not in cache.file_to_keys

        # Check embeddings are gone
        assert cache.get_embedding("/test/file.py", "content1") is None
        assert cache.get_embedding("/test/file.py", "content2") is None

    def test_clear(self):
        """Test clearing all cached embeddings."""
        cache = EmbeddingCache()

        cache.store_embedding("/test/file1.py", "content1", np.array([0.1, 0.2]))
        cache.store_embedding("/test/file2.py", "content2", np.array([0.3, 0.4]))

        cache.clear()

        assert len(cache.embedding_cache.cache) == 0
        assert len(cache.file_metadata.cache) == 0
        assert len(cache.file_to_keys) == 0

    def test_persistent_cache_save_load(self):
        """Test saving and loading persistent cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(max_size=100, cache_dir=temp_dir)

            # Store some data
            embedding1 = np.array([0.1, 0.2, 0.3])
            embedding2 = np.array([0.4, 0.5, 0.6])
            cache.store_embedding("/test/file1.py", "content1", embedding1)
            cache.store_embedding("/test/file2.py", "content2", embedding2)

            # Save cache
            cache._save_persistent_cache()

            # Create new cache instance
            new_cache = EmbeddingCache(max_size=100, cache_dir=temp_dir)

            # Check data was loaded
            loaded1 = new_cache.get_embedding("/test/file1.py", "content1")
            loaded2 = new_cache.get_embedding("/test/file2.py", "content2")

            assert loaded1 is not None
            assert loaded2 is not None
            np.testing.assert_array_equal(loaded1, embedding1)
            np.testing.assert_array_equal(loaded2, embedding2)

    def test_stats(self):
        """Test getting embedding cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(max_size=100, cache_dir=temp_dir)

            cache.store_embedding("/test/file.py", "content", np.array([0.1, 0.2]))

            stats = cache.stats()

            assert "embedding_cache" in stats
            assert "file_metadata_cache" in stats
            assert stats["files_tracked"] == 1


class TestSearchResultCache:
    """Test cases for SearchResultCache class."""

    def test_initialization(self):
        """Test SearchResultCache initialization."""
        cache = SearchResultCache(max_size=50, similarity_threshold=0.8)

        assert cache.max_size == 50
        assert cache.similarity_threshold == 0.8
        assert isinstance(cache.cache, LRUCache)
        assert isinstance(cache.query_embeddings, LRUCache)

    def test_hash_query(self):
        """Test query hashing."""
        cache = SearchResultCache()

        hash1 = cache._hash_query("test query")
        hash2 = cache._hash_query("test query")
        hash3 = cache._hash_query("different query")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hex length

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        cache = SearchResultCache()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([1.0, 0.0, 0.0])  # Same as emb1

        similarity1 = cache._cosine_similarity(emb1, emb2)
        similarity2 = cache._cosine_similarity(emb1, emb3)

        assert similarity1 == 0.0  # Orthogonal vectors
        assert similarity2 == 1.0  # Identical vectors

    def test_get_similar_results_exact_match(self):
        """Test getting exact match from cache."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}, {"file": "test2.py", "score": 0.8},
                       {"file": "test3.py", "score": 0.7}, {"file": "test4.py", "score": 0.6},
                       {"file": "test5.py", "score": 0.5}]  # 5 results

        # Store results
        cache.store_results("test query", np.array([1.0, 0.0]), test_results)

        # Retrieve exact match
        results = cache.get_similar_results("test query", np.array([1.0, 0.0]), top_k=5)

        assert results == test_results

    def test_get_similar_results_similar_query(self):
        """Test getting similar results based on embedding similarity."""
        cache = SearchResultCache(similarity_threshold=0.8)
        test_results = [{"file": "test.py", "score": 0.9}, {"file": "test2.py", "score": 0.8},
                       {"file": "test3.py", "score": 0.7}, {"file": "test4.py", "score": 0.6},
                       {"file": "test5.py", "score": 0.5}]  # 5 results

        # Store results with one embedding
        cache.store_results("original query", np.array([1.0, 0.0, 0.0]), test_results)

        # Try to get with similar embedding
        similar_embedding = np.array([0.9, 0.0, 0.0])  # 0.9 similarity
        results = cache.get_similar_results("similar query", similar_embedding, top_k=5)

        assert results == test_results

    def test_get_similar_results_below_threshold(self):
        """Test that dissimilar queries don't return cached results."""
        cache = SearchResultCache(similarity_threshold=0.8)
        test_results = [{"file": "test.py", "score": 0.9}]

        # Store results with one embedding
        cache.store_results("original query", np.array([1.0, 0.0, 0.0]), test_results)

        # Try to get with dissimilar embedding
        dissimilar_embedding = np.array([0.0, 1.0, 0.0])  # 0.0 similarity
        results = cache.get_similar_results("dissimilar query", dissimilar_embedding, top_k=5)

        assert results is None

    def test_store_results(self):
        """Test storing search results."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}, {"file": "test2.py", "score": 0.8},
                       {"file": "test3.py", "score": 0.7}, {"file": "test4.py", "score": 0.6},
                       {"file": "test5.py", "score": 0.5}]  # 5 results
        query_embedding = np.array([1.0, 0.0])

        cache.store_results("test query", query_embedding, test_results)

        # Check results are stored
        results = cache.get_similar_results("test query", query_embedding, top_k=5)
        assert results == test_results

        # Check query embedding is stored
        query_hash = cache._hash_query("test query")
        stored_embedding = cache.query_embeddings.cache.get(query_hash)
        assert stored_embedding is not None
        np.testing.assert_array_equal(stored_embedding, query_embedding)

    def test_clear(self):
        """Test clearing all cached search results."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}]

        cache.store_results("test query", np.array([1.0, 0.0]), test_results)

        cache.clear()

        assert len(cache.cache.cache) == 0
        assert len(cache.query_embeddings.cache) == 0

    def test_stats(self):
        """Test getting search result cache statistics."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}, {"file": "test2.py", "score": 0.8},
                       {"file": "test3.py", "score": 0.7}, {"file": "test4.py", "score": 0.6},
                       {"file": "test5.py", "score": 0.5}]  # 5 results

        cache.store_results("test query", np.array([1.0, 0.0]), test_results)

        stats = cache.stats()

        assert "result_cache" in stats
        assert "query_embedding_cache" in stats
        assert stats["similarity_threshold"] == 0.85  # Default value

    def test_insufficient_results_check(self):
        """Test that cache returns None when stored results don't have enough entries."""
        cache = SearchResultCache()
        # Store results with only 3 entries
        test_results = [{"file": "test.py", "score": 0.9}, {"file": "test2.py", "score": 0.8},
                       {"file": "test3.py", "score": 0.7}]

        cache.store_results("test query", np.array([1.0, 0.0, 0.0]), test_results)

        # Try to get with top_k=5 (more than stored)
        results = cache.get_similar_results("test query", np.array([1.0, 0.0, 0.0]), top_k=5)

        assert results is None  # Should return None because len(test_results) < top_k

        # Try to get with top_k=3 (exact match)
        results = cache.get_similar_results("test query", np.array([1.0, 0.0, 0.0]), top_k=3)

        assert results == test_results  # Should return results because len(test_results) >= top_k


class TestFileContentCache:
    """Test cases for FileContentCache class."""

    def test_initialization(self):
        """Test FileContentCache initialization."""
        cache = FileContentCache(max_size=50, max_file_size=1024*1024)

        assert cache.max_size == 50
        assert cache.max_file_size == 1024*1024
        assert isinstance(cache.cache, LRUCache)
        assert isinstance(cache.file_sizes, LRUCache)

    def test_get_content_hit(self):
        """Test getting cached file content."""
        cache = FileContentCache()

        cache.store_content("/test/file.py", "test content")

        content = cache.get_content("/test/file.py")

        assert content == "test content"

    def test_get_content_miss(self):
        """Test getting non-existent file content."""
        cache = FileContentCache()

        content = cache.get_content("/nonexistent/file.py")

        assert content is None

    def test_store_content_success(self):
        """Test successfully storing file content."""
        cache = FileContentCache(max_file_size=100)

        result = cache.store_content("/test/file.py", "short content")

        assert result is True

        # Check content and size are stored
        content = cache.get_content("/test/file.py")
        assert content == "short content"

        size = cache.file_sizes.cache.get("/test/file.py")
        assert size == len("short content".encode('utf-8'))

    def test_store_content_too_large(self):
        """Test storing content that's too large."""
        cache = FileContentCache(max_file_size=10)  # Very small limit

        result = cache.store_content("/test/file.py", "this content is too long")

        assert result is False

        # Check nothing was stored
        content = cache.get_content("/test/file.py")
        assert content is None

    def test_invalidate_file(self):
        """Test invalidating cached file content."""
        cache = FileContentCache()

        cache.store_content("/test/file.py", "test content")

        result = cache.invalidate_file("/test/file.py")

        assert result is True
        assert cache.get_content("/test/file.py") is None
        assert "/test/file.py" not in cache.file_sizes.cache

    def test_clear(self):
        """Test clearing all cached file contents."""
        cache = FileContentCache()

        cache.store_content("/test/file1.py", "content1")
        cache.store_content("/test/file2.py", "content2")

        cache.clear()

        assert len(cache.cache.cache) == 0
        assert len(cache.file_sizes.cache) == 0

    def test_stats(self):
        """Test getting file content cache statistics."""
        cache = FileContentCache()

        cache.store_content("/test/file.py", "test content")

        stats = cache.stats()

        assert "content_cache" in stats
        assert "total_cached_size_bytes" in stats
        assert "total_cached_size_mb" in stats
        assert "max_file_size_bytes" in stats
        assert "max_file_size_mb" in stats


class TestIntelligentCache:
    """Test cases for IntelligentCache class."""

    def test_initialization(self):
        """Test IntelligentCache initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"embedding_cache_size": 100, "search_cache_size": 50}
            cache = IntelligentCache(cache_dir=temp_dir, config=config)

            assert cache.cache_dir == Path(temp_dir)
            assert cache.config["embedding_cache_size"] == 100
            assert cache.config["search_cache_size"] == 50
            assert isinstance(cache.embedding_cache, EmbeddingCache)
            assert isinstance(cache.search_cache, SearchResultCache)
            assert isinstance(cache.file_cache, FileContentCache)

    def test_get_embedding_hit(self):
        """Test getting embedding from intelligent cache."""
        cache = IntelligentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store embedding
        cache.store_embedding("/test/file.py", "content", embedding)

        # Retrieve it
        retrieved, hit = cache.get_embedding("/test/file.py", "content")

        assert hit is True
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_get_embedding_miss(self):
        """Test getting non-existent embedding."""
        cache = IntelligentCache()

        retrieved, hit = cache.get_embedding("/test/file.py", "content")

        assert hit is False
        assert retrieved is None

    def test_store_embedding(self):
        """Test storing embedding in intelligent cache."""
        cache = IntelligentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        cache.store_embedding("/test/file.py", "content", embedding)

        # Check it was stored
        retrieved, hit = cache.get_embedding("/test/file.py", "content")
        assert hit is True
        np.testing.assert_array_equal(retrieved, embedding)

    def test_get_search_results_hit(self):
        """Test getting search results from cache."""
        cache = IntelligentCache()
        test_results = [{"file": "test.py", "score": 0.9}]
        query_embedding = np.array([1.0, 0.0])

        # Store results
        cache.store_search_results("test query", query_embedding, test_results)

        # Retrieve them
        results, hit = cache.get_search_results("test query", query_embedding, top_k=5)

        assert hit is True
        assert results == test_results

    def test_get_search_results_miss(self):
        """Test getting non-existent search results."""
        cache = IntelligentCache()
        query_embedding = np.array([1.0, 0.0])

        results, hit = cache.get_search_results("test query", query_embedding, top_k=5)

        assert hit is False
        assert results is None

    def test_store_search_results(self):
        """Test storing search results in intelligent cache."""
        cache = IntelligentCache()
        test_results = [{"file": "test.py", "score": 0.9}]
        query_embedding = np.array([1.0, 0.0])

        cache.store_search_results("test query", query_embedding, test_results)

        # Check they were stored
        results, hit = cache.get_search_results("test query", query_embedding, top_k=5)
        assert hit is True
        assert results == test_results

    def test_get_file_content_hit(self):
        """Test getting file content from cache."""
        cache = IntelligentCache()

        cache.store_file_content("/test/file.py", "test content")

        content, hit = cache.get_file_content("/test/file.py")

        assert hit is True
        assert content == "test content"

    def test_get_file_content_miss(self):
        """Test getting non-existent file content."""
        cache = IntelligentCache()

        content, hit = cache.get_file_content("/test/file.py")

        assert hit is False
        assert content is None

    def test_store_file_content(self):
        """Test storing file content in intelligent cache."""
        cache = IntelligentCache()

        result = cache.store_file_content("/test/file.py", "test content")

        assert result is True

        # Check it was stored
        content, hit = cache.get_file_content("/test/file.py")
        assert hit is True
        assert content == "test content"

    def test_invalidate_file(self):
        """Test invalidating all cached data for a file."""
        cache = IntelligentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store different types of data for the file
        cache.store_embedding("/test/file.py", "content", embedding)
        cache.store_file_content("/test/file.py", "file content")
        cache.store_search_results("query", np.array([1.0, 0.0]), [{"file": "/test/file.py"}])

        # Invalidate file
        invalidated = cache.invalidate_file("/test/file.py")

        assert "embeddings" in invalidated
        assert "file_content" in invalidated

        # Check data was invalidated
        retrieved, hit = cache.get_embedding("/test/file.py", "content")
        assert hit is False

        content, hit = cache.get_file_content("/test/file.py")
        assert hit is False

    def test_clear_all(self):
        """Test clearing all caches."""
        cache = IntelligentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store data in all caches
        cache.store_embedding("/test/file.py", "content", embedding)
        cache.store_file_content("/test/file.py", "file content")
        cache.store_search_results("query", np.array([1.0, 0.0]), [{"file": "test.py"}])

        # Clear all
        cache.clear_all()

        # Check all caches are empty
        retrieved, hit = cache.get_embedding("/test/file.py", "content")
        assert hit is False

        content, hit = cache.get_file_content("/test/file.py")
        assert hit is False

        results, hit = cache.get_search_results("query", np.array([1.0, 0.0]), top_k=5)
        assert hit is False

    def test_get_comprehensive_stats(self):
        """Test getting comprehensive cache statistics."""
        cache = IntelligentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store some data
        cache.store_embedding("/test/file.py", "content", embedding)
        cache.store_file_content("/test/file.py", "file content")

        stats = cache.get_comprehensive_stats()

        assert "uptime_seconds" in stats
        assert "performance_metrics" in stats
        assert "hit_rates" in stats
        assert "caches" in stats
        assert "recommendations" in stats

    def test_save_load_persistent_cache(self):
        """Test saving and loading persistent cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = IntelligentCache(cache_dir=temp_dir)

            # Store some data
            embedding = np.array([0.1, 0.2, 0.3])
            cache.store_embedding("/test/file.py", "content", embedding)
            cache.store_file_content("/test/file.py", "file content")

            # Save cache
            cache.save_persistent_cache()

            # Create new cache instance
            new_cache = IntelligentCache(cache_dir=temp_dir)

            # Load cache
            new_cache.load_persistent_cache()

            # Check data was loaded
            retrieved, hit = new_cache.get_embedding("/test/file.py", "content")
            assert hit is True
            np.testing.assert_array_equal(retrieved, embedding)

    def test_get_cache_instance_singleton(self):
        """Test that get_cache_instance returns a singleton."""
        # Reset global instance
        reset_cache_instance()

        cache1 = get_cache_instance()
        cache2 = get_cache_instance()

        assert cache1 is cache2
        assert isinstance(cache1, IntelligentCache)


class TestIntelligentCacheAdvanced:
    """Advanced test cases for IntelligentCache."""

    def test_adaptive_cache_sizes(self):
        """Test adaptive cache size adjustment."""
        cache = IntelligentCache()

        # Mock workload and memory conditions
        with patch.object(cache, '_get_memory_usage', return_value=0.9), \
             patch.dict(cache.workload_stats, {'accesses_last_minute': 150}), \
             patch.dict(cache.adaptive_config, {'last_adjustment': 0}):  # Allow immediate adjustment
                result = cache.adapt_cache_sizes()

                assert result["adapted"] is True
                assert "old_sizes" in result
                assert "new_sizes" in result

    def test_smart_prefetch(self):
        """Test smart prefetching based on usage patterns."""
        cache = IntelligentCache()

        # Add some usage patterns
        cache.usage_patterns["file_coaccess"]["file1.py"]["file2.py"] = 5
        cache.usage_patterns["file_access_counts"]["file1.py"] = 10

        # Mock sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch('codesage_mcp.cache.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('builtins.open', mock_open(read_data="test content")):
                result = cache.smart_prefetch("file1.py", "/test/codebase", mock_model)

                assert "prefetched" in result

    def test_analyze_cache_efficiency(self):
        """Test comprehensive cache efficiency analysis."""
        cache = IntelligentCache()

        # Add some test data and usage
        embedding = np.array([0.1, 0.2, 0.3])
        cache.store_embedding("/test/file.py", "content", embedding)
        cache.get_embedding("/test/file.py", "content")  # Hit

        # Mock the size method for the internal LRUCache
        with patch.object(cache.embedding_cache.embedding_cache, 'size', return_value=1):
            analysis = cache.analyze_cache_efficiency()

            assert "overall_efficiency" in analysis
            assert "cache_breakdown" in analysis
            assert "recommendations" in analysis
        assert "optimization_opportunities" in analysis

    # New comprehensive tests for adaptive sizing and prefetching logic

    @pytest.fixture
    def cache_with_data(self):
        """Fixture providing a cache pre-populated with test data."""
        cache = IntelligentCache(config={
            "embedding_cache_size": 10,
            "search_cache_size": 5,
            "file_cache_size": 5,
            "adaptive_sizing_enabled": True
        })

        # Add test data
        for i in range(8):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            cache.store_embedding(f"/test/file{i}.py", f"content{i}", embedding)
            cache.store_file_content(f"/test/file{i}.py", f"file content {i}")

        return cache

    @pytest.fixture
    def empty_cache(self):
        """Fixture providing an empty cache."""
        return IntelligentCache(config={
            "embedding_cache_size": 10,
            "search_cache_size": 5,
            "file_cache_size": 5,
            "adaptive_sizing_enabled": True
        })

    def test_cache_full_scenario_adaptive_sizing(self, cache_with_data):
        """Test adaptive sizing when cache is full."""
        cache = cache_with_data

        # Mock high memory usage to trigger size reduction
        with patch.object(cache, '_get_memory_usage', return_value=0.85):
            # Force immediate adaptation
            cache.adaptive_config["last_adjustment"] = 0
            cache.workload_stats["accesses_last_minute"] = 10

            result = cache.adapt_cache_sizes()

            assert result["adapted"] is True
            assert result["new_sizes"]["embedding_cache_size"] < cache.config["embedding_cache_size"]

    def test_cache_empty_scenario_adaptive_sizing(self, empty_cache):
        """Test adaptive sizing when cache is empty."""
        cache = empty_cache

        # Mock low memory usage to potentially increase sizes
        with patch.object(cache, '_get_memory_usage', return_value=0.2):
            cache.adaptive_config["last_adjustment"] = 0
            cache.workload_stats["accesses_last_minute"] = 50  # High workload

            result = cache.adapt_cache_sizes()

            # Should adapt due to high workload despite empty cache
            assert result["adapted"] is True

    def test_sequential_access_pattern_learning(self, empty_cache):
        """Test pattern learning with sequential file access."""
        cache = empty_cache

        # Simulate sequential access pattern
        files = [f"/test/file{i}.py" for i in range(10)]
        for file_path in files:
            cache._record_file_access(file_path)
            cache.store_file_content(file_path, f"content for {file_path}")

        # Check that access sequence is recorded
        assert len(cache.usage_patterns["file_access_sequence"]) == 10
        assert cache.usage_patterns["file_access_counts"]["/test/file0.py"] == 1

        # Test prediction based on sequential pattern
        predictions = cache.predict_next_files("/test/file5.py")
        assert isinstance(predictions, list)

    def test_random_access_pattern_learning(self, empty_cache):
        """Test pattern learning with random file access."""
        cache = empty_cache

        # Simulate random access pattern
        files = [f"/test/file{i}.py" for i in range(20)]
        import random
        random_files = random.sample(files, 15)

        for file_path in random_files:
            cache._record_file_access(file_path)
            cache.store_file_content(file_path, f"content for {file_path}")

        # Check access patterns are recorded
        assert len(cache.usage_patterns["file_access_counts"]) <= 15
        assert sum(cache.usage_patterns["file_access_counts"].values()) == 15

    def test_bursty_access_pattern_learning(self, empty_cache):
        """Test pattern learning with bursty access patterns."""
        cache = empty_cache

        # Simulate bursty access (multiple accesses to same file in short time)
        burst_file = "/test/burst_file.py"
        for _ in range(10):
            cache._record_file_access(burst_file)
            cache.store_file_content(burst_file, "burst content")

        # Check that bursty access is recorded
        assert cache.usage_patterns["file_access_counts"][burst_file] == 10

        # Test co-access patterns with bursty file
        other_file = "/test/other_file.py"
        cache._record_file_access(other_file)
        cache.usage_patterns["file_coaccess"][burst_file][other_file] = 5

        predictions = cache.predict_next_files(burst_file)
        assert other_file in predictions

    def test_concurrent_access_thread_safety(self, empty_cache):
        """Test thread safety under concurrent access."""
        cache = empty_cache
        import concurrent.futures
        import threading

        results = []
        errors = []

        def worker(thread_id):
            try:
                # Each thread performs multiple operations
                for i in range(20):
                    file_path = f"/test/thread{thread_id}_file{i}.py"
                    content = f"content from thread {thread_id}, operation {i}"

                    # Store file content
                    cache.store_file_content(file_path, content)

                    # Store embedding
                    embedding = np.array([0.1 * thread_id, 0.2 * i, 0.3])
                    cache.store_embedding(file_path, content, embedding)

                    # Retrieve to test concurrent reads
                    retrieved, hit = cache.get_file_content(file_path)
                    if hit:
                        results.append((thread_id, i, retrieved == content))

                    # Record access for pattern learning
                    cache._record_file_access(file_path)

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Verify some operations succeeded
        assert len(results) > 0

        # Verify pattern learning worked under concurrency
        assert len(cache.usage_patterns["file_access_counts"]) > 0

    def test_memory_pressure_simulation(self, cache_with_data):
        """Test cache behavior under simulated memory pressure."""
        cache = cache_with_data

        # Simulate high memory pressure
        with patch.object(cache, '_get_memory_usage', return_value=0.9):
            # Force adaptation
            cache.adaptive_config["last_adjustment"] = 0
            cache.workload_stats["accesses_last_minute"] = 20

            result = cache.adapt_cache_sizes()

            # Should reduce cache sizes due to high memory usage
            assert result["adapted"] is True
            assert result["new_sizes"]["embedding_cache_size"] < result["old_sizes"]["embedding_cache_size"]

        # Simulate low memory pressure
        with patch.object(cache, '_get_memory_usage', return_value=0.1):
            cache.adaptive_config["last_adjustment"] = 0
            cache.workload_stats["accesses_last_minute"] = 100  # High workload

            result = cache.adapt_cache_sizes()

            # Should increase cache sizes due to low memory and high workload
            assert result["adapted"] is True
            assert result["new_sizes"]["embedding_cache_size"] > result["old_sizes"]["embedding_cache_size"]

    def test_pattern_learning_edge_cases(self, empty_cache):
        """Test edge cases in pattern learning."""
        cache = empty_cache

        # Test with no access patterns
        predictions = cache.predict_next_files("/nonexistent.py")
        assert predictions == []

        # Test with single file access
        cache._record_file_access("/single.py")
        predictions = cache.predict_next_files("/single.py")
        assert isinstance(predictions, list)

        # Test co-access threshold edge case
        file1, file2 = "/test/file1.py", "/test/file2.py"
        cache.usage_patterns["file_coaccess"][file1][file2] = 2  # Below threshold (default 3)
        predictions = cache.predict_next_files(file1)
        assert file2 not in predictions

        # Test co-access above threshold
        cache.usage_patterns["file_coaccess"][file1][file2] = 4  # Above threshold
        predictions = cache.predict_next_files(file1)
        assert file2 in predictions

        # Test frequency-based prediction when co-access is insufficient
        frequent_file = "/test/frequent.py"
        cache.usage_patterns["file_access_counts"][frequent_file] = 100
        predictions = cache.predict_next_files("/other.py")
        assert frequent_file in predictions

    def test_prefetching_under_load(self, empty_cache):
        """Test prefetching behavior under high load conditions."""
        cache = empty_cache

        # Set up usage patterns for prefetching
        main_file = "/test/main.py"
        prefetch_file = "/test/prefetch.py"

        # Create strong co-access pattern
        cache.usage_patterns["file_coaccess"][main_file][prefetch_file] = 10
        cache.usage_patterns["file_access_counts"][main_file] = 50
        cache.usage_patterns["file_access_counts"][prefetch_file] = 30

        # Mock file system and model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch('codesage_mcp.cache.Path') as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance

            with patch('builtins.open', mock_open(read_data="test content")):
                # Test prefetching under normal conditions
                result = cache.smart_prefetch(main_file, "/test/codebase", mock_model)
                assert "prefetched" in result

                # Simulate high load by filling caches
                for i in range(20):
                    cache.store_file_content(f"/test/load_file{i}.py", f"load content {i}")

                # Test prefetching under load (should still work but may skip some)
                result = cache.smart_prefetch(main_file, "/test/codebase", mock_model)
                assert "prefetched" in result or "skipped" in result

    def test_adaptive_sizing_boundary_conditions(self, empty_cache):
        """Test adaptive sizing at boundary conditions."""
        cache = empty_cache

        # Test minimum size boundary
        cache.config["embedding_cache_size"] = cache.adaptive_config["min_cache_size"]

        with patch.object(cache, '_get_memory_usage', return_value=0.95):  # Very high memory
            cache.adaptive_config["last_adjustment"] = 0
            result = cache.adapt_cache_sizes()

            # Should not go below minimum
            assert result["new_sizes"]["embedding_cache_size"] >= cache.adaptive_config["min_cache_size"]

        # Test maximum size boundary
        cache.config["embedding_cache_size"] = cache.adaptive_config["max_cache_size"]

        with patch.object(cache, '_get_memory_usage', return_value=0.05):  # Very low memory
            cache.adaptive_config["last_adjustment"] = 0
            cache.workload_stats["accesses_last_minute"] = 200  # Very high workload

            result = cache.adapt_cache_sizes()

            # Should not exceed maximum
            assert result["new_sizes"]["embedding_cache_size"] <= cache.adaptive_config["max_cache_size"]

    def test_prefetching_disabled_scenario(self, empty_cache):
        """Test prefetching when disabled."""
        cache = empty_cache
        cache.prefetch_config["enabled"] = False

        mock_model = MagicMock()
        result = cache.smart_prefetch("/test/file.py", "/test/codebase", mock_model)

        assert result == {"prefetched": 0, "skipped": 0}

    def test_concurrent_adaptive_sizing(self, empty_cache):
        """Test adaptive sizing under concurrent access."""
        cache = empty_cache
        import concurrent.futures

        def sizing_worker():
            try:
                with patch.object(cache, '_get_memory_usage', return_value=0.8):
                    cache.adaptive_config["last_adjustment"] = 0
                    cache.workload_stats["accesses_last_minute"] = 50
                    result = cache.adapt_cache_sizes()
                    return result["adapted"]
            except Exception as e:
                return str(e)

        # Run multiple concurrent sizing operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(sizing_worker) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # At least one should succeed (due to locking)
        assert any(isinstance(r, bool) for r in results)

    def test_pattern_learning_with_file_invalidation(self, empty_cache):
        """Test pattern learning when files are invalidated."""
        cache = empty_cache

        file_path = "/test/file.py"
        cache._record_file_access(file_path)
        cache.store_file_content(file_path, "content")

        # Verify access is recorded
        assert file_path in cache.usage_patterns["file_access_counts"]

        # Invalidate file
        cache.invalidate_file(file_path)

        # Access patterns should still be maintained (for learning purposes)
        assert file_path in cache.usage_patterns["file_access_counts"]

        # But cache content should be cleared
        content, hit = cache.get_file_content(file_path)
        assert hit is False


if __name__ == "__main__":
    pytest.main([__file__])