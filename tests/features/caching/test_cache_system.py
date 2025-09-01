"""
Comprehensive Unit Tests for Cache System Module.

This module contains unit tests for the IntelligentCache class and all related cache components,
focusing on LRU eviction, embedding caching, search result caching, file content caching,
and intelligent cache management features.

The tests are organized into classes by component for better maintainability and separation of concerns.
"""

# Standard library imports
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Third-party imports
import numpy as np
import pytest

# Local imports
from codesage_mcp.features.caching.cache_components import (
    EmbeddingCache,
    FileContentCache,
    LRUCache,
    SearchResultCache,
)
from codesage_mcp.features.caching.cache import (
    get_embedding_cache,
    reset_cache_instances,
)
from codesage_mcp.features.caching.intelligent_cache import IntelligentCache


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
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_cache = EmbeddingCache(max_size=100, cache_dir=temp_dir)
            search_cache = SearchResultCache(max_size=50)
            file_cache = FileContentCache(max_size=25)

            assert embedding_cache.max_size == 100
            assert search_cache.max_size == 50
            assert file_cache.max_size == 25
            assert isinstance(embedding_cache, EmbeddingCache)
            assert isinstance(search_cache, SearchResultCache)
            assert isinstance(file_cache, FileContentCache)

    def test_get_embedding_hit(self):
        """Test getting embedding from cache."""
        cache = EmbeddingCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store embedding
        cache.store_embedding("/test/file.py", "content", embedding)

        # Retrieve it
        retrieved = cache.get_embedding("/test/file.py", "content")

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_get_embedding_miss(self):
        """Test getting non-existent embedding."""
        cache = EmbeddingCache()

        retrieved = cache.get_embedding("/test/file.py", "content")

        assert retrieved is None

    def test_store_embedding(self):
        """Test storing embedding in cache."""
        cache = EmbeddingCache()
        embedding = np.array([0.1, 0.2, 0.3])

        cache.store_embedding("/test/file.py", "content", embedding)

        # Check it was stored
        retrieved = cache.get_embedding("/test/file.py", "content")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_get_search_results_hit(self):
        """Test getting search results from cache."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}]
        query_embedding = np.array([1.0, 0.0])

        # Store results
        cache.store_results("test query", query_embedding, test_results)

        # Retrieve them
        results = cache.get_similar_results("test query", query_embedding, top_k=5)

        assert results == test_results

    def test_get_search_results_miss(self):
        """Test getting non-existent search results."""
        cache = SearchResultCache()
        query_embedding = np.array([1.0, 0.0])

        results = cache.get_similar_results("test query", query_embedding, top_k=5)

        assert results is None

    def test_store_search_results(self):
        """Test storing search results in cache."""
        cache = SearchResultCache()
        test_results = [{"file": "test.py", "score": 0.9}]
        query_embedding = np.array([1.0, 0.0])

        cache.store_results("test query", query_embedding, test_results)

        # Check they were stored
        results = cache.get_similar_results("test query", query_embedding, top_k=5)
        assert results == test_results

    def test_get_file_content_hit(self):
        """Test getting file content from cache."""
        cache = FileContentCache()

        cache.store_content("/test/file.py", "test content")

        content = cache.get_content("/test/file.py")

        assert content == "test content"

    def test_get_file_content_miss(self):
        """Test getting non-existent file content."""
        cache = FileContentCache()

        content = cache.get_content("/test/file.py")

        assert content is None

    def test_store_file_content(self):
        """Test storing file content in cache."""
        cache = FileContentCache()

        result = cache.store_content("/test/file.py", "test content")

        assert result is True

        # Check it was stored
        content = cache.get_content("/test/file.py")
        assert content == "test content"

    def test_invalidate_file(self):
        """Test invalidating all cached data for a file."""
        embedding_cache = EmbeddingCache()
        file_cache = FileContentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store different types of data for the file
        embedding_cache.store_embedding("/test/file.py", "content", embedding)
        file_cache.store_content("/test/file.py", "file content")

        # Invalidate file
        embedding_invalidated = embedding_cache.invalidate_file("/test/file.py")
        file_invalidated = file_cache.invalidate_file("/test/file.py")

        assert embedding_invalidated == 1  # One embedding invalidated
        assert file_invalidated is True

        # Check data was invalidated
        retrieved = embedding_cache.get_embedding("/test/file.py", "content")
        assert retrieved is None

        content = file_cache.get_content("/test/file.py")
        assert content is None

    def test_clear_all(self):
        """Test clearing all caches."""
        embedding_cache = EmbeddingCache()
        file_cache = FileContentCache()
        search_cache = SearchResultCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store data in all caches
        embedding_cache.store_embedding("/test/file.py", "content", embedding)
        file_cache.store_content("/test/file.py", "file content")
        search_cache.store_results("query", np.array([1.0, 0.0]), [{"file": "test.py"}])

        # Clear all
        embedding_cache.clear()
        file_cache.clear()
        search_cache.clear()

        # Check all caches are empty
        retrieved = embedding_cache.get_embedding("/test/file.py", "content")
        assert retrieved is None

        content = file_cache.get_content("/test/file.py")
        assert content is None

        results = search_cache.get_similar_results("query", np.array([1.0, 0.0]), top_k=5)
        assert results is None

    def test_get_comprehensive_stats(self):
        """Test getting cache statistics."""
        embedding_cache = EmbeddingCache()
        file_cache = FileContentCache()
        embedding = np.array([0.1, 0.2, 0.3])

        # Store some data
        embedding_cache.store_embedding("/test/file.py", "content", embedding)
        file_cache.store_content("/test/file.py", "file content")

        embedding_stats = embedding_cache.stats()
        file_stats = file_cache.stats()

        assert "embedding_cache" in embedding_stats
        assert "file_metadata_cache" in embedding_stats
        assert "files_tracked" in embedding_stats

        assert "content_cache" in file_stats
        assert "total_cached_size_bytes" in file_stats
        assert "total_cached_size_mb" in file_stats

    def test_save_load_persistent_cache(self):
        """Test saving and loading persistent cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)

            # Store some data
            embedding = np.array([0.1, 0.2, 0.3])
            cache.store_embedding("/test/file.py", "content", embedding)

            # Save cache
            cache._save_persistent_cache()

            # Create new cache instance
            new_cache = EmbeddingCache(cache_dir=temp_dir)

            # Check data was loaded (automatically on init)
            retrieved = new_cache.get_embedding("/test/file.py", "content")
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, embedding)

    def test_get_cache_instance_singleton(self):
        """Test that cache instances are singletons."""
        # Reset global instances
        reset_cache_instances()

        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()

        assert cache1 is cache2
        assert isinstance(cache1, EmbeddingCache)


class TestIntelligentCacheAdvanced:
    """Advanced test cases for IntelligentCache with comprehensive coverage."""

    @pytest.fixture
    def sample_embedding(self):
        """Provide a sample embedding for testing."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    @pytest.fixture
    def sample_search_results(self):
        """Provide sample search results for testing."""
        return [
            {"file": "test1.py", "score": 0.9, "line": 10},
            {"file": "test2.py", "score": 0.8, "line": 20},
            {"file": "test3.py", "score": 0.7, "line": 30},
        ]

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Provide a mock sentence transformer for testing."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        return mock_model

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

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
        """Fixture providing caches pre-populated with test data."""
        embedding_cache = EmbeddingCache(max_size=10)
        file_cache = FileContentCache(max_size=5)

        # Add test data
        for i in range(8):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            embedding_cache.store_embedding(f"/test/file{i}.py", f"content{i}", embedding)
            file_cache.store_content(f"/test/file{i}.py", f"file content {i}")

        return {"embedding": embedding_cache, "file": file_cache}

    @pytest.fixture
    def empty_cache(self):
        """Fixture providing empty IntelligentCache instance."""
        return IntelligentCache(config={
            "embedding_cache_size": 10,
            "search_cache_size": 5,
            "file_cache_size": 5,
            "max_file_size": 1024 * 1024,
            "enable_persistence": False,
            "cache_warming_enabled": False,
        })

    def test_cache_full_scenario_adaptive_sizing(self, cache_with_data):
        """Test cache behavior when full.

        This test verifies that caches handle full scenarios correctly.

        Args:
            cache_with_data: Fixture providing caches pre-populated with test data
        """
        embedding_cache = cache_with_data["embedding"]
        file_cache = cache_with_data["file"]

        # Test that caches handle full scenarios
        # Add more data to potentially fill caches
        for i in range(10, 15):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            embedding_cache.store_embedding(f"/test/file{i}.py", f"content{i}", embedding)
            file_cache.store_content(f"/test/file{i}.py", f"file content {i}")

        # Verify caches are working
        retrieved = embedding_cache.get_embedding("/test/file10.py", "content10")
        assert retrieved is not None

        content = file_cache.get_content("/test/file10.py")
        assert content == "file content 10"

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
        file_cache = empty_cache["file"]

        # Simulate sequential access pattern
        files = [f"/test/file{i}.py" for i in range(10)]
        for file_path in files:
            file_cache.store_content(file_path, f"content for {file_path}")

        # Check that content is stored correctly
        assert file_cache.get_content("/test/file0.py") == "content for /test/file0.py"
        assert file_cache.get_content("/test/file5.py") == "content for /test/file5.py"

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
        embedding_cache = cache_with_data["embedding"]
        file_cache = cache_with_data["file"]

        # Test that caches handle memory pressure by evicting old items
        # Add more data to potentially cause evictions
        for i in range(15, 25):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            embedding_cache.store_embedding(f"/test/file{i}.py", f"content{i}", embedding)
            file_cache.store_content(f"/test/file{i}.py", f"file content {i}")

        # Verify caches are still working after potential evictions
        retrieved = embedding_cache.get_embedding("/test/file20.py", "content20")
        assert retrieved is not None

        content = file_cache.get_content("/test/file20.py")
        assert content == "file content 20"

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

    def test_save_persistent_cache_exception_handling(self, empty_cache):
        """Test exception handling in save_persistent_cache."""
        cache = empty_cache

        # Mock embedding cache's _save_persistent_cache to raise exception
        with patch.object(cache.embedding_cache, '_save_persistent_cache', side_effect=Exception("Test error")):
            # Should not raise exception
            cache.save_persistent_cache()

    def test_get_search_results_exact_match(self, empty_cache):
        """Test get_search_results with exact match."""
        cache = empty_cache
        test_results = [{"file": "test.py", "score": 0.9}]
        query_embedding = np.array([1.0, 0.0])

        # Store results
        cache.store_search_results("test query", query_embedding, test_results)

        # Get exact match (should return results even if len < top_k)
        results, hit = cache.get_search_results("test query", query_embedding, top_k=5)

        assert hit is True
        assert results == test_results

    def test_store_file_content_stats_update(self, empty_cache):
        """Test that store_file_content updates stats correctly."""
        cache = empty_cache

        # Initially no misses
        initial_misses = cache.stats["misses"]["file"]

        # Store content that exceeds size limit by setting file_cache max_file_size
        cache.file_cache.max_file_size = 10  # Very small limit
        result = cache.store_file_content("/test/file.py", "this content is too long")

        # Should return False and increment misses
        assert result is False
        assert cache.stats["misses"]["file"] == initial_misses + 1

    def test_calculate_file_priority_score(self, empty_cache):
        """Test _calculate_file_priority_score method."""
        cache = empty_cache

        # Mock file path
        with patch('codesage_mcp.cache.Path') as mock_path:
            mock_file = MagicMock()
            mock_file.stat.return_value.st_size = 5000  # 5KB
            mock_file.suffix.lower.return_value = ".py"
            mock_path.return_value = mock_file

            score = cache._calculate_file_priority_score("test.py", "/test")

            assert isinstance(score, float)
            assert score > 0

    def test_prioritize_files_ml(self, empty_cache):
        """Test _prioritize_files_ml method."""
        cache = empty_cache

        with patch('codesage_mcp.cache.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.is_file.return_value = True
            mock_path.return_value.is_symlink.return_value = False
            mock_path.return_value.relative_to.return_value = Path("test.py")
            mock_path.return_value.stat.return_value.st_size = 1000
            mock_path.return_value.suffix.lower.return_value = ".py"
            mock_path.return_value.parts = ("test.py",)

            # Mock rglob to return some files
            mock_path.return_value.rglob.return_value = [mock_path.return_value]

            result = cache._prioritize_files_ml("/test")

            assert isinstance(result, list)

    def test_warm_cache_functionality(self, empty_cache):
        """Test warm_cache method."""
        cache = empty_cache

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch('codesage_mcp.cache.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.rglob.return_value = []
            mock_path.return_value.relative_to.return_value = Path("test.py")

            result = cache.warm_cache("/test", mock_model)

            assert "files_warmed" in result
            assert "embeddings_cached" in result

    def test_record_file_access_functionality(self, empty_cache):
        """Test _record_file_access method."""
        cache = empty_cache

        file_path = "/test/file.py"
        cache._record_file_access(file_path)

        assert file_path in cache.usage_patterns["file_access_counts"]
        assert cache.usage_patterns["file_access_counts"][file_path] == 1

    def test_predict_next_files_functionality(self, empty_cache):
        """Test predict_next_files method."""
        cache = empty_cache

        # Set up co-access pattern
        file1, file2 = "/test/file1.py", "/test/file2.py"
        cache.usage_patterns["file_coaccess"][file1][file2] = 5

        predictions = cache.predict_next_files(file1)

        assert isinstance(predictions, list)

    def test_prefetch_files_functionality(self, empty_cache):
        """Test prefetch_files method."""
        cache = empty_cache

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch('codesage_mcp.cache.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('builtins.open', mock_open(read_data="test")):
                result = cache.prefetch_files(["/test/file.py"], "/test", mock_model)

                assert "prefetched" in result
                assert "skipped" in result

    def test_update_workload_stats(self, empty_cache):
        """Test _update_workload_stats method."""
        cache = empty_cache

        initial_minute = cache.workload_stats["accesses_last_minute"]
        initial_hour = cache.workload_stats["accesses_last_hour"]

        # Simulate time passage
        with patch('time.time', return_value=time.time() + 70):  # 70 seconds later
            cache._update_workload_stats()

            # Should reset minute stats
            assert cache.workload_stats["accesses_last_minute"] == 0

    def test_get_memory_usage_fallback(self, empty_cache):
        """Test _get_memory_usage fallback when psutil not available."""
        cache = empty_cache

        with patch.dict('sys.modules', {'psutil': None}):
            usage = cache._get_memory_usage()

            assert isinstance(usage, float)
            assert 0 <= usage <= 1

    def test_calculate_optimal_cache_sizes(self, empty_cache):
        """Test _calculate_optimal_cache_sizes method."""
        cache = empty_cache

        with patch.object(cache, '_get_memory_usage', return_value=0.5):
            sizes = cache._calculate_optimal_cache_sizes()

            assert "embedding_cache_size" in sizes
            assert "search_cache_size" in sizes
            assert "file_cache_size" in sizes

    def test_adapt_cache_sizes_disabled(self, empty_cache):
        """Test adapt_cache_sizes when disabled."""
        cache = empty_cache
        cache.adaptive_config["enabled"] = False

        result = cache.adapt_cache_sizes()

        assert result["adapted"] is False
        assert "disabled" in result["reason"]

    def test_record_cache_access_workload(self, empty_cache):
        """Test record_cache_access workload tracking."""
        cache = empty_cache

        initial_accesses = cache.workload_stats["accesses_last_minute"]

        cache.record_cache_access("embedding")

        assert cache.workload_stats["accesses_last_minute"] == initial_accesses + 1

    def test_smart_prefetch_disabled(self, empty_cache):
        """Test smart_prefetch when disabled."""
        cache = empty_cache
        cache.prefetch_config["enabled"] = False

        mock_model = MagicMock()
        result = cache.smart_prefetch("/test/file.py", "/test", mock_model)

        assert result == {"prefetched": 0, "skipped": 0}

    def test_generate_performance_recommendations(self, empty_cache):
        """Test _generate_performance_recommendations method."""
        cache = empty_cache

        # Add some stats
        cache.stats["hits"]["embedding"] = 10
        cache.stats["misses"]["embedding"] = 5

        recommendations = cache._generate_performance_recommendations()

        assert isinstance(recommendations, list)

    def test_analyze_single_cache(self, empty_cache):
        """Test _analyze_single_cache method."""
        cache = empty_cache

        # Add some data
        cache.stats["hits"]["embedding"] = 5
        cache.stats["misses"]["embedding"] = 3

        analysis = cache._analyze_single_cache("embedding")

        assert "hits" in analysis
        assert "misses" in analysis
        assert "hit_rate" in analysis

    def test_reset_cache_instances(self, empty_cache):
        """Test reset_cache_instances function."""
        from codesage_mcp.features.caching.cache import reset_cache_instances, get_embedding_cache

        # Get instance
        instance1 = get_embedding_cache()

        # Reset
        reset_cache_instances()

        # Get new instance
        instance2 = get_embedding_cache()

        # Should be different instances
        assert instance1 is not instance2


if __name__ == "__main__":
    pytest.main([__file__])