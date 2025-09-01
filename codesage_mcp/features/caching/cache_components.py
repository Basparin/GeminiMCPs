"""
Cache Components Module.

This module contains the specialized cache classes extracted from cache.py
to break circular import dependencies.
"""

import json
import hashlib
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import OrderedDict, defaultdict
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache implementation with thread safety."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Put an item in the cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove least recently used
            self.cache[key] = value

    def delete(self, key: str) -> bool:
        """Delete an item from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get the current size of the cache."""
        with self.lock:
            return len(self.cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size
                if self.max_size > 0
                else 0,
            }


class EmbeddingCache:
    """Specialized cache for embeddings with file-based invalidation."""

    def __init__(self, max_size: int = 5000, cache_dir: str = ".codesage"):
        self.max_size = max_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.json"
        self.embeddings_file = self.cache_dir / "embeddings.npy"

        # In-memory cache for embeddings
        self.embedding_cache = LRUCache(max_size)
        # File metadata cache for invalidation
        self.file_metadata = LRUCache(max_size)
        # Reverse mapping from file paths to cache keys
        self.file_to_keys = defaultdict(set)

        self._load_persistent_cache()

    def _get_cache_key(self, file_path: str, content_hash: str) -> str:
        """Generate a cache key for an embedding."""
        return f"{file_path}:{content_hash}"

    def _hash_content(self, content: str) -> str:
        """Generate a hash of the file content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_embedding(self, file_path: str, content: str) -> Optional[np.ndarray]:
        """Get cached embedding if available and valid."""
        start_time = time.time()
        content_hash = self._hash_content(content)
        cache_key = self._get_cache_key(file_path, content_hash)

        # Check if we have this embedding cached
        cached_embedding = self.embedding_cache.get(cache_key)
        response_time = time.time() - start_time

        if cached_embedding is not None:
            logger.debug(f"Cache hit for {file_path}, embedding dimension: {len(cached_embedding) if hasattr(cached_embedding, '__len__') else 'unknown'}, response_time: {response_time:.6f}s")
            logger.info(f"CACHE_METRICS: embedding_hit file={file_path} response_time={response_time:.6f} cache_size={self.embedding_cache.size()}")
            return cached_embedding
        else:
            logger.debug(f"Cache miss for {file_path}, response_time: {response_time:.6f}s")
            logger.info(f"CACHE_METRICS: embedding_miss file={file_path} response_time={response_time:.6f} cache_size={self.embedding_cache.size()}")

        return None

    def store_embedding(
        self, file_path: str, content: str, embedding: np.ndarray
    ) -> None:
        """Store an embedding in the cache."""
        start_time = time.time()
        content_hash = self._hash_content(content)
        cache_key = self._get_cache_key(file_path, content_hash)

        # Store the embedding
        self.embedding_cache.put(cache_key, embedding)

        # Track which keys belong to this file for invalidation
        self.file_to_keys[file_path].add(cache_key)

        # Store file metadata for invalidation
        self.file_metadata.put(
            file_path, {"hash": content_hash, "timestamp": time.time()}
        )

        response_time = time.time() - start_time
        logger.debug(f"Stored embedding for {file_path}, response_time: {response_time:.6f}s")
        logger.info(f"CACHE_METRICS: embedding_store file={file_path} response_time={response_time:.6f} cache_size={self.embedding_cache.size()}")

    def invalidate_file(self, file_path: str) -> int:
        """Invalidate all cached embeddings for a file. Returns number of invalidated entries."""
        invalidated_count = 0

        # Remove all cache entries for this file
        if file_path in self.file_to_keys:
            for cache_key in self.file_to_keys[file_path]:
                if self.embedding_cache.delete(cache_key):
                    invalidated_count += 1

            # Clear the file-to-keys mapping
            del self.file_to_keys[file_path]

        # Remove file metadata
        self.file_metadata.delete(file_path)

        return invalidated_count

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.embedding_cache.clear()
        self.file_metadata.clear()
        self.file_to_keys.clear()

    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    data = json.load(f)

                # Load file metadata
                for file_path, metadata in data.get("file_metadata", {}).items():
                    self.file_metadata.put(file_path, metadata)

                # Load file-to-keys mapping
                for file_path, keys in data.get("file_to_keys", {}).items():
                    self.file_to_keys[file_path] = set(keys)

            # Load embeddings if available
            if self.embeddings_file.exists():
                embeddings_data = np.load(
                    self.embeddings_file, allow_pickle=True
                ).item()
                for cache_key, embedding in embeddings_data.items():
                    self.embedding_cache.put(cache_key, embedding)

        except Exception as e:
            logger.warning(f"Could not load persistent embedding cache: {e}")

    def _save_persistent_cache(self) -> None:
        """Save cache to disk for persistence."""
        try:
            # Save metadata
            data = {
                "file_metadata": dict(self.file_metadata.cache),
                "file_to_keys": {k: list(v) for k, v in self.file_to_keys.items()},
            }

            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)

            # Save embeddings
            embeddings_data = dict(self.embedding_cache.cache)
            np.save(self.embeddings_file, embeddings_data)

        except Exception as e:
            logger.exception(f"Could not save persistent embedding cache: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "embedding_cache": self.embedding_cache.stats(),
            "file_metadata_cache": self.file_metadata.stats(),
            "files_tracked": len(self.file_to_keys),
        }

    def size(self) -> int:
        """Get the current size of the embedding cache."""
        return self.embedding_cache.size()


class SearchResultCache:
    """Specialized cache for search results with similarity-based retrieval."""

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = LRUCache(max_size)
        # Store query embeddings for similarity comparison
        self.query_embeddings = LRUCache(max_size)

    def _hash_query(self, query: str) -> str:
        """Generate a hash for a search query."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        logger.debug(f"Calculating cosine similarity between embeddings of shapes: {emb1.shape if hasattr(emb1, 'shape') else len(emb1) if hasattr(emb1, '__len__') else 'unknown'} and {emb2.shape if hasattr(emb2, 'shape') else len(emb2) if hasattr(emb2, '__len__') else 'unknown'}")
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    def get_similar_results(
        self, query: str, query_embedding: np.ndarray, top_k: int = 5
    ) -> Optional[List[Dict]]:
        """Get cached results for a similar query if available."""
        # Check for exact match first
        query_hash = self._hash_query(query)
        exact_match = self.cache.get(query_hash)
        if exact_match is not None and len(exact_match) >= top_k:
            return exact_match[:top_k]

        # Look for similar queries
        best_match = None
        best_similarity = 0

        for cached_key, cached_embedding in self.query_embeddings.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = self.cache.get(cached_key)

        if best_match is not None and len(best_match) >= top_k:
            return best_match[:top_k]

        return None

    def store_results(
        self, query: str, query_embedding: np.ndarray, results: List[Dict]
    ) -> None:
        """Store search results in the cache."""
        query_hash = self._hash_query(query)

        # Store the results
        self.cache.put(query_hash, results)
        # Store the query embedding for similarity comparison
        self.query_embeddings.put(query_hash, query_embedding)

    def clear(self) -> None:
        """Clear all cached search results."""
        self.cache.clear()
        self.query_embeddings.clear()

    def stats(self) -> Dict[str, Any]:
        """Get search result cache statistics."""
        return {
            "result_cache": self.cache.stats(),
            "query_embedding_cache": self.query_embeddings.stats(),
            "similarity_threshold": self.similarity_threshold,
        }


class FileContentCache:
    """Specialized cache for file contents with memory-efficient storage."""

    def __init__(
        self, max_size: int = 100, max_file_size: int = 1024 * 1024
    ):  # 1MB default
        self.max_size = max_size
        self.max_file_size = max_file_size
        self.cache = LRUCache(max_size)
        self.file_sizes = LRUCache(max_size)  # Track file sizes for memory management

    def get_content(self, file_path: str) -> Optional[str]:
        """Get cached file content if available."""
        return self.cache.get(file_path)

    def store_content(self, file_path: str, content: str) -> bool:
        """Store file content in cache if it meets size requirements."""
        content_size = len(content.encode("utf-8"))

        # Check if content is too large
        if content_size > self.max_file_size:
            return False

        # Store content and size
        self.cache.put(file_path, content)
        self.file_sizes.put(file_path, content_size)

        return True

    def invalidate_file(self, file_path: str) -> bool:
        """Invalidate cached content for a file."""
        invalidated = self.cache.delete(file_path)
        self.file_sizes.delete(file_path)
        return invalidated

    def clear(self) -> None:
        """Clear all cached file contents."""
        self.cache.clear()
        self.file_sizes.clear()

    def stats(self) -> Dict[str, Any]:
        """Get file content cache statistics."""
        total_size = sum(self.file_sizes.cache.values()) if self.file_sizes.cache else 0

        return {
            "content_cache": self.cache.stats(),
            "total_cached_size_bytes": total_size,
            "total_cached_size_mb": total_size / (1024 * 1024),
            "max_file_size_bytes": self.max_file_size,
            "max_file_size_mb": self.max_file_size / (1024 * 1024),
        }