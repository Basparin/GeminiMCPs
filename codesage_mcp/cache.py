"""
Intelligent Caching System for CodeSage MCP Server.

This module provides a comprehensive caching system to improve performance by caching
embeddings, search results, and other expensive computations. It includes:

- Embedding Caching: Cache embeddings for unchanged files to avoid re-encoding
- Search Result Caching: Cache search results based on query similarity
- File Content Caching: Cache frequently accessed file contents
- Cache Management: IntelligentCache class with monitoring capabilities
- LRU eviction policies and configurable size limits
- Cache persistence for faster startup
- Performance monitoring and statistics

Classes:
    IntelligentCache: Main cache management class
    EmbeddingCache: Specialized cache for embeddings
    SearchResultCache: Specialized cache for search results
    FileContentCache: Specialized cache for file contents
"""

import json
import hashlib
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
        content_hash = self._hash_content(content)
        cache_key = self._get_cache_key(file_path, content_hash)

        # Check if we have this embedding cached
        cached_embedding = self.embedding_cache.get(cache_key)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for {file_path}, embedding dimension: {len(cached_embedding) if hasattr(cached_embedding, '__len__') else 'unknown'}")
            return cached_embedding

        return None

    def store_embedding(
        self, file_path: str, content: str, embedding: np.ndarray
    ) -> None:
        """Store an embedding in the cache."""
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
            logger.warning(f"Could not save persistent embedding cache: {e}")

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


class IntelligentCache:
    """Main cache management class with monitoring and coordination capabilities."""

    def __init__(
        self, cache_dir: str = ".codesage", config: Optional[Dict[str, Any]] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Default configuration
        default_config = {
            "embedding_cache_size": 5000,
            "search_cache_size": 1000,
            "file_cache_size": 100,
            "similarity_threshold": 0.85,
            "max_file_size": 1024 * 1024,  # 1MB
            "enable_persistence": True,
            "cache_warming_enabled": True,
        }

        # Merge with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize specialized caches
        self.embedding_cache = EmbeddingCache(
            max_size=self.config["embedding_cache_size"], cache_dir=str(self.cache_dir)
        )

        self.search_cache = SearchResultCache(
            max_size=self.config["search_cache_size"],
            similarity_threshold=self.config["similarity_threshold"],
        )

        self.file_cache = FileContentCache(
            max_size=self.config["file_cache_size"],
            max_file_size=self.config["max_file_size"],
        )

        # Statistics tracking
        self.stats = {
            "hits": defaultdict(int),
            "misses": defaultdict(int),
            "invalidations": defaultdict(int),
            "prefetch_hits": defaultdict(int),
            "prefetch_misses": defaultdict(int),
            "start_time": time.time(),
        }

        # Usage pattern tracking for prefetching
        self.usage_patterns = {
            "file_access_sequence": [],  # Recent file access sequence
            "file_access_counts": defaultdict(int),  # Access frequency
            "file_coaccess": defaultdict(
                lambda: defaultdict(int)
            ),  # Files accessed together
            "last_access_time": {},  # Last access time for each file
            "access_intervals": defaultdict(list),  # Time intervals between accesses
        }

        # Prefetching configuration
        self.prefetch_config = {
            "enabled": True,
            "max_prefetch_files": 10,
            "prefetch_threshold": 0.7,  # Similarity threshold for pattern matching
            "coaccess_threshold": 3,  # Minimum co-access count to establish pattern
            "enable_pattern_learning": True,
        }

        # Adaptive sizing configuration
        self.adaptive_config = {
            "enabled": True,
            "min_cache_size": 100,
            "max_cache_size": 10000,
            "memory_threshold_high": 0.8,  # Reduce cache when memory usage > 80%
            "memory_threshold_low": 0.3,  # Increase cache when memory usage < 30%
            "high_workload_threshold": 100,  # High workload = >100 accesses per minute
            "adjustment_interval": 300,  # Check every 5 minutes
            "last_adjustment": time.time(),
        }

        # Workload tracking
        self.workload_stats = {
            "accesses_last_minute": 0,
            "accesses_last_hour": 0,
            "last_minute_start": time.time(),
            "last_hour_start": time.time(),
        }

        # Thread safety
        self.lock = threading.RLock()

    def get_embedding(
        self, file_path: str, content: str
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Get embedding from cache or return None if not cached. Returns (embedding, was_hit)."""
        with self.lock:
            embedding = self.embedding_cache.get_embedding(file_path, content)
            if embedding is not None:
                self.stats["hits"]["embedding"] += 1
                self.record_cache_access("embedding")
                return embedding, True
            else:
                self.stats["misses"]["embedding"] += 1
                return None, False

    def store_embedding(
        self, file_path: str, content: str, embedding: np.ndarray
    ) -> None:
        """Store an embedding in the cache."""
        with self.lock:
            self.embedding_cache.store_embedding(file_path, content, embedding)

    def get_search_results(
        self, query: str, query_embedding: np.ndarray, top_k: int = 5
    ) -> Tuple[Optional[List[Dict]], bool]:
        """Get similar search results from cache. Returns (results, was_hit)."""
        with self.lock:
            results = self.search_cache.get_similar_results(
                query, query_embedding, top_k
            )
            if results is not None:
                self.stats["hits"]["search"] += 1
                self.record_cache_access("search")
                return results, True
            else:
                # Check if exact match exists (even if len < top_k)
                query_hash = self.search_cache._hash_query(query)
                stored_results = self.search_cache.cache.get(query_hash)
                if stored_results is not None:
                    results = stored_results[:top_k]
                    self.stats["hits"]["search"] += 1
                    self.record_cache_access("search")
                    return results, True
                else:
                    self.stats["misses"]["search"] += 1
                    return None, False

    def store_search_results(
        self, query: str, query_embedding: np.ndarray, results: List[Dict]
    ) -> None:
        """Store search results in the cache."""
        with self.lock:
            self.search_cache.store_results(query, query_embedding, results)

    def get_file_content(self, file_path: str) -> Tuple[Optional[str], bool]:
        """Get file content from cache. Returns (content, was_hit)."""
        with self.lock:
            content = self.file_cache.get_content(file_path)
            if content is not None:
                self.stats["hits"]["file"] += 1
                self._record_file_access(file_path)
                self.record_cache_access("file")
                return content, True
            else:
                self.stats["misses"]["file"] += 1
                return None, False

    def store_file_content(self, file_path: str, content: str) -> bool:
        """Store file content in cache. Returns True if stored successfully."""
        with self.lock:
            stored = self.file_cache.store_content(file_path, content)
            if not stored:
                self.stats["misses"]["file"] += (
                    1  # Count as miss if not stored due to size
                )
            return stored

    def invalidate_file(self, file_path: str) -> Dict[str, int]:
        """Invalidate all cached data for a file. Returns counts of invalidated items."""
        with self.lock:
            invalidated = {
                "embeddings": self.embedding_cache.invalidate_file(file_path),
                "file_content": 1 if self.file_cache.invalidate_file(file_path) else 0,
            }

            total_invalidated = sum(invalidated.values())
            if total_invalidated > 0:
                self.stats["invalidations"]["file"] += 1

            return invalidated

    def clear_all(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.embedding_cache.clear()
            self.search_cache.clear()
            self.file_cache.clear()

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            uptime = time.time() - self.stats["start_time"]

            # Calculate hit rates
            embedding_hits = self.stats["hits"]["embedding"]
            embedding_misses = self.stats["misses"]["embedding"]
            embedding_hit_rate = (
                embedding_hits / (embedding_hits + embedding_misses)
                if (embedding_hits + embedding_misses) > 0
                else 0
            )

            search_hits = self.stats["hits"]["search"]
            search_misses = self.stats["misses"]["search"]
            search_hit_rate = (
                search_hits / (search_hits + search_misses)
                if (search_hits + search_misses) > 0
                else 0
            )

            file_hits = self.stats["hits"]["file"]
            file_misses = self.stats["misses"]["file"]
            file_hit_rate = (
                file_hits / (file_hits + file_misses)
                if (file_hits + file_misses) > 0
                else 0
            )

            # Calculate performance metrics
            total_requests = (
                embedding_hits
                + embedding_misses
                + search_hits
                + search_misses
                + file_hits
                + file_misses
            )
            avg_request_time = uptime / max(total_requests, 1)  # Rough estimate

            # Memory efficiency metrics
            memory_usage = self._get_memory_usage()
            cache_efficiency = (embedding_hits + search_hits + file_hits) / max(
                total_requests, 1
            )

            # Workload analysis
            workload_intensity = self.workload_stats["accesses_last_minute"]

            return {
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "performance_metrics": {
                    "total_requests": total_requests,
                    "avg_request_time_seconds": avg_request_time,
                    "requests_per_second": total_requests / max(uptime, 1),
                    "memory_usage_percent": memory_usage * 100,
                    "cache_efficiency_percent": cache_efficiency * 100,
                    "workload_intensity": workload_intensity,
                },
                "hit_rates": {
                    "embedding": embedding_hit_rate,
                    "search": search_hit_rate,
                    "file": file_hit_rate,
                    "overall": (embedding_hits + search_hits + file_hits)
                    / (
                        embedding_hits
                        + embedding_misses
                        + search_hits
                        + search_misses
                        + file_hits
                        + file_misses
                    )
                    if (
                        embedding_hits
                        + embedding_misses
                        + search_hits
                        + search_misses
                        + file_hits
                        + file_misses
                    )
                    > 0
                    else 0,
                },
                "hits": dict(self.stats["hits"]),
                "misses": dict(self.stats["misses"]),
                "invalidations": dict(self.stats["invalidations"]),
                "prefetch_stats": {
                    "prefetch_hits": dict(self.stats.get("prefetch_hits", {})),
                    "prefetch_misses": dict(self.stats.get("prefetch_misses", {})),
                },
                "usage_patterns": {
                    "most_accessed_files": sorted(
                        self.usage_patterns["file_access_counts"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10],
                    "file_access_sequence_length": len(
                        self.usage_patterns["file_access_sequence"]
                    ),
                    "coaccess_patterns_count": len(
                        self.usage_patterns["file_coaccess"]
                    ),
                },
                "adaptive_sizing": {
                    "enabled": self.adaptive_config["enabled"],
                    "last_adaptation": self.adaptive_config.get(
                        "last_adaptation", "never"
                    ),
                    "current_sizes": {
                        "embedding_cache": self.config["embedding_cache_size"],
                        "search_cache": self.config["search_cache_size"],
                        "file_cache": self.config["file_cache_size"],
                    },
                },
                "caches": {
                    "embedding": self.embedding_cache.stats(),
                    "search": self.search_cache.stats(),
                    "file": self.file_cache.stats(),
                },
                "config": self.config,
                "recommendations": self._generate_performance_recommendations(),
            }

    def save_persistent_cache(self) -> None:
        """Save persistent cache data to disk."""
        if not self.config["enable_persistence"]:
            return

        with self.lock:
            try:
                self.embedding_cache._save_persistent_cache()

                # Save cache statistics
                stats_file = self.cache_dir / "cache_stats.json"
                with open(stats_file, "w") as f:
                    json.dump(self.get_comprehensive_stats(), f, indent=2, default=str)

            except Exception as e:
                logger.warning(f"Could not save persistent cache: {e}")

    def load_persistent_cache(self) -> None:
        """Load persistent cache data from disk."""
        if not self.config["enable_persistence"]:
            return

        with self.lock:
            try:
                # Embedding cache is loaded automatically in its constructor
                # Load and merge statistics if available
                stats_file = self.cache_dir / "cache_stats.json"
                if stats_file.exists():
                    with open(stats_file, "r") as f:
                        saved_stats = json.load(f)

                    # Merge saved statistics (don't overwrite current session stats)
                    for key in ["hits", "misses", "invalidations"]:
                        if key in saved_stats:
                            for subkey, value in saved_stats[key].items():
                                if subkey not in self.stats[key]:
                                    self.stats[key][subkey] = value

            except Exception as e:
                logger.warning(f"Could not load persistent cache: {e}")

    def _calculate_file_priority_score(
        self, file_path: str, codebase_path: str
    ) -> float:
        """Calculate a priority score for a file based on ML-based heuristics.

        Args:
            file_path: Relative path to the file
            codebase_path: Path to the codebase

        Returns:
            Priority score (higher = more important to cache)
        """
        score = 0.0
        abs_file_path = Path(codebase_path) / file_path

        try:
            # Factor 1: File size (smaller files are more likely to be read entirely)
            file_size = abs_file_path.stat().st_size
            if file_size < 10000:  # Less than 10KB
                score += 2.0
            elif file_size < 50000:  # Less than 50KB
                score += 1.0

            # Factor 2: File type priority
            if file_path.endswith(".py"):
                score += 3.0  # Python files are most important
            elif file_path.endswith((".js", ".ts", ".java", ".cpp", ".c", ".h")):
                score += 2.5  # Other programming languages
            elif file_path.endswith((".md", ".txt", ".yml", ".yaml", ".json")):
                score += 1.5  # Documentation and config files
            elif file_path.endswith((".html", ".css")):
                score += 1.0  # Web files

            # Factor 3: Directory structure (files in root or src are more important)
            path_parts = Path(file_path).parts
            if len(path_parts) <= 2:
                score += 1.5  # Root level files
            elif "src" in path_parts or "source" in path_parts:
                score += 1.0  # Source directory files
            elif "test" in path_parts or "tests" in path_parts:
                score += 0.5  # Test files (less critical)

            # Factor 4: Historical access patterns (if available)
            if file_path in self.usage_patterns["file_access_counts"]:
                access_count = self.usage_patterns["file_access_counts"][file_path]
                score += min(access_count * 0.1, 2.0)  # Cap at 2.0

            # Factor 5: File name patterns (common entry points)
            file_name = Path(file_path).name.lower()
            if any(
                keyword in file_name
                for keyword in ["main", "app", "index", "init", "config", "setup"]
            ):
                score += 1.0

            # Factor 6: Import relationships (files with many imports/exports are important)
            if hasattr(self, "_dependency_graph"):
                # This would be set by the indexing manager
                pass

        except Exception:
            # If we can't analyze the file, give it a neutral score
            score = 1.0

        return score

    def _prioritize_files_ml(
        self, codebase_path: str, max_files: int = 100
    ) -> List[Tuple[str, float]]:
        """Prioritize files for cache warming using ML-based scoring.

        Args:
            codebase_path: Path to the codebase
            max_files: Maximum number of files to consider

        Returns:
            List of (file_path, priority_score) tuples, sorted by priority
        """
        codebase_path = Path(codebase_path)
        file_priorities = []

        try:
            # Collect all relevant files
            for file_path in codebase_path.rglob("*"):
                if file_path.is_file() and not file_path.is_symlink():
                    # Skip common non-text files
                    if file_path.suffix.lower() in [
                        ".pyc",
                        ".pyo",
                        ".class",
                        ".jar",
                        ".zip",
                        ".tar",
                        ".gz",
                        ".bin",
                        ".exe",
                        ".dll",
                        ".so",
                    ]:
                        continue

                    # Skip hidden files and directories
                    if any(part.startswith(".") for part in file_path.parts):
                        continue

                    # Skip very large files (>1MB)
                    try:
                        if file_path.stat().st_size > 1024 * 1024:
                            continue
                    except OSError:
                        continue

                    # Calculate relative path
                    try:
                        relative_path = str(file_path.relative_to(codebase_path))
                        priority_score = self._calculate_file_priority_score(
                            relative_path, str(codebase_path)
                        )
                        file_priorities.append((relative_path, priority_score))
                    except ValueError:
                        continue

            # Sort by priority score (descending)
            file_priorities.sort(key=lambda x: x[1], reverse=True)

            # Return top files
            return file_priorities[:max_files]

        except Exception as e:
            logger.warning(f"ML-based file prioritization failed: {e}")
            # Fallback to simple prioritization
            return self._prioritize_files_simple(codebase_path, max_files)

    def _prioritize_files_simple(
        self, codebase_path: str, max_files: int = 100
    ) -> List[Tuple[str, float]]:
        """Simple fallback file prioritization based on basic heuristics.

        Args:
            codebase_path: Path to the codebase
            max_files: Maximum number of files to consider

        Returns:
            List of (file_path, priority_score) tuples
        """
        codebase_path = Path(codebase_path)
        file_priorities = []

        try:
            # Prioritize Python files, then other common formats
            priority_patterns = [
                ("*.py", 3.0),
                ("*.js", 2.5),
                ("*.ts", 2.5),
                ("*.java", 2.5),
                ("*.cpp", 2.5),
                ("*.c", 2.5),
                ("*.h", 2.5),
                ("*.md", 1.5),
                ("*.txt", 1.5),
                ("*.yml", 1.5),
                ("*.yaml", 1.5),
                ("*.json", 1.5),
                ("*.html", 1.0),
                ("*.css", 1.0),
            ]

            for pattern, base_score in priority_patterns:
                for file_path in codebase_path.rglob(pattern):
                    if len(file_priorities) >= max_files:
                        break

                    try:
                        relative_path = str(file_path.relative_to(codebase_path))
                        file_priorities.append((relative_path, base_score))
                    except ValueError:
                        continue

        except Exception as e:
            logger.warning(f"Simple file prioritization failed: {e}")

        return file_priorities[:max_files]

    def warm_cache(
        self, codebase_path: str, sentence_transformer_model
    ) -> Dict[str, int]:
        """Warm up the cache with intelligently prioritized files. Returns warming statistics."""
        if not self.config["cache_warming_enabled"]:
            return {"files_warmed": 0, "embeddings_cached": 0}

        warmed_files = 0
        cached_embeddings = 0
        prioritized_files = []

        try:
            # Use ML-based prioritization if enabled
            if self.prefetch_config.get("enable_pattern_learning", True):
                prioritized_files = self._prioritize_files_ml(
                    codebase_path, max_files=50
                )
                logger.info(
                    f"ML-based cache warming prioritized {len(prioritized_files)} files"
                )
            else:
                prioritized_files = self._prioritize_files_simple(
                    codebase_path, max_files=50
                )
                logger.info(
                    f"Simple cache warming prioritized {len(prioritized_files)} files"
                )

            # Warm cache in priority order
            for file_path, priority_score in prioritized_files:
                if warmed_files >= 50:  # Limit warming to prevent startup delay
                    break

                abs_file_path = Path(codebase_path) / file_path

                try:
                    with open(abs_file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Cache file content
                    if self.store_file_content(file_path, content):
                        warmed_files += 1
                        logger.debug(
                            f"Warmed cache for {file_path} (priority: {priority_score:.2f})"
                        )

                    # Generate and cache embedding
                    embedding = sentence_transformer_model.encode(content)
                    self.store_embedding(file_path, content, embedding)
                    cached_embeddings += 1

                except Exception as e:
                    logger.warning(f"Could not warm cache for {file_path}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")

        return {
            "files_warmed": warmed_files,
            "embeddings_cached": cached_embeddings,
            "prioritization_method": "ml"
            if self.prefetch_config.get("enable_pattern_learning", True)
            else "simple",
        }

    def _record_file_access(self, file_path: str) -> None:
        """Record file access for pattern learning.

        Args:
            file_path: Path to the accessed file
        """
        if not self.prefetch_config["enable_pattern_learning"]:
            return

        current_time = time.time()

        # Update access sequence (keep last 100 accesses)
        self.usage_patterns["file_access_sequence"].append(file_path)
        if len(self.usage_patterns["file_access_sequence"]) > 100:
            self.usage_patterns["file_access_sequence"].pop(0)

        # Update access counts
        self.usage_patterns["file_access_counts"][file_path] += 1

        # Update last access time
        if file_path in self.usage_patterns["last_access_time"]:
            # Calculate interval since last access
            last_time = self.usage_patterns["last_access_time"][file_path]
            interval = current_time - last_time
            self.usage_patterns["access_intervals"][file_path].append(interval)
            # Keep only last 10 intervals
            if len(self.usage_patterns["access_intervals"][file_path]) > 10:
                self.usage_patterns["access_intervals"][file_path].pop(0)

        self.usage_patterns["last_access_time"][file_path] = current_time

        # Update co-access patterns
        if len(self.usage_patterns["file_access_sequence"]) >= 2:
            prev_file = self.usage_patterns["file_access_sequence"][-2]
            if prev_file != file_path:  # Don't count self-access
                self.usage_patterns["file_coaccess"][prev_file][file_path] += 1

    def predict_next_files(
        self, current_file: str, max_predictions: int = 5
    ) -> List[str]:
        """Predict which files are likely to be accessed next based on usage patterns.

        Args:
            current_file: Currently accessed file
            max_predictions: Maximum number of files to predict

        Returns:
            List of predicted file paths
        """
        if not self.prefetch_config["enable_pattern_learning"]:
            return []

        predictions = []

        # 1. Co-access prediction (files frequently accessed together)
        if current_file in self.usage_patterns["file_coaccess"]:
            coaccess_files = self.usage_patterns["file_coaccess"][current_file]
            # Sort by co-access frequency
            sorted_coaccess = sorted(
                coaccess_files.items(), key=lambda x: x[1], reverse=True
            )

            for file_path, count in sorted_coaccess:
                if count >= self.prefetch_config["coaccess_threshold"]:
                    predictions.append(file_path)
                    if len(predictions) >= max_predictions:
                        break

        # 2. Frequency-based prediction (most frequently accessed files)
        if len(predictions) < max_predictions:
            frequent_files = sorted(
                self.usage_patterns["file_access_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for file_path, count in frequent_files:
                if file_path not in predictions and file_path != current_file:
                    predictions.append(file_path)
                    if len(predictions) >= max_predictions:
                        break

        return predictions[:max_predictions]

    def prefetch_files(
        self, file_paths: List[str], codebase_path: str, sentence_transformer_model
    ) -> Dict[str, int]:
        """Prefetch files into cache based on predictions.

        Args:
            file_paths: List of file paths to prefetch
            codebase_path: Path to the codebase
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Dictionary with prefetch statistics
        """
        if not self.prefetch_config["enabled"]:
            return {"prefetched": 0, "skipped": 0}

        prefetched = 0
        skipped = 0

        for file_path in file_paths[: self.prefetch_config["max_prefetch_files"]]:
            try:
                abs_file_path = Path(codebase_path) / file_path

                if not abs_file_path.exists():
                    skipped += 1
                    continue

                # Check if already cached
                if self.file_cache.get_content(file_path) is not None:
                    skipped += 1
                    continue

                # Read and cache file content
                with open(abs_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if self.store_file_content(file_path, content):
                    prefetched += 1
                    logger.debug(f"Prefetched file: {file_path}")

                    # Also prefetch embedding if model is available
                    try:
                        embedding = sentence_transformer_model.encode(content)
                        self.store_embedding(file_path, content, embedding)
                    except Exception as e:
                        logger.warning(
                            f"Could not prefetch embedding for {file_path}: {e}"
                        )
                else:
                    skipped += 1

            except Exception as e:
                logger.warning(f"Could not prefetch {file_path}: {e}")
                skipped += 1

        return {"prefetched": prefetched, "skipped": skipped}

    def _update_workload_stats(self) -> None:
        """Update workload statistics for adaptive sizing."""
        current_time = time.time()

        # Update minute stats
        if current_time - self.workload_stats["last_minute_start"] >= 60:
            self.workload_stats["accesses_last_minute"] = 0
            self.workload_stats["last_minute_start"] = current_time

        # Update hour stats
        if current_time - self.workload_stats["last_hour_start"] >= 3600:
            self.workload_stats["accesses_last_hour"] = 0
            self.workload_stats["last_hour_start"] = current_time

    def _get_memory_usage(self) -> float:
        """Get current memory usage as a fraction (0.0 to 1.0)."""
        try:
            import psutil

            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback: estimate based on cache sizes
            total_cached_items = (
                self.embedding_cache.size()
                + self.search_cache.cache.size()
                + self.file_cache.cache.size()
            )
            # Rough estimate: assume 1MB per 100 items
            estimated_mb = total_cached_items / 100
            return min(estimated_mb / 1000, 1.0)  # Cap at 1GB

    def _calculate_optimal_cache_sizes(self) -> Dict[str, int]:
        """Calculate optimal cache sizes based on current workload and memory."""
        if not self.adaptive_config["enabled"]:
            return {}

        current_memory = self._get_memory_usage()
        workload_intensity = self.workload_stats["accesses_last_minute"]

        # Base sizes from configuration
        base_embedding_size = self.config["embedding_cache_size"]
        base_search_size = self.config["search_cache_size"]
        base_file_size = self.config["file_cache_size"]

        # Adjust based on memory pressure
        if current_memory > self.adaptive_config["memory_threshold_high"]:
            # High memory usage - reduce cache sizes
            memory_factor = 0.5
        elif current_memory < self.adaptive_config["memory_threshold_low"]:
            # Low memory usage - can increase cache sizes
            memory_factor = 1.5
        else:
            memory_factor = 1.0

        # Adjust based on workload intensity
        if workload_intensity > self.adaptive_config["high_workload_threshold"]:
            # High workload - increase cache sizes
            workload_factor = 1.5
        elif workload_intensity < 10:
            # Low workload - can reduce cache sizes
            workload_factor = 0.7
        else:
            workload_factor = 1.0

        # Calculate new sizes
        new_sizes = {
            "embedding_cache_size": int(
                max(
                    self.adaptive_config["min_cache_size"],
                    min(
                        self.adaptive_config["max_cache_size"],
                        base_embedding_size * memory_factor * workload_factor,
                    ),
                )
            ),
            "search_cache_size": int(
                max(
                    self.adaptive_config["min_cache_size"] // 10,
                    min(
                        self.adaptive_config["max_cache_size"] // 10,
                        base_search_size * memory_factor * workload_factor,
                    ),
                )
            ),
            "file_cache_size": int(
                max(
                    self.adaptive_config["min_cache_size"] // 10,
                    min(
                        self.adaptive_config["max_cache_size"] // 10,
                        base_file_size * memory_factor * workload_factor,
                    ),
                )
            ),
        }

        return new_sizes

    def adapt_cache_sizes(self) -> Dict[str, Any]:
        """Adapt cache sizes based on current workload and memory usage.

        Returns:
            Dictionary with adaptation results
        """
        if not self.adaptive_config["enabled"]:
            return {"adapted": False, "reason": "adaptive sizing disabled"}

        current_time = time.time()

        # Check if enough time has passed since last adjustment
        if (
            current_time - self.adaptive_config["last_adjustment"]
            < self.adaptive_config["adjustment_interval"]
        ):
            return {"adapted": False, "reason": "too soon since last adjustment"}

        # Update workload stats
        self._update_workload_stats()

        # Calculate optimal sizes
        optimal_sizes = self._calculate_optimal_cache_sizes()

        if not optimal_sizes:
            return {"adapted": False, "reason": "no optimal sizes calculated"}

        # Check if any sizes need adjustment
        needs_adjustment = (
            optimal_sizes["embedding_cache_size"] != self.config["embedding_cache_size"]
            or optimal_sizes["search_cache_size"] != self.config["search_cache_size"]
            or optimal_sizes["file_cache_size"] != self.config["file_cache_size"]
        )

        if not needs_adjustment:
            return {"adapted": False, "reason": "sizes already optimal"}

        # Apply new sizes
        old_sizes = {
            "embedding_cache_size": self.config["embedding_cache_size"],
            "search_cache_size": self.config["search_cache_size"],
            "file_cache_size": self.config["file_cache_size"],
        }

        self.config.update(optimal_sizes)

        # Resize caches if necessary
        if optimal_sizes["embedding_cache_size"] != old_sizes["embedding_cache_size"]:
            self.embedding_cache = EmbeddingCache(
                max_size=optimal_sizes["embedding_cache_size"],
                cache_dir=str(self.cache_dir),
            )

        if optimal_sizes["search_cache_size"] != old_sizes["search_cache_size"]:
            self.search_cache = SearchResultCache(
                max_size=optimal_sizes["search_cache_size"],
                similarity_threshold=self.config["similarity_threshold"],
            )

        if optimal_sizes["file_cache_size"] != old_sizes["file_cache_size"]:
            self.file_cache = FileContentCache(
                max_size=optimal_sizes["file_cache_size"],
                max_file_size=self.config["max_file_size"],
            )

        self.adaptive_config["last_adjustment"] = current_time

        return {
            "adapted": True,
            "old_sizes": old_sizes,
            "new_sizes": optimal_sizes,
            "memory_usage": self._get_memory_usage(),
            "workload_intensity": self.workload_stats["accesses_last_minute"],
        }

    def record_cache_access(self, cache_type: str) -> None:
        """Record a cache access for workload tracking.

        Args:
            cache_type: Type of cache access ("embedding", "search", "file")
        """
        self.workload_stats["accesses_last_minute"] += 1
        self.workload_stats["accesses_last_hour"] += 1

        # Trigger adaptive sizing check periodically
        if self.workload_stats["accesses_last_minute"] % 50 == 0:
            try:
                adaptation_result = self.adapt_cache_sizes()
                if adaptation_result["adapted"]:
                    logger.info(f"Adaptive cache sizing: {adaptation_result}")
            except Exception as e:
                logger.warning(f"Adaptive cache sizing failed: {e}")

    def smart_prefetch(
        self, current_file: str, codebase_path: str, sentence_transformer_model
    ) -> Dict[str, int]:
        """Perform smart prefetching based on usage patterns.

        Args:
            current_file: Currently accessed file
            codebase_path: Path to the codebase
            sentence_transformer_model: Model for generating embeddings

        Returns:
            Dictionary with prefetch statistics
        """
        if not self.prefetch_config["enabled"]:
            return {"prefetched": 0, "skipped": 0}

        # Predict next files to access
        predicted_files = self.predict_next_files(current_file)

        if predicted_files:
            logger.info(
                f"Smart prefetching {len(predicted_files)} files based on usage patterns"
            )
            return self.prefetch_files(
                predicted_files, codebase_path, sentence_transformer_model
            )

        return {"prefetched": 0, "skipped": 0}

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on current statistics.

        Returns:
            List of performance recommendations
        """
        recommendations = []

        # Analyze hit rates
        embedding_hits = self.stats["hits"]["embedding"]
        embedding_misses = self.stats["misses"]["embedding"]
        embedding_hit_rate = (
            embedding_hits / (embedding_hits + embedding_misses)
            if (embedding_hits + embedding_misses) > 0
            else 0
        )

        search_hits = self.stats["hits"]["search"]
        search_misses = self.stats["misses"]["search"]
        search_hit_rate = (
            search_hits / (search_hits + search_misses)
            if (search_hits + search_misses) > 0
            else 0
        )

        file_hits = self.stats["hits"]["file"]
        file_misses = self.stats["misses"]["file"]
        file_hit_rate = (
            file_hits / (file_hits + file_misses)
            if (file_hits + file_misses) > 0
            else 0
        )

        # Cache efficiency recommendations
        if embedding_hit_rate < 0.5:
            recommendations.append(
                "Embedding cache hit rate is low - consider increasing embedding_cache_size"
            )
        if search_hit_rate < 0.3:
            recommendations.append(
                "Search cache hit rate is low - consider increasing search_cache_size"
            )
        if file_hit_rate < 0.6:
            recommendations.append(
                "File cache hit rate is low - consider increasing file_cache_size"
            )

        # Memory usage recommendations
        memory_usage = self._get_memory_usage()
        if memory_usage > 0.8:
            recommendations.append(
                "High memory usage detected - consider reducing cache sizes or enabling adaptive sizing"
            )
        elif memory_usage < 0.3:
            recommendations.append(
                "Low memory usage - cache sizes could potentially be increased"
            )

        # Workload-based recommendations
        workload_intensity = self.workload_stats["accesses_last_minute"]
        if workload_intensity > 200:
            recommendations.append(
                "High workload detected - consider increasing cache sizes for better performance"
            )
        elif workload_intensity < 10:
            recommendations.append(
                "Low workload detected - cache sizes could potentially be reduced to save memory"
            )

        # Pattern learning recommendations
        if len(self.usage_patterns["file_coaccess"]) > 50:
            recommendations.append(
                "Strong file co-access patterns detected - prefetching should be highly effective"
            )
        elif len(self.usage_patterns["file_coaccess"]) < 5:
            recommendations.append(
                "Limited co-access patterns - consider collecting more usage data for better prefetching"
            )

        # Adaptive sizing recommendations
        if not self.adaptive_config["enabled"]:
            recommendations.append(
                "Consider enabling adaptive cache sizing for automatic performance optimization"
            )

        # Prefetching recommendations
        prefetch_hits = sum(self.stats.get("prefetch_hits", {}).values())
        prefetch_misses = sum(self.stats.get("prefetch_misses", {}).values())
        if prefetch_hits > prefetch_misses * 2:
            recommendations.append(
                "Prefetching is highly effective - consider increasing max_prefetch_files"
            )
        elif prefetch_misses > prefetch_hits * 2:
            recommendations.append(
                "Prefetching hit rate is low - consider adjusting prefetching parameters"
            )

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics (alias for get_comprehensive_stats)."""
        return self.get_comprehensive_stats()

    def analyze_cache_efficiency(self) -> Dict[str, Any]:
        """Perform comprehensive cache efficiency analysis.

        Returns:
            Dictionary with detailed efficiency analysis and recommendations
        """
        analysis = {
            "overall_efficiency": {},
            "cache_breakdown": {},
            "temporal_analysis": {},
            "memory_analysis": {},
            "usage_patterns": {},
            "recommendations": [],
            "optimization_opportunities": [],
        }

        # Overall efficiency metrics
        total_hits = sum(self.stats["hits"].values())
        total_misses = sum(self.stats["misses"].values())
        total_requests = total_hits + total_misses

        analysis["overall_efficiency"] = {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": total_hits / total_requests
            if total_requests > 0
            else 0,
            "cache_effectiveness": self._calculate_cache_effectiveness(),
        }

        # Cache breakdown analysis
        analysis["cache_breakdown"] = {
            "embedding_cache": self._analyze_single_cache("embedding"),
            "search_cache": self._analyze_single_cache("search"),
            "file_cache": self._analyze_single_cache("file"),
        }

        # Temporal analysis
        analysis["temporal_analysis"] = self._analyze_temporal_patterns()

        # Memory analysis
        analysis["memory_analysis"] = self._analyze_memory_efficiency()

        # Usage pattern analysis
        analysis["usage_patterns"] = self._analyze_usage_patterns()

        # Generate recommendations
        analysis["recommendations"] = self._generate_detailed_recommendations(analysis)

        # Identify optimization opportunities
        analysis["optimization_opportunities"] = (
            self._identify_optimization_opportunities(analysis)
        )

        return analysis

    def _analyze_single_cache(self, cache_type: str) -> Dict[str, Any]:
        """Analyze a single cache type in detail.

        Args:
            cache_type: Type of cache to analyze ("embedding", "search", "file")

        Returns:
            Dictionary with cache analysis
        """
        hits = self.stats["hits"].get(cache_type, 0)
        misses = self.stats["misses"].get(cache_type, 0)
        total = hits + misses

        cache_analysis = {
            "hits": hits,
            "misses": misses,
            "total_requests": total,
            "hit_rate": hits / total if total > 0 else 0,
            "efficiency_score": self._calculate_cache_efficiency_score(cache_type),
        }

        # Cache-specific metrics
        if cache_type == "embedding":
            cache_analysis["cache_size"] = self.embedding_cache.size()
            cache_analysis["max_size"] = self.embedding_cache.max_size
            cache_analysis["utilization"] = (
                self.embedding_cache.size() / self.embedding_cache.max_size
            )
        elif cache_type == "search":
            cache_analysis["cache_size"] = self.search_cache.cache.size()
            cache_analysis["max_size"] = self.search_cache.max_size
            cache_analysis["utilization"] = (
                self.search_cache.cache.size() / self.search_cache.max_size
            )
        elif cache_type == "file":
            cache_analysis["cache_size"] = self.file_cache.cache.size()
            cache_analysis["max_size"] = self.file_cache.max_size
            cache_analysis["utilization"] = (
                self.file_cache.cache.size() / self.file_cache.max_size
            )
            cache_analysis["total_cached_size_mb"] = self.file_cache.stats()[
                "total_cached_size_mb"
            ]

        return cache_analysis

    def _calculate_cache_effectiveness(self) -> float:
        """Calculate overall cache effectiveness score (0.0 to 1.0).

        Returns:
            Effectiveness score
        """
        # Weighted effectiveness based on different factors
        hit_rate_weight = 0.4
        memory_efficiency_weight = 0.3
        prefetch_efficiency_weight = 0.3

        # Hit rate component
        total_hits = sum(self.stats["hits"].values())
        total_misses = sum(self.stats["misses"].values())
        hit_rate_score = (
            total_hits / (total_hits + total_misses)
            if (total_hits + total_misses) > 0
            else 0
        )

        # Memory efficiency component
        memory_usage = self._get_memory_usage()
        memory_efficiency = 1.0 - memory_usage  # Higher is better (less memory usage)

        # Prefetch efficiency component
        prefetch_hits = sum(self.stats.get("prefetch_hits", {}).values())
        prefetch_misses = sum(self.stats.get("prefetch_misses", {}).values())
        prefetch_efficiency = (
            prefetch_hits / (prefetch_hits + prefetch_misses)
            if (prefetch_hits + prefetch_misses) > 0
            else 0.5
        )

        return (
            hit_rate_weight * hit_rate_score
            + memory_efficiency_weight * memory_efficiency
            + prefetch_efficiency_weight * prefetch_efficiency
        )

    def _calculate_cache_efficiency_score(self, cache_type: str) -> float:
        """Calculate efficiency score for a specific cache type.

        Args:
            cache_type: Type of cache

        Returns:
            Efficiency score (0.0 to 1.0)
        """
        hits = self.stats["hits"].get(cache_type, 0)
        misses = self.stats["misses"].get(cache_type, 0)
        total = hits + misses

        if total == 0:
            return 0.5  # Neutral score for unused caches

        hit_rate = hits / total

        # Adjust score based on cache type priorities
        if cache_type == "embedding":
            # Embeddings are expensive to compute, so hit rate is very important
            return hit_rate * 0.9 + 0.1  # Slight boost for having the cache
        elif cache_type == "search":
            # Search results are moderately expensive
            return hit_rate * 0.8 + 0.2
        elif cache_type == "file":
            # File content is cheaper but still valuable
            return hit_rate * 0.7 + 0.3
        else:
            return hit_rate

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal access patterns.

        Returns:
            Dictionary with temporal pattern analysis
        """
        return {
            "access_frequency": {
                "per_minute": self.workload_stats["accesses_last_minute"],
                "per_hour": self.workload_stats["accesses_last_hour"],
            },
            "peak_usage_times": self._identify_peak_usage_times(),
            "access_patterns": {
                "most_frequent_files": sorted(
                    self.usage_patterns["file_access_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "recent_access_sequence": self.usage_patterns["file_access_sequence"][
                    -10:
                ],
            },
        }

    def _analyze_memory_efficiency(self) -> Dict[str, Any]:
        """Analyze memory efficiency of caches.

        Returns:
            Dictionary with memory efficiency analysis
        """
        memory_usage = self._get_memory_usage()

        return {
            "current_memory_usage_percent": memory_usage * 100,
            "cache_memory_breakdown": {
                "embedding_cache_mb": self._estimate_cache_memory_usage("embedding"),
                "search_cache_mb": self._estimate_cache_memory_usage("search"),
                "file_cache_mb": self._estimate_cache_memory_usage("file"),
            },
            "memory_efficiency_score": self._calculate_memory_efficiency_score(),
            "wasted_memory_mb": self._calculate_wasted_memory(),
        }

    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns for optimization opportunities.

        Returns:
            Dictionary with usage pattern analysis
        """
        return {
            "coaccess_patterns": {
                "total_patterns": len(self.usage_patterns["file_coaccess"]),
                "strong_patterns": self._identify_strong_coaccess_patterns(),
                "pattern_strength_distribution": self._calculate_pattern_strength_distribution(),
            },
            "access_frequency": {
                "unique_files_accessed": len(self.usage_patterns["file_access_counts"]),
                "most_accessed_files": sorted(
                    self.usage_patterns["file_access_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
                "access_concentration": self._calculate_access_concentration(),
            },
            "temporal_patterns": {
                "access_intervals": self._analyze_access_intervals(),
                "predictability_score": self._calculate_predictability_score(),
            },
        }

    def _generate_detailed_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate detailed recommendations based on comprehensive analysis.

        Args:
            analysis: Complete efficiency analysis

        Returns:
            List of detailed recommendations
        """
        recommendations = []

        # Hit rate based recommendations
        overall_hit_rate = analysis["overall_efficiency"]["overall_hit_rate"]
        if overall_hit_rate < 0.5:
            recommendations.append(
                "Overall cache hit rate is low - consider increasing cache sizes"
            )
        elif overall_hit_rate > 0.8:
            recommendations.append(
                "Excellent cache hit rate - current configuration is optimal"
            )

        # Cache-specific recommendations
        for cache_type, cache_analysis in analysis["cache_breakdown"].items():
            hit_rate = cache_analysis["hit_rate"]
            utilization = cache_analysis["utilization"]

            if hit_rate < 0.5:
                recommendations.append(
                    f"{cache_type} cache hit rate is low - consider increasing {cache_type}_cache_size"
                )
            elif utilization > 0.9:
                recommendations.append(
                    f"{cache_type} cache is nearly full - consider increasing {cache_type}_cache_size"
                )

        # Memory-based recommendations
        memory_usage = analysis["memory_analysis"]["current_memory_usage_percent"]
        if memory_usage > 85:
            recommendations.append(
                "High memory usage - consider reducing cache sizes or enabling adaptive sizing"
            )
        elif memory_usage < 30:
            recommendations.append(
                "Low memory usage - cache sizes could be increased for better performance"
            )

        # Pattern-based recommendations
        coaccess_patterns = analysis["usage_patterns"]["coaccess_patterns"][
            "total_patterns"
        ]
        if coaccess_patterns > 20:
            recommendations.append(
                "Strong co-access patterns detected - prefetching should be very effective"
            )
        elif coaccess_patterns < 5:
            recommendations.append(
                "Limited usage patterns - collect more data for better optimization"
            )

        return recommendations

    def _identify_optimization_opportunities(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities.

        Args:
            analysis: Complete efficiency analysis

        Returns:
            List of optimization opportunities with details
        """
        opportunities = []

        # Cache size optimization opportunities
        for cache_type, cache_analysis in analysis["cache_breakdown"].items():
            hit_rate = cache_analysis["hit_rate"]
            utilization = cache_analysis["utilization"]

            if hit_rate < 0.6 and utilization < 0.5:
                opportunities.append(
                    {
                        "type": "cache_size_reduction",
                        "cache": cache_type,
                        "description": f"Consider reducing {cache_type}_cache_size to save memory",
                        "potential_savings_mb": cache_analysis.get("cache_size", 0)
                        * 0.5,
                        "impact": "low",
                    }
                )
            elif hit_rate > 0.8 and utilization > 0.8:
                opportunities.append(
                    {
                        "type": "cache_size_increase",
                        "cache": cache_type,
                        "description": f"Consider increasing {cache_type}_cache_size for better performance",
                        "potential_benefit": "higher_hit_rate",
                        "impact": "medium",
                    }
                )

        # Prefetching opportunities
        prefetch_efficiency = analysis["overall_efficiency"].get(
            "prefetch_efficiency", 0
        )
        if prefetch_efficiency > 0.7:
            opportunities.append(
                {
                    "type": "prefetch_optimization",
                    "description": "Prefetching is highly effective - consider increasing max_prefetch_files",
                    "potential_benefit": "reduced_latency",
                    "impact": "high",
                }
            )

        return opportunities

    def _estimate_cache_memory_usage(self, cache_type: str) -> float:
        """Estimate memory usage of a specific cache type.

        Args:
            cache_type: Type of cache

        Returns:
            Estimated memory usage in MB
        """
        try:
            if cache_type == "embedding":
                # Rough estimate: 4 bytes per float * dimension * vectors * overhead
                return (self.embedding_cache.size() * 384 * 4 * 1.5) / (1024 * 1024)
            elif cache_type == "search":
                # Rough estimate for search results
                return (self.search_cache.cache.size() * 1000) / (1024 * 1024)
            elif cache_type == "file":
                return self.file_cache.stats()["total_cached_size_mb"]
        except Exception:
            pass

        return 0.0

    def _calculate_memory_efficiency_score(self) -> float:
        """Calculate memory efficiency score.

        Returns:
            Memory efficiency score (0.0 to 1.0)
        """
        memory_usage = self._get_memory_usage()
        hit_rate = sum(self.stats["hits"].values()) / max(
            sum(self.stats["hits"].values()) + sum(self.stats["misses"].values()), 1
        )

        # Higher hit rate justifies higher memory usage
        optimal_memory = hit_rate * 0.8  # Optimal memory is proportional to hit rate
        efficiency = 1.0 - abs(memory_usage - optimal_memory)

        return max(0.0, min(1.0, efficiency))

    def _calculate_wasted_memory(self) -> float:
        """Calculate wasted memory in MB.

        Returns:
            Wasted memory in MB
        """
        # Estimate wasted memory as memory used by low-hit-rate caches
        wasted = 0.0
        for cache_type in ["embedding", "search", "file"]:
            hits = self.stats["hits"].get(cache_type, 0)
            misses = self.stats["misses"].get(cache_type, 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

            if hit_rate < 0.3:  # Low hit rate indicates wasted memory
                wasted += self._estimate_cache_memory_usage(cache_type) * 0.7

        return wasted

    def _identify_peak_usage_times(self) -> List[str]:
        """Identify peak usage times (placeholder implementation).

        Returns:
            List of peak usage time descriptions
        """
        # This would require more sophisticated time-series analysis
        return ["Peak usage patterns require time-series data collection"]

    def _identify_strong_coaccess_patterns(self) -> List[Tuple[str, str, int]]:
        """Identify strong co-access patterns.

        Returns:
            List of (file1, file2, coaccess_count) tuples
        """
        strong_patterns = []
        for file1, coaccess_files in self.usage_patterns["file_coaccess"].items():
            for file2, count in coaccess_files.items():
                if count >= self.prefetch_config["coaccess_threshold"]:
                    strong_patterns.append((file1, file2, count))

        return sorted(strong_patterns, key=lambda x: x[2], reverse=True)[:10]

    def _calculate_pattern_strength_distribution(self) -> Dict[str, int]:
        """Calculate distribution of pattern strengths.

        Returns:
            Dictionary with pattern strength distribution
        """
        distribution = {"weak": 0, "medium": 0, "strong": 0}

        for file1, coaccess_files in self.usage_patterns["file_coaccess"].items():
            for file2, count in coaccess_files.items():
                if count >= 5:
                    distribution["strong"] += 1
                elif count >= 3:
                    distribution["medium"] += 1
                else:
                    distribution["weak"] += 1

        return distribution

    def _calculate_access_concentration(self) -> float:
        """Calculate access concentration (0.0 = spread out, 1.0 = concentrated).

        Returns:
            Access concentration score
        """
        if not self.usage_patterns["file_access_counts"]:
            return 0.0

        total_accesses = sum(self.usage_patterns["file_access_counts"].values())
        if total_accesses == 0:
            return 0.0

        # Calculate Gini coefficient-like concentration
        sorted_counts = sorted(
            self.usage_patterns["file_access_counts"].values(), reverse=True
        )
        cumulative = 0
        for i, count in enumerate(sorted_counts):
            cumulative += count
            if cumulative >= total_accesses * 0.8:  # 80% of accesses covered
                return (i + 1) / len(sorted_counts)

        return 1.0

    def _analyze_access_intervals(self) -> Dict[str, float]:
        """Analyze access intervals for temporal patterns.

        Returns:
            Dictionary with interval analysis
        """
        if not self.usage_patterns["access_intervals"]:
            return {"mean_interval": 0, "median_interval": 0, "predictability": 0}

        all_intervals = []
        for intervals in self.usage_patterns["access_intervals"].values():
            all_intervals.extend(intervals)

        if not all_intervals:
            return {"mean_interval": 0, "median_interval": 0, "predictability": 0}

        mean_interval = sum(all_intervals) / len(all_intervals)
        sorted_intervals = sorted(all_intervals)
        median_interval = sorted_intervals[len(sorted_intervals) // 2]

        # Calculate predictability (inverse of coefficient of variation)
        if mean_interval > 0:
            variance = sum((x - mean_interval) ** 2 for x in all_intervals) / len(
                all_intervals
            )
            std_dev = variance**0.5
            predictability = 1.0 / (1.0 + std_dev / mean_interval)
        else:
            predictability = 0.0

        return {
            "mean_interval": mean_interval,
            "median_interval": median_interval,
            "predictability": predictability,
        }

    def _calculate_predictability_score(self) -> float:
        """Calculate overall predictability score of access patterns.

        Returns:
            Predictability score (0.0 to 1.0)
        """
        if not self.usage_patterns["file_coaccess"]:
            return 0.0

        # Calculate based on strength and number of co-access patterns
        total_patterns = len(self.usage_patterns["file_coaccess"])
        strong_patterns = sum(
            1
            for coaccess_files in self.usage_patterns["file_coaccess"].values()
            for count in coaccess_files.values()
            if count >= self.prefetch_config["coaccess_threshold"]
        )

        pattern_strength = strong_patterns / max(total_patterns, 1)
        pattern_coverage = total_patterns / max(
            len(self.usage_patterns["file_access_counts"]), 1
        )

        return pattern_strength * 0.6 + pattern_coverage * 0.4


# Global cache instance
_cache_instance = None
_cache_lock = threading.Lock()


def get_cache_instance(config: Optional[Dict[str, Any]] = None) -> IntelligentCache:
    """Get the global cache instance (singleton pattern)."""
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = IntelligentCache(config=config)

    return _cache_instance


def reset_cache_instance() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _cache_instance
    with _cache_lock:
        if _cache_instance is not None:
            _cache_instance.save_persistent_cache()
            # Explicitly clear persistent cache files for testing
            try:
                if _cache_instance.cache_dir.exists():
                    import shutil
                    shutil.rmtree(_cache_instance.cache_dir)
            except Exception as e:
                logger.warning(f"Could not remove cache directory: {e}")
        _cache_instance = None
