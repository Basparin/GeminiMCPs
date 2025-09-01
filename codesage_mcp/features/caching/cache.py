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

import logging
import threading
from typing import Dict, List, Any, Optional

from .cache_components import LRUCache, EmbeddingCache, SearchResultCache, FileContentCache

# Configure logging
logger = logging.getLogger(__name__)

# Import custom exceptions


# IntelligentCache has been moved to intelligent_cache.py for better modularity






# Global cache instances for backward compatibility
_embedding_cache_instance = None
_search_cache_instance = None
_file_cache_instance = None
_cache_lock = threading.Lock()


def get_cache_instance():
    """Get the global IntelligentCache instance (singleton pattern)."""
    # Lazy import to avoid circular dependency
    from .intelligent_cache import IntelligentCache
    return IntelligentCache()


def get_embedding_cache(max_size: int = 5000, cache_dir: str = ".codesage") -> EmbeddingCache:
    """Get the global embedding cache instance (singleton pattern)."""
    global _embedding_cache_instance

    if _embedding_cache_instance is None:
        with _cache_lock:
            if _embedding_cache_instance is None:
                _embedding_cache_instance = EmbeddingCache(max_size=max_size, cache_dir=cache_dir)

    return _embedding_cache_instance


def get_search_cache(max_size: int = 1000, similarity_threshold: float = 0.85) -> SearchResultCache:
    """Get the global search cache instance (singleton pattern)."""
    global _search_cache_instance

    if _search_cache_instance is None:
        with _cache_lock:
            if _search_cache_instance is None:
                _search_cache_instance = SearchResultCache(max_size=max_size, similarity_threshold=similarity_threshold)

    return _search_cache_instance


def get_file_cache(max_size: int = 100, max_file_size: int = 1024 * 1024) -> FileContentCache:
    """Get the global file cache instance (singleton pattern)."""
    global _file_cache_instance

    if _file_cache_instance is None:
        with _cache_lock:
            if _file_cache_instance is None:
                _file_cache_instance = FileContentCache(max_size=max_size, max_file_size=max_file_size)

    return _file_cache_instance


def reset_cache_instances() -> None:
    """Reset all global cache instances (useful for testing)."""
    global _embedding_cache_instance, _search_cache_instance, _file_cache_instance
    with _cache_lock:
        if _embedding_cache_instance is not None:
            _embedding_cache_instance.clear()
        if _search_cache_instance is not None:
            _search_cache_instance.clear()
        if _file_cache_instance is not None:
            _file_cache_instance.clear()
        _embedding_cache_instance = None
        _search_cache_instance = None
        _file_cache_instance = None


# Alias for backward compatibility
reset_cache_instance = reset_cache_instances
