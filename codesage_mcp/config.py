"""Configuration Module for CodeSage MCP Server.

This module manages the configuration for the CodeSage MCP Server, including API keys
for various services like Groq, OpenRouter, and Google AI, as well as indexing options.

It uses `python-dotenv` to automatically load environment variables from a `.env` file
located in the project root directory. This allows for secure configuration without
hardcoding sensitive information in the source code.

Environment Variables:
      GROQ_API_KEY (str): API key for Groq.
      OPENROUTER_API_KEY (str): API key for OpenRouter.
      GOOGLE_API_KEY (str): API key for Google AI.
      ENABLE_INCREMENTAL_INDEXING (bool): Enable incremental indexing (default: true).
      FORCE_FULL_REINDEX (bool): Force full re-indexing even when incremental is enabled (default: false).
      ENABLE_CACHING (bool): Enable intelligent caching system (default: true).
      EMBEDDING_CACHE_SIZE (int): Maximum number of embeddings to cache (default: 5000).
      SEARCH_CACHE_SIZE (int): Maximum number of search results to cache (default: 1000).
      FILE_CACHE_SIZE (int): Maximum number of file contents to cache (default: 100).
      CACHE_SIMILARITY_THRESHOLD (float): Similarity threshold for search result caching (default: 0.85).
      MAX_FILE_SIZE_MB (int): Maximum file size in MB for content caching (default: 1).
      ENABLE_CACHE_PERSISTENCE (bool): Enable cache persistence across restarts (default: true).
      ENABLE_CACHE_WARMING (bool): Enable cache warming on startup (default: true).
      ENABLE_MEMORY_MAPPED_INDEXES (bool): Enable memory-mapped FAISS indexes (default: true).
      INDEX_TYPE (str): FAISS index type - 'flat', 'ivf', or 'auto' (default: 'auto').
      MAX_MEMORY_MB (int): Maximum memory usage in MB (default: 2048).
      MODEL_CACHE_TTL_MINUTES (int): Model cache TTL in minutes (default: 60).
      ENABLE_MODEL_QUANTIZATION (bool): Enable model quantization (default: false).
      CHUNK_SIZE_TOKENS (int): Document chunk size in tokens (default: 750).
      ENABLE_MEMORY_MONITORING (bool): Enable memory monitoring (default: true).

Example .env file:
      ```env
      GROQ_API_KEY="gsk_..."
      OPENROUTER_API_KEY="sk-or-..."
      GOOGLE_API_KEY="AIza..."
      ENABLE_INCREMENTAL_INDEXING=true
      FORCE_FULL_REINDEX=false
      ENABLE_CACHING=true
      EMBEDDING_CACHE_SIZE=5000
      SEARCH_CACHE_SIZE=1000
      FILE_CACHE_SIZE=100
      CACHE_SIMILARITY_THRESHOLD=0.85
      MAX_FILE_SIZE_MB=1
      ENABLE_CACHE_PERSISTENCE=true
      ENABLE_CACHE_WARMING=true
      ENABLE_MEMORY_MAPPED_INDEXES=true
      INDEX_TYPE=auto
      MAX_MEMORY_MB=2048
      MODEL_CACHE_TTL_MINUTES=60
      ENABLE_MODEL_QUANTIZATION=false
      CHUNK_SIZE_TOKENS=750
      ENABLE_MEMORY_MONITORING=true
      ```
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_required_env_var(var_name: str) -> str:
    """Get a required environment variable with validation.

    Args:
        var_name: Name of the environment variable

    Returns:
        The environment variable value

    Raises:
        ValueError: If the environment variable is not set or empty
    """
    value = os.getenv(var_name)
    if not value or value.strip() == "":
        raise ValueError(
            f"Required environment variable '{var_name}' is not set. "
            f"Please set it in your .env file or environment."
        )
    return value.strip()


def get_optional_env_var(var_name: str) -> Optional[str]:
    """Get an optional environment variable.

    Args:
        var_name: Name of the environment variable

    Returns:
        The environment variable value or None if not set
    """
    value = os.getenv(var_name)
    return value.strip() if value else None


# Load API keys from environment variables
try:
    GROQ_API_KEY = get_required_env_var("GROQ_API_KEY")
except ValueError:
    GROQ_API_KEY = None

try:
    OPENROUTER_API_KEY = get_required_env_var("OPENROUTER_API_KEY")
except ValueError:
    OPENROUTER_API_KEY = None

try:
    GOOGLE_API_KEY = get_required_env_var("GOOGLE_API_KEY")
except ValueError:
    GOOGLE_API_KEY = None

# Load incremental indexing configuration
try:
    ENABLE_INCREMENTAL_INDEXING = (
        get_optional_env_var("ENABLE_INCREMENTAL_INDEXING").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_INCREMENTAL_INDEXING = True  # Default to enabled

try:
    FORCE_FULL_REINDEX = get_optional_env_var("FORCE_FULL_REINDEX").lower() == "true"
except (ValueError, AttributeError):
    FORCE_FULL_REINDEX = False  # Default to incremental

# Load cache configuration
try:
    ENABLE_CACHING = get_optional_env_var("ENABLE_CACHING").lower() == "true"
except (ValueError, AttributeError):
    ENABLE_CACHING = True  # Default to enabled

try:
    EMBEDDING_CACHE_SIZE = int(get_optional_env_var("EMBEDDING_CACHE_SIZE") or "5000")
except (ValueError, AttributeError):
    EMBEDDING_CACHE_SIZE = 5000  # Default embedding cache size

try:
    SEARCH_CACHE_SIZE = int(get_optional_env_var("SEARCH_CACHE_SIZE") or "1000")
except (ValueError, AttributeError):
    SEARCH_CACHE_SIZE = 1000  # Default search cache size

try:
    FILE_CACHE_SIZE = int(get_optional_env_var("FILE_CACHE_SIZE") or "100")
except (ValueError, AttributeError):
    FILE_CACHE_SIZE = 100  # Default file cache size

try:
    CACHE_SIMILARITY_THRESHOLD = float(
        get_optional_env_var("CACHE_SIMILARITY_THRESHOLD") or "0.85"
    )
except (ValueError, AttributeError):
    CACHE_SIMILARITY_THRESHOLD = 0.85  # Default similarity threshold for search caching

try:
    MAX_FILE_SIZE_MB = int(get_optional_env_var("MAX_FILE_SIZE_MB") or "1")
except (ValueError, AttributeError):
    MAX_FILE_SIZE_MB = 1  # Default max file size for caching (1MB)

try:
    ENABLE_CACHE_PERSISTENCE = (
        get_optional_env_var("ENABLE_CACHE_PERSISTENCE").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_CACHE_PERSISTENCE = True  # Default to enabled

try:
    ENABLE_CACHE_WARMING = (
        get_optional_env_var("ENABLE_CACHE_WARMING").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_CACHE_WARMING = True  # Default to enabled

# Load memory optimization configuration
try:
    ENABLE_MEMORY_MAPPED_INDEXES = (
        get_optional_env_var("ENABLE_MEMORY_MAPPED_INDEXES").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_MEMORY_MAPPED_INDEXES = True  # Default to enabled

try:
    INDEX_TYPE = get_optional_env_var("INDEX_TYPE") or "auto"
except (ValueError, AttributeError):
    INDEX_TYPE = "auto"  # Default to auto

try:
    MAX_MEMORY_MB = int(get_optional_env_var("MAX_MEMORY_MB") or "2048")
except (ValueError, AttributeError):
    MAX_MEMORY_MB = 2048  # Default 2GB

try:
    MODEL_CACHE_TTL_MINUTES = int(
        get_optional_env_var("MODEL_CACHE_TTL_MINUTES") or "60"
    )
except (ValueError, AttributeError):
    MODEL_CACHE_TTL_MINUTES = 60  # Default 1 hour

try:
    ENABLE_MODEL_QUANTIZATION = (
        get_optional_env_var("ENABLE_MODEL_QUANTIZATION").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_MODEL_QUANTIZATION = False  # Default to disabled

try:
    CHUNK_SIZE_TOKENS = int(get_optional_env_var("CHUNK_SIZE_TOKENS") or "750")
except (ValueError, AttributeError):
    CHUNK_SIZE_TOKENS = 750  # Default chunk size

try:
    ENABLE_MEMORY_MONITORING = (
        get_optional_env_var("ENABLE_MEMORY_MONITORING").lower() == "true"
    )
except (ValueError, AttributeError):
    ENABLE_MEMORY_MONITORING = True  # Default to enabled


def validate_configuration() -> list[str]:
    """Validate the current configuration and return any issues found.

    Returns:
        List of configuration issues (empty if all valid)
    """
    issues = []

    if not GROQ_API_KEY:
        issues.append("GROQ_API_KEY is not configured")

    if not OPENROUTER_API_KEY:
        issues.append("OPENROUTER_API_KEY is not configured")

    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY is not configured")

    # Note: ENABLE_INCREMENTAL_INDEXING and FORCE_FULL_REINDEX are optional
    # and have defaults, so no validation needed

    return issues


def get_configuration_status() -> dict:
    """Get the current configuration status.

    Returns:
        Dictionary with configuration status information
    """
    issues = validate_configuration()

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "configured_providers": {
            "groq": GROQ_API_KEY is not None,
            "openrouter": OPENROUTER_API_KEY is not None,
            "google": GOOGLE_API_KEY is not None,
        },
        "indexing_config": {
            "incremental_indexing_enabled": ENABLE_INCREMENTAL_INDEXING,
            "force_full_reindex": FORCE_FULL_REINDEX,
        },
        "cache_config": {
            "caching_enabled": ENABLE_CACHING,
            "embedding_cache_size": EMBEDDING_CACHE_SIZE,
            "search_cache_size": SEARCH_CACHE_SIZE,
            "file_cache_size": FILE_CACHE_SIZE,
            "similarity_threshold": CACHE_SIMILARITY_THRESHOLD,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "cache_persistence_enabled": ENABLE_CACHE_PERSISTENCE,
            "cache_warming_enabled": ENABLE_CACHE_WARMING,
        },
        "memory_config": {
            "memory_mapped_indexes_enabled": ENABLE_MEMORY_MAPPED_INDEXES,
            "index_type": INDEX_TYPE,
            "max_memory_mb": MAX_MEMORY_MB,
            "model_cache_ttl_minutes": MODEL_CACHE_TTL_MINUTES,
            "model_quantization_enabled": ENABLE_MODEL_QUANTIZATION,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "memory_monitoring_enabled": ENABLE_MEMORY_MONITORING,
        },
    }
