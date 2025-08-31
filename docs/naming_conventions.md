# CodeSage MCP Server Naming Conventions

This document defines the naming conventions for the CodeSage MCP Server codebase. These rules ensure consistency, readability, and maintainability across the project. They are based on PEP 8 guidelines and align with the existing codebase patterns.

## Table of Contents

1. [Files and Directories](#files-and-directories)
2. [Modules and Packages](#modules-and-packages)
3. [Classes](#classes)
4. [Functions and Methods](#functions-and-methods)
5. [Variables and Constants](#variables-and-constants)
6. [Test Files](#test-files)
7. [Exceptions and Special Cases](#exceptions-and-special-cases)

## Files and Directories

### Python Files
- **Convention**: Use lowercase with underscores (`snake_case`)
- **Rationale**: Improves readability and follows PEP 8 standards
- **Examples**:
  - `cache.py`
  - `memory_manager.py`
  - `adaptive_cache_manager.py`
  - `performance_monitor.py`

### Directory Structure
- **Convention**: Use lowercase with underscores for package directories
- **Rationale**: Consistent with Python package naming and filesystem compatibility
- **Examples**:
  - `codesage_mcp/` (main package)
  - `codesage_mcp/tools/` (subpackage)
  - `benchmark_results/` (data directory)
  - `config/templates/` (configuration templates)

### Configuration and Documentation Files
- **Convention**: Use lowercase with underscores or hyphens as appropriate
- **Examples**:
  - `pyproject.toml`
  - `requirements.txt`
  - `docker-compose.yml`
  - `AGENT_WORKFLOW.md`
  - `error_handling_analysis.md`

## Modules and Packages

### Module Names
- **Convention**: `snake_case` matching the filename
- **Rationale**: Direct correspondence between filename and module name
- **Examples**:
  - `import cache` (from `cache.py`)
  - `import memory_manager` (from `memory_manager.py`)
  - `from codesage_mcp import adaptive_cache_manager`

### Package Names
- **Convention**: `snake_case` for package directories
- **Rationale**: Follows Python packaging standards and avoids conflicts
- **Examples**:
  - `codesage_mcp` (main package)
  - `codesage_mcp.tools` (subpackage)

## Classes

### Class Names
- **Convention**: `CamelCase` (PascalCase)
- **Rationale**: Standard Python convention for class names, distinguishes from functions/variables
- **Examples**:
  - `IntelligentCache`
  - `EmbeddingCache`
  - `PerformanceMonitor`
  - `AdaptiveCacheManager`
  - `JSONRPCRequest`
  - `GeminiCompatibleJSONResponse`

### Exception Classes
- **Convention**: `CamelCase` ending with "Error" or "Exception"
- **Rationale**: Clear identification as exceptions and follows Python standards
- **Examples**:
  - `BaseMCPError`
  - `ToolExecutionError`
  - `InvalidRequestError`
  - `IndexingError`

### Data Classes and Enums
- **Convention**: `CamelCase`
- **Rationale**: Consistent with class naming
- **Examples**:
  - `CacheMetrics` (dataclass)
  - `AdaptationStrategy` (enum)
  - `PerformanceMeasurement` (dataclass)
  - `LoadLevel` (enum)

## Functions and Methods

### Function Names
- **Convention**: `snake_case`
- **Rationale**: PEP 8 standard for functions, improves readability
- **Examples**:
  - `get_embedding()`
  - `store_embedding()`
  - `analyze_cache_effectiveness()`
  - `record_cache_access()`
  - `get_comprehensive_stats()`

### Method Names
- **Convention**: `snake_case`
- **Rationale**: Consistent with function naming
- **Examples**:
  - `__init__()` (special method)
  - `get_embedding()` (instance method)
  - `store_embedding()` (instance method)
  - `clear_all()` (instance method)

### Private Methods
- **Convention**: `snake_case` prefixed with single underscore
- **Rationale**: Indicates internal implementation details
- **Examples**:
  - `_load_persistent_cache()`
  - `_calculate_file_priority_score()`
  - `_generate_performance_recommendations()`
  - `_analyze_single_cache()`

### Static Methods and Class Methods
- **Convention**: `snake_case`
- **Rationale**: Consistent with other methods
- **Examples**:
  - `sanitize_filename()` (static method)
  - `get_cache_instance()` (module-level factory function)

## Variables and Constants

### Instance Variables
- **Convention**: `snake_case`
- **Rationale**: PEP 8 standard for variables
- **Examples**:
  - `max_size`
  - `cache_dir`
  - `similarity_threshold`
  - `embedding_cache`

### Local Variables
- **Convention**: `snake_case`
- **Rationale**: Improves readability and follows PEP 8
- **Examples**:
  - `content_hash`
  - `response_time`
  - `cached_embedding`
  - `file_path`

### Constants
- **Convention**: `UPPER_CASE` with underscores
- **Rationale**: Distinguishes constants from variables
- **Examples**:
  - `CHUNK_SIZE_TOKENS`
  - `DEFAULT_TIMEOUT`
  - `MAX_RETRIES`

### Module-Level Variables
- **Convention**: `snake_case` for regular variables, `UPPER_CASE` for constants
- **Rationale**: Clear distinction between configurable and constant values
- **Examples**:
  - `_cache_instance` (private module variable)
  - `logger` (module-level logger)

## Test Files

### Test File Names
- **Convention**: `test_*.py` or `test_*.py` with descriptive names
- **Rationale**: Standard pytest convention, easy identification
- **Examples**:
  - `test_cache.py`
  - `test_exceptions.py`
  - `test_performance_benchmarks.py`
  - `test_gemini_integration.py`
  - `test_error_handling_integration.py`

### Test Function Names
- **Convention**: `test_*` followed by descriptive name
- **Rationale**: pytest standard for test discovery
- **Examples**:
  - `test_get_embedding_hit()`
  - `test_store_embedding_success()`
  - `test_cache_invalidation()`
  - `test_performance_regression()`

### Test Class Names
- **Convention**: `Test*` in CamelCase
- **Rationale**: Standard for test classes
- **Examples**:
  - `TestIntelligentCache`
  - `TestPerformanceMonitor`
  - `TestAdaptiveCacheManager`

## Exceptions and Special Cases

### Acronyms in Names
- **Convention**: Treat acronyms as regular words, maintain case consistency
- **Rationale**: Improves readability, avoids inconsistent capitalization
- **Examples**:
  - `get_embedding()` (not `getEmbedding()`)
  - `LLMAnalysisManager` (not `LLMAnalysisManager` or `llmAnalysisManager`)
  - `JSONRPCRequest` (not `JsonRpcRequest`)
  - `MCPError` (not `McpError`)

### API and Tool Names
- **Convention**: `snake_case` for internal tool names, maintain external API names as-is
- **Rationale**: Internal consistency while respecting external interfaces
- **Examples**:
  - Internal: `read_code_file_tool`
  - External: May preserve original casing if from external APIs

### Legacy Code
- **Convention**: Gradually refactor to match conventions when modifying
- **Rationale**: Avoid breaking changes while improving consistency
- **Examples**: Update variable names when functions are modified

### Third-Party Integration
- **Convention**: Follow external library conventions when interfacing
- **Rationale**: Respect external APIs and maintain compatibility
- **Examples**:
  - Use `faiss_index` to match FAISS library naming
  - Use `sentence_transformer_model` to match SentenceTransformers library

## Enforcement and Tools

### Linting
- Use Ruff or similar tools configured for PEP 8 compliance
- Configure custom rules for project-specific patterns

### Code Reviews
- Naming conventions should be checked during code review
- Automated tools should flag violations

### Documentation
- Update this document when new patterns are established
- Reference this document in contribution guidelines

## Rationale Summary

These naming conventions provide:
- **Consistency**: Uniform style across the codebase
- **Readability**: Clear, descriptive names that follow Python standards
- **Maintainability**: Easy to understand and modify code
- **Collaboration**: Common standards for team development
- **Tool Support**: Compatible with Python development tools and linters

Following these conventions ensures the CodeSage MCP Server codebase remains professional, maintainable, and aligned with Python community standards.