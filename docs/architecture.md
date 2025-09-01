# CodeSage MCP Server - System Architecture

## Overview

CodeSage MCP Server is a high-performance, production-ready Model Context Protocol (MCP) server designed for enterprise-scale code analysis and search. The system delivers exceptional performance through advanced optimization techniques while maintaining comprehensive functionality for code intelligence tasks.

## Core Architecture Principles

### 🏗️ Modular Design
The system follows a modular architecture with clear separation of concerns:

- **Main Server Layer**: FastAPI-based MCP server handling JSON-RPC requests
- **Core Components**: Specialized modules for indexing, caching, memory management
- **Tool Layer**: Extensible tool system for various code analysis tasks
- **Storage Layer**: Optimized persistence and retrieval systems

### ⚡ Performance-First Approach
- **Sub-millisecond search responses** through optimized indexing
- **Memory-efficient operations** with intelligent resource management
- **Parallel processing capabilities** for large-scale codebases
- **Adaptive optimization** based on workload patterns

### 🔧 Enterprise-Ready Features
- **Production deployment support** with Docker and orchestration
- **Comprehensive monitoring** and health checking
- **Security hardening** with encrypted configuration
- **Scalability patterns** for growing codebases

## System Components

### 1. Main Server Layer (`main.py`)

**FastAPI-based MCP Server**
- **Protocol**: JSON-RPC 2.0 over HTTP
- **Framework**: FastAPI with async support
- **Endpoints**:
  - `/mcp`: Main JSON-RPC endpoint
  - `/`: Health check endpoint
- **Features**:
  - Tool discovery and registration
  - Request routing and validation
  - Error handling and logging
  - Performance monitoring integration

**Key Classes:**
- `JSONRPCRequest/Response`: Protocol message handling
- `get_all_tools_definitions_as_object()`: Tool metadata management
- `TOOL_FUNCTIONS`: Tool function mapping

### 2. Core Components

#### Memory Management System (`codesage_mcp/features/memory_management/memory_manager.py`)

**Intelligent Memory Optimization**
- **Memory Monitoring**: Real-time memory usage tracking with psutil
- **Model Caching**: TTL-based caching for sentence transformers
- **Memory-mapped Indexes**: FAISS indexes with memory mapping support
- **Automatic Cleanup**: Proactive memory management during high usage
- **Quantization Support**: Model size reduction for memory efficiency

**Key Classes:**
- `MemoryManager`: Main memory management orchestrator
- `ModelCache`: Model loading and caching with TTL
- `get_memory_manager()`: Global singleton instance

#### Intelligent Caching System (`codesage_mcp/features/caching/cache.py`)

**Multi-Strategy Caching Architecture**
- **Embedding Cache**: File-based invalidation for embeddings
- **Search Result Cache**: Similarity-based retrieval for search results
- **File Content Cache**: Memory-efficient file content storage
- **LRU Eviction**: Configurable cache sizes with LRU policies
- **Persistent Storage**: Disk-based cache persistence
- **Adaptive Sizing**: Dynamic cache adjustment based on workload

**Key Classes:**
- `IntelligentCache`: Main cache orchestrator
- `EmbeddingCache`: Specialized embedding cache
- `SearchResultCache`: Similarity-based search result cache
- `FileContentCache`: File content cache with size limits

#### Indexing System (`codesage_mcp/core/indexing.py`)

**Advanced Codebase Indexing**
- **Incremental Indexing**: Dependency tracking with intelligent change detection
- **Parallel Processing**: Concurrent file processing for large codebases
- **FAISS Integration**: Vector similarity search with optimized indexes
- **Chunking Support**: Large file processing with intelligent chunking
- **Index Optimization**: Compression and defragmentation capabilities
- **Health Monitoring**: Index performance and fragmentation analysis

**Key Classes:**
- `IndexingManager`: Main indexing orchestrator
- `DocumentChunker`: Intelligent document chunking

## Recent Modularization Changes

### 📁 Updated Module Structure

Following the recent modularization effort, the codebase has been reorganized into a clear hierarchical structure under the `codesage_mcp` package:

```
codesage_mcp/
├── __init__.py
├── main.py                    # Main server entry point
├── configuration.py           # Configuration management
├── config/                    # Configuration modules
│   ├── __init__.py
│   └── config.py
├── core/                      # Core system components
│   ├── __init__.py
│   ├── api_handling.py
│   ├── chunking.py
│   ├── code_model.py
│   ├── data_structures.py
│   ├── error_handling.py
│   ├── error_reporting.py
│   ├── exceptions.py
│   ├── gemini_compatibility.py
│   ├── indexing_search.py
│   ├── indexing.py
│   ├── logging_config.py
│   ├── searching.py
│   └── utils.py
├── features/                  # Feature-specific modules
│   ├── __init__.py
│   ├── intelligent_prefetcher.py
│   ├── caching/
│   │   ├── __init__.py
│   │   ├── adaptive_cache_manager.py
│   │   ├── cache_analysis.py
│   │   ├── cache_components.py
│   │   ├── cache.py
│   │   └── intelligent_cache.py
│   ├── codebase_manager/
│   │   ├── __init__.py
│   │   ├── advanced_analysis.py
│   │   └── codebase_manager.py
│   ├── llm_analysis/
│   │   ├── __init__.py
│   │   └── llm_analysis.py
│   ├── memory_management/
│   │   ├── __init__.py
│   │   ├── memory_manager.py
│   │   ├── memory_pattern_monitor.py
│   │   ├── workload_adaptive_memory.py
│   │   └── workload_pattern_recognition.py
│   ├── performance_monitoring/
│   │   ├── __init__.py
│   │   ├── auto_performance_tuner.py
│   │   ├── performance_monitor.py
│   │   ├── performance_report_generator.py
│   │   └── prometheus_client.py
│   └── user_feedback/
│       ├── __init__.py
│       └── user_feedback.py
└── tools/                     # Tool implementations
    ├── __init__.py
    ├── adaptive_cache_tools.py
    ├── advanced_analysis_tools.py
    ├── auto_performance_tuning_tools.py
    ├── cache_analysis_tools.py
    ├── code_generation.py
    ├── codebase_analysis.py
    ├── configuration.py
    ├── continuous_improvement.py
    ├── intelligent_prefetch_tools.py
    ├── llm_analysis.py
    ├── memory_pattern_tools.py
    ├── performance_monitoring.py
    ├── trend_analysis_tools.py
    └── user_feedback_tools.py
```

### 🔧 Import Path Updates

All import statements have been updated to use absolute paths from the `codesage_mcp` package root:

**Before (old structure):**
```python
from indexing import IndexingManager
from cache import get_cache_instance
from memory_manager import get_memory_manager
```

**After (new modular structure):**
```python
from codesage_mcp.core.indexing import IndexingManager
from codesage_mcp.features.caching.cache import get_cache_instance
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager
```

### 📋 Module Organization Benefits

- **Clear Separation**: Core functionality separated from feature-specific code
- **Scalability**: Easy to add new features without affecting core components
- **Maintainability**: Reduced circular dependencies and improved code organization
- **Testability**: Isolated modules enable better unit testing
- **Import Clarity**: Absolute imports prevent ambiguity and improve IDE support

### 3. Tool Layer

**Extensible Tool System**
- **Core Tools**: Indexing, searching, file reading, code analysis
- **Analysis Tools**: Code improvement suggestions, performance profiling
- **Configuration Tools**: API key management, system configuration
- **Utility Tools**: Boilerplate generation, documentation automation

**Tool Categories:**
- **Code Intelligence**: `read_code_file`, `search_codebase`, `semantic_search_codebase`
- **Analysis**: `analyze_codebase_improvements`, `suggest_code_improvements`
- **Management**: `index_codebase`, `get_configuration`, `get_cache_statistics`
- **Generation**: `generate_unit_tests`, `generate_boilerplate`, `auto_document_tool`

### 4. Storage Layer

#### FAISS Vector Indexes
- **Index Types**: Flat, IVF, PQ, Scalar Quantization
- **Memory Mapping**: Support for memory-mapped indexes
- **Optimization**: Automatic index optimization based on data characteristics
- **Compression**: Multiple compression strategies for memory efficiency

#### Persistent Storage
- **Index Persistence**: FAISS indexes saved to disk
- **Metadata Storage**: JSON-based metadata and file tracking
- **Cache Persistence**: Optional cache persistence across restarts
- **Configuration**: Environment-based configuration management

## Data Flow Architecture

### 1. Indexing Workflow

```
Codebase Files → Gitignore Filtering → File Processing → Chunking → Embedding Generation → FAISS Indexing → Persistence
```

**Detailed Flow:**
1. **File Discovery**: Recursive directory scanning with `.gitignore` support
2. **Dependency Analysis**: Python import analysis for incremental indexing
3. **Parallel Processing**: Concurrent file processing with thread pools
4. **Chunking Strategy**: Intelligent document chunking for large files
5. **Embedding Generation**: Sentence transformer encoding with caching
6. **Index Optimization**: FAISS index creation with optimal parameters
7. **Persistence**: Index and metadata storage to disk

### 2. Search Workflow

```
Query → Embedding Generation → Cache Check → FAISS Search → Result Ranking → Response
```

**Detailed Flow:**
1. **Query Processing**: Input validation and preprocessing
2. **Embedding**: Query text to vector embedding
3. **Cache Lookup**: Similarity-based cache retrieval
4. **Vector Search**: FAISS index search with optimized parameters
5. **Result Processing**: Ranking and metadata attachment
6. **Response Formatting**: JSON-RPC response generation

### 3. Memory Management Workflow

```
System Monitoring → Memory Threshold Check → Cleanup Trigger → Resource Optimization → Adaptive Adjustment
```

**Detailed Flow:**
1. **Continuous Monitoring**: Background memory usage tracking
2. **Threshold Analysis**: Configurable memory limit checking
3. **Cleanup Procedures**: Model cache clearing and garbage collection
4. **Adaptive Sizing**: Dynamic cache size adjustment
5. **Resource Optimization**: Index compression and memory mapping

## Performance Optimization Strategies

### 1. Memory Optimization

#### Memory-mapped Indexes
- **Technique**: FAISS memory mapping for large indexes
- **Benefit**: Reduced memory footprint for large codebases
- **Implementation**: `memory_manager.load_faiss_index()` with mmap support

#### Model Quantization
- **Technique**: 8-bit quantization for sentence transformers
- **Benefit**: 50-75% reduction in model memory usage
- **Implementation**: Configurable quantization in `ENABLE_MODEL_QUANTIZATION`

#### Intelligent Cleanup
- **Technique**: Proactive memory cleanup during high usage
- **Benefit**: Prevents memory exhaustion in long-running processes
- **Implementation**: Background monitoring with automatic cleanup

### 2. Caching Strategies

#### Multi-Level Caching
- **Embedding Cache**: Avoids re-encoding unchanged files
- **Search Cache**: Similarity-based result caching
- **File Cache**: Frequently accessed file content
- **Implementation**: LRU with configurable sizes and persistence

#### Adaptive Cache Sizing
- **Technique**: Dynamic cache size adjustment based on workload
- **Benefit**: Optimal memory utilization across different usage patterns
- **Implementation**: Workload monitoring with automatic adjustment

### 3. Parallel Processing

#### Thread Pool Optimization
- **Technique**: Configurable thread pools for CPU-bound tasks
- **Benefit**: Improved throughput for large codebase processing
- **Implementation**: `ThreadPoolExecutor` with adaptive worker counts

#### Batch Processing
- **Technique**: File processing in optimized batches
- **Benefit**: Reduced overhead and improved memory locality
- **Implementation**: Configurable batch sizes with memory monitoring

### 4. Index Optimization

#### Adaptive Index Types
- **Technique**: Automatic selection of optimal FAISS index type
- **Benefit**: Best performance for different data sizes
- **Implementation**: Dataset size analysis with index type selection

#### Index Compression
- **Technique**: Product quantization and scalar quantization
- **Benefit**: Reduced memory usage with minimal accuracy loss
- **Implementation**: Multiple compression strategies with automatic selection

## Scalability Patterns

### 1. Horizontal Scaling

#### Multi-Instance Deployment
- **Pattern**: Multiple CodeSage instances behind load balancer
- **Benefit**: Increased throughput for high-traffic scenarios
- **Implementation**: Stateless design with shared storage

#### Database Sharding
- **Pattern**: Index sharding across multiple instances
- **Benefit**: Support for very large codebases
- **Implementation**: Configurable sharding strategies

### 2. Vertical Scaling

#### Memory Optimization
- **Pattern**: Memory-efficient data structures and algorithms
- **Benefit**: Support for larger codebases on single instance
- **Implementation**: Memory mapping, compression, and caching

#### Resource Pooling
- **Pattern**: Shared resource pools for expensive operations
- **Benefit**: Better resource utilization under varying loads
- **Implementation**: Connection pooling and resource management

## Security Architecture

### 1. Configuration Security

#### Encrypted Storage
- **Technique**: API keys stored with encryption at rest
- **Benefit**: Protection of sensitive configuration data
- **Implementation**: Environment variable masking and secure storage

#### Access Control
- **Technique**: MCP server access control and authentication
- **Benefit**: Authorized access to code analysis capabilities
- **Implementation**: Configurable authentication mechanisms

### 2. Data Protection

#### Input Validation
- **Technique**: Comprehensive input validation and sanitization
- **Benefit**: Protection against malicious inputs and injection attacks
- **Implementation**: Pydantic models with strict validation

#### Secure Communication
- **Technique**: TLS/HTTPS for MCP communication
- **Benefit**: Encrypted communication channels
- **Implementation**: Production deployment with TLS termination

## Monitoring and Observability

### 1. Performance Monitoring

#### Real-time Metrics
- **Metrics**: Response times, memory usage, cache hit rates
- **Benefit**: Continuous performance visibility
- **Implementation**: Integrated monitoring with Prometheus-compatible metrics

#### Health Checks
- **Endpoints**: Health check endpoints for load balancers
- **Benefit**: Automatic instance management and failover
- **Implementation**: Comprehensive health status reporting

### 2. Logging and Tracing

#### Structured Logging
- **Format**: JSON-structured logs with context
- **Benefit**: Better log analysis and debugging
- **Implementation**: Configurable logging levels and formats

#### Performance Tracing
- **Technique**: Request tracing with performance metrics
- **Benefit**: Detailed performance bottleneck identification
- **Implementation**: Integrated tracing with timing information

## Deployment Architecture

### 1. Containerized Deployment

#### Docker Optimization
- **Base Images**: Optimized Python base images
- **Layer Caching**: Efficient Docker layer utilization
- **Resource Limits**: Configurable CPU and memory limits
- **Health Checks**: Container-level health monitoring

#### Orchestration
- **Docker Compose**: Single-node deployment with monitoring
- **Kubernetes**: Production orchestration with auto-scaling
- **Load Balancing**: Multi-instance load distribution

### 2. Configuration Management

#### Environment-based Config
- **Technique**: 12-factor app configuration via environment
- **Benefit**: Flexible deployment across different environments
- **Implementation**: Comprehensive environment variable support

#### Configuration Templates
- **Templates**: Pre-configured setups for different use cases
- **Benefit**: Quick deployment with optimized settings
- **Implementation**: Environment-specific configuration templates

## Future Architecture Considerations

### 1. Distributed Processing

#### Index Distribution
- **Pattern**: Distributed FAISS indexes across multiple nodes
- **Benefit**: Support for massive codebases
- **Implementation**: Planned distributed index management

#### Query Federation
- **Pattern**: Federated search across multiple CodeSage instances
- **Benefit**: Unified search across distributed deployments
- **Implementation**: Planned federation protocol

### 2. Advanced Caching

#### Distributed Cache
- **Pattern**: Redis-based distributed caching
- **Benefit**: Shared cache across multiple instances
- **Implementation**: Planned Redis integration

#### Machine Learning Optimization
- **Pattern**: ML-based cache prediction and prefetching
- **Benefit**: Proactive caching based on usage patterns
- **Implementation**: Planned ML optimization layer

### 3. Advanced Analytics

#### Usage Analytics
- **Pattern**: Detailed usage analytics and reporting
- **Benefit**: Insights into system usage and optimization opportunities
- **Implementation**: Planned analytics dashboard

#### Performance Prediction
- **Pattern**: ML-based performance prediction and alerting
- **Benefit**: Proactive performance management
- **Implementation**: Planned predictive analytics

## Conclusion

CodeSage MCP Server's architecture represents a comprehensive approach to high-performance code analysis, combining advanced optimization techniques with enterprise-ready features. The modular design ensures maintainability while the performance-first approach delivers exceptional results across different scales and use cases.

The system's architecture supports both current operational needs and future growth through scalable patterns and extensible design principles.