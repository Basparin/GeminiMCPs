# CodeSage MCP Server - Configuration Guide

## Overview

CodeSage MCP Server offers **comprehensive configuration options** that allow you to optimize performance for your specific environment and use case. The configuration system supports environment variables, configuration files, and adaptive tuning.

## Configuration Architecture

### Configuration Sources

#### 1. Environment Variables (Primary)
- **Method**: Environment variables with `CODESAGE_` prefix
- **Benefits**: Easy deployment, container-friendly, secure
- **Precedence**: Highest priority

#### 2. Configuration Files (Secondary)
- **Method**: JSON/YAML configuration files
- **Benefits**: Structured configuration, validation
- **Precedence**: Medium priority

#### 3. Default Values (Fallback)
- **Method**: Hardcoded sensible defaults
- **Benefits**: Always works out-of-the-box
- **Precedence**: Lowest priority

### Configuration Loading Priority
```
Environment Variables > Configuration File > Defaults
```

## Core Configuration Categories

### 1. Performance Configuration

#### Memory Management
```bash
# Memory limits and monitoring
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_MAX_MEMORY_MB=1024
CODESAGE_ENABLE_MEMORY_MONITORING=true
CODESAGE_MEMORY_CHECK_INTERVAL=30

# Model management
CODESAGE_MODEL_CACHE_TTL_MINUTES=60
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
```

#### Caching Configuration
```bash
# Cache sizes
CODESAGE_EMBEDDING_CACHE_SIZE=5000
CODESAGE_SEARCH_CACHE_SIZE=1000
CODESAGE_FILE_CACHE_SIZE=100

# Cache behavior
CODESAGE_SIMILARITY_THRESHOLD=0.85
CODESAGE_MAX_FILE_SIZE_MB=1
CODESAGE_CACHE_PERSISTENCE_ENABLED=true
CODESAGE_CACHE_WARMING_ENABLED=true

# Adaptive caching
CODESAGE_ADAPTIVE_CACHE_ENABLED=true
CODESAGE_MEMORY_THRESHOLD_HIGH=0.8
CODESAGE_MEMORY_THRESHOLD_LOW=0.3
```

#### Processing Configuration
```bash
# Parallel processing
CODESAGE_PARALLEL_WORKERS=4
CODESAGE_BATCH_SIZE=50
CODESAGE_MAX_FILE_SIZE_MB=1

# Index optimization
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_TYPE=auto
CODESAGE_FAISS_NLIST_AUTO=true
```

### 2. LLM Configuration

#### API Keys
```bash
# Primary LLM provider (choose one)
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-..."
GOOGLE_API_KEY="AIza..."

# Secondary providers (optional)
ANTHROPIC_API_KEY="sk-ant-..."
COHERE_API_KEY="..."
```

#### Model Selection
```bash
# Default models
CODESAGE_DEFAULT_LLM_MODEL="llama3-8b-8192"
CODESAGE_EMBEDDING_MODEL="all-MiniLM-L6-v2"
CODESAGE_SUMMARIZATION_MODEL="llama3-8b-8192"

# Model parameters
CODESAGE_MAX_TOKENS=4096
CODESAGE_TEMPERATURE=0.1
CODESAGE_TOP_P=0.9
```

#### LLM Behavior
```bash
# Request settings
CODESAGE_LLM_TIMEOUT=30
CODESAGE_LLM_MAX_RETRIES=3
CODESAGE_LLM_RETRY_DELAY=1

# Response processing
CODESAGE_LLM_RESPONSE_FORMAT="json"
CODESAGE_LLM_STRIP_WHITESPACE=true
```

### 3. Indexing Configuration

#### Index Management
```bash
# Index location
CODESAGE_INDEX_DIR=".codesage"
CODESAGE_INDEX_FILE="codebase_index.json"
CODESAGE_FAISS_INDEX_FILE="codebase_index.faiss"

# Index optimization
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="auto"
CODESAGE_TARGET_MEMORY_MB=500

# Incremental indexing
CODESAGE_INCREMENTAL_INDEXING_ENABLED=true
CODESAGE_DEPENDENCY_TRACKING_ENABLED=true
CODESAGE_DEPENDENCY_INVALIDATION_DEPTH=3
```

#### File Processing
```bash
# File filtering
CODESAGE_RESPECT_GITIGNORE=true
CODESAGE_INCLUDE_PATTERNS="*.py,*.js,*.ts,*.java,*.cpp,*.c,*.h,*.md,*.txt,*.yml,*.yaml,*.json"
CODESAGE_EXCLUDE_PATTERNS="*.pyc,*.pyo,*.class,*.jar,*.zip,*.tar,*.gz,*.bin,*.exe,*.dll,*.so"

# Chunking
CODESAGE_CHUNK_SIZE=1000
CODESAGE_CHUNK_OVERLAP=200
CODESAGE_MAX_CHUNKS_PER_FILE=50
```

### 4. Search Configuration

#### Search Behavior
```bash
# Search parameters
CODESAGE_DEFAULT_TOP_K=5
CODESAGE_MAX_TOP_K=20
CODESAGE_SEMANTIC_SEARCH_ENABLED=true
CODESAGE_TEXT_SEARCH_ENABLED=true

# Similarity thresholds
CODESAGE_SEMANTIC_SIMILARITY_THRESHOLD=0.7
CODESAGE_TEXT_SIMILARITY_THRESHOLD=0.8
CODESAGE_HYBRID_SEARCH_WEIGHT=0.6
```

#### Search Optimization
```bash
# Search caching
CODESAGE_SEARCH_CACHE_ENABLED=true
CODESAGE_SEARCH_CACHE_SIZE=1000
CODESAGE_SEARCH_CACHE_SIMILARITY_THRESHOLD=0.85

# Performance tuning
CODESAGE_SEARCH_BATCH_SIZE=10
CODESAGE_SEARCH_TIMEOUT=10
```

### 5. System Configuration

#### Logging
```bash
# Logging configuration
CODESAGE_LOG_LEVEL=INFO
CODESAGE_LOG_FORMAT="json"
CODESAGE_LOG_FILE="codesage.log"
CODESAGE_LOG_MAX_SIZE=100MB
CODESAGE_LOG_BACKUP_COUNT=5

# Performance logging
CODESAGE_PERFORMANCE_LOGGING_ENABLED=true
CODESAGE_SLOW_QUERY_THRESHOLD=1.0
```

#### Monitoring
```bash
# Health monitoring
CODESAGE_HEALTH_CHECK_ENABLED=true
CODESAGE_HEALTH_CHECK_INTERVAL=30
CODESAGE_METRICS_PORT=9090

# Performance monitoring
CODESAGE_PERFORMANCE_MONITORING_ENABLED=true
CODESAGE_METRICS_COLLECTION_INTERVAL=60
```

#### Security
```bash
# API security
CODESAGE_API_KEY_REQUIRED=false
CODESAGE_API_KEY="your-api-key"
CODESAGE_ALLOWED_ORIGINS="*"

# Data protection
CODESAGE_ENCRYPT_SENSITIVE_DATA=true
CODESAGE_MASK_API_KEYS_IN_LOGS=true
```

## Environment-Specific Configurations

### Development Environment

#### Basic Development Setup
```bash
# Development configuration
CODESAGE_MEMORY_LIMIT=256MB
CODESAGE_EMBEDDING_CACHE_SIZE=2000
CODESAGE_SEARCH_CACHE_SIZE=500
CODESAGE_FILE_CACHE_SIZE=50

# Development features
CODESAGE_LOG_LEVEL=DEBUG
CODESAGE_CACHE_PERSISTENCE_ENABLED=false
CODESAGE_ADAPTIVE_CACHE_ENABLED=false
CODESAGE_CACHE_WARMING_ENABLED=false

# Development performance
CODESAGE_PARALLEL_WORKERS=2
CODESAGE_BATCH_SIZE=25
```

#### Full Development Configuration
```bash
# Complete development setup
# Memory and performance
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_MAX_MEMORY_MB=1024
CODESAGE_ENABLE_MEMORY_MONITORING=true

# Caching (reduced for development)
CODESAGE_EMBEDDING_CACHE_SIZE=2000
CODESAGE_SEARCH_CACHE_SIZE=500
CODESAGE_FILE_CACHE_SIZE=50
CODESAGE_SIMILARITY_THRESHOLD=0.8

# Processing
CODESAGE_PARALLEL_WORKERS=2
CODESAGE_BATCH_SIZE=25
CODESAGE_MAX_FILE_SIZE_MB=0.5

# LLM (use smaller models)
CODESAGE_DEFAULT_LLM_MODEL="llama3-8b-8192"
CODESAGE_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Logging and debugging
CODESAGE_LOG_LEVEL=DEBUG
CODESAGE_PERFORMANCE_LOGGING_ENABLED=true
CODESAGE_SLOW_QUERY_THRESHOLD=0.5

# Development features
CODESAGE_CACHE_PERSISTENCE_ENABLED=false
CODESAGE_ADAPTIVE_CACHE_ENABLED=false
CODESAGE_CACHE_WARMING_ENABLED=false
```

### Production Environment

#### Standard Production Setup
```bash
# Production configuration
CODESAGE_MEMORY_LIMIT=1GB
CODESAGE_EMBEDDING_CACHE_SIZE=10000
CODESAGE_SEARCH_CACHE_SIZE=2000
CODESAGE_FILE_CACHE_SIZE=200

# Production optimizations
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_CACHE_PERSISTENCE_ENABLED=true
CODESAGE_ADAPTIVE_CACHE_ENABLED=true

# Production performance
CODESAGE_PARALLEL_WORKERS=8
CODESAGE_BATCH_SIZE=100
```

#### High-Performance Production Configuration
```bash
# High-performance production setup
# Memory and performance
CODESAGE_MEMORY_LIMIT=2GB
CODESAGE_MAX_MEMORY_MB=4096
CODESAGE_ENABLE_MEMORY_MONITORING=true
CODESAGE_MEMORY_CHECK_INTERVAL=15

# Advanced caching
CODESAGE_EMBEDDING_CACHE_SIZE=20000
CODESAGE_SEARCH_CACHE_SIZE=5000
CODESAGE_FILE_CACHE_SIZE=500
CODESAGE_SIMILARITY_THRESHOLD=0.9
CODESAGE_MAX_FILE_SIZE_MB=2

# Processing optimization
CODESAGE_PARALLEL_WORKERS=12
CODESAGE_BATCH_SIZE=200
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="ivf_pq"

# LLM optimization
CODESAGE_DEFAULT_LLM_MODEL="llama3-70b-8192"
CODESAGE_EMBEDDING_MODEL="text-embedding-ada-002"
CODESAGE_MAX_TOKENS=8192

# Advanced features
CODESAGE_ADAPTIVE_CACHE_ENABLED=true
CODESAGE_CACHE_WARMING_ENABLED=true
CODESAGE_PREFETCH_ENABLED=true
CODESAGE_DEPENDENCY_TRACKING_ENABLED=true

# Monitoring and logging
CODESAGE_LOG_LEVEL=INFO
CODESAGE_PERFORMANCE_MONITORING_ENABLED=true
CODESAGE_METRICS_PORT=9090
CODESAGE_HEALTH_CHECK_ENABLED=true
```

#### Enterprise Production Configuration
```bash
# Enterprise production setup
# Memory and resources
CODESAGE_MEMORY_LIMIT=4GB
CODESAGE_MAX_MEMORY_MB=8192
CODESAGE_ENABLE_MEMORY_MONITORING=true
CODESAGE_MEMORY_CHECK_INTERVAL=10

# Large-scale caching
CODESAGE_EMBEDDING_CACHE_SIZE=50000
CODESAGE_SEARCH_CACHE_SIZE=10000
CODESAGE_FILE_CACHE_SIZE=1000
CODESAGE_SIMILARITY_THRESHOLD=0.95

# Enterprise processing
CODESAGE_PARALLEL_WORKERS=16
CODESAGE_BATCH_SIZE=500
CODESAGE_MAX_FILE_SIZE_MB=5

# Enterprise features
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="ivf_pq"
CODESAGE_TARGET_MEMORY_MB=2000
CODESAGE_DEPENDENCY_TRACKING_ENABLED=true
CODESAGE_DEPENDENCY_INVALIDATION_DEPTH=5

# Advanced optimization
CODESAGE_ADAPTIVE_CACHE_ENABLED=true
CODESAGE_CACHE_WARMING_ENABLED=true
CODESAGE_PREFETCH_ENABLED=true
CODESAGE_MAX_PREFETCH_FILES=20

# Enterprise monitoring
CODESAGE_LOG_LEVEL=INFO
CODESAGE_PERFORMANCE_MONITORING_ENABLED=true
CODESAGE_METRICS_PORT=9090
CODESAGE_HEALTH_CHECK_ENABLED=true
CODESAGE_METRICS_COLLECTION_INTERVAL=30
```

### Memory-Constrained Environment

#### Low-Memory Configuration
```bash
# Memory-constrained configuration
CODESAGE_MEMORY_LIMIT=128MB
CODESAGE_EMBEDDING_CACHE_SIZE=1000
CODESAGE_SEARCH_CACHE_SIZE=500
CODESAGE_FILE_CACHE_SIZE=50

# Memory optimizations
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_MAX_FILE_SIZE_MB=0.5

# Reduced processing
CODESAGE_PARALLEL_WORKERS=1
CODESAGE_BATCH_SIZE=10
```

#### Ultra-Low-Memory Configuration
```bash
# Ultra memory-constrained setup
CODESAGE_MEMORY_LIMIT=64MB
CODESAGE_EMBEDDING_CACHE_SIZE=500
CODESAGE_SEARCH_CACHE_SIZE=200
CODESAGE_FILE_CACHE_SIZE=25

# Aggressive memory optimization
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="scalar_quant"
CODESAGE_MAX_FILE_SIZE_MB=0.25

# Minimal processing
CODESAGE_PARALLEL_WORKERS=1
CODESAGE_BATCH_SIZE=5
CODESAGE_CACHE_PERSISTENCE_ENABLED=false
CODESAGE_ADAPTIVE_CACHE_ENABLED=false
```

### Large Codebase Configuration

#### Big Codebase Setup
```bash
# Large codebase configuration
CODESAGE_MEMORY_LIMIT=2GB
CODESAGE_EMBEDDING_CACHE_SIZE=25000
CODESAGE_SEARCH_CACHE_SIZE=5000
CODESAGE_FILE_CACHE_SIZE=500

# Large codebase optimizations
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="ivf_pq"
CODESAGE_TARGET_MEMORY_MB=1000

# Processing for large codebases
CODESAGE_PARALLEL_WORKERS=8
CODESAGE_BATCH_SIZE=200
CODESAGE_MAX_FILE_SIZE_MB=2
```

#### Massive Codebase Configuration
```bash
# Massive codebase setup (>100k files)
CODESAGE_MEMORY_LIMIT=8GB
CODESAGE_EMBEDDING_CACHE_SIZE=100000
CODESAGE_SEARCH_CACHE_SIZE=20000
CODESAGE_FILE_CACHE_SIZE=2000

# Massive scale optimizations
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_INDEX_COMPRESSION_TYPE="ivf_pq"
CODESAGE_TARGET_MEMORY_MB=4000

# Distributed processing
CODESAGE_PARALLEL_WORKERS=24
CODESAGE_BATCH_SIZE=1000
CODESAGE_MAX_FILE_SIZE_MB=10

# Advanced features
CODESAGE_DEPENDENCY_TRACKING_ENABLED=true
CODESAGE_DEPENDENCY_INVALIDATION_DEPTH=10
CODESAGE_ADAPTIVE_CACHE_ENABLED=true
CODESAGE_CACHE_WARMING_ENABLED=true
CODESAGE_PREFETCH_ENABLED=true
```

## Configuration Templates

### Quick Start Templates

#### Minimal Configuration
```bash
# Minimal working configuration
GROQ_API_KEY="your-groq-api-key"
CODESAGE_MEMORY_LIMIT=256MB
CODESAGE_EMBEDDING_CACHE_SIZE=1000
```

#### Standard Configuration
```bash
# Standard configuration for most use cases
GROQ_API_KEY="your-groq-api-key"
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_EMBEDDING_CACHE_SIZE=5000
CODESAGE_SEARCH_CACHE_SIZE=1000
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_INDEX_COMPRESSION=true
```

#### Performance Configuration
```bash
# High-performance configuration
GROQ_API_KEY="your-groq-api-key"
CODESAGE_MEMORY_LIMIT=1GB
CODESAGE_EMBEDDING_CACHE_SIZE=10000
CODESAGE_SEARCH_CACHE_SIZE=2000
CODESAGE_PARALLEL_WORKERS=8
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_ENABLE_MEMORY_MAPPED_INDEXES=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_ADAPTIVE_CACHE_ENABLED=true
CODESAGE_CACHE_WARMING_ENABLED=true
```

### Advanced Templates

#### Research and Development
```bash
# R&D configuration with extensive logging
GROQ_API_KEY="your-groq-api-key"
CODESAGE_LOG_LEVEL=DEBUG
CODESAGE_PERFORMANCE_LOGGING_ENABLED=true
CODESAGE_SLOW_QUERY_THRESHOLD=0.1
CODESAGE_METRICS_COLLECTION_INTERVAL=10
CODESAGE_CACHE_PERSISTENCE_ENABLED=false
```

#### CI/CD Pipeline
```bash
# CI/CD optimized configuration
GROQ_API_KEY="your-groq-api-key"
CODESAGE_MEMORY_LIMIT=1GB
CODESAGE_EMBEDDING_CACHE_SIZE=5000
CODESAGE_CACHE_PERSISTENCE_ENABLED=false
CODESAGE_CACHE_WARMING_ENABLED=false
CODESAGE_LOG_LEVEL=INFO
```

#### Multi-Tenant SaaS
```bash
# Multi-tenant configuration
CODESAGE_MEMORY_LIMIT=2GB
CODESAGE_EMBEDDING_CACHE_SIZE=10000
CODESAGE_SEARCH_CACHE_SIZE=2000
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_API_KEY_REQUIRED=true
CODESAGE_METRICS_PORT=9090
CODESAGE_LOG_LEVEL=INFO
```

## Configuration Validation

### Configuration Checking

#### Validate Configuration
```bash
# Validate configuration on startup
python -c "from codesage_mcp.config import validate_config; validate_config()"
```

#### Configuration Health Check
```bash
# Check configuration health
python -c "from codesage_mcp.config import check_config_health; check_config_health()"
```

### Configuration Migration

#### Migrate from Old Versions
```bash
# Migrate configuration
python -c "from codesage_mcp.config import migrate_config; migrate_config()"
```

#### Backup Configuration
```bash
# Backup current configuration
cp .env .env.backup
```

## Best Practices

### 1. Configuration Management

#### Version Control
- Keep configuration files in version control
- Use different configurations for different environments
- Document configuration changes

#### Environment Separation
- Use different configurations for dev/staging/production
- Never share production API keys in development configs
- Use environment-specific secrets management

#### Regular Review
- Review configuration quarterly
- Update based on performance metrics
- Adjust for new features and optimizations

### 2. Performance Tuning

#### Start Conservative
- Begin with default or minimal configurations
- Monitor performance for 1-2 weeks
- Gradually increase resource allocation

#### Monitor and Adjust
- Set up monitoring dashboards
- Track key performance metrics
- Adjust configuration based on usage patterns

#### Capacity Planning
- Plan for 2x current usage for future growth
- Consider peak usage patterns
- Account for memory overhead

### 3. Security Considerations

#### API Key Management
- Use environment variables for API keys
- Rotate keys regularly
- Use different keys for different environments

#### Access Control
- Configure allowed origins for production
- Use API key authentication when required
- Implement rate limiting if needed

#### Data Protection
- Enable sensitive data encryption
- Mask API keys in logs
- Use secure configuration storage

## Troubleshooting Configuration

### Common Configuration Issues

#### Configuration Not Loading
**Symptoms**: Settings not taking effect
**Solutions**:
1. Check environment variable names (case-sensitive)
2. Verify variable values are valid
3. Restart the service after configuration changes
4. Check configuration file permissions

#### Performance Issues
**Symptoms**: Poor performance despite configuration
**Solutions**:
1. Verify memory limits are appropriate
2. Check cache sizes are adequate
3. Ensure parallel workers are configured correctly
4. Review index compression settings

#### Memory Issues
**Symptoms**: Out of memory errors
**Solutions**:
1. Reduce cache sizes
2. Enable quantization and compression
3. Use memory-mapped indexes
4. Decrease parallel workers

### Configuration Debugging

#### Debug Configuration Loading
```bash
# Debug configuration
CODESAGE_DEBUG_CONFIG=true python -c "import codesage_mcp.config"
```

#### Validate Configuration Values
```bash
# Validate all settings
python -c "from codesage_mcp.config import print_config; print_config()"
```

#### Test Configuration Performance
```bash
# Test configuration impact
python -c "from codesage_mcp.config import benchmark_config; benchmark_config()"
```

## Conclusion

CodeSage MCP Server's configuration system provides **flexible, comprehensive control** over all aspects of the system. The environment-specific templates and best practices ensure optimal performance across different deployment scenarios.

**Key Configuration Principles:**
- **Start Simple**: Use minimal configurations initially
- **Monitor Performance**: Track metrics and adjust accordingly
- **Environment-Specific**: Tailor configurations to your environment
- **Security-First**: Protect sensitive data and control access
- **Regular Maintenance**: Review and update configurations regularly

The configuration system supports everything from memory-constrained environments to large-scale enterprise deployments, ensuring optimal performance and reliability.