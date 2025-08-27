# CodeSage MCP Server - Configuration Templates

This directory contains **production-ready configuration templates** for different deployment environments. Each template is optimized for specific use cases and system constraints.

## Available Templates

### 🚀 Quick Start Templates

#### `minimal.env`
**Use Case**: Basic setup for testing and development
**Memory**: ~256MB
**Features**: Essential configuration only
```bash
cp config/templates/minimal.env .env
# Edit API keys and basic settings
```

#### `standard.env`
**Use Case**: General production use
**Memory**: ~512MB-1GB
**Features**: Balanced performance and reliability
```bash
cp config/templates/standard.env .env
# Configure API keys and adjust resource limits
```

### 🛠️ Environment-Specific Templates

#### `development.env`
**Use Case**: Local development and debugging
**Memory**: ~256MB
**Features**:
- Debug logging enabled
- Reduced resource limits
- Cache persistence disabled
- Performance monitoring enabled
```bash
cp config/templates/development.env .env
```

#### `production.env`
**Use Case**: Production deployment
**Memory**: ~1GB
**Features**:
- Security hardening
- Comprehensive monitoring
- Performance optimization
- Enterprise features
```bash
cp config/templates/production.env .env
```

#### `high-performance.env`
**Use Case**: Maximum performance requirements
**Memory**: ~4GB+
**Features**:
- Enterprise-grade optimization
- Advanced caching strategies
- Comprehensive monitoring
- Maximum resource utilization
```bash
cp config/templates/high-performance.env .env
```

### 📊 Specialized Templates

#### `memory-constrained.env`
**Use Case**: Limited memory systems
**Memory**: ~128MB
**Features**:
- Aggressive memory optimization
- Maximum compression
- Minimal resource usage
- Efficiency-focused configuration
```bash
cp config/templates/memory-constrained.env .env
```

#### `large-codebase.env`
**Use Case**: Large codebases (>10k files)
**Memory**: ~2GB
**Features**:
- Incremental indexing
- Advanced memory management
- Large-scale optimization
- Scalability features
```bash
cp config/templates/large-codebase.env .env
```

## How to Use Templates

### 1. Select Appropriate Template

Choose the template that best matches your environment:

```bash
# For development
cp config/templates/development.env .env

# For production
cp config/templates/production.env .env

# For limited memory
cp config/templates/memory-constrained.env .env
```

### 2. Configure API Keys

Edit the `.env` file to add your API keys:

```bash
# Required: Choose your LLM provider
GROQ_API_KEY="your-groq-api-key"
# OPENROUTER_API_KEY="your-openrouter-api-key"
# GOOGLE_API_KEY="your-google-api-key"
```

### 3. Adjust Resource Limits

Modify resource limits based on your system:

```bash
# Memory limits (adjust based on available RAM)
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_MAX_MEMORY_MB=1024

# CPU resources (adjust based on available cores)
CODESAGE_PARALLEL_WORKERS=4
```

### 4. Environment-Specific Customization

#### Development Environment
```bash
# Enable debugging features
CODESAGE_LOG_LEVEL=DEBUG
CODESAGE_PERFORMANCE_LOGGING_ENABLED=true

# Reduce resource usage
CODESAGE_EMBEDDING_CACHE_SIZE=2000
CODESAGE_MEMORY_LIMIT=256MB
```

#### Production Environment
```bash
# Enable security features
CODESAGE_ENCRYPT_SENSITIVE_DATA=true
CODESAGE_MASK_API_KEYS_IN_LOGS=true

# Optimize for performance
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_INDEX_COMPRESSION=true
```

#### Memory-Constrained Environment
```bash
# Aggressive optimization
CODESAGE_ENABLE_MODEL_QUANTIZATION=true
CODESAGE_INDEX_COMPRESSION=true
CODESAGE_CACHE_COMPRESSION_ENABLED=true

# Minimal resource usage
CODESAGE_MEMORY_LIMIT=128MB
CODESAGE_EMBEDDING_CACHE_SIZE=1000
```

## Template Features Overview

| Template | Memory | Performance | Security | Monitoring | Use Case |
|----------|--------|-------------|----------|------------|----------|
| `development.env` | 256MB | Good | Basic | Debug | Local development |
| `production.env` | 1GB | Excellent | Enterprise | Comprehensive | Production deployment |
| `high-performance.env` | 4GB+ | Maximum | Enterprise | Advanced | High-performance needs |
| `memory-constrained.env` | 128MB | Good | Basic | Essential | Limited resources |
| `large-codebase.env` | 2GB | Excellent | Enterprise | Comprehensive | Large projects |

## Configuration Validation

### Validate Configuration
```bash
# Check configuration syntax
python -c "from codesage_mcp.config import validate_config; validate_config()"

# Test configuration loading
python -c "from codesage_mcp.config import print_config; print_config()"
```

### Performance Testing
```bash
# Test configuration performance
python -c "from codesage_mcp.config import benchmark_config; benchmark_config()"
```

## Best Practices

### 1. Environment Management

#### Version Control
- Keep configuration templates in version control
- Use different configurations for different environments
- Document configuration changes

#### Security
- Never commit API keys to version control
- Use environment-specific secrets management
- Rotate API keys regularly

#### Documentation
- Document your configuration choices
- Maintain configuration change log
- Update documentation with configuration changes

### 2. Performance Tuning

#### Start Conservative
- Begin with default template settings
- Monitor performance for 1-2 weeks
- Adjust based on observed usage patterns

#### Monitor and Adjust
- Set up monitoring dashboards
- Track key performance metrics
- Adjust configuration based on workload

#### Capacity Planning
- Plan for 2x current usage for future growth
- Consider peak usage patterns
- Account for memory overhead

### 3. Maintenance

#### Regular Review
- Review configuration quarterly
- Update based on performance metrics
- Adjust for new features and optimizations

#### Backup Configurations
```bash
# Backup current configuration
cp .env .env.backup.$(date +%Y%m%d)

# Restore configuration
cp .env.backup.20231201 .env
```

## Troubleshooting

### Common Issues

#### Configuration Not Loading
**Symptoms**: Settings not taking effect
**Solutions**:
1. Check environment variable syntax
2. Verify file permissions
3. Restart the application
4. Check configuration file location

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
# Enable debug logging
CODESAGE_DEBUG_CONFIG=true python -c "import codesage_mcp.config"
```

#### Validate Configuration Values
```bash
# Print all configuration values
python -c "from codesage_mcp.config import print_config; print_config()"
```

## Support and Resources

### Documentation
- [Configuration Guide](../docs/configuration.md) - Detailed configuration options
- [Performance Optimization](../docs/performance_optimization.md) - Performance tuning
- [Production Deployment](../docs/docker_deployment.md) - Deployment strategies

### Community Support
- [GitHub Issues](https://github.com/your-repo/codesage-mcp/issues) - Report configuration issues
- [Discussions](https://github.com/your-repo/codesage-mcp/discussions) - Configuration help

### Enterprise Support
- Contact: enterprise-support@yourcompany.com
- SLA: Enterprise support available
- Custom Configuration: Enterprise-specific templates available

---

**Configuration Template Version**: 1.0.0
**Last Updated**: 2024-12-27
**Supported Environments**: Development, Production, Enterprise