# CodeSage MCP Server - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the CodeSage MCP Server in production environments with all optimizations enabled.

## Prerequisites

- Docker and Docker Compose installed
- At least one LLM API key (Groq, OpenRouter, or Google AI)
- Minimum 4GB RAM, 8GB recommended
- Linux/Windows/macOS with bash support

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd codesage-mcp
```

### 2. Configure Environment

```bash
# Copy production environment template
cp config/templates/production.env .env

# Edit with your API keys
nano .env
```

### 3. Run Production Readiness Check

```bash
./scripts/production_readiness_check.sh
```

### 4. Deploy

```bash
./scripts/deploy.sh
```

## Production Configuration

### Environment Variables

The production environment (`config/templates/production.env`) includes:

#### LLM API Configuration
```env
GROQ_API_KEY="your-production-groq-api-key"
OPENROUTER_API_KEY="your-production-openrouter-api-key"
GOOGLE_API_KEY="your-production-google-api-key"
```

#### Performance & Memory Configuration
```env
MAX_MEMORY_MB=2048
ENABLE_MEMORY_MONITORING=true
ENABLE_MODEL_QUANTIZATION=false
MODEL_CACHE_TTL_MINUTES=120
ENABLE_MEMORY_MAPPED_INDEXES=true
```

#### Self-Optimization Features (All Enabled)
```env
ENABLE_AUTO_PERFORMANCE_TUNING=true
ENABLE_ADAPTIVE_CACHE_MANAGEMENT=true
ENABLE_WORKLOAD_PATTERN_RECOGNITION=true
```

#### Connection Pool & Timeout Configuration
```env
HTTP_CONNECTION_POOL_SIZE=20
HTTP_MAX_KEEPALIVE_CONNECTIONS=10
FAST_OPERATION_TIMEOUT=0.5
MEDIUM_OPERATION_TIMEOUT=2.0
SLOW_OPERATION_TIMEOUT=10.0
VERY_SLOW_OPERATION_TIMEOUT=60.0
ADAPTIVE_TIMEOUT_ENABLED=true
ENABLE_CONNECTION_POOL_MONITORING=true
```

#### Circuit Breaker Configuration
```env
CIRCUIT_BREAKER_FAILURE_THRESHOLD=10
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD=3
```

## Docker Deployment

### Basic Production Deployment

```bash
# Build and deploy
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f codesage-mcp
```

### With Monitoring Stack

```bash
# Deploy with full monitoring
docker-compose --profile monitoring up -d

# Access monitoring
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# Alertmanager: http://localhost:9093
```

### With Reverse Proxy

```bash
# Deploy with Nginx reverse proxy
docker-compose --profile with-nginx up -d
```

### With Redis Caching

```bash
# Deploy with Redis for enhanced caching
docker-compose --profile with-redis up -d
```

## Production Optimizations

### Enabled by Default

1. **Intelligent Caching System**
   - Multi-level caching (embedding, search, file)
   - Cache persistence across restarts
   - Cache warming on startup
   - Adaptive cache management

2. **Memory Management**
   - Memory-mapped indexes
   - Memory monitoring and alerts
   - Adaptive memory allocation
   - Workload pattern recognition

3. **Connection Pool Optimization**
   - Configurable connection pool size
   - Keep-alive connections
   - Connection pool monitoring
   - Progressive timeout configuration

4. **Self-Optimization Features**
   - Auto performance tuning
   - Adaptive cache management
   - Workload pattern recognition
   - Intelligent prefetching

5. **Resilience Features**
   - Circuit breaker pattern
   - Adaptive timeouts
   - Retry logic with backoff
   - Rate limiting protection

## Monitoring and Observability

### Metrics Endpoint

The server exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

### Health Checks

```bash
# Run comprehensive health check
./scripts/health_check.sh

# Check specific components
./scripts/health_check.sh --docker
./scripts/health_check.sh --http
./scripts/health_check.sh --mcp
```

### Logging

Logs are available through:

```bash
# Docker logs
docker-compose logs -f codesage-mcp

# Application logs (inside container)
docker-compose exec codesage-mcp tail -f /app/logs/*.log
```

## Scaling and Performance

### Resource Configuration

```yaml
# docker-compose.yml resource limits
deploy:
  resources:
    limits:
      memory: 2g
      cpus: '1.0'
    reservations:
      memory: 1g
      cpus: '0.5'
```

### Performance Tuning

The server includes automatic performance tuning:

```bash
# Trigger manual performance tuning
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "trigger_performance_tuning"}, "id": 1}'
```

### Cache Optimization

```bash
# Get cache statistics
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "get_cache_statistics"}, "id": 1}'

# Trigger cache adaptation
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "trigger_cache_adaptation"}, "id": 1}'
```

## Security Considerations

### API Key Security

- Store API keys securely using Docker secrets or KMS
- Never commit API keys to version control
- Use environment-specific key rotation
- Implement API key masking in logs

### Network Security

```yaml
# Security options in docker-compose.yml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp:noexec,nosuid,size=100m
```

### SSL/TLS Configuration

For production SSL/TLS:

```bash
# Using Nginx reverse proxy with SSL
docker-compose --profile with-nginx up -d

# Configure SSL certificates in nginx/nginx.conf
```

## Backup and Recovery

### Cache Persistence

The system automatically persists cache data:

```yaml
volumes:
  codesage_cache:
    driver: local
```

### Backup Strategy

```bash
# Backup cache and configuration
docker run --rm -v codesage_cache:/data -v $(pwd):/backup alpine tar czf /backup/cache-backup.tar.gz -C /data .
docker run --rm -v $(pwd):/backup alpine cp /backup/.env /backup/.env.backup
```

### Recovery

```bash
# Restore from backup
docker run --rm -v codesage_cache:/data -v $(pwd):/backup alpine tar xzf /backup/cache-backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

1. **Logger Error Fixed**
   - The "name 'logger' is not defined" error has been resolved
   - Logging is now properly configured in `codesage_mcp/llm_analysis.py`

2. **Memory Issues**
   ```bash
   # Check memory usage
   ./scripts/health_check.sh --memory

   # Optimize memory settings
   docker-compose exec codesage-mcp python -c "from codesage_mcp.features.memory_management.memory_manager import get_memory_manager; print(get_memory_manager().get_memory_optimization_recommendations())"
   ```

3. **Rate Limiting**
   ```bash
   # Check rate limiting status
   curl http://localhost:8000/metrics | grep rate_limited

   # Monitor connection pool
   curl http://localhost:8000/metrics | grep connection_pool
   ```

### Logs and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose up -d

# Check application logs
docker-compose logs codesage-mcp

# Check monitoring stack
docker-compose logs prometheus grafana
```

## Maintenance

### Updates

```bash
# Update with zero downtime
docker-compose pull
docker-compose up -d --no-deps codesage-mcp

# Clean up old images
docker image prune -f
```

### Monitoring Maintenance

```bash
# Update monitoring stack
docker-compose pull prometheus grafana alertmanager
docker-compose up -d prometheus grafana alertmanager

# Clean up monitoring data (if needed)
docker-compose exec prometheus rm -rf /prometheus/*
docker-compose restart prometheus
```

## Performance Benchmarks

Run the included benchmark suite:

```bash
python tests/benchmark_performance.py
```

View results:
```bash
cat benchmark_results/benchmark_report_*.json
```

## Support

For issues and questions:

1. Check the troubleshooting guide: `docs/troubleshooting_guide.md`
2. Run production readiness check: `./scripts/production_readiness_check.sh`
3. Review operational runbook: `docs/operational_runbook.md`
4. Check monitoring dashboards

## Summary

This deployment guide covers:

✅ **Logger error resolution** - Fixed "name 'logger' is not defined" error
✅ **Production Docker setup** - Multi-stage builds with security hardening
✅ **Environment configuration** - All optimizations enabled by default
✅ **Monitoring stack** - Prometheus, Grafana, Alertmanager integration
✅ **Deployment scripts** - Automated deployment and management
✅ **Production optimizations** - Self-tuning, adaptive caching, connection pooling

The CodeSage MCP Server is now ready for production deployment with all optimizations active and comprehensive monitoring in place.