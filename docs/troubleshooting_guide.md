# CodeSage MCP Server - Troubleshooting Guide

## Quick Reference for Common Issues

This guide provides quick solutions for the most common issues encountered when operating the CodeSage MCP Server in production environments.

## 🔍 Issue Diagnosis

### 1. Service Not Starting

#### Symptoms
- Container exits immediately after start
- `docker-compose ps` shows "Exit" status
- Logs show startup failures

#### Quick Diagnosis
```bash
# Check container status
docker-compose ps

# View startup logs
docker-compose logs codesage-mcp

# Check environment variables
docker-compose exec codesage-mcp env | grep -E "(API_KEY|MEMORY|CACHE)"
```

#### Common Causes & Solutions

**Missing API Keys:**
```bash
# Check if API keys are set
grep "API_KEY" .env

# Solution: Add missing keys to .env
echo "GROQ_API_KEY=your_key_here" >> .env
docker-compose restart codesage-mcp
```

**Port Already in Use:**
```bash
# Check what's using port 8000
lsof -i :8000

# Solution: Change port in docker-compose.yml or free the port
```

**Insufficient Memory:**
```bash
# Check system memory
free -h

# Solution: Reduce memory limits in docker-compose.yml
```

### 2. High Memory Usage

#### Symptoms
- Container memory usage >80%
- Slow performance or crashes
- OOM (Out of Memory) errors

#### Quick Diagnosis
```bash
# Check memory usage
docker stats codesage-mcp

# Check application memory
docker-compose exec codesage-mcp ps aux --sort=-%mem
```

#### Solutions

**Reduce Cache Sizes:**
```bash
# Edit .env file
sed -i 's/EMBEDDING_CACHE_SIZE=.*/EMBEDDING_CACHE_SIZE=5000/' .env
sed -i 's/SEARCH_CACHE_SIZE=.*/SEARCH_CACHE_SIZE=1000/' .env
sed -i 's/FILE_CACHE_SIZE=.*/FILE_CACHE_SIZE=100/' .env

# Restart service
docker-compose restart codesage-mcp
```

**Increase Container Memory Limit:**
```bash
# Edit docker-compose.yml
sed -i 's/memory: 2g/memory: 4g/' docker-compose.yml
docker-compose up -d codesage-mcp
```

**Clear Cache:**
```bash
# Clear cache files
docker-compose exec codesage-mcp rm -rf /app/.codesage/cache/*
docker-compose restart codesage-mcp
```

### 3. Slow Performance

#### Symptoms
- Search responses >2 seconds
- High latency on API calls
- Cache miss rate >30%

#### Quick Diagnosis
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/

# Check cache statistics
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "get_cache_statistics"}}' \
  http://localhost:8000/mcp
```

#### Solutions

**Optimize Cache Settings:**
```bash
# Increase cache sizes for better hit rates
sed -i 's/EMBEDDING_CACHE_SIZE=.*/EMBEDDING_CACHE_SIZE=10000/' .env
sed -i 's/SEARCH_CACHE_SIZE=.*/SEARCH_CACHE_SIZE=2000/' .env

docker-compose restart codesage-mcp
```

**Enable Memory Mapped Indexes:**
```bash
# Add to .env
echo "ENABLE_MEMORY_MAPPED_INDEXES=true" >> .env
docker-compose restart codesage-mcp
```

**Check System Resources:**
```bash
# Monitor CPU and I/O
docker stats codesage-mcp
iostat -x 1 5
```

### 4. Service Unresponsive

#### Symptoms
- HTTP requests timeout
- Health checks fail
- Container shows "unhealthy"

#### Quick Diagnosis
```bash
# Check container health
docker-compose ps

# Test basic connectivity
curl -f http://localhost:8000/

# Check container logs
docker-compose logs --tail=20 codesage-mcp
```

#### Solutions

**Restart Service:**
```bash
# Graceful restart
docker-compose restart codesage-mcp

# Force restart if needed
docker-compose up -d --force-recreate codesage-mcp
```

**Check Resource Limits:**
```bash
# Verify resource usage isn't hitting limits
docker stats codesage-mcp

# Check system resources
df -h  # Disk space
free -h  # Memory
```

**Network Issues:**
```bash
# Check network connectivity
docker-compose exec codesage-mcp ping -c 3 google.com

# Restart network if needed
docker-compose down
docker-compose up -d
```

### 5. API Errors

#### Symptoms
- JSON-RPC calls return errors
- LLM API failures
- Tool execution failures

#### Quick Diagnosis
```bash
# Test MCP functionality
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' \
  http://localhost:8000/mcp

# Check API key validity
docker-compose exec codesage-mcp python -c "
import os
from groq import Groq
try:
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    print('Groq API: OK')
except Exception as e:
    print(f'Groq API: ERROR - {e}')
"
```

#### Solutions

**API Key Issues:**
```bash
# Check API key format
grep "GROQ_API_KEY" .env

# Test key validity
curl -H "Authorization: Bearer YOUR_KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"test"}],"model":"llama2-70b-4096"}' \
     https://api.groq.com/openai/v1/chat/completions
```

**LLM Service Outages:**
```bash
# Switch to different LLM provider
sed -i 's/OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=your_openrouter_key/' .env
docker-compose restart codesage-mcp
```

**Rate Limiting:**
```bash
# Check rate limit headers in responses
curl -I http://localhost:8000/

# Implement request throttling
# Add rate limiting configuration to .env
echo "RATE_LIMIT_REQUESTS=60" >> .env
echo "RATE_LIMIT_WINDOW=60" >> .env
```

### 6. Index Corruption

#### Symptoms
- Search results are incomplete or wrong
- Indexing operations fail
- "Index corruption" errors in logs

#### Quick Diagnosis
```bash
# Check index files
docker-compose exec codesage-mcp ls -la /app/.codesage/

# Check index file sizes
docker-compose exec codesage-mcp du -h /app/.codesage/*.faiss
```

#### Solutions

**Rebuild Index:**
```bash
# Remove corrupted index
docker-compose exec codesage-mcp rm -f /app/.codesage/codebase_index.faiss

# Restart to rebuild
docker-compose restart codesage-mcp
```

**Index Optimization:**
```bash
# Run index optimization
docker-compose exec codesage-mcp python -c "
from codesage_mcp.core.indexing import IndexingManager
manager = IndexingManager()
manager.optimize_index_for_memory()
"
```

### 7. Monitoring Issues

#### Symptoms
- Prometheus metrics not updating
- Grafana dashboards show no data
- Alertmanager not sending alerts

#### Quick Diagnosis
```bash
# Check monitoring services
docker-compose ps | grep -E "(prometheus|grafana|alertmanager)"

# Test metrics endpoint
curl -s http://localhost:9100/metrics | head -5

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'
```

#### Solutions

**Restart Monitoring Stack:**
```bash
# Restart monitoring services
docker-compose restart prometheus grafana alertmanager

# Recreate monitoring stack
docker-compose up -d --force-recreate prometheus grafana alertmanager
```

**Fix Prometheus Configuration:**
```bash
# Check Prometheus configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# Reload configuration
curl -X POST http://localhost:9090/-/reload
```

## 🚨 Emergency Procedures

### Critical Service Down
```bash
# Immediate actions
docker-compose logs codesage-mcp --tail=50
docker-compose restart codesage-mcp
curl -f http://localhost:8000/

# If restart fails
docker-compose up -d --force-recreate codesage-mcp

# Last resort
docker-compose down
docker system prune -f
docker-compose up -d
```

### Data Loss Recovery
```bash
# Quick cache recovery
docker-compose exec codesage-mcp rm -rf /app/.codesage/cache/*
docker-compose restart codesage-mcp

# Full recovery from backup
./scripts/restore_from_backup.sh ./backups/latest/
```

## 📊 Performance Benchmarks

### Expected Performance
- **Indexing Speed:** >5 files/second
- **Search Response:** <2 seconds (average)
- **Memory Usage:** <500MB during indexing
- **Cache Hit Rate:** >70%
- **Uptime:** >99.9%

### Performance Testing
```bash
# Run quick performance test
curl -w "@curl-format.txt" -o /dev/null -s \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' \
  http://localhost:8000/mcp
```

## 📞 Support Contacts

- **Operations Team:** ops@company.com
- **Development Team:** dev@company.com
- **Emergency Hotline:** +1-555-0123
- **Documentation:** https://docs.company.com/codesage-mcp

## 📝 Log Analysis

### Common Log Patterns

**Success Patterns:**
```
INFO: Cache hit for search query
INFO: Indexed X files in Y seconds
INFO: Application startup complete
```

**Warning Patterns:**
```
WARNING: High memory usage detected
WARNING: Cache miss ratio above threshold
WARNING: API rate limit approaching
```

**Error Patterns:**
```
ERROR: Failed to connect to LLM API
ERROR: Index corruption detected
ERROR: Insufficient memory for operation
```

### Log Locations
```bash
# Application logs
docker-compose logs codesage-mcp

# System logs
docker-compose logs

# Monitoring logs
docker-compose logs prometheus
docker-compose logs grafana
```

---

**Quick Commands Reference:**

```bash
# Health check
curl -f http://localhost:8000/

# Restart service
docker-compose restart codesage-mcp

# Check logs
docker-compose logs -f codesage-mcp

# Monitor resources
docker stats codesage-mcp

# Clear cache
docker-compose exec codesage-mcp rm -rf /app/.codesage/cache/*
```

**Document Version:** 1.0
**Last Updated:** 2025-08-27