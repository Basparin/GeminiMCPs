# CodeSage MCP Server - Operational Runbook

## Overview

This operational runbook provides procedures for operating and maintaining the CodeSage MCP Server in production environments. It covers startup/shutdown, monitoring, troubleshooting, and maintenance procedures.

## 1. Service Management

### 1.1 Starting the Service

#### Docker Compose Deployment (Recommended)
```bash
# Start all services
docker-compose up -d

# Start with monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Start with specific resource limits
docker-compose up -d --scale codesage-mcp=3
```

#### Direct Docker Deployment
```bash
# Build and run
docker build -t codesage-mcp .
docker run -d \
  --name codesage-server \
  --memory=2g \
  --cpus=2.0 \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  codesage-mcp
```

#### Manual Deployment
```bash
# Activate virtual environment
source venv/bin/activate

# Start with high-performance settings
uvicorn codesage_mcp.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --http httptools \
  --access-log \
  --log-level info
```

### 1.2 Stopping the Service

#### Docker Compose
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (CAUTION: This removes data)
docker-compose down --volumes

# Stop but keep containers for inspection
docker-compose stop
```

#### Direct Docker
```bash
docker stop codesage-server
docker rm codesage-server
```

#### Manual Process
```bash
# Find and stop the process
ps aux | grep uvicorn
kill -TERM <PID>

# Or use pkill
pkill -f uvicorn
```

### 1.3 Restarting the Service

```bash
# Docker Compose
docker-compose restart codesage-mcp

# Direct Docker
docker restart codesage-server

# Manual
kill -HUP <PID>  # Graceful restart
```

## 2. Health Monitoring

### 2.1 Health Check Procedures

#### Automated Health Checks
```bash
# Run comprehensive health check
./scripts/health_check.sh

# Check specific components
./scripts/health_check.sh --docker    # Docker services only
./scripts/health_check.sh --http      # HTTP endpoint only
./scripts/health_check.sh --mcp       # MCP functionality only
./scripts/health_check.sh --memory    # Memory usage only
```

#### Manual Health Verification
```bash
# Check service status
curl -s http://localhost:8000/

# Check MCP functionality
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' \
  http://localhost:8000/mcp

# Check container health
docker-compose ps
docker stats codesage-mcp
```

### 2.2 Monitoring Dashboards

#### Accessing Monitoring
```bash
# Prometheus metrics
open http://localhost:9090

# Grafana dashboards
open http://localhost:3000  # admin/admin

# Node Exporter metrics
open http://localhost:9100/metrics
```

#### Key Metrics to Monitor
- **Response Time:** Should be <100ms for most operations
- **Memory Usage:** Should stay below 80% of allocated memory
- **CPU Usage:** Monitor for sustained high usage
- **Cache Hit Rate:** Should be >70%
- **Error Rate:** Should be <1%
- **Container Health:** Should show "healthy"

## 3. Performance Monitoring

### 3.1 Performance Benchmarks

#### Running Benchmarks
```bash
# Run full benchmark suite
python tests/benchmark_performance.py

# Run specific benchmark
python -m pytest tests/benchmark_performance.py::BenchmarkSuite::benchmark_search_performance -v
```

#### Performance Targets
- **Indexing Speed:** >5 files/second
- **Search Response:** <2 seconds (average), <5 seconds (P95)
- **Memory Usage:** <500MB during indexing
- **Cache Hit Rate:** >70%

### 3.2 Resource Usage Monitoring

```bash
# Monitor container resources
docker stats codesage-mcp

# Check memory usage
docker exec codesage-mcp ps aux --sort=-%mem | head -10

# Monitor disk usage
df -h
du -sh .codesage/
```

## 4. Troubleshooting Procedures

### 4.1 Common Issues and Solutions

#### Issue: Service Not Starting
```bash
# Check logs
docker-compose logs codesage-mcp

# Check environment variables
docker exec codesage-mcp env | grep -E "(API_KEY|MEMORY|CACHE)"

# Verify API keys are set
grep "API_KEY" .env
```

#### Issue: High Memory Usage
```bash
# Check memory usage
docker stats codesage-mcp

# Reduce cache sizes in .env
EMBEDDING_CACHE_SIZE=5000
SEARCH_CACHE_SIZE=1000
FILE_CACHE_SIZE=100

# Restart service
docker-compose restart codesage-mcp
```

#### Issue: Slow Performance
```bash
# Check cache hit rates
curl -s http://localhost:8000/mcp -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "get_cache_statistics"}}'

# Clear cache if needed
docker exec codesage-mcp rm -rf /app/.codesage/cache/*

# Restart service
docker-compose restart codesage-mcp
```

#### Issue: Container Health Check Failing
```bash
# Check container logs
docker-compose logs codesage-mcp

# Check health endpoint
curl -f http://localhost:8000/

# Restart unhealthy container
docker-compose restart codesage-mcp
```

### 4.2 Log Analysis

#### Accessing Logs
```bash
# Docker Compose logs
docker-compose logs -f codesage-mcp

# Direct Docker logs
docker logs -f codesage-server

# Application logs
docker exec codesage-mcp tail -f /app/logs/application.log
```

#### Common Log Patterns
```
# Successful operations
INFO: Cache hit for search query
INFO: Indexed X files in Y seconds

# Warnings
WARNING: High memory usage detected
WARNING: Cache miss ratio above threshold

# Errors
ERROR: Failed to connect to LLM API
ERROR: Index corruption detected
```

## 5. Maintenance Procedures

### 5.1 Cache Management

#### Cache Statistics
```bash
# Get cache statistics
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "get_cache_statistics"}}' \
  http://localhost:8000/mcp
```

#### Cache Maintenance
```bash
# Clear old cache entries (if needed)
docker exec codesage-mcp find /app/.codesage/cache -name "*.cache" -mtime +7 -delete

# Optimize cache size based on usage
# Adjust values in .env file:
EMBEDDING_CACHE_SIZE=8000  # Increase if high miss rate
SEARCH_CACHE_SIZE=1500     # Increase if high miss rate
```

### 5.2 Index Management

#### Index Optimization
```bash
# Check index health
docker exec codesage-mcp ls -la /app/.codesage/

# Rebuild index if corrupted
docker exec codesage-mcp rm -f /app/.codesage/codebase_index.faiss
docker-compose restart codesage-mcp
```

#### Index Backup
```bash
# Create index backup
docker exec codesage-mcp tar -czf /tmp/index_backup.tar.gz /app/.codesage/
docker cp codesage-mcp:/tmp/index_backup.tar.gz ./backups/

# Restore index
docker cp ./backups/index_backup.tar.gz codesage-mcp:/tmp/
docker exec codesage-mcp tar -xzf /tmp/index_backup.tar.gz -C /
```

### 5.3 Configuration Updates

#### Updating Environment Variables
```bash
# Edit .env file
nano .env

# Update specific values
sed -i 's/MAX_MEMORY_MB=.*/MAX_MEMORY_MB=4096/' .env

# Restart to apply changes
docker-compose restart codesage-mcp
```

#### Updating Docker Images
```bash
# Build new image
docker-compose build --no-cache codesage-mcp

# Rolling update
docker-compose up -d codesage-mcp

# Verify new version
docker-compose exec codesage-mcp python -c "import codesage_mcp; print(codesage_mcp.__version__)"
```

## 6. Backup and Recovery

### 6.1 Data Backup

#### Cache and Index Backup
```bash
# Create timestamped backup
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup persistent data
docker exec codesage-mcp tar -czf /tmp/cache_backup.tar.gz /app/.codesage/
docker cp codesage-mcp:/tmp/cache_backup.tar.gz $BACKUP_DIR/

# Backup configuration
cp .env $BACKUP_DIR/
cp docker-compose.yml $BACKUP_DIR/
```

#### Database Backup (if applicable)
```bash
# Backup any persistent databases
docker exec codesage-mcp pg_dump -U user -h host database > $BACKUP_DIR/database.sql
```

### 6.2 Recovery Procedures

#### Complete Service Recovery
```bash
# Stop current service
docker-compose down

# Restore from backup
BACKUP_DIR="./backups/latest"
docker cp $BACKUP_DIR/cache_backup.tar.gz codesage-mcp:/tmp/
docker exec codesage-mcp tar -xzf /tmp/cache_backup.tar.gz -C /

# Restore configuration
cp $BACKUP_DIR/.env .env

# Start service
docker-compose up -d
```

#### Partial Recovery
```bash
# Restore only cache
docker cp ./backups/latest/cache_backup.tar.gz codesage-mcp:/tmp/
docker exec codesage-mcp tar -xzf /tmp/cache_backup.tar.gz -C /

# Restart service
docker-compose restart codesage-mcp
```

## 7. Scaling Procedures

### 7.1 Horizontal Scaling

#### Adding More Instances
```bash
# Scale to 3 instances
docker-compose up -d --scale codesage-mcp=3

# Check instance status
docker-compose ps
```

#### Load Balancer Configuration
```nginx
# nginx.conf for load balancing
upstream codesage_backend {
    server codesage-mcp-1:8000;
    server codesage-mcp-2:8000;
    server codesage-mcp-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://codesage_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 7.2 Vertical Scaling

#### Increasing Resources
```bash
# Update docker-compose.yml with higher limits
services:
  codesage-mcp:
    deploy:
      resources:
        limits:
          memory: 4g
          cpus: '4.0'
        reservations:
          memory: 2g
          cpus: '2.0'

# Apply changes
docker-compose up -d codesage-mcp
```

## 8. Emergency Procedures

### 8.1 Service Outage Response

#### Immediate Actions
```bash
# Check service status
docker-compose ps

# Check logs for errors
docker-compose logs --tail=50 codesage-mcp

# Restart service
docker-compose restart codesage-mcp

# If restart fails, force recreate
docker-compose up -d --force-recreate codesage-mcp
```

#### Escalation Procedures
1. **Level 1:** Service restart (5 minutes)
2. **Level 2:** Container recreation (10 minutes)
3. **Level 3:** Full service rebuild (30 minutes)
4. **Level 4:** Infrastructure team involvement (1+ hours)

### 8.2 Data Loss Recovery

#### Cache Loss Recovery
```bash
# Clear corrupted cache
docker exec codesage-mcp rm -rf /app/.codesage/cache/*

# Restart to rebuild cache
docker-compose restart codesage-mcp
```

#### Complete Data Loss
```bash
# Restore from latest backup
./scripts/restore_from_backup.sh ./backups/latest/

# Verify service health
./scripts/health_check.sh
```

## 9. Monitoring and Alerting

### 9.1 Setting Up Alerts

#### Prometheus Alerting Rules
```yaml
# alerting_rules.yml
groups:
  - name: codesage
    rules:
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 1800
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}MB"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CodeSage MCP service is down"
          description: "Service has been down for more than 1 minute"
```

#### Email Notifications
```bash
# Configure email alerts
# Edit alertmanager.yml with SMTP settings
nano monitoring/alertmanager.yml
docker-compose restart alertmanager
```

### 9.2 Log Aggregation

#### Centralized Logging Setup
```bash
# Configure rsyslog for Docker containers
# Edit monitoring/rsyslog.conf
# Restart rsyslog service
sudo systemctl restart rsyslog

# View aggregated logs
tail -f /var/log/codesage/codesage.log
```

## 10. Security Procedures

### 10.1 API Key Management

#### Rotating API Keys
```bash
# Update .env file with new keys
nano .env

# Test new keys
docker-compose exec codesage-mcp python -c "
import os
from groq import Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
print('API key valid:', client.models.list() is not None)
"

# Restart service
docker-compose restart codesage-mcp
```

#### Key Security Best Practices
- Store keys in environment variables only
- Use Docker secrets for production
- Rotate keys quarterly
- Monitor key usage patterns
- Never commit keys to version control

### 10.2 Access Control

#### Network Security
```bash
# Configure firewall rules
sudo ufw allow 8000/tcp
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana

# Use reverse proxy for SSL termination
# Configure nginx with SSL certificates
```

## Summary

This operational runbook provides comprehensive procedures for:

✅ **Service Management:** Start, stop, restart procedures
✅ **Health Monitoring:** Automated and manual health checks
✅ **Performance Monitoring:** Resource usage and performance metrics
✅ **Troubleshooting:** Common issues and resolution steps
✅ **Maintenance:** Cache, index, and configuration management
✅ **Backup/Recovery:** Data protection and restoration
✅ **Scaling:** Horizontal and vertical scaling procedures
✅ **Emergency Response:** Outage handling and escalation
✅ **Security:** API key management and access control

**Next Steps:**
1. Customize alert thresholds for your environment
2. Set up automated backups
3. Configure monitoring dashboards
4. Train operations team on these procedures
5. Establish regular maintenance schedules

**Contact Information:**
- Operations Team: ops@company.com
- Development Team: dev@company.com
- Emergency Hotline: +1-555-0123

**Document Version:** 1.0
**Last Updated:** 2025-08-27
**Review Schedule:** Quarterly