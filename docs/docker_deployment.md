# CodeSage MCP Server - Docker Deployment Guide

## Overview

This guide covers **production-ready Docker deployment** of CodeSage MCP Server with optimized configurations for performance, security, and scalability.

## Quick Start Deployment

### Basic Docker Deployment

#### 1. Build the Image
```bash
# Clone the repository
git clone <repository_url>
cd GeminiMCPs

# Build optimized Docker image
docker build -t codesage-mcp:latest .
```

#### 2. Create Environment Configuration
```bash
# Create .env file
cat > .env << EOF
# LLM API Keys
GROQ_API_KEY="your-groq-api-key"

# Performance Configuration
CODESAGE_MEMORY_LIMIT=512MB
CODESAGE_EMBEDDING_CACHE_SIZE=5000
CODESAGE_PARALLEL_WORKERS=4

# Production Settings
CODESAGE_LOG_LEVEL=INFO
CODESAGE_MONITORING_ENABLED=true
EOF
```

#### 3. Run the Container
```bash
# Run with basic configuration
docker run -d \
  --name codesage-server \
  -p 8000:8000 \
  --env-file .env \
  codesage-mcp:latest

# Check if it's running
curl http://localhost:8000/health
```

### Optimized Production Deployment

#### High-Performance Setup
```bash
# Run with performance optimizations
docker run -d \
  --name codesage-server \
  --memory=1g \
  --cpus=2.0 \
  --restart=unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v codesage_data:/app/.codesage \
  codesage-mcp:latest
```

#### Enterprise Setup
```bash
# Enterprise deployment with advanced features
docker run -d \
  --name codesage-server \
  --memory=2g \
  --cpus=4.0 \
  --restart=unless-stopped \
  --security-opt=no-new-privileges \
  --cap-drop=all \
  --cap-add=NET_BIND_SERVICE \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v codesage_data:/app/.codesage \
  -v /etc/ssl/certs:/etc/ssl/certs:ro \
  codesage-mcp:latest
```

## Docker Compose Deployment

### Basic Docker Compose Setup

#### docker-compose.yml
```yaml
version: '3.8'

services:
  codesage:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - codesage_data:/app/.codesage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  codesage_data:
```

#### Production Docker Compose
```yaml
version: '3.8'

services:
  codesage:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - BUILD_ENV=production
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - codesage_data:/app/.codesage
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2.0'
        reservations:
          memory: 512M
          cpus: '1.0'

  # Optional: Prometheus metrics
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

volumes:
  codesage_data:
```

### Advanced Docker Compose Setup

#### Multi-Service Deployment
```yaml
version: '3.8'

services:
  codesage:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - codesage_data:/app/.codesage
      - ./ssl:/app/ssl:ro
    restart: unless-stopped
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis for distributed caching (optional)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    restart: unless-stopped
    depends_on:
      - codesage

  # Monitoring stack
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  codesage_data:
  redis_data:
  grafana_data:
```

## Kubernetes Deployment

### Basic Kubernetes Deployment

#### codesage-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codesage-mcp
  labels:
    app: codesage-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: codesage-mcp
  template:
    metadata:
      labels:
        app: codesage-mcp
    spec:
      containers:
      - name: codesage
        image: codesage-mcp:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: codesage-config
        - secretRef:
            name: codesage-secrets
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: codesage-data
          mountPath: /app/.codesage
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: codesage-data
        persistentVolumeClaim:
          claimName: codesage-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: codesage-service
spec:
  selector:
    app: codesage-mcp
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: codesage-config
data:
  CODESAGE_MEMORY_LIMIT: "512MB"
  CODESAGE_EMBEDDING_CACHE_SIZE: "5000"
  CODESAGE_LOG_LEVEL: "INFO"

---
apiVersion: v1
kind: Secret
metadata:
  name: codesage-secrets
type: Opaque
data:
  GROQ_API_KEY: <base64-encoded-key>

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: codesage-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

#### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: codesage-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codesage-mcp
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Advanced Kubernetes Setup

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: codesage-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - codesage.yourdomain.com
    secretName: codesage-tls
  rules:
  - host: codesage.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: codesage-service
            port:
              number: 8000
```

#### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: codesage-network-policy
spec:
  podSelector:
    matchLabels:
      app: codesage-mcp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

## Docker Image Optimization

### Multi-Stage Dockerfile

#### Optimized Dockerfile
```dockerfile
# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r codesage && useradd -r -g codesage codesage

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/codesage/.local
ENV PATH="/home/codesage/.local/bin:$PATH"

# Copy application code
COPY codesage_mcp/ ./codesage_mcp/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/.codesage /app/logs && \
    chown -R codesage:codesage /app

# Switch to non-root user
USER codesage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "codesage_mcp.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Security Hardening

#### Security-Optimized Dockerfile
```dockerfile
FROM python:3.12-slim

# Security: Update packages and install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -r -g 1000 codesage && \
    useradd -r -u 1000 -g codesage -s /sbin/nologin -c "CodeSage User" codesage

# Security: Don't run as root
USER codesage

WORKDIR /app

# Security: Copy only necessary files
COPY --chown=codesage:codesage requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=codesage:codesage codesage_mcp/ ./codesage_mcp/

# Security: Remove unnecessary packages
RUN pip uninstall -y pip setuptools wheel

# Security: Use read-only filesystem where possible
VOLUME ["/app/.codesage", "/app/logs"]

EXPOSE 8000

# Security: Drop all capabilities and add only necessary ones
# (set in docker-compose.yml or docker run)

CMD ["uvicorn", "codesage_mcp.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Deployment Strategies

### Blue-Green Deployment

#### Docker Compose Blue-Green
```bash
# Deploy blue version
docker tag codesage-mcp:latest codesage-mcp:blue
docker-compose up -d codesage-blue

# Test blue version
curl -f http://localhost:8000/health || exit 1

# Switch traffic to blue
docker-compose up -d codesage-blue
docker-compose stop codesage-green
docker-compose rm codesage-green

# Cleanup old version
docker rmi codesage-mcp:green
```

### Rolling Deployment

#### Kubernetes Rolling Update
```bash
# Update deployment image
kubectl set image deployment/codesage-mcp codesage=codesage-mcp:v2.0.0

# Monitor rollout
kubectl rollout status deployment/codesage-mcp

# Rollback if needed
kubectl rollout undo deployment/codesage-mcp
```

### Load Balancing

#### Docker with Load Balancer
```yaml
version: '3.8'

services:
  codesage:
    build: .
    deploy:
      replicas: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  loadbalancer:
    image: nginx:alpine
    ports:
      - "8000:8000"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - codesage
```

#### Nginx Configuration
```nginx
upstream codesage_backend {
    server codesage-1:8000;
    server codesage-2:8000;
    server codesage-3:8000;
}

server {
    listen 8000;

    location / {
        proxy_pass http://codesage_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health check for load balancer
        health_check;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## Monitoring and Observability

### Docker Monitoring

#### Container Metrics
```bash
# Monitor container resources
docker stats codesage-server

# Container logs
docker logs -f codesage-server

# Container health
docker inspect codesage-server | grep -A 10 "Health"
```

#### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'codesage'
    static_configs:
      - targets: ['codesage:8000']
    metrics_path: '/metrics'
```

### Kubernetes Monitoring

#### ServiceMonitor for Prometheus
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: codesage-servicemonitor
  labels:
    team: backend
spec:
  selector:
    matchLabels:
      app: codesage-mcp
  endpoints:
  - port: metrics
    interval: 30s
```

#### Grafana Dashboard
```json
// Grafana dashboard JSON for CodeSage metrics
{
  "dashboard": {
    "title": "CodeSage MCP Server",
    "tags": ["codesage", "mcp"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{pod=~\"codesage-.*\"}",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "codesage_cache_hit_rate",
            "legendFormat": "Cache Hit Rate"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Data Backup

#### Docker Volume Backup
```bash
# Backup Docker volume
docker run --rm -v codesage_data:/data -v $(pwd):/backup alpine tar czf /backup/codesage_backup.tar.gz -C /data .

# Restore Docker volume
docker run --rm -v codesage_data:/data -v $(pwd):/backup alpine tar xzf /backup/codesage_backup.tar.gz -C /data
```

#### Kubernetes Backup
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: codesage-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: alpine
            command: ["/bin/sh", "-c"]
            args:
            - |
              tar czf /backup/codesage-$(date +%Y%m%d).tar.gz -C /data .
            volumeMounts:
            - name: codesage-data
              mountPath: /data
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: codesage-data
            persistentVolumeClaim:
              claimName: codesage-pvc
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Disaster Recovery

#### Recovery Procedures
```bash
# 1. Stop the current deployment
docker-compose down

# 2. Restore from backup
docker run --rm -v codesage_data:/data -v $(pwd):/backup alpine \
  sh -c "cd /data && tar xzf /backup/codesage_backup.tar.gz"

# 3. Start the deployment
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health
```

## Performance Tuning

### Docker Performance Optimization

#### Resource Limits
```yaml
services:
  codesage:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2.0'
        reservations:
          memory: 512M
          cpus: '1.0'
```

#### JVM-like Tuning for Python
```bash
# Python performance environment variables
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
```

### Network Optimization

#### Docker Network Configuration
```yaml
services:
  codesage:
    networks:
      - codesage_network
    dns:
      - 8.8.8.8
      - 1.1.1.1

networks:
  codesage_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Security Best Practices

### Docker Security

#### Security Scanning
```bash
# Scan Docker image for vulnerabilities
docker scan codesage-mcp:latest

# Use Trivy for comprehensive scanning
trivy image codesage-mcp:latest
```

#### Image Signing
```bash
# Sign Docker image
docker trust sign codesage-mcp:latest

# Verify signed image
docker trust inspect codesage-mcp:latest
```

### Production Security

#### Secrets Management
```yaml
# Use Docker secrets
secrets:
  groq_api_key:
    file: ./secrets/groq_api_key.txt

services:
  codesage:
    secrets:
      - groq_api_key
```

#### Network Security
```bash
# Run with security options
docker run --rm \
  --security-opt=no-new-privileges \
  --cap-drop=all \
  --cap-add=NET_BIND_SERVICE \
  --network=host \
  codesage-mcp:latest
```

## Troubleshooting Docker Deployments

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs codesage-server

# Check container status
docker ps -a

# Verify environment variables
docker exec codesage-server env
```

#### Performance Issues
```bash
# Check resource usage
docker stats codesage-server

# Profile container
docker exec codesage-server python -m cProfile -s time /app/codesage_mcp/main.py
```

#### Networking Issues
```bash
# Check network connectivity
docker network ls
docker network inspect bridge

# Test connectivity
docker exec codesage-server curl -f http://localhost:8000/health
```

### Health Checks

#### Custom Health Check Script
```bash
#!/bin/bash
# healthcheck.sh

# Check if service is responding
if curl -f -s http://localhost:8000/health > /dev/null; then
    # Check memory usage
    memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep codesage | awk '{print $3}')
    memory_percent=$(echo $memory_usage | sed 's/%.*//')

    if (( $(echo "$memory_percent < 90" | bc -l) )); then
        exit 0
    fi
fi

exit 1
```

## Conclusion

CodeSage MCP Server's Docker deployment provides **enterprise-grade deployment capabilities** with comprehensive optimization for performance, security, and scalability.

**Key Deployment Features:**
- **Multi-platform support**: Docker, Docker Compose, Kubernetes
- **Production ready**: Health checks, monitoring, security hardening
- **Scalable architecture**: Load balancing, auto-scaling, rolling updates
- **Security focused**: Image scanning, secrets management, network policies
- **Monitoring integrated**: Prometheus metrics, Grafana dashboards

The deployment strategies support everything from simple single-container deployments to complex multi-service Kubernetes clusters, ensuring optimal performance and reliability across different scales and environments.