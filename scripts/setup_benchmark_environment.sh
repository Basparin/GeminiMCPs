#!/bin/bash
# Setup script for CodeSage MCP Benchmarking Environment

set -e

echo "Setting up CodeSage MCP Benchmarking Environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p benchmark_results
mkdir -p logs
mkdir -p config
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p benchmark/load_test.py

# Create Prometheus configuration for benchmarking
echo "Creating Prometheus configuration..."
cat > monitoring/prometheus/benchmark.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'codesage-mcp'
    static_configs:
      - targets: ['codesage-mcp:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'benchmark-runner'
    static_configs:
      - targets: ['benchmark-runner:8000']
    scrape_interval: 10s

  - job_name: 'docker'
    static_configs:
      - targets: ['docker.for.mac.host.internal:9323']
    scrape_interval: 30s
EOF

# Create Grafana datasource configuration
echo "Creating Grafana datasource configuration..."
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create Grafana dashboard configuration
echo "Creating Grafana dashboard configuration..."
cat > monitoring/grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create a sample load test file for Locust
echo "Creating sample load test file..."
cat > benchmark/load_test.py << EOF
from locust import HttpUser, task, between
import json
import random

class MCPUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def test_initialize(self):
        """Test initialize request"""
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": random.randint(1, 1000),
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "load-test", "version": "1.0.0"}
            }
        }
        self.client.post("/mcp", json=payload)

    @task(5)
    def test_tools_list(self):
        """Test tools/list request"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": random.randint(1, 1000),
            "params": {}
        }
        self.client.post("/mcp", json=payload)

    @task(2)
    def test_read_file(self):
        """Test read_code_file tool"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": random.randint(1, 1000),
            "params": {
                "name": "read_code_file",
                "arguments": {"file_path": "codesage_mcp/main.py"}
            }
        }
        self.client.post("/mcp", json=payload)

    @task(1)
    def test_search_codebase(self):
        """Test search_codebase tool"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": random.randint(1, 1000),
            "params": {
                "name": "search_codebase",
                "arguments": {
                    "codebase_path": ".",
                    "pattern": "def",
                    "file_types": ["*.py"]
                }
            }
        }
        self.client.post("/mcp", json=payload)
EOF

# Create environment configuration files
echo "Creating environment configuration files..."

# Development configuration
cat > config/benchmark.env << EOF
# Benchmark Environment Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000
LOG_LEVEL=INFO
CACHE_SIZE_MB=512
MAX_MEMORY_USAGE_PERCENT=80
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_AUTO_TUNING=true
RESULTS_DIR=/app/benchmark_results
TEST_ITERATIONS=5
CONCURRENT_USERS=10
SERVER_URL=http://localhost:8000/mcp
EOF

# High-performance configuration
cat > config/high_performance.env << EOF
# High-Performance Benchmark Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000
LOG_LEVEL=WARNING
CACHE_SIZE_MB=1024
MAX_MEMORY_USAGE_PERCENT=90
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_AUTO_TUNING=true
RESULTS_DIR=/app/benchmark_results
TEST_ITERATIONS=10
CONCURRENT_USERS=50
SERVER_URL=http://localhost:8000/mcp
EOF

# Memory-constrained configuration
cat > config/memory_constrained.env << EOF
# Memory-Constrained Benchmark Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000
LOG_LEVEL=WARNING
CACHE_SIZE_MB=128
MAX_MEMORY_USAGE_PERCENT=60
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_AUTO_TUNING=false
RESULTS_DIR=/app/benchmark_results
TEST_ITERATIONS=3
CONCURRENT_USERS=5
SERVER_URL=http://localhost:8000/mcp
EOF

# Create benchmark runner script
echo "Creating benchmark runner script..."
cat > scripts/run_docker_benchmarks.sh << 'EOF'
#!/bin/bash
# Docker Benchmark Runner Script

set -e

echo "Starting Docker-based benchmarks..."

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service to be ready on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.benchmark.yml exec -T $service curl -f http://localhost:$port >/dev/null 2>&1; then
            echo "$service is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 10
        ((attempt++))
    done

    echo "Error: $service failed to start"
    return 1
}

# Start services
echo "Starting benchmark services..."
docker-compose -f docker-compose.benchmark.yml up -d

# Wait for services to be ready
wait_for_service codesage-mcp 8000

# Run benchmarks
echo "Running comprehensive benchmarks..."
docker-compose -f docker-compose.benchmark.yml exec -T benchmark-runner python /app/scripts/run_parameterized_benchmark.py --config default --users 10 --iterations 5

echo "Running modular tool benchmarks..."
docker-compose -f docker-compose.benchmark.yml exec -T benchmark-runner python /app/tests/benchmark_mcp_tools.py http://codesage-mcp:8000/mcp 5

# Generate reports
echo "Generating benchmark reports..."
docker-compose -f docker-compose.benchmark.yml exec -T benchmark-runner python /app/scripts/generate_benchmark_report.py

# Stop services
echo "Stopping benchmark services..."
docker-compose -f docker-compose.benchmark.yml down

echo "Benchmarking complete! Results are in the benchmark_results/ directory."
EOF

chmod +x scripts/run_docker_benchmarks.sh

# Create cleanup script
echo "Creating cleanup script..."
cat > scripts/cleanup_benchmark_environment.sh << 'EOF'
#!/bin/bash
# Cleanup script for benchmark environment

echo "Cleaning up benchmark environment..."

# Stop and remove containers
docker-compose -f docker-compose.benchmark.yml down -v --remove-orphans

# Remove Docker images (optional)
# docker rmi codesage-mcp:latest

# Clean up result files (optional - uncomment if needed)
# rm -rf benchmark_results/*
# rm -rf logs/*

echo "Cleanup complete!"
EOF

chmod +x scripts/cleanup_benchmark_environment.sh

# Make scripts executable
chmod +x scripts/setup_benchmark_environment.sh
chmod +x scripts/run_docker_benchmarks.sh
chmod +x scripts/cleanup_benchmark_environment.sh

echo ""
echo "Benchmark environment setup complete!"
echo ""
echo "Available commands:"
echo "  ./scripts/run_docker_benchmarks.sh     - Run benchmarks in Docker"
echo "  ./scripts/cleanup_benchmark_environment.sh - Clean up Docker environment"
echo ""
echo "To run benchmarks manually:"
echo "  docker-compose -f docker-compose.benchmark.yml up -d"
echo "  docker-compose -f docker-compose.benchmark.yml exec benchmark-runner python /app/tests/benchmark_performance.py"
echo ""
echo "Access points:"
echo "  MCP Server: http://localhost:8000"
echo "  Grafana: http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Locust: http://localhost:8089"