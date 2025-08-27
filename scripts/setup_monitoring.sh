#!/bin/bash

# ================================
# CodeSage MCP Server - Monitoring Setup Script
# ================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_monitoring_directories() {
    log_info "Creating monitoring directories..."

    mkdir -p "$MONITORING_DIR/prometheus"
    mkdir -p "$MONITORING_DIR/grafana/provisioning/datasources"
    mkdir -p "$MONITORING_DIR/grafana/provisioning/dashboards"
    mkdir -p "$MONITORING_DIR/grafana/dashboards"
    mkdir -p "$MONITORING_DIR/alertmanager"

    log_success "Monitoring directories created"
}

create_prometheus_config() {
    log_info "Creating Prometheus configuration..."

    cat > "$MONITORING_DIR/prometheus.yml" << 'EOF'
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
    scrape_interval: 5s
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
EOF

    log_success "Prometheus configuration created"
}

create_grafana_datasource() {
    log_info "Creating Grafana datasource configuration..."

    cat > "$MONITORING_DIR/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    log_success "Grafana datasource configuration created"
}

create_grafana_dashboard() {
    log_info "Creating Grafana dashboard configuration..."

    cat > "$MONITORING_DIR/grafana/provisioning/dashboards/dashboard.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'CodeSage MCP Dashboard'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    cat > "$MONITORING_DIR/grafana/dashboards/codesage-overview.json" << 'EOF'
{
  "dashboard": {
    "title": "CodeSage MCP Server Overview",
    "tags": ["codesage", "mcp"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests per second"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) * 100",
            "legendFormat": "Cache hit rate (%)"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    log_success "Grafana dashboard configuration created"
}

create_alertmanager_config() {
    log_info "Creating Alertmanager configuration..."

    cat > "$MONITORING_DIR/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@codesage.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'
  routes:
    - match:
        severity: critical
      receiver: 'email'

receivers:
  - name: 'email'
    email_configs:
      - to: 'admin@codesage.com'
        subject: 'CodeSage MCP Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Runbook: {{ .Annotations.runbook_url }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
EOF

    log_success "Alertmanager configuration created"
}

create_logrotate_config() {
    log_info "Creating log rotation configuration..."

    cat > "$MONITORING_DIR/logrotate.conf" << 'EOF'
/app/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 0644 codesage codesage
    postrotate
        docker-compose exec -T codesage-mcp kill -HUP 1
    endscript
}
EOF

    log_success "Log rotation configuration created"
}

create_monitoring_docker_compose() {
    log_info "Creating monitoring Docker Compose override..."

    cat > "$PROJECT_ROOT/docker-compose.monitoring.yml" << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: codesage-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - codesage-network

  alertmanager:
    image: prom/alertmanager:latest
    container_name: codesage-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - codesage-network

  grafana:
    image: grafana/grafana:latest
    container_name: codesage-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
    networks:
      - codesage-network

  node-exporter:
    image: prom/node-exporter:latest
    container_name: codesage-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/root'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - codesage-network

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  codesage-network:
    external: true
EOF

    log_success "Monitoring Docker Compose configuration created"
}

setup_log_aggregation() {
    log_info "Setting up log aggregation..."

    # Create rsyslog configuration for Docker containers
    cat > "$MONITORING_DIR/rsyslog.conf" << 'EOF'
# Docker container logging
input(type="imtcp" port="514" ruleset="docker")

ruleset(name="docker") {
    action(type="omfile" file="/var/log/docker/containers.log")
    stop
}

# CodeSage MCP specific logging
input(type="imfile" file="/app/logs/*.log" ruleset="codesage")

ruleset(name="codesage") {
    action(type="omfile" file="/var/log/codesage/codesage.log")
    stop
}
EOF

    log_success "Log aggregation configuration created"
}

# Main setup function
main() {
    log_info "Setting up CodeSage MCP monitoring stack..."

    create_monitoring_directories
    create_prometheus_config
    create_grafana_datasource
    create_grafana_dashboard
    create_alertmanager_config
    create_logrotate_config
    create_monitoring_docker_compose
    setup_log_aggregation

    echo ""
    log_success "Monitoring setup completed!"
    echo ""
    log_info "To start monitoring stack:"
    echo "  docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d"
    echo ""
    log_info "Access URLs:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Alertmanager: http://localhost:9093"
    echo "  - Node Exporter: http://localhost:9100"
    echo ""
    log_warning "Remember to:"
    echo "  1. Update email configuration in alertmanager.yml"
    echo "  2. Change default Grafana password"
    echo "  3. Configure SSL/TLS for production"
    echo "  4. Set up proper firewall rules"
}

# Run main function
main "$@"