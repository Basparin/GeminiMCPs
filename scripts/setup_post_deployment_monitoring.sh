#!/bin/bash

# ================================
# CodeSage MCP Server - Post-Deployment Monitoring Setup
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "docker-compose is not installed."
        exit 1
    fi

    # Check if CodeSage MCP is running
    if ! curl -s http://localhost:8000/metrics >/dev/null 2>&1; then
        log_warning "CodeSage MCP server is not running or metrics endpoint is not accessible."
        log_warning "Please ensure the server is running before setting up monitoring."
    fi

    log_success "Prerequisites check completed"
}

# Setup monitoring stack
setup_monitoring_stack() {
    log_info "Setting up monitoring stack..."

    # Create monitoring network if it doesn't exist
    docker network create codesage-network 2>/dev/null || true

    # Start monitoring stack
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.monitoring.yml up -d

    # Wait for services to be ready
    log_info "Waiting for monitoring services to start..."
    sleep 10

    # Check service health
    if docker-compose -f docker-compose.monitoring.yml ps | grep -q "Up"; then
        log_success "Monitoring stack started successfully"
    else
        log_error "Failed to start monitoring stack"
        exit 1
    fi
}

# Configure Grafana dashboards
configure_grafana() {
    log_info "Configuring Grafana dashboards..."

    # Wait for Grafana to be ready
    local retries=0
    while ! curl -s http://localhost:3000/api/health >/dev/null 2>&1; do
        if [ $retries -ge 30 ]; then
            log_error "Grafana failed to start within expected time"
            exit 1
        fi
        sleep 2
        retries=$((retries + 1))
    done

    log_success "Grafana is ready"

    # Note: Dashboards are provisioned automatically via docker-compose
    # No additional configuration needed
}

# Configure Prometheus alerting
configure_alerting() {
    log_info "Configuring Prometheus alerting rules..."

    # Alerting rules are already configured in docker-compose
    # Check if Prometheus can access the rules
    if curl -s http://localhost:9090/api/v1/rules >/dev/null 2>&1; then
        log_success "Prometheus alerting rules configured"
    else
        log_warning "Prometheus may not be fully ready yet"
    fi
}

# Test monitoring setup
test_monitoring_setup() {
    log_info "Testing monitoring setup..."

    # Test Prometheus metrics collection
    if curl -s http://localhost:9090/api/v1/targets | grep -q "codesage-mcp"; then
        log_success "Prometheus is collecting CodeSage MCP metrics"
    else
        log_warning "Prometheus may not be scraping CodeSage MCP metrics yet"
    fi

    # Test Grafana accessibility
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        log_success "Grafana is accessible"
    else
        log_error "Grafana is not accessible"
    fi

    # Test Alertmanager
    if curl -s http://localhost:9093 >/dev/null 2>&1; then
        log_success "Alertmanager is accessible"
    else
        log_error "Alertmanager is not accessible"
    fi
}

# Setup continuous improvement automation
setup_continuous_improvement() {
    log_info "Setting up continuous improvement automation..."

    # Create cron job for daily analysis
    cat > /tmp/codesage_monitoring.cron << 'EOF'
# CodeSage MCP Monitoring - Daily Analysis
0 2 * * * /path/to/codesage-mcp/scripts/daily_monitoring_check.sh

# CodeSage MCP Monitoring - Weekly Report
0 3 * * 1 /path/to/codesage-mcp/scripts/weekly_monitoring_report.sh

# CodeSage MCP Monitoring - Continuous Improvement
*/30 * * * * /path/to/codesage-mcp/scripts/continuous_improvement_check.sh
EOF

    log_info "Continuous improvement automation configured"
    log_warning "Please update the paths in the cron file and install manually"
    echo "Cron configuration saved to: /tmp/codesage_monitoring.cron"
}

# Display access information
display_access_info() {
    echo ""
    log_success "Post-deployment monitoring setup completed!"
    echo ""
    log_info "Access URLs:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Alertmanager: http://localhost:9093"
    echo "  - Node Exporter: http://localhost:9100"
    echo "  - CodeSage MCP Metrics: http://localhost:8000/metrics"
    echo ""
    log_info "Available Dashboards:"
    echo "  - CodeSage MCP Server Overview"
    echo "  - CodeSage MCP Performance Analysis"
    echo "  - CodeSage MCP User Experience"
    echo "  - CodeSage MCP Self-Optimization"
    echo ""
    log_warning "Important next steps:"
    echo "  1. Change the default Grafana password (admin/admin)"
    echo "  2. Configure email notifications in alertmanager.yml"
    echo "  3. Review and adjust alert thresholds for your environment"
    echo "  4. Set up automated backups of monitoring data"
    echo "  5. Configure SSL/TLS for production access"
    echo ""
    log_info "Documentation:"
    echo "  - Post-deployment monitoring guide: docs/post_deployment_monitoring_guide.md"
    echo "  - Operational runbook: docs/operational_runbook.md"
    echo "  - Production deployment guide: docs/production_deployment_guide.md"
}

# Main setup function
main() {
    log_info "Starting CodeSage MCP post-deployment monitoring setup..."

    check_prerequisites
    setup_monitoring_stack
    configure_grafana
    configure_alerting
    test_monitoring_setup
    setup_continuous_improvement
    display_access_info

    echo ""
    log_success "Setup completed successfully!"
    echo ""
    log_info "The monitoring framework includes:"
    echo "  ✅ Prometheus metrics collection"
    echo "  ✅ Grafana dashboards (4 specialized dashboards)"
    echo "  ✅ Alertmanager with multi-tier alerting"
    echo "  ✅ Continuous improvement tools"
    echo "  ✅ Automated optimization capabilities"
    echo "  ✅ Comprehensive documentation"
}

# Run main function
main "$@"