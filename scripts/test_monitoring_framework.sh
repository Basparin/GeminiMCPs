#!/bin/bash

# ================================
# CodeSage MCP Server - Monitoring Framework Test
# ================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CODESAGE_URL="http://localhost:8000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
ALERTMANAGER_URL="http://localhost:9093"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

run_test() {
    local test_name="$1"
    local test_function="$2"

    TESTS_RUN=$((TESTS_RUN + 1))
    log_info "Running test: $test_name"

    if $test_function; then
        log_success "✓ $test_name passed"
    else
        log_error "✗ $test_name failed"
    fi
}

# Test CodeSage MCP server health
test_codesage_health() {
    if curl -s -f "$CODESAGE_URL/" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Test metrics endpoint
test_metrics_endpoint() {
    local response=$(curl -s "$CODESAGE_URL/metrics" 2>/dev/null)
    if [[ $response == *"codesage_mcp"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test MCP tools/list endpoint
test_tools_list() {
    local response=$(curl -s -X POST "$CODESAGE_URL/mcp" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' 2>/dev/null)

    if [[ $response == *"tools"* ]] && [[ $response == *"name"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test continuous improvement tools availability
test_continuous_improvement_tools() {
    local response=$(curl -s -X POST "$CODESAGE_URL/mcp" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' 2>/dev/null)

    if [[ $response == *"analyze_continuous_improvement_opportunities"* ]] && \
       [[ $response == *"implement_automated_improvements"* ]] && \
       [[ $response == *"monitor_improvement_effectiveness"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test continuous improvement tool functionality
test_continuous_improvement_functionality() {
    local response=$(curl -s -X POST "$CODESAGE_URL/mcp" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "analyze_continuous_improvement_opportunities"}}' 2>/dev/null)

    if [[ $response == *"optimization_opportunities"* ]] && [[ $response == *"recommendations"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test Prometheus accessibility
test_prometheus_access() {
    if curl -s -f "$PROMETHEUS_URL/-/healthy" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Test Prometheus targets
test_prometheus_targets() {
    local response=$(curl -s "$PROMETHEUS_URL/api/v1/targets" 2>/dev/null)
    if [[ $response == *"codesage-mcp"* ]] && [[ $response == *"up"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test Prometheus alerting rules
test_prometheus_alerting() {
    local response=$(curl -s "$PROMETHEUS_URL/api/v1/rules" 2>/dev/null)
    if [[ $response == *"codesage_performance_alerts"* ]] || [[ $response == *"codesage_predictive_alerts"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test Grafana accessibility
test_grafana_access() {
    if curl -s -f "$GRAFANA_URL/api/health" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Test Grafana dashboards
test_grafana_dashboards() {
    local response=$(curl -s "$GRAFANA_URL/api/search" 2>/dev/null)
    if [[ $response == *"CodeSage MCP"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test Alertmanager accessibility
test_alertmanager_access() {
    if curl -s -f "$ALERTMANAGER_URL/-/healthy" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Test monitoring data collection
test_monitoring_data_collection() {
    # Wait a moment for data collection
    sleep 2

    local metrics_response=$(curl -s "$CODESAGE_URL/metrics" 2>/dev/null)
    local prometheus_response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=up" 2>/dev/null)

    if [[ $metrics_response == *"codesage_mcp_requests_total"* ]] && \
       [[ $prometheus_response == *"value"* ]]; then
        return 0
    else
        return 1
    fi
}

# Test configuration validation
test_configuration_validation() {
    # Check if required files exist
    local required_files=(
        "monitoring/prometheus.yml"
        "monitoring/alerting_rules.yml"
        "monitoring/alertmanager.yml"
        "monitoring/grafana/dashboards/codesage-overview.json"
        "monitoring/grafana/dashboards/codesage-performance.json"
        "monitoring/grafana/dashboards/codesage-user-experience.json"
        "monitoring/grafana/dashboards/codesage-self-optimization.json"
        "docs/post_deployment_monitoring_guide.md"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file missing: $file"
            return 1
        fi
    done

    return 0
}

# Test docker services
test_docker_services() {
    if command -v docker-compose >/dev/null 2>&1; then
        local services=$(docker-compose ps 2>/dev/null | grep -E "(prometheus|grafana|alertmanager)" | wc -l)
        if [[ $services -ge 3 ]]; then
            return 0
        fi
    fi
    return 1
}

# Display test summary
display_test_summary() {
    echo ""
    echo "========================================"
    echo "Monitoring Framework Test Summary"
    echo "========================================"
    echo "Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed! Monitoring framework is working correctly."
    else
        log_error "Some tests failed. Please review the errors above."
        echo ""
        log_info "Common issues:"
        echo "  - Ensure CodeSage MCP server is running"
        echo "  - Check if monitoring stack is deployed"
        echo "  - Verify network connectivity"
        echo "  - Review logs for detailed error information"
    fi
}

# Main test function
main() {
    log_info "Starting CodeSage MCP monitoring framework validation..."

    # Basic connectivity tests
    run_test "CodeSage MCP Server Health" test_codesage_health
    run_test "Metrics Endpoint Accessibility" test_metrics_endpoint
    run_test "MCP Tools/List Endpoint" test_tools_list

    # Continuous improvement tools tests
    run_test "Continuous Improvement Tools Availability" test_continuous_improvement_tools
    run_test "Continuous Improvement Tool Functionality" test_continuous_improvement_functionality

    # Prometheus tests
    run_test "Prometheus Accessibility" test_prometheus_access
    run_test "Prometheus Targets Configuration" test_prometheus_targets
    run_test "Prometheus Alerting Rules" test_prometheus_alerting

    # Grafana tests
    run_test "Grafana Accessibility" test_grafana_access
    run_test "Grafana Dashboards" test_grafana_dashboards

    # Alertmanager tests
    run_test "Alertmanager Accessibility" test_alertmanager_access

    # Integration tests
    run_test "Monitoring Data Collection" test_monitoring_data_collection
    run_test "Configuration Validation" test_configuration_validation
    run_test "Docker Services Status" test_docker_services

    # Display results
    display_test_summary

    # Provide next steps
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo ""
        log_info "Next steps:"
        echo "  1. Access Grafana dashboards at: $GRAFANA_URL"
        echo "  2. Review monitoring metrics at: $PROMETHEUS_URL"
        echo "  3. Check alert status at: $ALERTMANAGER_URL"
        echo "  4. Run continuous improvement analysis using the MCP tools"
        echo "  5. Review the comprehensive guide: docs/post_deployment_monitoring_guide.md"
    fi
}

# Run main function
main "$@"