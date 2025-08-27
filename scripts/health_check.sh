#!/bin/bash

# ================================
# CodeSage MCP Server - Health Check Script
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
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-30}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Health check functions
check_docker_services() {
    log_info "Checking Docker services status..."

    if ! docker-compose ps | grep -q "Up"; then
        log_error "Docker services are not running"
        return 1
    fi

    log_success "Docker services are running"
    return 0
}

check_http_endpoint() {
    log_info "Checking HTTP endpoint health..."

    if ! curl -f -s --max-time "$TIMEOUT" "$HEALTH_CHECK_URL/" > /dev/null; then
        log_error "HTTP endpoint is not responding"
        return 1
    fi

    log_success "HTTP endpoint is responding"
    return 0
}

check_mcp_server() {
    log_info "Checking MCP server functionality..."

    # Test the initialize endpoint
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' \
        --max-time "$TIMEOUT" \
        "$HEALTH_CHECK_URL/mcp" 2>/dev/null)

    if [ $? -ne 0 ] || ! echo "$response" | grep -q '"protocolVersion"'; then
        log_error "MCP server is not responding correctly"
        return 1
    fi

    log_success "MCP server is functioning correctly"
    return 0
}

check_tools_list() {
    log_info "Checking tools/list endpoint..."

    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' \
        --max-time "$TIMEOUT" \
        "$HEALTH_CHECK_URL/mcp" 2>/dev/null)

    if [ $? -ne 0 ] || ! echo "$response" | grep -q '"tools"'; then
        log_error "tools/list endpoint is not working"
        return 1
    fi

    log_success "tools/list endpoint is working"
    return 0
}

check_memory_usage() {
    log_info "Checking memory usage..."

    cd "$PROJECT_ROOT"

    # Get container memory usage
    container_name=$(docker-compose ps | grep codesage-mcp | awk '{print $1}')
    if [ -z "$container_name" ]; then
        log_error "Could not find container name"
        return 1
    fi

    memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        | grep "$container_name" \
        | awk '{print $3}')

    log_info "Container memory usage: $memory_usage"

    # Check if memory usage is too high (>90%)
    if echo "$memory_usage" | grep -q "%" && [ "$(echo "$memory_usage" | sed 's/%//')" -gt 90 ]; then
        log_warning "High memory usage detected: $memory_usage"
    else
        log_success "Memory usage is within acceptable limits"
    fi

    return 0
}

check_disk_space() {
    log_info "Checking disk space..."

    # Check if cache directory exists and has reasonable size
    cache_dir="$PROJECT_ROOT/.codesage"
    if [ -d "$cache_dir" ]; then
        cache_size=$(du -sh "$cache_dir" 2>/dev/null | awk '{print $1}')
        log_info "Cache directory size: $cache_size"

        # Check if cache is too large (>1GB)
        cache_bytes=$(du -sb "$cache_dir" 2>/dev/null | awk '{print $1}')
        if [ "$cache_bytes" -gt 1073741824 ]; then
            log_warning "Cache directory is very large: $cache_size"
        fi
    else
        log_info "Cache directory does not exist yet"
    fi

    return 0
}

check_logs() {
    log_info "Checking recent logs for errors..."

    cd "$PROJECT_ROOT"

    # Check for recent errors in logs
    error_count=$(docker-compose logs --tail=100 codesage-mcp 2>&1 | grep -i error | wc -l)

    if [ "$error_count" -gt 0 ]; then
        log_warning "Found $error_count error(s) in recent logs"
        docker-compose logs --tail=10 codesage-mcp | grep -i error
    else
        log_success "No recent errors found in logs"
    fi

    return 0
}

# Main health check function
perform_health_check() {
    local checks_passed=0
    local total_checks=0

    # Array of check functions
    checks=(
        "check_docker_services"
        "check_http_endpoint"
        "check_mcp_server"
        "check_tools_list"
        "check_memory_usage"
        "check_disk_space"
        "check_logs"
    )

    echo "========================================"
    echo "CodeSage MCP Server Health Check"
    echo "========================================"

    for check in "${checks[@]}"; do
        ((total_checks++))
        if $check; then
            ((checks_passed++))
        fi
        echo ""
    done

    echo "========================================"
    echo "Health Check Summary"
    echo "========================================"
    echo "Checks passed: $checks_passed/$total_checks"

    if [ "$checks_passed" -eq "$total_checks" ]; then
        log_success "All health checks passed!"
        return 0
    else
        log_error "$((total_checks - checks_passed)) health check(s) failed!"
        return 1
    fi
}

# Main script logic
main() {
    local action="${1:-check}"

    case $action in
        check|--check)
            perform_health_check
            ;;
        docker|--docker)
            check_docker_services
            ;;
        http|--http)
            check_http_endpoint
            ;;
        mcp|--mcp)
            check_mcp_server
            ;;
        memory|--memory)
            check_memory_usage
            ;;
        logs|--logs)
            check_logs
            ;;
        --help|-h)
            echo "Usage: $0 [ACTION]"
            echo ""
            echo "Actions:"
            echo "  check     Run full health check (default)"
            echo "  docker    Check Docker services only"
            echo "  http      Check HTTP endpoint only"
            echo "  mcp       Check MCP server functionality only"
            echo "  memory    Check memory usage only"
            echo "  logs      Check logs for errors only"
            echo ""
            echo "Environment variables:"
            echo "  HEALTH_CHECK_URL  URL to check (default: http://localhost:8000)"
            echo "  TIMEOUT           Request timeout in seconds (default: 30)"
            exit 0
            ;;
        *)
            log_error "Unknown action: $action"
            log_info "Use '$0 --help' for available actions"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"