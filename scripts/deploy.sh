#!/bin/bash

# ================================
# CodeSage MCP Server - Deployment Script
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
ENV_FILE="$PROJECT_ROOT/.env"

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

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    log_success "Dependencies check passed"
}

check_environment() {
    log_info "Checking environment configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file not found: $ENV_FILE"
        log_info "Creating environment file from production template..."

        if [ -f "$PROJECT_ROOT/config/templates/production.env" ]; then
            cp "$PROJECT_ROOT/config/templates/production.env" "$ENV_FILE"
            log_warning "Please edit $ENV_FILE with your actual API keys and configuration"
            log_warning "Required: GROQ_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY"
            exit 1
        else
            log_error "Production template not found"
            exit 1
        fi
    fi

    # Check for required API keys
    if ! grep -q "GROQ_API_KEY=.*[a-zA-Z0-9]" "$ENV_FILE" && \
       ! grep -q "OPENROUTER_API_KEY=.*[a-zA-Z0-9]" "$ENV_FILE" && \
       ! grep -q "GOOGLE_API_KEY=.*[a-zA-Z0-9]" "$ENV_FILE"; then
        log_error "No valid API keys found in $ENV_FILE"
        log_error "Please set at least one of: GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY"
        exit 1
    fi

    log_success "Environment configuration check passed"
}

build_images() {
    log_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Build with no cache for clean production build
    if [ "$1" = "--no-cache" ]; then
        docker-compose build --no-cache
    else
        docker-compose build
    fi

    log_success "Docker images built successfully"
}

deploy_services() {
    log_info "Deploying services..."

    cd "$PROJECT_ROOT"

    # Start services
    docker-compose up -d

    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 10

    # Check service health
    if docker-compose ps | grep -q "Up"; then
        log_success "Services deployed successfully"
        show_status
    else
        log_error "Service deployment failed"
        docker-compose logs
        exit 1
    fi
}

show_status() {
    log_info "Service Status:"
    cd "$PROJECT_ROOT"
    docker-compose ps

    log_info "Service Logs:"
    docker-compose logs --tail=20 codesage-mcp
}

rollback() {
    log_warning "Rolling back deployment..."

    cd "$PROJECT_ROOT"
    docker-compose down

    # Start previous version if available
    if docker tag codesage-mcp:previous codesage-mcp:latest 2>/dev/null; then
        log_info "Starting previous version..."
        docker-compose up -d
        log_success "Rollback completed"
    else
        log_error "No previous version available for rollback"
        exit 1
    fi
}

cleanup() {
    log_info "Cleaning up old Docker images and containers..."

    cd "$PROJECT_ROOT"

    # Remove unused containers
    docker container prune -f

    # Remove unused images
    docker image prune -f

    # Remove unused volumes
    docker volume prune -f

    log_success "Cleanup completed"
}

# Main script logic
main() {
    local action="${1:-deploy}"
    local no_cache=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache)
                no_cache=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [ACTION] [OPTIONS]"
                echo ""
                echo "Actions:"
                echo "  deploy     Deploy services (default)"
                echo "  build      Build Docker images only"
                echo "  start      Start services"
                echo "  stop       Stop services"
                echo "  restart    Restart services"
                echo "  status     Show service status"
                echo "  logs       Show service logs"
                echo "  rollback   Rollback to previous version"
                echo "  cleanup    Clean up Docker resources"
                echo ""
                echo "Options:"
                echo "  --no-cache    Build images without cache"
                echo "  --help, -h    Show this help message"
                exit 0
                ;;
            *)
                action="$1"
                shift
                ;;
        esac
    done

    case $action in
        deploy)
            check_dependencies
            check_environment
            build_images $no_cache
            deploy_services
            ;;
        build)
            check_dependencies
            build_images $no_cache
            ;;
        start)
            cd "$PROJECT_ROOT"
            docker-compose start
            show_status
            ;;
        stop)
            cd "$PROJECT_ROOT"
            docker-compose stop
            ;;
        restart)
            cd "$PROJECT_ROOT"
            docker-compose restart
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            cd "$PROJECT_ROOT"
            docker-compose logs -f codesage-mcp
            ;;
        rollback)
            rollback
            ;;
        cleanup)
            cleanup
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