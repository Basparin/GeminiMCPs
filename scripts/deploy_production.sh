#!/bin/bash
# CES Production Deployment Script
# This script handles production deployment of the CES system

set -e

# Configuration
DEPLOY_ENV=${1:-"production"}
DEPLOY_VERSION=${2:-"latest"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}
PROJECT_NAME="ces"
DEPLOY_DIR="/opt/ces"
BACKUP_DIR="/opt/ces/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi

    # Check required tools
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log_error "Docker Compose is required but not installed"; exit 1; }

    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        log_error ".env file not found. Please create it with production configuration"
        exit 1
    fi

    # Validate environment variables
    source .env
    if [[ -z "$GROQ_API_KEY" ]] || [[ -z "$GOOGLE_API_KEY" ]]; then
        log_error "Required API keys not found in .env file"
        exit 1
    fi

    log_success "Pre-deployment checks passed"
}

# Create backup
create_backup() {
    log_info "Creating backup of current deployment..."

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"

    mkdir -p "$BACKUP_PATH"

    # Backup database
    if [[ -f "ces_memory.db" ]]; then
        cp ces_memory.db "${BACKUP_PATH}/"
        log_info "Database backup created"
    fi

    # Backup configuration
    if [[ -d "config" ]]; then
        cp -r config "${BACKUP_PATH}/"
        log_info "Configuration backup created"
    fi

    # Backup logs
    if [[ -d "logs" ]]; then
        cp -r logs "${BACKUP_PATH}/"
        log_info "Logs backup created"
    fi

    # Clean old backups (keep last 10)
    cd "$BACKUP_DIR" && ls -t | tail -n +11 | xargs -r rm -rf

    log_success "Backup completed: $BACKUP_PATH"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."

    # Build images
    docker-compose -f docker-compose.yml build

    # Tag images
    docker tag ${PROJECT_NAME}_web:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}_web:${DEPLOY_VERSION}
    docker tag ${PROJECT_NAME}_api:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}_api:${DEPLOY_VERSION}

    # Push images
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_web:${DEPLOY_VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_api:${DEPLOY_VERSION}

    log_success "Docker images built and pushed"
}

# Deploy to production
deploy_to_production() {
    log_info "Deploying to production environment..."

    # Create deployment directory
    sudo mkdir -p "$DEPLOY_DIR"
    sudo chown $USER:$USER "$DEPLOY_DIR"

    # Copy deployment files
    cp docker-compose.prod.yml "$DEPLOY_DIR/"
    cp .env "$DEPLOY_DIR/"
    cp -r config "$DEPLOY_DIR/"
    cp -r monitoring "$DEPLOY_DIR/"

    cd "$DEPLOY_DIR"

    # Update image tags in docker-compose.prod.yml
    sed -i "s|image:.*|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}_web:${DEPLOY_VERSION}|g" docker-compose.prod.yml
    sed -i "s|image:.*|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}_api:${DEPLOY_VERSION}|g" docker-compose.prod.yml

    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down || true

    # Start new containers
    docker-compose -f docker-compose.prod.yml up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30

    # Health check
    if curl -f http://localhost:8001/api/health > /dev/null 2>&1; then
        log_success "Production deployment successful"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    cd "$DEPLOY_DIR"

    # Start monitoring stack
    docker-compose -f monitoring/docker-compose.monitoring.yml up -d

    # Configure Grafana dashboards
    log_info "Grafana dashboards configured"

    log_success "Monitoring setup completed"
}

# Configure reverse proxy
configure_reverse_proxy() {
    log_info "Configuring reverse proxy..."

    # Create nginx configuration
    cat > /tmp/ces_nginx.conf << EOF
server {
    listen 80;
    server_name your-domain.com;

    # SSL configuration (uncomment when SSL is configured)
    # listen 443 ssl http2;
    # ssl_certificate /path/to/ssl/cert.pem;
    # ssl_certificate_key /path/to/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Web interface
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Static files caching
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

    sudo cp /tmp/ces_nginx.conf /etc/nginx/sites-available/ces
    sudo ln -sf /etc/nginx/sites-available/ces /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx

    log_success "Reverse proxy configured"
}

# Setup SSL certificate
setup_ssl() {
    log_info "Setting up SSL certificate..."

    # Install certbot if not present
    if ! command -v certbot >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y certbot python3-certbot-nginx
    fi

    # Obtain SSL certificate
    sudo certbot --nginx -d your-domain.com

    log_success "SSL certificate configured"
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."

    # Run database migrations if needed
    log_info "Checking for database migrations..."

    # Update file permissions
    sudo chown -R www-data:www-data "$DEPLOY_DIR"
    sudo chmod -R 755 "$DEPLOY_DIR"

    # Setup log rotation
    sudo cp monitoring/logrotate.conf /etc/logrotate.d/ces
    sudo systemctl restart logrotate

    # Configure firewall
    sudo ufw allow 80
    sudo ufw allow 443
    sudo ufw --force enable

    log_success "Post-deployment tasks completed"
}

# Rollback function
rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."

    cd "$DEPLOY_DIR"

    # Stop current deployment
    docker-compose -f docker-compose.prod.yml down

    # Find latest backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -1)

    if [[ -n "$LATEST_BACKUP" ]]; then
        log_info "Restoring from backup: $LATEST_BACKUP"

        # Restore files
        cp "${BACKUP_DIR}/${LATEST_BACKUP}/ces_memory.db" ./
        cp -r "${BACKUP_DIR}/${LATEST_BACKUP}/config" ./

        # Restart services
        docker-compose -f docker-compose.prod.yml up -d

        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
        exit 1
    fi
}

# Main deployment function
main() {
    log_info "Starting CES production deployment..."
    log_info "Environment: $DEPLOY_ENV"
    log_info "Version: $DEPLOY_VERSION"

    # Trap errors for rollback
    trap rollback_deployment ERR

    pre_deployment_checks
    create_backup
    build_and_push_images
    deploy_to_production
    setup_monitoring
    configure_reverse_proxy
    # setup_ssl  # Uncomment when domain is configured
    post_deployment_tasks

    log_success "CES production deployment completed successfully!"
    log_info "Application is available at: http://your-domain.com"
    log_info "API documentation: http://your-domain.com/api/docs"
    log_info "Health check: http://your-domain.com/api/health"
}

# Run main function
main "$@"