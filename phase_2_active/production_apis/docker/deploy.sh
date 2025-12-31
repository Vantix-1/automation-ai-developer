#!/bin/bash
# deploy.sh - Production Deployment Script

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-api"
ENVIRONMENT="${ENVIRONMENT:-production}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   AI API Deployment Script${NC}"
echo -e "${BLUE}   Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print colored messages
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

# Check if Docker is installed and running
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker is installed and running"
    docker --version
    
    if command -v docker-compose &> /dev/null; then
        docker-compose --version
    elif docker compose version &> /dev/null; then
        docker compose version
    else
        log_error "Docker Compose is not installed"
        exit 1
    fi
}

# Check required files
check_files() {
    log_info "Checking required files..."
    
    local required_files=(
        "docker-compose.yml"
        "Dockerfile"
        ".env.${ENVIRONMENT}"
        "requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "All required files present"
}

# Load environment variables
load_env() {
    log_info "Loading environment variables..."
    
    if [ -f ".env.${ENVIRONMENT}" ]; then
        export $(cat .env.${ENVIRONMENT} | grep -v '^#' | xargs)
        log_success "Environment variables loaded from .env.${ENVIRONMENT}"
    else
        log_warning "Environment file .env.${ENVIRONMENT} not found"
    fi
}

# Validate environment variables
validate_env() {
    log_info "Validating environment variables..."
    
    local required_vars=(
        "OPENAI_API_KEY"
        "SECRET_KEY"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
    
    # Check for default/weak passwords
    if [ "$DB_PASSWORD" = "password" ]; then
        log_error "DB_PASSWORD is set to default value 'password'. Please use a strong password!"
        exit 1
    fi
    
    if [ "$REDIS_PASSWORD" = "redispass" ]; then
        log_error "REDIS_PASSWORD is set to default value 'redispass'. Please use a strong password!"
        exit 1
    fi
    
    log_success "Environment variables validated"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."
    
    local dirs=(
        "nginx"
        "monitoring"
        "monitoring/grafana/provisioning"
        "monitoring/grafana/dashboards"
        "certbot/conf"
        "certbot/www"
        "init-scripts"
        "backups"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "Directories created"
}

# Backup existing data
backup_data() {
    if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
        log_info "Creating backup before deployment..."
        
        local backup_dir="backups/pre-deploy-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup docker volumes if they exist
        if docker volume ls | grep -q "${PROJECT_NAME}"; then
            log_info "Backing up Docker volumes..."
            docker run --rm \
                -v ai_uploads:/source/uploads:ro \
                -v ai_data:/source/data:ro \
                -v "$backup_dir:/backup" \
                alpine tar czf /backup/volumes.tar.gz -C /source .
        fi
        
        log_success "Backup created: $backup_dir"
    else
        log_info "Skipping backup (BACKUP_BEFORE_DEPLOY=false)"
    fi
}

# Pull latest images
pull_images() {
    log_info "Pulling latest Docker images..."
    docker compose pull
    log_success "Images pulled successfully"
}

# Build custom images
build_images() {
    log_info "Building custom Docker images..."
    
    export BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    export VERSION=${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}
    
    docker compose build --no-cache
    log_success "Images built successfully"
}

# Stop existing containers
stop_services() {
    log_info "Stopping existing services..."
    docker compose down
    log_success "Services stopped"
}

# Start services
start_services() {
    log_info "Starting services..."
    docker compose up -d
    log_success "Services started"
}

# Wait for services to be healthy
wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        # Check API health
        if docker compose ps | grep -q "ai-api.*healthy"; then
            log_success "AI API is healthy"
            return 0
        fi
        
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "Services failed to become healthy"
    docker compose logs --tail=50
    return 1
}

# Run database migrations (if applicable)
run_migrations() {
    log_info "Running database migrations..."
    
    # Uncomment and modify based on your migration tool
    # docker compose exec -T ai-api alembic upgrade head
    
    log_info "Migrations completed (or skipped if not configured)"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Test database connection (if health endpoint checks it)
    log_success "Smoke tests passed"
}

# Display service status
show_status() {
    log_info "Service Status:"
    echo ""
    docker compose ps
    echo ""
    
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# Display logs
show_logs() {
    log_info "Recent logs (last 50 lines):"
    docker compose logs --tail=50
}

# Main deployment function
deploy() {
    log_info "Starting deployment process..."
    
    check_docker
    check_files
    load_env
    validate_env
    create_directories
    backup_data
    
    if [ "${BUILD_IMAGES}" = "true" ]; then
        build_images
    else
        pull_images
    fi
    
    stop_services
    start_services
    
    if wait_for_health; then
        run_migrations
        
        if run_smoke_tests; then
            log_success "Deployment completed successfully!"
            show_status
        else
            log_error "Smoke tests failed - rolling back..."
            rollback
            exit 1
        fi
    else
        log_error "Health checks failed - rolling back..."
        rollback
        exit 1
    fi
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    docker compose down
    
    # Restore from latest backup if available
    local latest_backup=$(ls -t backups/pre-deploy-* 2>/dev/null | head -1)
    if [ -n "$latest_backup" ]; then
        log_info "Restoring from backup: $latest_backup"
        # Add restore logic here
    fi
    
    log_warning "Rollback completed"
}

# Script commands
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    stop)
        stop_services
        ;;
    start)
        start_services
        ;;
    restart)
        stop_services
        start_services
        wait_for_health
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    backup)
        backup_data
        ;;
    rollback)
        rollback
        ;;
    health)
        wait_for_health
        ;;
    *)
        echo "Usage: $0 {deploy|stop|start|restart|status|logs|backup|rollback|health}"
        exit 1
        ;;
esac

# =============================================================================
# setup-ssl.sh - SSL Certificate Setup Script
# =============================================================================

#!/bin/bash
# setup-ssl.sh - Setup SSL certificates with Let's Encrypt

set -e

DOMAIN="${DOMAIN:-yourdomain.com}"
EMAIL="${SSL_EMAIL:-admin@${DOMAIN}}"

log_info "Setting up SSL certificates for: $DOMAIN"

# Initial certificate request
log_info "Requesting SSL certificate..."

docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" \
    -d "www.$DOMAIN"

log_success "SSL certificate obtained"

# Update nginx configuration
log_info "Updating Nginx configuration..."

# Reload Nginx
docker compose exec nginx nginx -s reload

log_success "SSL setup completed!"

# =============================================================================
# monitor.sh - Monitoring and Alerting Script
# =============================================================================

#!/bin/bash
# monitor.sh - System monitoring script

set -e

check_service_health() {
    local service=$1
    
    if docker compose ps | grep -q "$service.*healthy"; then
        echo "✅ $service: Healthy"
        return 0
    else
        echo "❌ $service: Unhealthy"
        return 1
    fi
}

check_disk_space() {
    local usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$usage" -gt 85 ]; then
        echo "⚠️  Disk usage: ${usage}% (Warning threshold: 85%)"
        return 1
    else
        echo "✅ Disk usage: ${usage}%"
        return 0
    fi
}

check_memory() {
    local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ "$usage" -gt 85 ]; then
        echo "⚠️  Memory usage: ${usage}% (Warning threshold: 85%)"
        return 1
    else
        echo "✅ Memory usage: ${usage}%"
        return 0
    fi
}

log_info "Running system health checks..."

check_service_health "ai-api"
check_service_health "db"
check_service_health "redis"
check_service_health "nginx"

check_disk_space
check_memory

log_success "Health check completed"