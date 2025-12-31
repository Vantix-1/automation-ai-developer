# Complete Docker Deployment Guide

## 📋 Pre-Deployment Checklist

### 1. Generate Strong Passwords

```bash
# Generate SECRET_KEY (32 bytes)
openssl rand -base64 32

# Generate database password
openssl rand -base64 24

# Generate Redis password
openssl rand -base64 24

# Generate Grafana password
openssl rand -base64 16
```

### 2. Update Environment File

Edit `.env.production` with your generated passwords:

```bash
cp .env.production.example .env.production
nano .env.production  # or use your preferred editor
```

**Required changes:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `SECRET_KEY` - Generated secret key
- `DB_PASSWORD` - Strong database password
- `REDIS_PASSWORD` - Strong Redis password
- `GRAFANA_PASSWORD` - Strong Grafana password
- `DOMAIN` - Your domain name
- `SSL_EMAIL` - Your email for Let's Encrypt

### 3. Create Required Files

Create the following directory structure:

```bash
mkdir -p nginx
mkdir -p monitoring/grafana/{provisioning,dashboards}
mkdir -p certbot/{conf,www}
mkdir -p init-scripts
mkdir -p backups
```

### 4. File Checklist

Ensure you have:
- ✅ `docker-compose.yml` (improved version)
- ✅ `Dockerfile` (your existing file)
- ✅ `.env.production`
- ✅ `.dockerignore`
- ✅ `nginx/nginx.conf`
- ✅ `nginx/Dockerfile`
- ✅ `monitoring/prometheus.yml`
- ✅ `monitoring/alerts.yml`
- ✅ `deploy.sh` (make executable: `chmod +x deploy.sh`)

## 🚀 Quick Start Deployment

### Option 1: Using the Deploy Script (Recommended)

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy to production
./deploy.sh deploy
```

### Option 2: Manual Deployment

```bash
# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

# Build and start services
docker-compose build
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔒 Security Setup

### 1. SSL/TLS Configuration

#### Using Let's Encrypt (Recommended for Production)

```bash
# First deployment with self-signed certificates (already configured)
docker-compose up -d

# Once DNS is pointing to your server, get real certificates
./setup-ssl.sh

# Or manually:
docker-compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email your-email@domain.com \
  --agree-tos \
  --no-eff-email \
  -d yourdomain.com \
  -d www.yourdomain.com

# Reload Nginx
docker-compose exec nginx nginx -s reload
```

#### Certificate Auto-Renewal

The `certbot` service automatically renews certificates. To test renewal:

```bash
docker-compose run --rm certbot renew --dry-run
```

### 2. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 3. Security Headers Verification

Test your security headers:

```bash
curl -I https://yourdomain.com
```

Should include:
- `Strict-Transport-Security`
- `X-Frame-Options`
- `X-Content-Type-Options`
- `X-XSS-Protection`

## 📊 Monitoring Setup

### Access Monitoring Tools

- **Grafana**: `http://your-domain:3000`
  - Username: `admin`
  - Password: (from `.env.production`)
  
- **Prometheus**: `http://your-domain:9090`
  - No authentication by default (internal use)

### Set Up Grafana Dashboards

1. Log in to Grafana
2. Go to Dashboards → Import
3. Import these dashboard IDs:
   - **1860**: Node Exporter Full
   - **763**: Docker Prometheus Monitoring
   - **11074**: FastAPI Observability

### Configure Alerts

Edit `monitoring/alerts.yml` to customize:
- Alert thresholds
- Notification channels
- Alert severities

## 🔄 Common Operations

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ai-api

# Last 100 lines
docker-compose logs --tail=100 ai-api
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart ai-api

# Full restart (recreate containers)
docker-compose down && docker-compose up -d
```

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
./deploy.sh deploy

# Or manually
docker-compose build ai-api
docker-compose up -d ai-api
```

### Scale Services

```bash
# Scale API to 3 instances
docker-compose up -d --scale ai-api=3

# Note: You may need to adjust Nginx upstream configuration
```

### Execute Commands in Containers

```bash
# Open shell in API container
docker-compose exec ai-api bash

# Run database migrations
docker-compose exec ai-api alembic upgrade head

# Check Python packages
docker-compose exec ai-api pip list

# Run tests
docker-compose exec ai-api pytest
```

## 💾 Backup and Recovery

### Automatic Backups

Backups run daily at midnight by default. Configure in `.env.production`:

```bash
BACKUP_RETENTION_DAYS=7  # Keep backups for 7 days
```

### Manual Backup

```bash
# Create backup
./deploy.sh backup

# Or using Docker
docker-compose exec backup sh -c 'tar -czf /backups/manual_backup_$(date +%Y%m%d_%H%M%S).tar.gz /backup'
```

### Restore from Backup

```bash
# List available backups
ls -lh backups/

# Extract backup
tar -xzf backups/backup_20240101_120000.tar.gz -C /tmp/restore

# Stop services
docker-compose down

# Restore volumes (example for uploads)
docker volume rm ai_uploads
docker volume create ai_uploads
docker run --rm -v ai_uploads:/restore -v /tmp/restore/backup/uploads:/source alpine cp -r /source/. /restore/

# Start services
docker-compose up -d
```

## 🔍 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check container status
docker-compose ps

# Inspect container
docker inspect container-name

# Check resource usage
docker stats
```

### Database Connection Issues

```bash
# Check database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U postgres -d ai_api -c "SELECT 1;"

# Check environment variables
docker-compose exec ai-api env | grep DATABASE
```

### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli -a your-redis-password ping

# Check Redis logs
docker-compose logs redis
```

### Nginx Configuration Issues

```bash
# Test configuration
docker-compose exec nginx nginx -t

# Reload configuration
docker-compose exec nginx nginx -s reload

# Check error logs
docker-compose logs nginx
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a

# Remove old backups
find backups/ -name "*.tar.gz" -mtime +7 -delete

# Remove unused volumes
docker volume prune
```

### High Memory Usage

```bash
# Check memory usage per container
docker stats --no-stream

# Restart high-memory containers
docker-compose restart ai-api

# Adjust resource limits in docker-compose.yml
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Connect to database
docker-compose exec db psql -U postgres -d ai_api

-- Check slow queries
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Analyze tables
ANALYZE VERBOSE;

-- Vacuum database
VACUUM ANALYZE;
```

### Redis Optimization

```bash
# Check Redis memory
docker-compose exec redis redis-cli -a your-password info memory

# Check hit rate
docker-compose exec redis redis-cli -a your-password info stats | grep hit
```

### Application Optimization

```bash
# Adjust worker count in docker-compose.yml
# Current: --workers 4
# Formula: (2 * CPU cores) + 1

# Monitor worker performance
docker-compose logs ai-api | grep worker
```

## 🧹 Maintenance

### Weekly Tasks

```bash
# Check logs for errors
docker-compose logs --since 7d | grep -i error

# Review disk usage
df -h
docker system df

# Check backup integrity
ls -lh backups/ | tail -10

# Update containers
docker-compose pull
docker-compose up -d
```

### Monthly Tasks

```bash
# Review and rotate logs
docker-compose logs > logs/archive_$(date +%Y%m).log
docker-compose down && docker-compose up -d  # Truncates logs

# Database maintenance
docker-compose exec db psql -U postgres -d ai_api -c "VACUUM FULL ANALYZE;"

# Security updates
apt update && apt upgrade -y
docker-compose pull
docker-compose up -d --build
```

## 🆘 Emergency Procedures

### Complete System Failure

```bash
# 1. Stop all services
docker-compose down

# 2. Check system resources
df -h
free -h
top

# 3. Check Docker daemon
systemctl status docker

# 4. Restart Docker
systemctl restart docker

# 5. Start with fresh state
docker-compose up -d

# 6. Restore from backup if needed
./deploy.sh rollback
```

### Data Corruption

```bash
# 1. Stop services immediately
docker-compose down

# 2. Create emergency backup
tar -czf emergency_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  /var/lib/docker/volumes

# 3. Restore from last known good backup
# (see Restore from Backup section)

# 4. Verify data integrity
docker-compose exec db psql -U postgres -d ai_api -c "SELECT count(*) FROM users;"
```

## 📞 Support Contacts

### Monitoring Alerts

Configure alert notifications in `monitoring/alerts.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - 'alertmanager:9093'
```

### Log Aggregation

For centralized logging, consider:
- Loki (included in setup)
- ELK Stack
- Datadog
- CloudWatch

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com)
- [Docker Compose Documentation](https://docs.docker.com/compose)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment)
- [PostgreSQL Documentation](https://www.postgresql.org/docs)
- [Redis Documentation](https://redis.io/documentation)
- [Nginx Documentation](https://nginx.org/en/docs)
- [Prometheus Documentation](https://prometheus.io/docs)
- [Grafana Documentation](https://grafana.com/docs)

---

## 🎯 Next Steps

1. ✅ Generate all passwords
2. ✅ Update `.env.production`
3. ✅ Run `./deploy.sh deploy`
4. ✅ Configure SSL certificates
5. ✅ Set up monitoring dashboards
6. ✅ Configure backups
7. ✅ Test disaster recovery
8. ✅ Document your specific setup

**Your system is now production-ready! 🚀**