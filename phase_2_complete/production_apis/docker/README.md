# Docker Setup Instructions

## 📁 File Structure

Create this directory structure in your project:

```
your-project/
├── .dockerignore
├── .env.production
├── docker-compose.yml
├── Dockerfile (your existing file)
├── Makefile
├── requirements.txt (your existing file)
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
├── monitoring/
│   ├── prometheus.yml
│   ├── alerts.yml
│   ├── loki-config.yml
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasource.yml
├── certbot/
│   ├── conf/
│   └── www/
├── init-scripts/
└── backups/
```

## 🚀 Quick Setup Steps

### Step 1: Create Directories

```bash
# Create all required directories
mkdir -p nginx
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/dashboards
mkdir -p certbot/conf
mkdir -p certbot/www
mkdir -p init-scripts
mkdir -p backups
```

### Step 2: Copy All Files

Copy each file from the artifacts into the correct location:

1. **Root directory files:**
   - `.dockerignore`
   - `.env.production`
   - `docker-compose.yml`
   - `Makefile`

2. **nginx/ directory:**
   - `nginx/Dockerfile`
   - `nginx/nginx.conf`

3. **monitoring/ directory:**
   - `monitoring/prometheus.yml`
   - `monitoring/alerts.yml`
   - `monitoring/loki-config.yml`
   - `monitoring/grafana/provisioning/datasources/datasource.yml`

### Step 3: Generate Passwords

On Linux/Mac, generate secure passwords:

```bash
# SECRET_KEY (32 bytes)
openssl rand -base64 32

# DB_PASSWORD (24 bytes)
openssl rand -base64 24

# REDIS_PASSWORD (24 bytes)
openssl rand -base64 24

# GRAFANA_PASSWORD (16 bytes)
openssl rand -base64 16
```

On Windows (PowerShell):

```powershell
# Generate random passwords
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
```

### Step 4: Update .env.production

Edit `.env.production` and replace these values:

```bash
# Replace these:
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
SECRET_KEY=paste-generated-secret-key-here
DB_PASSWORD=paste-generated-db-password-here
REDIS_PASSWORD=paste-generated-redis-password-here
GRAFANA_PASSWORD=paste-generated-grafana-password-here

# Update these:
DOMAIN=yourdomain.com
SSL_EMAIL=admin@yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Step 5: Verify File Structure

Run this command to verify all files are in place:

```bash
ls -R
```

You should see:
- All directories created
- All configuration files in place
- .env.production with your passwords

### Step 6: Deploy

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

Or use the Makefile:

```bash
make build
make up
make status
make logs
```

## 🔍 Verify Deployment

### Check Services are Running

```bash
docker-compose ps
```

All services should show "Up" and "healthy" status.

### Test API

```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy"}`

### Access Monitoring

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: (from .env.production)

- **Prometheus**: http://localhost:9090

## 🔒 SSL Setup (After Initial Deployment)

If you have a domain pointing to your server:

```bash
# Get SSL certificate from Let's Encrypt
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

## 📊 Common Commands

```bash
# View all logs
make logs

# View specific service logs
docker-compose logs -f ai-api

# Restart services
make restart

# Check health
make health

# Create backup
make backup

# Stop all services
make down

# Clean up (WARNING: removes all data)
make clean
```

## 🆘 Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose logs service-name

# Check if port is already in use
netstat -tulpn | grep :8000
```

### Database connection error

```bash
# Verify database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U postgres -d ai_api
```

### Redis connection error

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis
docker-compose exec redis redis-cli -a your-redis-password ping
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a

# Remove old backups
find backups/ -name "*.tar.gz" -mtime +7 -delete
```

## 📝 Next Steps

1. ✅ Configure your domain's DNS to point to your server
2. ✅ Set up SSL certificates with Let's Encrypt
3. ✅ Configure firewall to allow only ports 80, 443, and 22
4. ✅ Set up automated backups
5. ✅ Configure Grafana dashboards
6. ✅ Set up monitoring alerts
7. ✅ Test disaster recovery procedures

## 🔐 Security Checklist

- [ ] All passwords are strong and unique
- [ ] Database port (5432) is NOT exposed to host
- [ ] Redis port (6379) is NOT exposed to host
- [ ] SSL/TLS is configured and working
- [ ] Firewall only allows necessary ports
- [ ] Grafana password has been changed
- [ ] CORS_ORIGINS is set to your actual domains
- [ ] Backups are running and tested

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify environment variables: `docker-compose config`
3. Check service health: `make health`
4. Review the troubleshooting section above

