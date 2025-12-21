# 🐳 Docker Containerization

## 📋 Overview
This section covers containerizing AI applications with Docker for production deployment. You'll learn Docker fundamentals, multi-container orchestration, and production optimization techniques.

## 📁 File Structure
```
day_43_45_docker/
├── Dockerfile              # Production Docker configuration
├── docker-compose.yml      # Multi-service orchestration
├── container_config.py     # Docker management utilities
└── docker_commands.md      # Comprehensive Docker cheat sheet
```

## 🛠️ Setup & Installation

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git (for version control)

### Installation Verification
```bash
# Check Docker installation
docker --version
docker-compose --version  # or docker compose version

# Verify Docker daemon is running
docker info

# Run test container
docker run hello-world
```

## 📚 Learning Objectives

### Day 43: Docker Fundamentals
- ✅ Understand Docker architecture and components
- ✅ Create Dockerfile for Python applications
- ✅ Build and run Docker containers
- ✅ Manage container lifecycle

### Day 44: Multi-container Applications
- ✅ Use Docker Compose for orchestration
- ✅ Set up multi-service applications
- ✅ Configure networking between containers
- ✅ Manage volumes for data persistence

### Day 45: Production Optimization
- ✅ Implement multi-stage builds
- ✅ Optimize Docker images for size and security
- ✅ Add health checks and monitoring
- ✅ Create production deployment configurations

## 🚦 Quick Start

### 1. Build Docker Image
```bash
# Build the production image
docker build -t ai-api:latest .

# Build with specific Dockerfile
docker build -t ai-api:prod -f Dockerfile.prod .

# View built images
docker images
```

### 2. Run Container
```bash
# Run a single container
docker run -d --name ai-api -p 8000:8000 ai-api:latest

# Run with environment variables
docker run -d --name ai-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e ENVIRONMENT=production \
  ai-api:latest

# View running containers
docker ps
```

### 3. Use Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale specific service
docker-compose up -d --scale ai-api=3

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### 4. Use Management Utilities
```bash
# Run container configuration manager
python container_config.py

# Check Docker health
python container_config.py --check

# Clean up Docker resources
python container_config.py --cleanup
```

## 🔧 Key Features

### 1. Production Dockerfile
- Multi-stage builds for minimal image size
- Non-root user for security
- Health checks for container monitoring
- Optimized layer caching
- Environment variable configuration

### 2. Docker Compose Setup
- Multi-service orchestration (API, DB, Redis, Nginx)
- Service dependencies and startup order
- Volume management for data persistence
- Network configuration for service communication
- Resource limits and constraints

### 3. Management Utilities
- Docker environment validation
- Image building and tagging
- Container lifecycle management
- Resource cleanup and optimization
- Configuration generation

### 4. Production Optimizations
- Security scanning and vulnerability management
- Performance optimization
- Logging and monitoring integration
- Backup and recovery procedures
- Scaling configurations

## 📖 Code Examples

### Basic Dockerfile
```dockerfile
# Multi-stage build for Python application
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Service
```yaml
services:
  ai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/ai_api
    volumes:
      - ./uploads:/app/uploads
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Python Docker Management
```python
from container_config import DockerManager

manager = DockerManager()

# Build image
manager.build_image(tag="ai-api:v1.0")

# Run container
manager.run_container(
    image="ai-api:v1.0",
    name="ai-api-production",
    ports={"8000": "8000"},
    env_vars={"ENVIRONMENT": "production"}
)

# List containers
manager.list_containers(all_containers=True)
```

## 🧪 Testing & Validation

### 1. Test Docker Build
```bash
# Build and test locally
docker build -t test-image .
docker run --rm test-image python --version

# Scan for vulnerabilities
docker scan test-image

# Check image size
docker images test-image
```

### 2. Test Docker Compose
```bash
# Test configuration
docker-compose config

# Build without cache
docker-compose build --no-cache

# Run tests in container
docker-compose run ai-api pytest
```

### 3. Validate Production Readiness
```bash
# Check security best practices
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  goodwithtech/dockerfile-checker Dockerfile

# Lint Dockerfile
docker run --rm -i hadolint/hadolint < Dockerfile
```

## 🔍 Configuration Reference

### Dockerfile Directives
| Directive | Purpose | Example |
|-----------|---------|---------|
| FROM | Base image | `FROM python:3.11-slim` |
| WORKDIR | Working directory | `WORKDIR /app` |
| COPY | Copy files | `COPY . .` |
| RUN | Execute commands | `RUN pip install -r requirements.txt` |
| ENV | Environment variables | `ENV PYTHONUNBUFFERED=1` |
| EXPOSE | Port exposure | `EXPOSE 8000` |
| CMD | Default command | `CMD ["python", "app.py"]` |
| HEALTHCHECK | Health check | `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health` |

### Docker Compose Configuration
| Service | Purpose | Configuration |
|---------|---------|---------------|
| ai-api | Main API service | Build from Dockerfile, port 8000 |
| db | PostgreSQL database | PostgreSQL 15 with volume |
| redis | Redis cache | Redis 7 with authentication |
| nginx | Reverse proxy | Nginx with SSL termination |
| prometheus | Metrics | Prometheus for monitoring |
| grafana | Visualization | Grafana dashboards |

### Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| OPENAI_API_KEY | OpenAI API key | `sk-...` |
| DATABASE_URL | Database connection | `postgresql://user:pass@host/db` |
| REDIS_URL | Redis connection | `redis://:pass@host:6379/0` |
| SECRET_KEY | JWT secret | `your-secret-key` |
| ENVIRONMENT | Runtime environment | `production` |

## 🎯 Best Practices Implemented

### Security
- **Non-root User:** Run containers as non-root user
- **Minimal Base Images:** Use slim/alpine images
- **Regular Updates:** Keep base images updated
- **Secrets Management:** Use Docker secrets or environment files
- **Network Segmentation:** Isolate services with networks

### Performance
- **Multi-stage Builds:** Reduce final image size
- **Layer Caching:** Optimize build times
- **Resource Limits:** Set CPU and memory limits
- **Health Checks:** Ensure service availability
- **Logging:** Structured JSON logging

### Operations
- **Version Tagging:** Semantic versioning for images
- **Rollback Strategy:** Easy rollback with version tags
- **Backup Procedures:** Regular volume backups
- **Monitoring Integration:** Prometheus metrics
- **Disaster Recovery:** Backup and restore procedures

## 📈 Production Deployment

### 1. Build Pipeline
```yaml
# GitHub Actions example
name: Build and Push
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t ${{ secrets.REGISTRY }}/ai-api:${{ github.sha }} .
      - name: Push to Registry
        run: docker push ${{ secrets.REGISTRY }}/ai-api:${{ github.sha }}
```

### 2. Deployment Script
```bash
#!/bin/bash
# deploy.sh
set -e

# Pull latest image
docker pull registry.example.com/ai-api:latest

# Update services
docker-compose pull
docker-compose up -d --remove-orphans

# Run migrations
docker-compose exec ai-api python manage.py migrate

# Cleanup old images
docker image prune -f
```

### 3. Monitoring Setup
```bash
# Check container health
docker inspect --format='{{json .State.Health}}' ai-api

# View resource usage
docker stats ai-api

# View logs
docker logs -f ai-api

# Execute commands
docker exec -it ai-api python manage.py shell
```

## 🚨 Troubleshooting

### Common Issues

**Port conflicts**
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Change port mapping
docker run -p 8001:8000 ai-api:latest
```

**Permission issues**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./uploads

# Or use named volumes
docker volume create ai_uploads
```

**Build cache issues**
```bash
# Clear build cache
docker builder prune

# Build without cache
docker build --no-cache -t ai-api:latest .
```

**Container won't start**
```bash
# Check logs
docker logs ai-api

# Run with interactive shell
docker run -it --rm ai-api:latest sh
```

### Debugging Commands
```bash
# Inspect container
docker inspect ai-api

# Check processes in container
docker top ai-api

# Copy files from container
docker cp ai-api:/app/logs/app.log ./app.log

# View resource usage
docker stats ai-api db redis
```

## 📚 Resources
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [OCI Image Specification](https://github.com/opencontainers/image-spec)

## 🏆 Completion Checklist
- [ ] Created production Dockerfile
- [ ] Implemented multi-stage builds
- [ ] Set up Docker Compose orchestration
- [ ] Added health checks and monitoring
- [ ] Implemented security best practices
- [ ] Created management utilities
- [ ] Optimized for production deployment
- [ ] Added backup and recovery procedures