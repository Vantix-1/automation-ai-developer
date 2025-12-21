# Docker Commands Cheat Sheet

## Basic Docker Commands

### Container Management

```bash
# Run a container
docker run -d --name my-container -p 8000:8000 my-image:latest

# Start/stop containers
docker start my-container
docker stop my-container
docker restart my-container

# Remove container
docker rm my-container
docker rm -f my-container  # Force remove running container

# List containers
docker ps                    # Running containers
docker ps -a                 # All containers
docker ps -q                 # Only container IDs

# View container logs
docker logs my-container
docker logs -f my-container  # Follow logs
docker logs --tail 100 my-container  # Last 100 lines

# Execute command in container
docker exec -it my-container bash
docker exec my-container python --version
```

### Image Management

```bash
# Build image
docker build -t my-image:latest .
docker build -t my-image:1.0.0 -f Dockerfile.prod .

# List images
docker images
docker image ls

# Remove image
docker rmi my-image:latest
docker rmi -f $(docker images -q)  # Remove all images

# Push/pull images
docker push my-registry/my-image:latest
docker pull my-registry/my-image:latest

# Save/load images
docker save -o my-image.tar my-image:latest
docker load -i my-image.tar
```

## Docker Compose Commands

```bash
# Start services
docker-compose up
docker-compose up -d          # Detached mode
docker-compose up --build     # Build then start

# Stop services
docker-compose down
docker-compose down -v        # Remove volumes
docker-compose down --rmi all # Remove images

# View logs
docker-compose logs
docker-compose logs -f
docker-compose logs service-name

# Manage specific services
docker-compose start service-name
docker-compose stop service-name
docker-compose restart service-name

# Rebuild service
docker-compose build service-name

# View running services
docker-compose ps
```

## Production Docker Commands

### Monitoring and Inspection

```bash
# Container stats
docker stats
docker stats --no-stream

# Resource usage
docker system df
docker system df -v

# Inspect container
docker inspect my-container
docker inspect --format='{{.State.Status}}' my-container

# Health check
docker inspect --format='{{json .State.Health}}' my-container
```

### Networking

```bash
# List networks
docker network ls
docker network inspect bridge

# Create network
docker network create my-network

# Connect container to network
docker network connect my-network my-container

# Disconnect from network
docker network disconnect my-network my-container
```

### Volumes

```bash
# List volumes
docker volume ls
docker volume inspect volume-name

# Create volume
docker volume create my-volume

# Remove volume
docker volume rm my-volume
docker volume prune  # Remove unused volumes
```

## Docker Swarm Commands (For Production Clusters)

```bash
# Initialize swarm
docker swarm init
docker swarm init --advertise-addr <MANAGER-IP>

# Join swarm
docker swarm join --token <TOKEN> <MANAGER-IP>:2377

# Deploy stack
docker stack deploy -c docker-compose.yml my-stack

# List services
docker service ls
docker stack services my-stack

# Scale service
docker service scale my-service=5

# Update service
docker service update --image new-image:latest my-service
```

## Security Commands

```bash
# Scan image for vulnerabilities
docker scan my-image:latest

# Check container security
docker container diff my-container

# Run with security constraints
docker run --read-only --security-opt="no-new-privileges" my-image

# User namespace
docker run --userns-remap="default" my-image
```

## Cleanup Commands

```bash
# Clean all unused data
docker system prune
docker system prune -a  # Remove all unused images

# Clean specific resources
docker container prune
docker image prune
docker volume prune
docker network prune

# Remove stopped containers
docker container prune -f

# Remove dangling images
docker image prune -f
```

## Dockerfile Best Practices Commands

```bash
# Build with specific target in multi-stage build
docker build --target builder -t my-app:builder .

# Build with build arguments
docker build --build-arg VERSION=1.0.0 -t my-app:1.0.0 .

# Optimize build cache
docker build --no-cache -t my-app:latest .
```

## Docker Registry Commands

```bash
# Login to registry
docker login
docker login my-registry.com

# Tag image for registry
docker tag my-image:latest my-registry.com/my-image:latest

# Push to registry
docker push my-registry.com/my-image:latest

# Pull from registry
docker pull my-registry.com/my-image:latest
```

## Useful Aliases for ~/.bashrc or ~/.zshrc

```bash
# Docker aliases
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dlog='docker logs'
alias dlogf='docker logs -f'
alias dexec='docker exec -it'
alias dstop='docker stop'
alias drm='docker rm'
alias drmi='docker rmi'
alias dcp='docker-compose'
alias dcup='docker-compose up'
alias dcupd='docker-compose up -d'
alias dcdown='docker-compose down'
alias dcrestart='docker-compose restart'
alias dclog='docker-compose logs'
alias dclogf='docker-compose logs -f'
alias dcbuild='docker-compose build'

# Docker cleanup
alias dclean='docker system prune -af'
alias dcleanv='docker volume prune -f'
alias dcleani='docker image prune -f'
alias dcleanc='docker container prune -f'
```

## Common Production Scenarios

### 1. Database Backup

```bash
# Backup PostgreSQL
docker exec postgres-container pg_dump -U postgres database_name > backup.sql

# Backup with volume
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/postgres_backup.tar.gz /data
```

### 2. Log Rotation

```bash
# Rotate Docker logs (configure in /etc/docker/daemon.json)
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### 3. Resource Limits

```bash
# Run with resource constraints
docker run -d \
    --name my-app \
    --memory="512m" \
    --memory-swap="1g" \
    --cpus="1.5" \
    my-image:latest
```

### 4. Health Check Script

```bash
#!/bin/bash
# healthcheck.sh
set -e

# Check if API is responding
curl -f http://localhost:8000/health || exit 1

# Check database connection
docker exec my-db pg_isready -U postgres || exit 1

# Check Redis
docker exec my-redis redis-cli ping | grep PONG || exit 1

exit 0
```

## Troubleshooting Commands

```bash
# Check Docker daemon
sudo systemctl status docker
journalctl -u docker.service

# Debug build
docker build --progress=plain -t my-app .

# Check container processes
docker top my-container

# Copy files to/from container
docker cp my-container:/app/logs/app.log ./app.log
docker cp ./config.json my-container:/app/config.json

# Inspect container ports
docker port my-container
```

## Performance Monitoring

```bash
# Monitor in real-time
watch -n 2 'docker stats --no-stream'

# Resource usage per container
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Monitor with cAdvisor
docker run -d \
  --name=cadvisor \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:rw \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  google/cadvisor:latest
```

## Quick Reference Card

### Build & Run
```
docker build -t tag .                # Build image
docker run -d -p 80:80 tag           # Run container
docker-compose up -d                 # Start stack
```

### Management
```
docker ps                           # List containers
docker logs container               # View logs
docker exec -it container bash      # Enter container
docker stop container               # Stop container
docker rm container                 # Remove container
```

### Cleanup
```
docker system prune                # Clean everything
docker image prune                 # Clean images
docker volume prune                # Clean volumes
```

### Debugging
```
docker inspect container           # Detailed info
docker stats                       # Resource usage
docker top container               # Container processes
```