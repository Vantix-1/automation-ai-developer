.PHONY: help build up down restart logs clean deploy status backup health

help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs"
	@echo "  make status     - Show service status"
	@echo "  make backup     - Create backup"
	@echo "  make clean      - Clean up resources"
	@echo "  make health     - Check service health"

build:
	docker-compose build --no-cache

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f --tail=100

status:
	docker-compose ps
	@echo ""
	@docker stats --no-stream

backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@docker run --rm \
		-v ai_uploads:/backup/uploads:ro \
		-v ai_data:/backup/data:ro \
		-v postgres_data:/backup/postgres:ro \
		-v redis_data:/backup/redis:ro \
		-v $(PWD)/backups:/backups \
		alpine tar -czf /backups/manual_backup_$$(date +%Y%m%d_%H%M%S).tar.gz /backup
	@echo "Backup completed!"

clean:
	docker-compose down -v
	docker system prune -af
	@echo "Warning: This removes all volumes and data!"

health:
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health && echo "✅ API: Healthy" || echo "❌ API: Down"
	@curl -f http://localhost:9090/-/healthy && echo "✅ Prometheus: Healthy" || echo "❌ Prometheus: Down"
	@curl -f http://localhost:3000/api/health && echo "✅ Grafana: Healthy" || echo "❌ Grafana: Down"