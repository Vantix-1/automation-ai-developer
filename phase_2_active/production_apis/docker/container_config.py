"""
Docker Container Configuration and Management
"""
import os
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class DockerManager:
    """Manage Docker containers and configurations"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.docker_compose_path = self.project_dir / "docker-compose.yml"
        self.dockerfile_path = self.project_dir / "Dockerfile"
        self.env_file = self.project_dir / ".env"
        
    def check_docker_installed(self) -> bool:
        """Check if Docker is installed and running"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Docker: {result.stdout.strip()}")
            
            # Check Docker Compose
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ Docker Compose: {result.stdout.strip()}")
            else:
                # Try docker compose v2
                result = subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✅ Docker Compose V2: {result.stdout.strip()}")
                else:
                    print("⚠️  Docker Compose not found")
                    return False
            
            # Check if Docker daemon is running
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("❌ Docker daemon not running")
                return False
            
            return True
            
        except FileNotFoundError:
            print("❌ Docker not installed")
            return False
    
    def build_image(self, tag: str = "ai-api:latest", build_args: Dict[str, str] = None):
        """Build Docker image"""
        print(f"🔨 Building Docker image: {tag}")
        
        cmd = ["docker", "build", "-t", tag, "."]
        
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Image built successfully")
                return True
            else:
                print(f"❌ Build failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False
    
    def run_container(self, image: str, name: str, ports: Dict[str, str] = None,
                     volumes: Dict[str, str] = None, env_vars: Dict[str, str] = None):
        """Run a Docker container"""
        print(f"🚀 Running container: {name}")
        
        cmd = ["docker", "run", "-d", "--name", name]
        
        # Add ports
        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add volumes
        if volumes:
            for host_path, container_path in volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Add image
        cmd.append(image)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Container {name} started")
                return True
            else:
                print(f"❌ Failed to start container: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def docker_compose_up(self, services: List[str] = None, detach: bool = True):
        """Start services with Docker Compose"""
        print("🚀 Starting services with Docker Compose...")
        
        cmd = ["docker-compose", "up"]
        
        if detach:
            cmd.append("-d")
        
        if services:
            cmd.extend(services)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Services started successfully")
                return True
            else:
                print(f"❌ Failed to start services: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def docker_compose_down(self, remove_volumes: bool = False):
        """Stop and remove Docker Compose services"""
        print("🛑 Stopping services...")
        
        cmd = ["docker-compose", "down"]
        
        if remove_volumes:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Services stopped")
                return True
            else:
                print(f"❌ Failed to stop services: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def list_containers(self, all_containers: bool = False):
        """List Docker containers"""
        print("📋 Listing containers...")
        
        cmd = ["docker", "ps"]
        if all_containers:
            cmd.append("-a")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            return result.stdout
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_container_logs(self, container_name: str, tail: int = 100, follow: bool = False):
        """Get container logs"""
        print(f"📄 Logs for {container_name}:")
        
        cmd = ["docker", "logs"]
        
        if tail:
            cmd.extend(["--tail", str(tail)])
        
        if follow:
            cmd.append("-f")
        
        cmd.append(container_name)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            return result.stdout
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def exec_command(self, container_name: str, command: str):
        """Execute command in running container"""
        print(f"💻 Executing in {container_name}: {command}")
        
        cmd = ["docker", "exec", container_name, "sh", "-c", command]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"⚠️  Stderr: {result.stderr}")
            return result.stdout
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def check_health(self, container_name: str) -> Dict[str, Any]:
        """Check container health status"""
        print(f"🏥 Health check for {container_name}")
        
        cmd = ["docker", "inspect", "--format='{{json .State}}'", container_name]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                state = json.loads(result.stdout.strip().strip("'"))
                print(f"Status: {state.get('Status', 'unknown')}")
                print(f"Running: {state.get('Running', False)}")
                print(f"ExitCode: {state.get('ExitCode', 0)}")
                return state
            else:
                print(f"❌ Failed to inspect container: {result.stderr}")
                return {}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {}
    
    def cleanup(self, remove_images: bool = False):
        """Clean up Docker resources"""
        print("🧹 Cleaning up Docker resources...")
        
        # Stop all containers
        cmd = ["docker", "ps", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            container_ids = result.stdout.strip().split()
            for cid in container_ids:
                subprocess.run(["docker", "stop", cid], capture_output=True)
                subprocess.run(["docker", "rm", cid], capture_output=True)
        
        # Remove dangling images
        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
        
        # Remove volumes
        subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        
        # Remove networks
        subprocess.run(["docker", "network", "prune", "-f"], capture_output=True)
        
        if remove_images:
            # Remove all images
            subprocess.run(["docker", "rmi", "-f", "$(docker images -q)"], 
                          shell=True, capture_output=True)
        
        print("✅ Cleanup complete")

class EnvironmentConfig:
    """Manage environment configuration for Docker"""
    
    def __init__(self):
        self.configs = {
            "development": {
                "OPENAI_API_KEY": "",
                "DATABASE_URL": "postgresql://postgres:password@localhost:5432/ai_api_dev",
                "REDIS_URL": "redis://localhost:6379/0",
                "ENVIRONMENT": "development",
                "LOG_LEVEL": "debug",
                "DEBUG": "true"
            },
            "production": {
                "OPENAI_API_KEY": "",
                "SECRET_KEY": "change-this-in-production",
                "DATABASE_URL": "postgresql://postgres:strong-password@db:5432/ai_api",
                "REDIS_URL": "redis://redis:6379/0",
                "ENVIRONMENT": "production",
                "LOG_LEVEL": "info",
                "DEBUG": "false",
                "CORS_ORIGINS": "https://yourdomain.com"
            },
            "testing": {
                "OPENAI_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://postgres:password@localhost:5434/ai_api_test",
                "REDIS_URL": "redis://localhost:6380/0",
                "ENVIRONMENT": "testing",
                "LOG_LEVEL": "warning",
                "DEBUG": "false"
            }
        }
    
    def generate_env_file(self, environment: str = "development", output_file: str = ".env"):
        """Generate .env file for specific environment"""
        if environment not in self.configs:
            raise ValueError(f"Unknown environment: {environment}")
        
        config = self.configs[environment]
        
        env_content = f"# {environment.upper()} Environment Configuration\n"
        env_content += f"# Generated: {datetime.now().isoformat()}\n\n"
        
        for key, value in config.items():
            env_content += f"{key}={value}\n"
        
        # Add additional production recommendations
        if environment == "production":
            env_content += "\n# Production Security Recommendations\n"
            env_content += "# Set these in your production environment:\n"
            env_content += "# SECRET_KEY=your-very-secure-random-key-here\n"
            env_content += "# DB_PASSWORD=strong-database-password\n"
            env_content += "# REDIS_PASSWORD=strong-redis-password\n"
            env_content += "# GRAFANA_PASSWORD=strong-grafana-password\n"
        
        with open(output_file, "w") as f:
            f.write(env_content)
        
        print(f"✅ Generated {environment} environment file: {output_file}")
        return env_content

class DockerOptimizer:
    """Optimize Docker configurations"""
    
    @staticmethod
    def analyze_dockerfile(dockerfile_path: str) -> Dict[str, Any]:
        """Analyze Dockerfile for optimization opportunities"""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        analysis = {
            "total_lines": len(content.splitlines()),
            "has_multi_stage": "FROM" in content and content.count("FROM") > 1,
            "has_non_root_user": "USER" in content,
            "has_healthcheck": "HEALTHCHECK" in content,
            "has_workdir": "WORKDIR" in content,
            "has_expose": "EXPOSE" in content,
            "recommendations": []
        }
        
        # Check for common optimizations
        if not analysis["has_multi_stage"]:
            analysis["recommendations"].append(
                "Consider using multi-stage builds to reduce image size"
            )
        
        if not analysis["has_non_root_user"]:
            analysis["recommendations"].append(
                "Consider running as non-root user for security"
            )
        
        if not analysis["has_healthcheck"]:
            analysis["recommendations"].append(
                "Add HEALTHCHECK instruction for container health monitoring"
            )
        
        return analysis
    
    @staticmethod
    def optimize_dockerfile(input_path: str, output_path: str):
        """Generate optimized Dockerfile"""
        template = """# Optimized Dockerfile for Python FastAPI Application
# Multi-stage build for minimal image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies if needed
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy application
COPY . .

# Set environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(output_path, 'w') as f:
            f.write(template)
        
        print(f"✅ Optimized Dockerfile generated: {output_path}")

# Example usage
def main():
    """Main function demonstrating Docker management"""
    print("=" * 60)
    print("🐳 Docker Container Configuration Manager")
    print("=" * 60)
    
    # Initialize manager
    manager = DockerManager()
    
    # Check Docker installation
    if not manager.check_docker_installed():
        print("Please install Docker and Docker Compose first")
        return
    
    # Generate environment files
    env_config = EnvironmentConfig()
    print("\n1. Generating environment files...")
    env_config.generate_env_file("development", ".env.development")
    env_config.generate_env_file("production", ".env.production")
    
    # Analyze Dockerfile
    print("\n2. Analyzing Dockerfile...")
    if manager.dockerfile_path.exists():
        optimizer = DockerOptimizer()
        analysis = optimizer.analyze_dockerfile(str(manager.dockerfile_path))
        
        print(f"   Total lines: {analysis['total_lines']}")
        print(f"   Multi-stage: {analysis['has_multi_stage']}")
        print(f"   Non-root user: {analysis['has_non_root_user']}")
        print(f"   Healthcheck: {analysis['has_healthcheck']}")
        
        if analysis["recommendations"]:
            print("\n   Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"   • {rec}")
    
    # List current containers
    print("\n3. Current Docker containers:")
    manager.list_containers()
    
    # Show available commands
    print("\n4. Available Commands:")
    print("   • Build image: manager.build_image('ai-api:latest')")
    print("   • Start services: manager.docker_compose_up()")
    print("   • Stop services: manager.docker_compose_down()")
    print("   • View logs: manager.get_container_logs('container-name')")
    print("   • Cleanup: manager.cleanup()")
    
    # Create docker-compose.override.yml for development
    override_config = {
        "version": "3.8",
        "services": {
            "ai-api": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.dev"
                },
                "volumes": [".:/app"],
                "environment": [
                    "ENVIRONMENT=development",
                    "DEBUG=true"
                ],
                "ports": ["8000:8000"]
            }
        }
    }
    
    override_path = manager.project_dir / "docker-compose.override.yml"
    with open(override_path, "w") as f:
        yaml.dump(override_config, f)
    
    print(f"\n✅ Created development override: {override_path}")
    print("\n🚀 Ready to deploy! Use 'docker-compose up' to start services.")

if __name__ == "__main__":
    main()