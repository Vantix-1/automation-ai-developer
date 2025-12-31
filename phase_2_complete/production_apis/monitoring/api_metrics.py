"""
Advanced API Metrics Collection and Analysis
"""
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import asyncio
import json
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, Summary
import redis
import psutil

class APIMetricsCollector:
    """Collect and analyze API metrics"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
        # In-memory storage (use Redis in production)
        self.metrics = {
            "requests": deque(maxlen=10000),
            "errors": deque(maxlen=1000),
            "performance": deque(maxlen=1000),
            "users": defaultdict(lambda: defaultdict(int))
        }
        
        # Prometheus metrics
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
        
        self.active_requests = Gauge(
            'api_active_requests',
            'Active API requests'
        )
        
        self.error_counter = Counter(
            'api_errors_total',
            'Total API errors',
            ['method', 'endpoint', 'error_type']
        )
        
        self.cache_hits = Counter(
            'api_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            'api_cache_misses_total',
            'Total cache misses'
        )
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record an API request"""
        timestamp = datetime.now()
        
        # Store in memory
        self.metrics["requests"].append({
            "timestamp": timestamp,
            "method": method,
            "endpoint": endpoint,
            "status": status,
            "duration": duration
        })
        
        # Update Prometheus metrics
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Store in Redis if available
        if self.redis:
            key = f"metrics:requests:{timestamp.strftime('%Y%m%d%H%M%S')}"
            self.redis.hset(key, mapping={
                "method": method,
                "endpoint": endpoint,
                "status": str(status),
                "duration": str(duration),
                "timestamp": timestamp.isoformat()
            })
            self.redis.expire(key, 86400)  # Expire after 24 hours
    
    def record_error(self, method: str, endpoint: str, error_type: str, error_msg: str):
        """Record an API error"""
        timestamp = datetime.now()
        
        # Store in memory
        self.metrics["errors"].append({
            "timestamp": timestamp,
            "method": method,
            "endpoint": endpoint,
            "error_type": error_type,
            "error_message": error_msg
        })
        
        # Update Prometheus
        self.error_counter.labels(
            method=method,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits.inc()
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses.inc()
    
    def start_request(self):
        """Mark request start"""
        self.active_requests.inc()
    
    def end_request(self):
        """Mark request end"""
        self.active_requests.dec()
    
    def get_summary(self, timeframe_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for timeframe"""
        cutoff = datetime.now() - timedelta(minutes=timeframe_minutes)
        
        # Filter recent requests
        recent_requests = [
            req for req in self.metrics["requests"]
            if req["timestamp"] > cutoff
        ]
        
        if not recent_requests:
            return {
                "total_requests": 0,
                "avg_duration": 0,
                "error_rate": 0,
                "endpoints": {}
            }
        
        durations = [req["duration"] for req in recent_requests]
        
        # Calculate endpoint statistics
        endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "durations": [],
            "errors": 0
        })
        
        for req in recent_requests:
            endpoint = req["endpoint"]
            endpoint_stats[endpoint]["count"] += 1
            endpoint_stats[endpoint]["durations"].append(req["duration"])
        
        # Calculate error rate
        recent_errors = [
            err for err in self.metrics["errors"]
            if err["timestamp"] > cutoff
        ]
        
        error_rate = len(recent_errors) / len(recent_requests) * 100
        
        return {
            "total_requests": len(recent_requests),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else 0,
            "p99_duration": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else 0,
            "error_rate": error_rate,
            "endpoints": {
                endpoint: {
                    "count": stats["count"],
                    "avg_duration": statistics.mean(stats["durations"]) if stats["durations"] else 0,
                    "p95_duration": statistics.quantiles(stats["durations"], n=20)[18] if len(stats["durations"]) >= 20 else 0
                }
                for endpoint, stats in endpoint_stats.items()
            }
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        trends = {
            "timestamps": [],
            "avg_durations": [],
            "request_counts": [],
            "error_rates": []
        }
        
        now = datetime.now()
        
        for i in range(hours * 12):  # 5-minute intervals
            interval_end = now - timedelta(minutes=i * 5)
            interval_start = interval_end - timedelta(minutes=5)
            
            # Get requests in this interval
            interval_requests = [
                req for req in self.metrics["requests"]
                if interval_start <= req["timestamp"] <= interval_end
            ]
            
            interval_errors = [
                err for err in self.metrics["errors"]
                if interval_start <= err["timestamp"] <= interval_end
            ]
            
            if interval_requests:
                durations = [req["duration"] for req in interval_requests]
                avg_duration = statistics.mean(durations)
                error_rate = len(interval_errors) / len(interval_requests) * 100
            else:
                avg_duration = 0
                error_rate = 0
            
            trends["timestamps"].append(interval_end.isoformat())
            trends["avg_durations"].append(avg_duration)
            trends["request_counts"].append(len(interval_requests))
            trends["error_rates"].append(error_rate)
        
        return trends
    
    def get_top_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top endpoints by request count"""
        endpoint_counts = defaultdict(int)
        
        for req in self.metrics["requests"]:
            endpoint_counts[req["endpoint"]] += 1
        
        sorted_endpoints = sorted(
            endpoint_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {"endpoint": endpoint, "count": count}
            for endpoint, count in sorted_endpoints
        ]
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_analysis = defaultdict(lambda: defaultdict(int))
        
        for error in self.metrics["errors"]:
            error_analysis[error["error_type"]][error["endpoint"]] += 1
        
        # Calculate error rates by endpoint
        endpoint_requests = defaultdict(int)
        for req in self.metrics["requests"]:
            endpoint_requests[req["endpoint"]] += 1
        
        endpoint_error_rates = {}
        for error_type, endpoints in error_analysis.items():
            for endpoint, error_count in endpoints.items():
                total_requests = endpoint_requests.get(endpoint, 1)
                error_rate = error_count / total_requests * 100
                
                if endpoint not in endpoint_error_rates:
                    endpoint_error_rates[endpoint] = {}
                
                endpoint_error_rates[endpoint][error_type] = {
                    "count": error_count,
                    "rate": error_rate
                }
        
        return {
            "total_errors": len(self.metrics["errors"]),
            "by_type": dict(error_analysis),
            "by_endpoint": endpoint_error_rates
        }

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "auth": {"requests": 10, "window": 60},      # 10 auth attempts per minute
            "upload": {"requests": 5, "window": 300},    # 5 uploads per 5 minutes
            "api": {"requests": 1000, "window": 3600},   # 1000 requests per hour
        }
    
    def is_rate_limited(self, client_id: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if client is rate limited"""
        limit_key = self._get_limit_key(endpoint)
        limit = self.limits.get(limit_key, self.limits["default"])
        
        window = limit["window"]
        max_requests = limit["requests"]
        
        if self.redis:
            # Use Redis for distributed rate limiting
            key = f"ratelimit:{client_id}:{limit_key}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            now = int(time.time())
            
            # Remove old requests
            pipe.zremrangebyscore(key, 0, now - window)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window)
            
            results = pipe.execute()
            current_requests = results[1]
            
            remaining = max(0, max_requests - current_requests)
            reset_time = now + window
            
            return current_requests >= max_requests, {
                "limit": max_requests,
                "remaining": remaining,
                "reset": reset_time,
                "window": window
            }
        else:
            # In-memory rate limiting (for single instance)
            current_time = time.time()
            key = f"{client_id}:{limit_key}"
            
            if not hasattr(self, 'request_timestamps'):
                self.request_timestamps = defaultdict(list)
            
            # Remove old timestamps
            timestamps = self.request_timestamps[key]
            timestamps[:] = [ts for ts in timestamps if ts > current_time - window]
            
            remaining = max(0, max_requests - len(timestamps))
            reset_time = current_time + window
            
            if len(timestamps) >= max_requests:
                return True, {
                    "limit": max_requests,
                    "remaining": remaining,
                    "reset": reset_time,
                    "window": window
                }
            
            timestamps.append(current_time)
            return False, {
                "limit": max_requests,
                "remaining": remaining,
                "reset": reset_time,
                "window": window
            }
    
    def _get_limit_key(self, endpoint: str) -> str:
        """Determine limit key based on endpoint"""
        if endpoint.startswith("/auth"):
            return "auth"
        elif "/upload" in endpoint:
            return "upload"
        elif "/api" in endpoint:
            return "api"
        return "default"

class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_metrics = deque(maxlen=1000)
        
    async def monitor(self):
        """Monitor system and application performance"""
        while True:
            metrics = self.collect_system_metrics()
            self.system_metrics.append(metrics)
            
            # Check for performance degradation
            self.check_performance_degradation(metrics)
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "network_sent_mb": network.bytes_sent / (1024**2),
            "network_recv_mb": network.bytes_recv / (1024**2),
            "process_memory_mb": process_memory.rss / (1024**2),
            "process_cpu_percent": process_cpu,
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    
    def check_performance_degradation(self, metrics: Dict[str, Any]):
        """Check for performance degradation"""
        alerts = []
        
        # CPU alert
        if metrics["cpu_percent"] > 80:
            alerts.append({
                "type": "high_cpu",
                "severity": "high" if metrics["cpu_percent"] > 90 else "medium",
                "value": metrics["cpu_percent"],
                "threshold": 80
            })
        
        # Memory alert
        if metrics["memory_percent"] > 85:
            alerts.append({
                "type": "high_memory",
                "severity": "high" if metrics["memory_percent"] > 95 else "medium",
                "value": metrics["memory_percent"],
                "threshold": 85
            })
        
        # Process memory alert
        if metrics["process_memory_mb"] > 1000:  # 1GB
            alerts.append({
                "type": "high_process_memory",
                "severity": "high",
                "value": metrics["process_memory_mb"],
                "threshold": 1000
            })
        
        return alerts
    
    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Generate performance report"""
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        
        recent_metrics = [
            m for m in self.system_metrics
            if m["timestamp"] > cutoff_iso
        ]
        
        if not recent_metrics:
            return {"message": "No metrics available"}
        
        # Calculate averages
        cpu_values = [m["cpu_percent"] for m in recent_metrics]
        memory_values = [m["memory_percent"] for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else 0,
            "max_cpu_percent": max(cpu_values) if cpu_values else 0,
            "avg_memory_percent": statistics.mean(memory_values) if memory_values else 0,
            "max_memory_percent": max(memory_values) if memory_values else 0,
            "process_memory_mb": recent_metrics[-1]["process_memory_mb"],
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "metrics_collected": len(self.system_metrics)
        }

class MetricsExporter:
    """Export metrics to various formats"""
    
    @staticmethod
    def export_to_prometheus(metrics_collector: APIMetricsCollector) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Request counter
        # In a real implementation, you would iterate through all label combinations
        lines.append('# HELP api_requests_total Total API requests')
        lines.append('# TYPE api_requests_total counter')
        lines.append('api_requests_total 123')  # Example
        
        # Request duration
        lines.append('# HELP api_request_duration_seconds API request duration')
        lines.append('# TYPE api_request_duration_seconds histogram')
        
        return "\n".join(lines)
    
    @staticmethod
    def export_to_json(metrics_collector: APIMetricsCollector, 
                      performance_monitor: PerformanceMonitor) -> Dict[str, Any]:
        """Export metrics to JSON"""
        summary = metrics_collector.get_summary(timeframe_minutes=5)
        performance = performance_monitor.get_performance_report(hours=1)
        error_analysis = metrics_collector.get_error_analysis()
        top_endpoints = metrics_collector.get_top_endpoints(limit=10)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "performance": performance,
            "error_analysis": error_analysis,
            "top_endpoints": top_endpoints,
            "system": performance_monitor.collect_system_metrics()
        }
    
    @staticmethod
    def export_to_csv(metrics_collector: APIMetricsCollector, 
                     filepath: str = "metrics.csv"):
        """Export metrics to CSV"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp', 'method', 'endpoint', 'status', 
                'duration', 'error_type', 'error_message'
            ])
            
            # Write requests
            for req in metrics_collector.metrics["requests"]:
                writer.writerow([
                    req["timestamp"].isoformat(),
                    req["method"],
                    req["endpoint"],
                    req["status"],
                    req["duration"],
                    "",  # No error type for successful requests
                    ""   # No error message
                ])
            
            # Write errors
            for err in metrics_collector.metrics["errors"]:
                writer.writerow([
                    err["timestamp"].isoformat(),
                    err["method"],
                    err["endpoint"],
                    "",  # No status for errors
                    0,   # No duration
                    err["error_type"],
                    err["error_message"]
                ])

# Usage example
async def main():
    """Example usage of metrics collection"""
    
    # Initialize components
    metrics_collector = APIMetricsCollector()
    rate_limiter = RateLimiter()
    performance_monitor = PerformanceMonitor()
    
    # Start monitoring
    monitoring_task = asyncio.create_task(performance_monitor.monitor())
    
    # Simulate some requests
    for i in range(100):
        metrics_collector.start_request()
        
        # Simulate request processing
        duration = 0.1 + (i % 10) * 0.05  # Varying durations
        time.sleep(0.01)  # Simulate processing
        
        # Record successful request
        metrics_collector.record_request(
            method="GET",
            endpoint=f"/api/users/{i % 10}",
            status=200,
            duration=duration
        )
        
        metrics_collector.end_request()
        
        # Simulate occasional errors
        if i % 20 == 0:
            metrics_collector.record_error(
                method="GET",
                endpoint=f"/api/users/{i % 10}",
                error_type="NotFound",
                error_msg=f"User {i} not found"
            )
    
    # Get summary
    summary = metrics_collector.get_summary(timeframe_minutes=1)
    print("1-minute Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get performance report
    performance = performance_monitor.get_performance_report(hours=0.1)
    print("\nPerformance Report:")
    print(json.dumps(performance, indent=2))
    
    # Export metrics
    exporter = MetricsExporter()
    json_metrics = exporter.export_to_json(metrics_collector, performance_monitor)
    
    print("\nExported Metrics:")
    print(json.dumps(json_metrics, indent=2))
    
    # Clean up
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())