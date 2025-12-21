"""
Production Monitoring Dashboard
Real-time metrics, alerts, and performance monitoring
"""
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
import psutil
import redis
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
from prometheus_client.core import CollectorRegistry

# Create FastAPI app
app = FastAPI(title="Monitoring Dashboard", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Redis (optional)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available, using in-memory storage")

# Prometheus metrics registry
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests',
    registry=registry
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type'],
    registry=registry
)

API_USAGE = Counter(
    'api_usage_total',
    'API usage by endpoint',
    ['endpoint', 'user_id'],
    registry=registry
)

SYSTEM_CPU = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage',
    registry=registry
)

SYSTEM_MEMORY = Gauge(
    'system_memory_percent',
    'System memory usage percentage',
    registry=registry
)

SYSTEM_DISK = Gauge(
    'system_disk_percent',
    'System disk usage percentage',
    registry=registry
)

# Data storage for metrics (in-memory, use database in production)
metrics_store = {
    "requests": [],
    "errors": [],
    "performance": [],
    "alerts": [],
    "system": []
}

# WebSocket connections
active_connections = []

# Models
class MetricData(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    metric_name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)

class Alert(BaseModel):
    id: str
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    message: str
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

class PerformanceMetrics(BaseModel):
    endpoint: str
    p50: float
    p90: float
    p95: float
    p99: float
    avg: float
    min: float
    max: float
    request_count: int
    error_rate: float

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        # Store detailed metrics
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
            "client_ip": request.client.host if request.client else "unknown"
        }
        
        metrics_store["requests"].append(metric_data)
        
        # Keep only last 10,000 requests
        if len(metrics_store["requests"]) > 10000:
            metrics_store["requests"] = metrics_store["requests"][-10000:]
        
        return response
        
    except Exception as e:
        ERROR_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            error_type=type(e).__name__
        ).inc()
        
        # Record error
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "endpoint": request.url.path,
            "error": str(e),
            "error_type": type(e).__name__
        }
        metrics_store["errors"].append(error_data)
        
        raise e
        
    finally:
        ACTIVE_REQUESTS.dec()

# System monitoring
async def monitor_system():
    """Monitor system resources"""
    while True:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_DISK.set(disk.percent)
            
            # Store system metrics
            system_data = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
            
            metrics_store["system"].append(system_data)
            
            # Keep only last 1000 system metrics
            if len(metrics_store["system"]) > 1000:
                metrics_store["system"] = metrics_store["system"][-1000:]
            
            # Check for alerts
            await check_alerts(system_data)
            
        except Exception as e:
            print(f"System monitoring error: {e}")
        
        await asyncio.sleep(10)  # Update every 10 seconds

async def check_alerts(system_data: Dict[str, Any]):
    """Check system metrics and trigger alerts"""
    alerts = []
    
    # CPU alert
    if system_data["cpu_percent"] > 80:
        alerts.append({
            "id": f"alert_cpu_{datetime.now().timestamp()}",
            "severity": "high" if system_data["cpu_percent"] > 90 else "medium",
            "message": f"High CPU usage: {system_data['cpu_percent']}%",
            "source": "system",
            "metric": "cpu_percent",
            "value": system_data["cpu_percent"]
        })
    
    # Memory alert
    if system_data["memory_percent"] > 85:
        alerts.append({
            "id": f"alert_memory_{datetime.now().timestamp()}",
            "severity": "high" if system_data["memory_percent"] > 95 else "medium",
            "message": f"High memory usage: {system_data['memory_percent']}%",
            "source": "system",
            "metric": "memory_percent",
            "value": system_data["memory_percent"]
        })
    
    # Disk alert
    if system_data["disk_percent"] > 90:
        alerts.append({
            "id": f"alert_disk_{datetime.now().timestamp()}",
            "severity": "critical" if system_data["disk_percent"] > 95 else "high",
            "message": f"High disk usage: {system_data['disk_percent']}%",
            "source": "system",
            "metric": "disk_percent",
            "value": system_data["disk_percent"]
        })
    
    # Add alerts to store
    for alert in alerts:
        metrics_store["alerts"].append(alert)
        
        # Broadcast alert via WebSocket
        await broadcast_alert(alert)
    
    # Keep only last 100 alerts
    if len(metrics_store["alerts"]) > 100:
        metrics_store["alerts"] = metrics_store["alerts"][-100:]

async def broadcast_alert(alert: Dict[str, Any]):
    """Broadcast alert to all WebSocket connections"""
    for connection in active_connections:
        try:
            await connection.send_json({
                "type": "alert",
                "data": alert
            })
        except:
            pass

# WebSocket endpoint
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket for real-time metrics streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send system metrics periodically
            if metrics_store["system"]:
                latest_system = metrics_store["system"][-1]
                await websocket.send_json({
                    "type": "system_metrics",
                    "data": latest_system
                })
            
            # Send performance summary
            performance = calculate_performance_metrics()
            await websocket.send_json({
                "type": "performance_metrics",
                "data": performance
            })
            
            # Send active alerts
            active_alerts = [a for a in metrics_store["alerts"] if not a.get("resolved", False)]
            await websocket.send_json({
                "type": "active_alerts",
                "data": active_alerts[-10:]  # Last 10 alerts
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

def calculate_performance_metrics() -> List[Dict[str, Any]]:
    """Calculate performance metrics from stored requests"""
    if not metrics_store["requests"]:
        return []
    
    # Group by endpoint
    endpoint_requests = defaultdict(list)
    for req in metrics_store["requests"][-1000:]:  # Last 1000 requests
        endpoint_requests[req["endpoint"]].append(req["duration"])
    
    performance = []
    for endpoint, durations in endpoint_requests.items():
        if durations:
            durations.sort()
            n = len(durations)
            
            performance.append({
                "endpoint": endpoint,
                "p50": durations[int(n * 0.5)] if n > 0 else 0,
                "p90": durations[int(n * 0.9)] if n > 9 else 0,
                "p95": durations[int(n * 0.95)] if n > 19 else 0,
                "p99": durations[int(n * 0.99)] if n > 99 else 0,
                "avg": statistics.mean(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "request_count": n,
                "error_rate": 0  # Would calculate from errors
            })
    
    return performance

# Routes
@app.get("/")
async def dashboard(request: Request):
    """Serve monitoring dashboard"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Production Monitoring Dashboard",
            "version": "2.0.0"
        }
    )

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry)

@app.get("/api/metrics/requests")
async def get_request_metrics(
    endpoint: Optional[str] = None,
    limit: int = 100,
    last_minutes: Optional[int] = None
):
    """Get request metrics"""
    metrics = metrics_store["requests"]
    
    # Filter by endpoint
    if endpoint:
        metrics = [m for m in metrics if m["endpoint"] == endpoint]
    
    # Filter by time
    if last_minutes:
        cutoff = datetime.now() - timedelta(minutes=last_minutes)
        cutoff_iso = cutoff.isoformat()
        metrics = [m for m in metrics if m["timestamp"] > cutoff_iso]
    
    return {
        "count": len(metrics),
        "metrics": metrics[-limit:]  # Return last N metrics
    }

@app.get("/api/metrics/performance")
async def get_performance_summary():
    """Get performance summary"""
    performance = calculate_performance_metrics()
    
    # Overall statistics
    all_durations = [req["duration"] for req in metrics_store["requests"][-1000:]]
    
    if all_durations:
        overall_stats = {
            "total_requests": len(all_durations),
            "avg_response_time": statistics.mean(all_durations),
            "p95_response_time": statistics.quantiles(all_durations, n=20)[18],
            "error_rate": len(metrics_store["errors"]) / max(len(all_durations), 1),
            "active_alerts": len([a for a in metrics_store["alerts"] if not a.get("resolved", False)])
        }
    else:
        overall_stats = {}
    
    return {
        "overall": overall_stats,
        "endpoints": performance
    }

@app.get("/api/metrics/system")
async def get_system_metrics(
    last_hours: Optional[int] = None,
    limit: int = 100
):
    """Get system metrics"""
    metrics = metrics_store["system"]
    
    if last_hours:
        cutoff = datetime.now() - timedelta(hours=last_hours)
        cutoff_iso = cutoff.isoformat()
        metrics = [m for m in metrics if m["timestamp"] > cutoff_iso]
    
    return {
        "count": len(metrics),
        "metrics": metrics[-limit:]
    }

@app.get("/api/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 50
):
    """Get alerts"""
    alerts = metrics_store["alerts"]
    
    # Filter by severity
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    
    # Filter by resolved status
    if resolved is not None:
        alerts = [a for a in alerts if a.get("resolved") == resolved]
    
    return {
        "count": len(alerts),
        "alerts": alerts[-limit:]
    }

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    for alert in metrics_store["alerts"]:
        if alert.get("id") == alert_id:
            alert["acknowledged"] = True
            
            # Broadcast acknowledgment
            for connection in active_connections:
                try:
                    await connection.send_json({
                        "type": "alert_acknowledged",
                        "alert_id": alert_id
                    })
                except:
                    pass
            
            return {"message": f"Alert {alert_id} acknowledged"}
    
    raise HTTPException(status_code=404, detail="Alert not found")

@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    for alert in metrics_store["alerts"]:
        if alert.get("id") == alert_id:
            alert["resolved"] = True
            alert["resolved_at"] = datetime.now().isoformat()
            
            # Broadcast resolution
            for connection in active_connections:
                try:
                    await connection.send_json({
                        "type": "alert_resolved",
                        "alert_id": alert_id
                    })
                except:
                    pass
            
            return {"message": f"Alert {alert_id} resolved"}
    
    raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "total_requests": len(metrics_store["requests"]),
            "total_errors": len(metrics_store["errors"]),
            "total_alerts": len(metrics_store["alerts"]),
            "active_connections": len(active_connections),
            "system_metrics": len(metrics_store["system"])
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }

@app.get("/api/usage")
async def get_usage_stats(
    timeframe: str = "day",  # day, week, month
    endpoint: Optional[str] = None
):
    """Get API usage statistics"""
    now = datetime.now()
    
    if timeframe == "day":
        cutoff = now - timedelta(days=1)
    elif timeframe == "week":
        cutoff = now - timedelta(weeks=1)
    else:  # month
        cutoff = now - timedelta(days=30)
    
    cutoff_iso = cutoff.isoformat()
    
    # Filter requests by timeframe
    recent_requests = [
        req for req in metrics_store["requests"]
        if req["timestamp"] > cutoff_iso
    ]
    
    if endpoint:
        recent_requests = [
            req for req in recent_requests
            if req["endpoint"] == endpoint
        ]
    
    # Calculate statistics
    if recent_requests:
        durations = [req["duration"] for req in recent_requests]
        status_codes = [req["status_code"] for req in recent_requests]
        
        success_count = sum(1 for code in status_codes if 200 <= code < 300)
        error_count = sum(1 for code in status_codes if code >= 400)
        
        return {
            "timeframe": timeframe,
            "total_requests": len(recent_requests),
            "success_rate": success_count / len(recent_requests) * 100,
            "error_rate": error_count / len(recent_requests) * 100,
            "avg_response_time": statistics.mean(durations),
            "p95_response_time": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else 0,
            "endpoint_distribution": {
                endpoint: sum(1 for req in recent_requests if req["endpoint"] == endpoint)
                for endpoint in set(req["endpoint"] for req in recent_requests)
            }
        }
    
    return {
        "timeframe": timeframe,
        "total_requests": 0,
        "message": "No data for the specified timeframe"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Start monitoring tasks on startup"""
    print("🚀 Monitoring Dashboard starting...")
    print("📊 Starting system monitoring...")
    
    # Start system monitoring task
    asyncio.create_task(monitor_system())
    
    print("🌐 WebSocket available at: ws://localhost:8000/ws/metrics")
    print("📈 Prometheus metrics at: http://localhost:8000/metrics/prometheus")
    print("📊 Dashboard at: http://localhost:8000/")

# Create HTML template directory and files
def create_template_files():
    """Create HTML template files"""
    import os
    from pathlib import Path
    
    # Create directories
    templates_dir = Path("templates")
    static_dir = Path("static")
    
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    # Create dashboard.html
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #f1f5f9;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #60a5fa;
        }
        
        .status-bar {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        
        .status-item {
            background: #334155;
            padding: 15px;
            border-radius: 8px;
            flex: 1;
            text-align: center;
        }
        
        .status-item h3 {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 5px;
        }
        
        .status-item .value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .status-healthy { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .card {
            background: #1e293b;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #cbd5e1;
            border-bottom: 2px solid #334155;
            padding-bottom: 10px;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 15px;
        }
        
        .alerts-container {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .alert {
            background: #334155;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }
        
        .alert-critical { border-left-color: #ef4444; }
        .alert-high { border-left-color: #f97316; }
        .alert-medium { border-left-color: #f59e0b; }
        .alert-low { border-left-color: #10b981; }
        
        .alert-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .alert-title {
            font-weight: bold;
        }
        
        .alert-time {
            font-size: 0.9rem;
            color: #94a3b8;
        }
        
        .alert-message {
            color: #cbd5e1;
        }
        
        .alert-actions {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        
        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
        }
        
        button:hover {
            background: #2563eb;
        }
        
        button.resolve {
            background: #10b981;
        }
        
        button.resolve:hover {
            background: #059669;
        }
        
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .performance-table th,
        .performance-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }
        
        .performance-table th {
            background: #334155;
            font-weight: bold;
            color: #cbd5e1;
        }
        
        .performance-table tr:hover {
            background: #2d3748;
        }
        
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9rem;
            z-index: 1000;
        }
        
        .connected { background: #10b981; }
        .disconnected { background: #ef4444; }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .status-bar {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }} v{{ version }}</h1>
            <p>Real-time monitoring and alerting system</p>
            
            <div class="status-bar">
                <div class="status-item">
                    <h3>System Health</h3>
                    <div class="value status-healthy" id="systemHealth">Loading...</div>
                </div>
                <div class="status-item">
                    <h3>Active Alerts</h3>
                    <div class="value" id="activeAlerts">0</div>
                </div>
                <div class="status-item">
                    <h3>Avg Response Time</h3>
                    <div class="value" id="avgResponseTime">0ms</div>
                </div>
                <div class="status-item">
                    <h3>Requests (Last 5 min)</h3>
                    <div class="value" id="requestCount">0</div>
                </div>
                <div class="status-item">
                    <h3>Error Rate</h3>
                    <div class="value" id="errorRate">0%</div>
                </div>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>System Resources</h2>
                <div class="chart-container">
                    <canvas id="systemChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Response Times</h2>
                <div class="chart-container">
                    <canvas id="responseTimeChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Active Alerts</h2>
                <div class="alerts-container" id="alertsList">
                    <div class="alert">
                        <div class="alert-header">
                            <span class="alert-title">No active alerts</span>
                            <span class="alert-time">--:--:--</span>
                        </div>
                        <div class="alert-message">System is running smoothly</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Endpoint Performance</h2>
                <div class="chart-container">
                    <canvas id="endpointChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Performance Summary</h2>
                <table class="performance-table" id="performanceTable">
                    <thead>
                        <tr>
                            <th>Endpoint</th>
                            <th>Requests</th>
                            <th>Avg (ms)</th>
                            <th>P95 (ms)</th>
                            <th>Error %</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Request Distribution</h2>
                <div class="chart-container">
                    <canvas id="requestDistributionChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <div class="connection-status disconnected" id="connectionStatus">
        Disconnected
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        let charts = {};
        let performanceData = {};
        
        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateConnectionStatus(true);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateConnectionStatus(false);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
        }
        
        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connectionStatus');
            statusEl.textContent = connected ? 'Connected' : 'Disconnected';
            statusEl.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        }
        
        function handleWebSocketMessage(data) {
            switch (data.type) {
                case 'system_metrics':
                    updateSystemMetrics(data.data);
                    break;
                case 'performance_metrics':
                    updatePerformanceMetrics(data.data);
                    break;
                case 'active_alerts':
                    updateAlerts(data.data);
                    break;
                case 'alert':
                    showNewAlert(data.data);
                    break;
                case 'alert_acknowledged':
                    console.log(`Alert ${data.alert_id} acknowledged`);
                    break;
                case 'alert_resolved':
                    console.log(`Alert ${data.alert_id} resolved`);
                    break;
            }
        }
        
        function updateSystemMetrics(metrics) {
            // Update status bar
            document.getElementById('systemHealth').textContent = 
                metrics.cpu_percent < 80 ? 'Healthy' : 
                metrics.cpu_percent < 90 ? 'Warning' : 'Critical';
            
            // Update system chart
            if (!charts.system) {
                createSystemChart();
            }
            updateSystemChart(metrics);
        }
        
        function updatePerformanceMetrics(metrics) {
            performanceData = metrics;
            
            // Update status bar
            if (metrics.length > 0) {
                const overall = metrics[0]; // Assuming first is overall
                document.getElementById('avgResponseTime').textContent = 
                    `${(overall.avg * 1000).toFixed(0)}ms`;
                document.getElementById('requestCount').textContent = 
                    overall.request_count;
                document.getElementById('errorRate').textContent = 
                    `${overall.error_rate.toFixed(1)}%`;
            }
            
            // Update performance table
            updatePerformanceTable(metrics);
            
            // Update charts
            if (!charts.responseTime) {
                createResponseTimeChart();
            }
            if (!charts.endpoint) {
                createEndpointChart();
            }
            if (!charts.distribution) {
                createDistributionChart();
            }
        }
        
        function updateAlerts(alerts) {
            document.getElementById('activeAlerts').textContent = alerts.length;
            
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsList.innerHTML = `
                    <div class="alert">
                        <div class="alert-header">
                            <span class="alert-title">No active alerts</span>
                            <span class="alert-time">--:--:--</span>
                        </div>
                        <div class="alert-message">System is running smoothly</div>
                    </div>
                `;
                return;
            }
            
            alerts.forEach(alert => {
                const time = new Date(alert.timestamp).toLocaleTimeString();
                const alertEl = document.createElement('div');
                alertEl.className = `alert alert-${alert.severity}`;
                alertEl.innerHTML = `
                    <div class="alert-header">
                        <span class="alert-title">${alert.severity.toUpperCase()}: ${alert.source}</span>
                        <span class="alert-time">${time}</span>
                    </div>
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-actions">
                        <button onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>
                        <button class="resolve" onclick="resolveAlert('${alert.id}')">Resolve</button>
                    </div>
                `;
                alertsList.appendChild(alertEl);
            });
        }
        
        function showNewAlert(alert) {
            // Show notification
            if (Notification.permission === 'granted') {
                new Notification(`Alert: ${alert.severity.toUpperCase()}`, {
                    body: alert.message,
                    icon: '/static/alert.png'
                });
            }
            
            // Add to alerts list
            const alertsList = document.getElementById('alertsList');
            if (alertsList.firstChild?.textContent?.includes('No active alerts')) {
                alertsList.innerHTML = '';
            }
            
            const time = new Date(alert.timestamp).toLocaleTimeString();
            const alertEl = document.createElement('div');
            alertEl.className = `alert alert-${alert.severity}`;
            alertEl.innerHTML = `
                <div class="alert-header">
                    <span class="alert-title">NEW: ${alert.severity.toUpperCase()}: ${alert.source}</span>
                    <span class="alert-time">${time}</span>
                </div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-actions">
                    <button onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>
                    <button class="resolve" onclick="resolveAlert('${alert.id}')">Resolve</button>
                </div>
            `;
            alertsList.prepend(alertEl);
        }
        
        // Chart creation functions
        function createSystemChart() {
            const ctx = document.getElementById('systemChart').getContext('2d');
            charts.system = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'CPU %',
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Memory %',
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Disk %',
                            borderColor: '#8b5cf6',
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function updateSystemChart(metrics) {
            const chart = charts.system;
            const now = new Date();
            
            // Add data points
            chart.data.datasets[0].data.push({x: now, y: metrics.cpu_percent});
            chart.data.datasets[1].data.push({x: now, y: metrics.memory_percent});
            chart.data.datasets[2].data.push({x: now, y: metrics.disk_percent});
            
            // Keep only last 20 points
            chart.data.datasets.forEach(dataset => {
                if (dataset.data.length > 20) {
                    dataset.data.shift();
                }
            });
            
            chart.update('none');
        }
        
        function createResponseTimeChart() {
            const ctx = document.getElementById('responseTimeChart').getContext('2d');
            charts.responseTime = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Response Time (ms)',
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Milliseconds'
                            }
                        }
                    }
                }
            });
        }
        
        function createEndpointChart() {
            const ctx = document.getElementById('endpointChart').getContext('2d');
            charts.endpoint = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Avg Response Time (ms)',
                            backgroundColor: '#3b82f6',
                            borderColor: '#2563eb',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Milliseconds'
                            }
                        }
                    }
                }
            });
        }
        
        function createDistributionChart() {
            const ctx = document.getElementById('requestDistributionChart').getContext('2d');
            charts.distribution = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#3b82f6',
                            '#10b981',
                            '#8b5cf6',
                            '#f59e0b',
                            '#ef4444',
                            '#ec4899'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right'
                        }
                    }
                }
            });
        }
        
        function updatePerformanceTable(metrics) {
            const tbody = document.querySelector('#performanceTable tbody');
            tbody.innerHTML = '';
            
            metrics.forEach(metric => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${metric.endpoint}</td>
                    <td>${metric.request_count}</td>
                    <td>${(metric.avg * 1000).toFixed(1)}</td>
                    <td>${(metric.p95 * 1000).toFixed(1)}</td>
                    <td>${metric.error_rate.toFixed(1)}%</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        // Alert actions
        async function acknowledgeAlert(alertId) {
            await fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
        }
        
        async function resolveAlert(alertId) {
            await fetch(`/api/alerts/${alertId}/resolve`, {
                method: 'POST'
            });
        }
        
        // Request notification permission
        if ('Notification' in window) {
            Notification.requestPermission();
        }
        
        // Initialize
        connectWebSocket();
        
        // Fetch initial data
        async function fetchInitialData() {
            try {
                const [performanceRes, alertsRes] = await Promise.all([
                    fetch('/api/metrics/performance'),
                    fetch('/api/alerts?resolved=false')
                ]);
                
                const performance = await performanceRes.json();
                const alerts = await alertsRes.json();
                
                updatePerformanceMetrics(performance.endpoints || []);
                updateAlerts(alerts.alerts || []);
            } catch (error) {
                console.error('Failed to fetch initial data:', error);
            }
        }
        
        fetchInitialData();
    </script>
</body>
</html>
    """
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    print("✅ Created dashboard template")
    return True

if __name__ == "__main__":
    # Create template files
    create_template_files()
    
    import uvicorn
    
    print("🚀 Starting Monitoring Dashboard...")
    print("📊 Dashboard: http://localhost:8000")
    print("📈 Metrics: http://localhost:8000/metrics/prometheus")
    print("🔌 WebSocket: ws://localhost:8000/ws/metrics")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )