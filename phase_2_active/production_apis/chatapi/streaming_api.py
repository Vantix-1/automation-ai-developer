# phase_2_complete/week_7_8_production/day_37_39_chatapi/streaming_api.py
"""
Advanced Streaming API Implementation
Server-Sent Events (SSE) and WebSocket streaming
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
import openai

app = FastAPI(title="Streaming API Demo")

# Store active connections
active_connections = []

# HTML page for WebSocket demo
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Streaming API Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .panel { flex: 1; border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        h2 { color: #333; }
        textarea, input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #sseOutput, #wsOutput { 
            height: 300px; 
            overflow-y: auto; 
            border: 1px solid #ddd; 
            padding: 10px; 
            margin: 10px 0;
            background: #f9f9f9;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .connection-status { padding: 5px 10px; border-radius: 4px; margin: 5px 0; }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .event { margin: 5px 0; padding: 5px; background: #e9ecef; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Streaming API Demo</h1>
    <p>Test Server-Sent Events (SSE) and WebSocket streaming</p>
    
    <div class="container">
        <div class="panel">
            <h2>Server-Sent Events (SSE)</h2>
            <textarea id="sseMessage" placeholder="Enter your message..." rows="3">What is artificial intelligence?</textarea>
            <button onclick="sendSSEMessage()">Send Message via SSE</button>
            <div>
                <span class="connection-status" id="sseStatus">SSE: Not connected</span>
            </div>
            <div id="sseOutput"></div>
        </div>
        
        <div class="panel">
            <h2>WebSocket</h2>
            <textarea id="wsMessage" placeholder="Enter your message..." rows="3">Explain machine learning in simple terms.</textarea>
            <button onclick="sendWSMessage()">Send Message via WebSocket</button>
            <button onclick="connectWebSocket()">Connect WebSocket</button>
            <button onclick="disconnectWebSocket()">Disconnect</button>
            <div>
                <span class="connection-status" id="wsStatus">WebSocket: Not connected</span>
            </div>
            <div id="wsOutput"></div>
        </div>
    </div>
    
    <script>
        // SSE Connection
        let sseConnection = null;
        
        function connectSSE() {
            if (sseConnection) return;
            
            sseConnection = new EventSource('/sse-stream');
            
            sseConnection.onopen = function() {
                document.getElementById('sseStatus').textContent = 'SSE: Connected';
                document.getElementById('sseStatus').className = 'connection-status connected';
                logSSE('SSE connection established');
            };
            
            sseConnection.onmessage = function(event) {
                const data = JSON.parse(event.data);
                logSSE(`[${data.timestamp}] ${data.type}: ${data.content}`);
            };
            
            sseConnection.onerror = function(error) {
                logSSE(`SSE Error: ${error}`);
                document.getElementById('sseStatus').textContent = 'SSE: Error';
                document.getElementById('sseStatus').className = 'connection-status disconnected';
                sseConnection = null;
            };
        }
        
        function sendSSEMessage() {
            connectSSE();
            const message = document.getElementById('sseMessage').value;
            
            fetch('/sse-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                logSSE(`[${new Date().toISOString()}] Message sent: ${message}`);
            })
            .catch(error => {
                logSSE(`Error sending message: ${error}`);
            });
        }
        
        function logSSE(text) {
            const output = document.getElementById('sseOutput');
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event';
            eventDiv.textContent = text;
            output.appendChild(eventDiv);
            output.scrollTop = output.scrollHeight;
        }
        
        // WebSocket Connection
        let wsConnection = null;
        
        function connectWebSocket() {
            if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
                logWS('WebSocket already connected');
                return;
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            wsConnection = new WebSocket(wsUrl);
            
            wsConnection.onopen = function() {
                document.getElementById('wsStatus').textContent = 'WebSocket: Connected';
                document.getElementById('wsStatus').className = 'connection-status connected';
                logWS('WebSocket connection established');
            };
            
            wsConnection.onmessage = function(event) {
                const data = JSON.parse(event.data);
                logWS(`[${data.timestamp}] ${data.type}: ${data.content}`);
            };
            
            wsConnection.onerror = function(error) {
                logWS(`WebSocket Error: ${error}`);
            };
            
            wsConnection.onclose = function() {
                document.getElementById('wsStatus').textContent = 'WebSocket: Disconnected';
                document.getElementById('wsStatus').className = 'connection-status disconnected';
                logWS('WebSocket connection closed');
                wsConnection = null;
            };
        }
        
        function sendWSMessage() {
            if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
                alert('Please connect WebSocket first');
                return;
            }
            
            const message = document.getElementById('wsMessage').value;
            wsConnection.send(JSON.stringify({
                type: 'message',
                content: message,
                timestamp: new Date().toISOString()
            }));
            
            logWS(`[${new Date().toISOString()}] You: ${message}`);
        }
        
        function disconnectWebSocket() {
            if (wsConnection) {
                wsConnection.close();
            }
        }
        
        function logWS(text) {
            const output = document.getElementById('wsOutput');
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event';
            eventDiv.textContent = text;
            output.appendChild(eventDiv);
            output.scrollTop = output.scrollHeight;
        }
        
        // Auto-connect SSE on page load
        window.onload = connectSSE;
    </script>
</body>
</html>
"""

# Routes
@app.get("/")
async def get():
    """Serve the demo page"""
    return HTMLResponse(html)

@app.get("/sse-stream")
async def sse_stream():
    """SSE stream endpoint"""
    
    async def event_generator():
        """Generate Server-Sent Events"""
        try:
            # Initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({
                    "message": "Connected to SSE stream",
                    "timestamp": datetime.now().isoformat()
                })
            }
            
            # Keep connection alive with heartbeat
            while True:
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({
                        "message": "Heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                }
                
        except asyncio.CancelledError:
            print("SSE connection closed by client")
    
    return EventSourceResponse(event_generator())

@app.post("/sse-message")
async def sse_message(message: Dict[str, Any]):
    """Handle incoming messages and stream simulated AI response"""
    
    async def generate_response():
        """Stream AI-like response for SSE"""
        response_text = f"AI Response to: {message.get('message', 'No message')}\n\n"
        words = [
            "This", "is", "a", "simulated", "AI", "response", "to", "demonstrate",
            "streaming", "capabilities.", "Each", "word", "is", "sent", "as", "a",
            "separate", "chunk", "in", "real-time."
        ]
        
        for word in words:
            yield {
                "event": "chunk",
                "data": json.dumps({
                    "type": "text_chunk",
                    "content": word + " ",
                    "timestamp": datetime.now().isoformat()
                })
            }
            await asyncio.sleep(0.1)  # Simulate processing delay
        
        yield {
            "event": "complete",
            "data": json.dumps({
                "type": "completion",
                "content": "\n\n[Response complete]",
                "timestamp": datetime.now().isoformat()
            })
        }
    
    return EventSourceResponse(generate_response())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time bidirectional communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message = data.get("content", "")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "acknowledgment",
                    "content": f"Processing: {message[:50]}...",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Simulate AI processing and stream response
                await simulate_ai_response(websocket, message)
                
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"WebSocket disconnected. Active connections: {len(active_connections)}")

async def simulate_ai_response(websocket: WebSocket, message: str):
    """Simulate streaming AI response"""
    responses = [
        f"I understand you're asking about '{message}'.",
        "Let me think about that for a moment...",
        "This is an interesting question.",
        "Based on my knowledge, here's what I can tell you:",
        f"The topic '{message}' is quite broad.",
        "Would you like me to elaborate on any specific aspect?"
    ]
    
    for i, response in enumerate(responses):
        await websocket.send_json({
            "type": "response_chunk",
            "content": response,
            "chunk_number": i + 1,
            "total_chunks": len(responses),
            "timestamp": datetime.now().isoformat()
        })
        await asyncio.sleep(0.5)  # Simulate thinking time
    
    await websocket.send_json({
        "type": "response_complete",
        "content": f"Complete response to '{message}' has been streamed.",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/stats")
async def get_stats():
    """Get streaming API statistics"""
    return {
        "active_connections": len(active_connections),
        "timestamp": datetime.now().isoformat(),
        "supported_protocols": ["SSE", "WebSocket"],
        "features": {
            "sse": {
                "heartbeat": "10 seconds",
                "reconnection": "automatic"
            },
            "websocket": {
                "bidirectional": True,
                "binary_messages": False
            }
        }
    }

@app.get("/demo/stream-text")
async def stream_text():
    """Demo endpoint that streams text character by character"""
    
    async def generate():
        text = "This is a demonstration of real-time text streaming. "
        text += "Each character appears with a small delay to simulate processing. "
        text += "This technique is useful for AI responses, live updates, and progress indicators."
        
        for char in text:
            yield char
            await asyncio.sleep(0.05)  # 50ms delay between characters
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

@app.get("/demo/stream-json")
async def stream_json():
    """Demo endpoint that streams JSON objects"""
    
    async def generate():
        items = [
            {"id": 1, "name": "Item 1", "status": "processing"},
            {"id": 2, "name": "Item 2", "status": "processing"},
            {"id": 3, "name": "Item 3", "status": "processing"},
            {"id": 4, "name": "Item 4", "status": "complete"},
            {"id": 5, "name": "Item 5", "status": "failed"}
        ]
        
        for item in items:
            # Simulate processing time
            await asyncio.sleep(0.3)
            
            # Update status based on some logic
            if item["id"] % 2 == 0:
                item["status"] = "complete"
            elif item["id"] % 3 == 0:
                item["status"] = "failed"
            
            yield json.dumps(item) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

# WebSocket manager for multiple connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Chat room via WebSocket"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "join":
                await manager.broadcast({
                    "type": "notification",
                    "content": f"User {data['username']} joined the chat",
                    "timestamp": datetime.now().isoformat()
                })
                
            elif data["type"] == "message":
                await manager.broadcast({
                    "type": "message",
                    "username": data.get("username", "Anonymous"),
                    "content": data["content"],
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({
            "type": "notification",
            "content": "A user left the chat",
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Streaming API Demo starting...")
    print("🌐 Open: http://localhost:8000")
    print("📡 Endpoints:")
    print("  - GET  /                 - Demo page")
    print("  - GET  /sse-stream       - SSE connection")
    print("  - POST /sse-message      - Send SSE message")
    print("  - WS   /ws               - WebSocket endpoint")
    print("  - WS   /ws/chat          - Chat room WebSocket")
    print("  - GET  /demo/stream-text - Text streaming demo")
    print("  - GET  /demo/stream-json - JSON streaming demo")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)