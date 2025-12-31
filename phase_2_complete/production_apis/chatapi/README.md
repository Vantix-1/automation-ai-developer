# 🤖 Chatbot REST API

## 📋 Overview
This section focuses on building production-ready chatbot APIs with streaming support, conversation management, and comprehensive testing. You'll learn to create AI-powered chat interfaces that scale.

## 📁 File Structure
```
day_37_39_chatapi/
├── requirements.txt        # Python dependencies
├── ai_chat_api.py         # Production chatbot API with streaming
├── streaming_api.py       # SSE and WebSocket streaming implementation
└── api_testing.py         # Comprehensive testing suite
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- OpenAI API key (for AI features)
- Redis (optional, for production)

### Installation
```bash
# Navigate to the directory
cd phase_2_complete/week_7_8_production/day_37_39_chatapi

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 📚 Learning Objectives

### Day 37: Basic Chat API
- ✅ Create RESTful chat endpoints
- ✅ Implement JWT authentication
- ✅ Manage conversation state
- ✅ Handle AI model responses

### Day 38: Streaming & Real-time
- ✅ Implement Server-Sent Events (SSE)
- ✅ Add WebSocket support
- ✅ Handle real-time streaming responses
- ✅ Create interactive chat interfaces

### Day 39: Testing & Quality
- ✅ Write comprehensive unit tests
- ✅ Implement integration tests
- ✅ Create load testing utilities
- ✅ Add security testing

## 🚦 Quick Start

### 1. Run the Chat API
```bash
python ai_chat_api.py
```

Access the API at:
- **API:** http://localhost:8000
- **Documentation:** http://localhost:8000/docs
- **Demo:** http://localhost:8000 (for streaming demo)

### 2. Test Authentication
```bash
# Get demo token (configured in code)
# Default token: "demo-token"

# Send chat message
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, AI!"}
    ],
    "model": "gpt-3.5-turbo"
  }'
```

### 3. Test Streaming
```bash
# Stream response (using curl with SSE)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain AI in simple terms"}
    ],
    "stream": true
  }'
```

### 4. Run Streaming Demo
```bash
python streaming_api.py
# Open browser: http://localhost:8000
```

### 5. Run Tests
```bash
python api_testing.py
# Or run specific test types
python api_testing.py mock      # Start mock server
python api_testing.py load      # Run load tests
```

## 🔧 Key Features

### 1. Production Chat API
- JWT authentication with token validation
- Conversation management with history
- Multiple AI model support (GPT-3.5, GPT-4)
- Token usage tracking and cost estimation

### 2. Real-time Streaming
- Server-Sent Events (SSE) for HTTP streaming
- WebSocket support for bidirectional communication
- Real-time response streaming
- Connection management and heartbeats

### 3. Conversation Management
- Create, read, update, delete conversations
- Message history persistence
- Context window management
- Conversation metadata (titles, timestamps)

### 4. Comprehensive Testing
- Unit tests for individual components
- Integration tests for API endpoints
- Load testing for performance
- Security testing (SQL injection, XSS)

### 5. Security
- JWT token authentication
- Rate limiting (implemented in middleware)
- Input validation and sanitization
- Secure headers and CORS configuration

## 📖 Code Examples

### Basic Chat Request
```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "demo-token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
}

response = requests.post(f"{BASE_URL}/chat", json=payload, headers=headers)
print(response.json())
```

### Streaming with SSE
```python
import requests
import json

def stream_chat():
    url = f"{BASE_URL}/chat/stream"
    response = requests.post(url, json=payload, headers=headers, stream=True)
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                data = json.loads(decoded_line[6:])
                print(data.get('content', ''), end='', flush=True)
```

### WebSocket Connection
```python
import asyncio
import websockets
import json

async def chat_via_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "message",
            "content": "Hello, AI!"
        }))
        
        # Receive responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"AI: {data.get('content', '')}")
```

## 🧪 Testing Suite

### 1. Unit Tests
```bash
# Run unit tests
python -m pytest api_testing.py::TestAPIEndpoints -v
```

### 2. Integration Tests
```bash
# Run integration tests (requires API to be running)
python -c "from api_testing import IntegrationTests; import asyncio; asyncio.run(IntegrationTests().run_all_tests())"
```

### 3. Load Tests
```bash
# Run load test (100 requests to /health)
python api_testing.py load
```

### 4. Security Tests
```bash
# Test for SQL injection
python -c "from api_testing import SecurityTests; print(SecurityTests.test_sql_injection('/test/echo'))"
```

## 🔍 API Reference

### Authentication
All endpoints require JWT token in Authorization header:
```
Authorization: Bearer your_token_here
```

### Main Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API information |
| GET | /health | Health check |
| POST | /chat | Send chat message |
| POST | /chat/stream | Stream chat response (SSE) |
| GET | /conversations | List conversations |
| POST | /conversations | Create conversation |
| GET | /conversations/{id} | Get conversation |
| DELETE | /conversations/{id} | Delete conversation |
| POST | /conversations/{id}/messages | Add message to conversation |

### WebSocket Endpoints
- `ws://localhost:8000/ws` - Basic WebSocket chat
- `ws://localhost:8000/ws/chat` - Chat room WebSocket

### Chat Request Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

### Chat Response Format
```json
{
  "id": "chat-123",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  },
  "processing_time": 0.45
}
```

## 🎯 Best Practices Implemented
- **Authentication:** JWT tokens with proper validation
- **Error Handling:** Comprehensive error responses
- **Streaming:** Efficient real-time responses
- **State Management:** Conversation persistence
- **Testing:** Full test coverage
- **Security:** Input validation and rate limiting
- **Documentation:** Automatic OpenAPI docs
- **Monitoring:** Health checks and metrics

## 📈 Performance Tips

### 1. Streaming Optimization
```python
# Use async generators for streaming
async def stream_response():
    async for chunk in ai_stream():
        yield chunk
        await asyncio.sleep(0)  # Yield control
```

### 2. Connection Pooling
```python
import httpx

async with httpx.AsyncClient() as client:
    # Reuse client for multiple requests
    responses = await asyncio.gather(
        client.get(url1),
        client.get(url2)
    )
```

### 3. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_response(prompt: str) -> str:
    # Cache frequent responses
    return ai_response(prompt)
```

## 🚨 Troubleshooting

### Common Issues

**OpenAI API errors**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test OpenAI API directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**WebSocket connection issues**
```python
# Check WebSocket server is running
# Ensure proper CORS configuration
```

**Streaming stops abruptly**
```python
# Add heartbeat to keep connection alive
async def heartbeat():
    while True:
        await asyncio.sleep(10)
        yield ": heartbeat\n\n"
```

## 📚 Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Server-Sent Events MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [JWT Authentication](https://jwt.io/)
- [WebSocket Protocol](https://datatracker.ietf.org/doc/html/rfc6455)

## 🏆 Completion Checklist
- [ ] Created production chatbot API
- [ ] Implemented JWT authentication
- [ ] Added streaming with SSE
- [ ] Implemented WebSocket support
- [ ] Created conversation management
- [ ] Added comprehensive testing
- [ ] Implemented load testing
- [ ] Added security testing