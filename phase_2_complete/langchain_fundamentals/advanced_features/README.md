# Advanced Features & Integration

---

## 🚀 Quick Start
```bash
# Navigate to Days 20-21
cd week2_day20-21

# Create virtual environment (Python 3.13 compatible)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements with version ranges
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Optional: Set database connection
echo "DATABASE_URL=sqlite:///./chat.db" >> .env
```

---

## 📊 Day-by-Day Learning

### Persistent Memory Systems
**File:** `persistent_memory.py`

- ✅ SQLite database storage for conversations
- ✅ Vector embeddings for semantic memory search
- ✅ Hybrid memory systems (SQL + vector)
- ✅ Multi-user memory management
- ✅ Conversation summarization and search

### LangChain Integration
**File:** `langchain_integration.py`

- ✅ FastAPI production API with LangChain
- ✅ Database and caching service integration
- ✅ Streaming responses with Server-Sent Events
- ✅ Webhook handlers for external services
- ✅ Analytics and monitoring
- ✅ Production deployment configuration

---

## 📁 Files Overview

### `persistent_memory.py`
- **SQLiteMemoryStore:** Structured conversation storage
- **VectorMemoryStore:** Semantic memory with embeddings
- **HybridMemorySystem:** Combined storage approach
- **PersistentMemoryAgent:** Agent with persistent memory
- **MultiUserMemoryManager:** Scalable user management

### `langchain_integration.py`
- FastAPI application with LangChain backend
- DatabaseService & CacheService integration
- AnalyticsService for monitoring
- Webhook handlers (Slack, Discord)
- Production deployment configuration
- Complete REST API with documentation

---

## 🛠️ Key Features

### Persistent Memory
- **SQLite Storage:** Structured conversation storage with full history
- **Vector Embeddings:** Semantic search across all conversations
- **Hybrid System:** Combine structured and semantic search
- **Multi-User:** Separate memory spaces per user
- **Automatic Summarization:** LLM-generated conversation summaries
- **Cleanup System:** Automatic cleanup of old conversations

### Integration Patterns
- **FastAPI Integration:** Production-ready REST API
- **Streaming Support:** Real-time response streaming
- **Service Architecture:** Modular service design
- **Webhook Support:** External service integration
- **Health Checks:** Monitoring and diagnostics
- **Configuration Management:** Environment-based configuration

---

## 🔧 Installation & Setup

### Python 3.13 Compatibility
All packages in `requirements.txt` use version ranges (>=) compatible with Python 3.13:
```bash
# Verify Python version
python --version  # Should be 3.13.x

# Install with version ranges
pip install -r requirements.txt

# Verify installation
python -c "
import sys
print(f'Python {sys.version}')
import langchain, fastapi, chromadb
print('✅ All packages installed successfully')
"
```

### Optional Database Setup
```bash
# For PostgreSQL (recommended for production)
pip install psycopg2-binary
echo "DATABASE_URL=postgresql://user:password@localhost/chatdb" >> .env

# For MongoDB
pip install pymongo
echo "MONGODB_URL=mongodb://localhost:27017/chatdb" >> .env
```

---

## 📚 Example Usage

### Run Persistent Memory Demo
```bash
python persistent_memory.py
```

### Run Integration API
```bash
# Start the API server
python langchain_integration.py server

# In another terminal, test the API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?", "user_id": "test_user"}'
```

### Test All Features
```bash
# Run comprehensive tests
python -c "
import asyncio
from langchain_integration import test_integration
asyncio.run(test_integration())
"
```

---

## 🎮 Interactive Demo

### CLI Interface
```bash
python langchain_integration.py
```

### API Endpoints
```
🌐 API Base URL: http://localhost:8000

├── POST /chat           - Chat with LangChain agent
├── GET  /health         - Health check
├── GET  /metrics        - System metrics
├── POST /documents      - Upload documents
├── POST /search         - Search documents
├── POST /webhooks/slack - Slack integration
└── POST /webhooks/discord - Discord integration
```

---

## 🔍 Code Examples

### Persistent Memory Usage
```python
from persistent_memory import PersistentMemoryAgent

# Create agent with persistent memory
agent = PersistentMemoryAgent(user_id="alice")

# Start conversation
agent.start_conversation("AI Ethics")

# Chat with memory
response = agent.process_message("What are ethical concerns with AI?")
print(response)

# Get conversation history
history = agent.get_conversation_history()
```

### FastAPI Integration
```python
from fastapi import FastAPI
from langchain_integration import LangChainIntegration

app = FastAPI()
integration = LangChainIntegration()

@app.post("/chat")
async def chat_endpoint(request: dict):
    return await integration.process_chat(request)
```

---

## 🚀 Production Deployment

### Docker Configuration
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "langchain_integration:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
WORKERS=4
```

### Deployment Commands
```bash
# Local development
uvicorn langchain_integration:app --reload

# Production with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker langchain_integration:app

# Docker deployment
docker build -t langchain-api .
docker run -p 8000:8000 --env-file .env langchain-api
```

---

## 📊 Monitoring & Analytics

### Built-in Metrics
```python
# Get system metrics
metrics = await integration.get_system_metrics()
print(f"Total requests: {metrics['analytics']['total_requests']}")
print(f"Success rate: {metrics['analytics']['success_rate']}%")
```

### Health Check
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy", "timestamp": "...", "service": "..."}
```

---

## 🆘 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check SQLite file
ls -la chat.db

# Test database connection
python -c "import sqlite3; conn = sqlite3.connect('chat.db'); print('✅ SQLite working')"
```

#### Vector Store Issues
```bash
# Check ChromaDB
python -c "import chromadb; client = chromadb.PersistentClient(); print('✅ ChromaDB working')"
```

#### API Server Issues
```bash
# Check port availability
netstat -an | grep 8000

# Test API endpoint
curl http://localhost:8000/health
```

### Quick Fix Commands
```bash
# Recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Clear cache and restart
rm -rf vector_memory/
rm chat.db
python langchain_integration.py server
```

---

## 📖 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM](https://www.sqlalchemy.org/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Production Guide](https://python.langchain.com/docs/guides/deployment)

### Best Practices
- **Database:** Use PostgreSQL for production, SQLite for development
- **Caching:** Implement Redis for distributed caching
- **Monitoring:** Add Prometheus metrics and Grafana dashboards
- **Security:** Implement authentication and rate limiting
- **Scaling:** Use message queues for background processing

### Next Steps
1. **Authentication:** Add JWT or OAuth2 authentication
2. **Rate Limiting:** Implement request throttling
3. **Background Jobs:** Add Celery for async processing
4. **Containerization:** Create Docker and Kubernetes configs
5. **CI/CD:** Set up automated testing and deployment

---

## 📄 License

This project is part of a comprehensive LangChain learning course. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!