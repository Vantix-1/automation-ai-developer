# 🚀 FastAPI Fundamentals

## 📋 Overview
This section covers the fundamentals of building production-ready APIs with FastAPI. You'll learn to create RESTful APIs with proper validation, documentation, and error handling.

## 📁 File Structure
```
day_34_36_fastapi/
├── requirements.txt              # Python dependencies
├── hello_api.py                  # Basic FastAPI application with Pydantic models
├── fastapi_basics.py             # Advanced FastAPI features
└── api_documentation.py          # Automated documentation generator
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- pip (Python package manager)

### Installation
```bash
# Navigate to the directory
cd phase_2_complete/week_7_8_production/day_34_36_fastapi

# Install dependencies
pip install -r requirements.txt
```

## 📚 Learning Objectives

### Day 34-35: FastAPI Basics
- ✅ Understand FastAPI framework architecture
- ✅ Create RESTful endpoints with proper HTTP methods
- ✅ Implement request/response validation with Pydantic
- ✅ Use path parameters, query parameters, and request bodies
- ✅ Generate automatic OpenAPI documentation

### Day 36: Advanced Features
- ✅ Implement dependency injection
- ✅ Add middleware for CORS and security headers
- ✅ Handle errors with custom exception handlers
- ✅ Create background tasks
- ✅ Build comprehensive API documentation

## 🚦 Quick Start

### 1. Run the Basic API
```bash
python hello_api.py
```

Access the API at:
- **API:** http://localhost:8000
- **Documentation:** http://localhost:8000/docs
- **Redoc:** http://localhost:8000/redoc

### 2. Test Basic Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get all users
curl http://localhost:8000/users

# Create a user
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"id": 100, "name": "John Doe", "email": "john@example.com"}'

# Send chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AI!", "user_id": 100}'
```

### 3. Run Advanced Features
```bash
python fastapi_basics.py
```

### 4. Generate Documentation
```bash
python api_documentation.py
```

## 🔧 Key Features

### 1. Pydantic Models
- Type validation with Python type hints
- Automatic JSON schema generation
- Custom validators and field constraints

### 2. Automatic Documentation
- Interactive OpenAPI (Swagger) documentation
- ReDoc alternative documentation
- Automatic parameter documentation

### 3. Dependency Injection
- Reusable dependencies for authentication
- Database session management
- Request context dependencies

### 4. Error Handling
- HTTPException for standard errors
- Custom exception handlers
- Validation error responses

### 5. Security
- CORS middleware configuration
- Security headers
- Input validation and sanitization

## 📖 Code Examples

### Basic Endpoint
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item, "created": True}
```

### Advanced Features
```python
# Dependency injection
async def verify_token(token: str = Depends(oauth2_scheme)):
    # Verify token logic
    return user

# Background tasks
@app.post("/notifications/")
async def send_notification(
    email: str,
    message: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, message)
    return {"message": "Notification queued"}
```

## 🧪 Testing

### Manual Testing
```bash
# Using curl
curl http://localhost:8000/health

# Using Python requests
import requests
response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Automated Testing
```python
from fastapi.testclient import TestClient
from hello_api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## 🔍 API Reference

### Main Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API information |
| GET | /health | Health check |
| GET | /users | Get all users |
| POST | /users | Create new user |
| GET | /users/{id} | Get user by ID |
| POST | /chat | Send chat message |
| GET | /chat/history | Get chat history |

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

## 🎯 Best Practices Implemented
- **Type Safety:** Full type hints with Pydantic models
- **Documentation:** Automatic OpenAPI generation
- **Error Handling:** Comprehensive error responses
- **Security:** CORS and input validation
- **Modularity:** Separated concerns with routers (in advanced version)
- **Testing:** Easy-to-test endpoints
- **Performance:** Async/await support

## 📈 Next Steps
After mastering this section, you should:
1. Add database integration (SQLAlchemy, databases)
2. Implement JWT authentication
3. Add rate limiting
4. Create API versioning
5. Set up logging and monitoring
6. Containerize with Docker

## 🚨 Troubleshooting

### Common Issues

**Port already in use**
```bash
# Change port
uvicorn hello_api:app --port 8001
```

**Module not found errors**
```bash
# Ensure you're in the right directory
pip install -r requirements.txt
```

**CORS issues**
```python
# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

## 📚 Resources
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI Specification](https://swagger.io/specification/)
- REST API Best Practices

## 🏆 Completion Checklist
- [ ] Created first FastAPI application
- [ ] Implemented Pydantic models for validation
- [ ] Added comprehensive error handling
- [ ] Generated automatic documentation
- [ ] Tested all endpoints
- [ ] Understood dependency injection
- [ ] Learned about middleware
- [ ] Created background tasks