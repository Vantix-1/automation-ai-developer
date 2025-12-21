"""
FastAPI Fundamentals - Day 34-36
Simple FastAPI application with Pydantic models
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Learning API",
    description="A simple API for learning FastAPI fundamentals",
    version="1.0.0"
)

# Pydantic Models
class User(BaseModel):
    id: int = Field(..., description="User ID")
    name: str = Field(..., min_length=2, max_length=50, description="User name")
    email: str = Field(..., description="User email")
    role: str = Field("user", description="User role")
    created_at: datetime = Field(default_factory=datetime.now)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, description="Message content")
    user_id: int = Field(..., description="User ID")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    message_id: int = Field(..., description="Message ID")
    processing_time: float = Field(..., description="Processing time in seconds")

# In-memory storage
users_db = []
chat_history = []
message_counter = 0

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to AI Learning API",
        "version": "1.0.0",
        "endpoints": {
            "/users": "GET/POST users",
            "/users/{id}": "GET specific user",
            "/chat": "POST chat messages",
            "/chat/history": "GET chat history"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/users", response_model=List[User])
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum users to return"),
    role: Optional[str] = Query(None, description="Filter by role")
):
    """Get all users with pagination and filtering"""
    filtered_users = users_db
    if role:
        filtered_users = [u for u in users_db if u.role == role]
    
    return filtered_users[skip:skip + limit]

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """Get a specific user by ID"""
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

@app.post("/users", response_model=User, status_code=201)
async def create_user(user: User):
    """Create a new user"""
    # Check if user ID already exists
    if any(u.id == user.id for u in users_db):
        raise HTTPException(status_code=400, detail="User ID already exists")
    
    users_db.append(user)
    return user

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat message and return AI response"""
    global message_counter
    
    # Simple AI response logic (placeholder for actual AI)
    responses = [
        f"I received your message: '{message.message}'",
        f"Hello user {message.user_id}! You said: {message.message}",
        f"Processing your request: {message.message}",
        f"AI response to: '{message.message}'"
    ]
    
    import random
    import time
    
    start_time = time.time()
    response_text = random.choice(responses)
    processing_time = time.time() - start_time
    
    message_counter += 1
    
    # Store in history
    chat_history.append({
        "message": message.message,
        "user_id": message.user_id,
        "timestamp": message.timestamp,
        "response": response_text
    })
    
    return ChatResponse(
        response=response_text,
        message_id=message_counter,
        processing_time=round(processing_time, 3)
    )

@app.get("/chat/history")
async def get_chat_history(
    limit: int = Query(10, ge=1, le=100, description="Number of messages to return"),
    user_id: Optional[int] = Query(None, description="Filter by user ID")
):
    """Get chat history with optional user filtering"""
    history = chat_history
    if user_id:
        history = [msg for msg in chat_history if msg["user_id"] == user_id]
    
    return {
        "count": len(history[-limit:]),
        "history": history[-limit:]
    }

if __name__ == "__main__":
    # Pre-populate with some users
    users_db.extend([
        User(id=1, name="Alice", email="alice@example.com", role="admin"),
        User(id=2, name="Bob", email="bob@example.com", role="user"),
        User(id=3, name="Charlie", email="charlie@example.com", role="user")
    ])
    
    print("🚀 Starting FastAPI server on http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📖 Redoc Documentation: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)