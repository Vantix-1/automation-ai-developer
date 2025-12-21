"""
AI Chatbot REST API - Day 37-39
Production-ready chatbot API with streaming support
"""
import os
import time
import uuid
from typing import List, Optional, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

import openai
from sse_starlette.sse import EventSourceResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="Production-ready chatbot API with streaming and conversation management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")

# Security
security = HTTPBearer()

# Models
class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be system, user, or assistant')
        return v

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation history")
    model: str = Field("gpt-3.5-turbo", description="OpenAI model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Creativity control")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    stream: bool = Field(False, description="Enable streaming response")

class ChatResponse(BaseModel):
    id: str = Field(..., description="Chat completion ID")
    message: Message = Field(..., description="Assistant's response")
    model: str = Field(..., description="Model used")
    usage: dict = Field(..., description="Token usage statistics")
    created: float = Field(..., description="Unix timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Conversation ID")
    user_id: str = Field(..., description="User identifier")
    messages: List[Message] = Field(default_factory=list, description="All messages in conversation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

# In-memory storage (use database in production)
conversations_db = {}
user_tokens = {"demo-token": "demo-user"}  # Simple token system

# Dependencies
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token and return user info"""
    token = credentials.credentials
    if token not in user_tokens:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return {"user_id": user_tokens[token], "token": token}

async def get_conversation(
    conversation_id: str,
    user_info: dict = Depends(verify_token)
) -> dict:
    """Get conversation by ID with authorization check"""
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversations_db[conversation_id]
    if conversation["user_id"] != user_info["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    
    return conversation

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Chatbot API",
        "version": "2.0.0",
        "description": "Production-ready chatbot API",
        "endpoints": {
            "/chat": "POST - Send chat message",
            "/conversations": "GET - List conversations",
            "/conversations/{id}": "GET - Get conversation",
            "/conversations/{id}": "DELETE - Delete conversation",
            "/health": "GET - Health check"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": "connected" if openai.api_key else "disconnected"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    user_info: dict = Depends(verify_token),
    conversation_id: Optional[str] = Query(None, description="Conversation ID to continue")
):
    """Send chat message and get AI response"""
    start_time = time.time()
    
    try:
        # Prepare messages for OpenAI
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Make API call
        response = openai.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )
        
        # Create response
        assistant_message = Message(
            role="assistant",
            content=response.choices[0].message.content
        )
        
        # Update conversation if conversation_id provided
        if conversation_id and conversation_id in conversations_db:
            conv = conversations_db[conversation_id]
            if conv["user_id"] == user_info["user_id"]:
                conv["messages"].extend([request.messages[-1], assistant_message])
                conv["updated_at"] = datetime.now()
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            id=response.id,
            message=assistant_message,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            created=response.created,
            processing_time=round(processing_time, 3)
        )
        
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_info: dict = Depends(verify_token)
):
    """Stream chat response using Server-Sent Events"""
    
    async def event_generator():
        """Generate SSE events for streaming response"""
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Make streaming API call
            stream = openai.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            full_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # Send each chunk as an SSE event
                    yield {
                        "event": "message",
                        "data": content
                    }
                
                # Simulate delay for demonstration
                # await asyncio.sleep(0.01)
            
            # Send completion event
            yield {
                "event": "complete",
                "data": full_response
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": f"Error: {str(e)}"
            }
    
    return EventSourceResponse(event_generator())

@app.get("/conversations")
async def list_conversations(
    user_info: dict = Depends(verify_token),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's conversations"""
    user_conversations = [
        conv for conv in conversations_db.values()
        if conv["user_id"] == user_info["user_id"]
    ]
    
    # Sort by updated_at descending
    user_conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    
    return {
        "count": len(user_conversations),
        "conversations": user_conversations[offset:offset + limit]
    }

@app.post("/conversations")
async def create_conversation(
    user_info: dict = Depends(verify_token),
    title: Optional[str] = Query(None, description="Optional conversation title")
):
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    
    conversation = {
        "id": conversation_id,
        "user_id": user_info["user_id"],
        "title": title or f"Conversation {len(conversations_db) + 1}",
        "messages": [],
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    conversations_db[conversation_id] = conversation
    
    return {
        "conversation_id": conversation_id,
        "message": "Conversation created successfully"
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation_detail(
    conversation: dict = Depends(get_conversation)
):
    """Get conversation details"""
    return conversation

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_info: dict = Depends(verify_token)
):
    """Delete a conversation"""
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversations_db[conversation_id]
    if conversation["user_id"] != user_info["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    del conversations_db[conversation_id]
    
    return {"message": "Conversation deleted successfully"}

@app.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(
    message: Message,
    conversation: dict = Depends(get_conversation)
):
    """Add a message to existing conversation"""
    conversation["messages"].append(message.dict())
    conversation["updated_at"] = datetime.now()
    
    return {
        "message": "Message added successfully",
        "total_messages": len(conversation["messages"])
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 AI Chatbot API starting up...")
    print(f"📅 Started at: {datetime.now().isoformat()}")
    
    # Create a demo conversation
    demo_conv_id = str(uuid.uuid4())
    conversations_db[demo_conv_id] = {
        "id": demo_conv_id,
        "user_id": "demo-user",
        "title": "Demo Conversation",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello, AI!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ],
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    print("✅ Demo conversation created")
    print("🔑 Demo token: demo-token")
    print("👤 Demo user: demo-user")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Chatbot API...")
    print("📚 OpenAPI docs: http://localhost:8000/docs")
    print("🔐 Use Authorization: Bearer demo-token")
    print("💬 Test endpoint: POST /chat")
    print("📡 Streaming: POST /chat/stream")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )