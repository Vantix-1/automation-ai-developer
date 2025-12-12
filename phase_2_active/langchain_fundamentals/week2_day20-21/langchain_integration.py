"""
🔗 LangChain Integration Patterns
Day 21: Production integration with web APIs, databases, and external services
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

load_dotenv()

# =========== DATA MODELS ===========

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous")
    conversation_id: Optional[str] = None
    stream: bool = Field(default=False)

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    conversation_id: str
    message_id: str
    timestamp: str

class DocumentUpload(BaseModel):
    """Request model for document upload"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    """Request model for search"""
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=100)

# =========== INTEGRATION SERVICES ===========

class DatabaseService:
    """Service for database integration"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("DATABASE_URL", "sqlite:///./chat.db")
        
    async def save_message(self, user_id: str, conversation_id: str, 
                          role: str, content: str) -> str:
        """Save message to database (async)"""
        # In production, use async database driver
        message_id = f"msg_{datetime.now().timestamp()}"
        
        # Simulate database operation
        await asyncio.sleep(0.01)
        return message_id
    
    async def get_conversation_history(self, conversation_id: str, 
                                      limit: int = 50) -> List[Dict]:
        """Get conversation history from database"""
        # Simulated data
        return [
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-15T10:30:00"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you! How can I help you today?",
                "timestamp": "2024-01-15T10:30:05"
            }
        ]

class CacheService:
    """Service for caching"""
    
    def __init__(self):
        self.cache = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        self.cache[key] = {
            "value": value,
            "expires_at": datetime.now().timestamp() + ttl
        }
    
    async def delete(self, key: str):
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]

class AnalyticsService:
    """Service for analytics and monitoring"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0
        }
    
    async def track_request(self, user_id: str, endpoint: str):
        """Track API request"""
        self.metrics["total_requests"] += 1
        
    async def track_response(self, success: bool, response_time: float):
        """Track API response"""
        if success:
            self.metrics["successful_responses"] += 1
        else:
            self.metrics["failed_responses"] += 1
        
        # Update average response time
        total = self.metrics["successful_responses"] + self.metrics["failed_responses"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

# =========== LANGCHAIN INTEGRATION ===========

class LangChainIntegration:
    """Main integration class combining all services"""
    
    def __init__(self):
        # Initialize services
        self.db_service = DatabaseService()
        self.cache_service = CacheService()
        self.analytics_service = AnalyticsService()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            streaming=True
        )
        
        # Create tools for agent
        self.tools = self._create_tools()
        
        # Create conversation memory
        self.conversations = {}  # In production, use distributed cache
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        def search_knowledge_base(query: str) -> str:
            """Search knowledge base"""
            return f"Found information about: {query}"
        
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions"""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except:
                return "Error: Invalid expression"
        
        def get_current_time() -> str:
            """Get current time"""
            return datetime.now().isoformat()
        
        return [
            Tool(
                name="KnowledgeSearch",
                func=search_knowledge_base,
                description="Search knowledge base for information"
            ),
            Tool(
                name="Calculator",
                func=calculator,
                description="Calculate mathematical expressions"
            ),
            Tool(
                name="Time",
                func=get_current_time,
                description="Get current date and time"
            )
        ]
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with full integration"""
        
        start_time = datetime.now()
        
        try:
            # Track request
            await self.analytics_service.track_request(
                request.user_id, 
                "chat"
            )
            
            # Check cache for similar queries
            cache_key = f"chat:{request.user_id}:{hash(request.message)}"
            cached_response = await self.cache_service.get(cache_key)
            
            if cached_response:
                response_text = cached_response["value"]
            else:
                # Create or get conversation memory
                if request.conversation_id not in self.conversations:
                    self.conversations[request.conversation_id] = ConversationBufferMemory()
                
                memory = self.conversations[request.conversation_id]
                
                # Create conversation chain
                conversation = ConversationChain(
                    llm=self.llm,
                    memory=memory,
                    verbose=True
                )
                
                # Generate response
                response_text = conversation.predict(input=request.message)
                
                # Cache response
                await self.cache_service.set(cache_key, response_text, ttl=300)
            
            # Save to database
            message_id = await self.db_service.save_message(
                request.user_id,
                request.conversation_id or "new",
                "user",
                request.message
            )
            
            # Track successful response
            response_time = (datetime.now() - start_time).total_seconds()
            await self.analytics_service.track_response(True, response_time)
            
            return ChatResponse(
                response=response_text,
                conversation_id=request.conversation_id or "new",
                message_id=message_id,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Track failed response
            response_time = (datetime.now() - start_time).total_seconds()
            await self.analytics_service.track_response(False, response_time)
            
            raise HTTPException(
                status_code=500,
                detail=f"Chat processing failed: {str(e)}"
            )
    
    async def process_chat_stream(self, request: ChatRequest):
        """Process chat with streaming response"""
        
        async def generate_stream():
            """Generate streaming response"""
            # Simulate streaming for demonstration
            words = f"Processing your message: {request.message}".split()
            
            for word in words:
                yield f"data: {json.dumps({'token': word})}\n\n"
                await asyncio.sleep(0.1)
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        return await self.db_service.get_conversation_history(conversation_id)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            "analytics": self.analytics_service.get_metrics(),
            "cache_size": len(self.cache_service.cache),
            "active_conversations": len(self.conversations),
            "timestamp": datetime.now().isoformat()
        }

# =========== FASTAPI APPLICATION ===========

@dataclass
class AppState:
    """Application state"""
    langchain_integration: LangChainIntegration

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting LangChain Integration API...")
    state = AppState(langchain_integration=LangChainIntegration())
    app.state.state = state
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    # Cleanup resources

app = FastAPI(
    title="LangChain Integration API",
    description="Production-ready LangChain integration with FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========== API ENDPOINTS ===========

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Chat endpoint with LangChain integration"""
    state: AppState = app.state.state
    
    if request.stream:
        return await state.langchain_integration.process_chat_stream(request)
    else:
        return await state.langchain_integration.process_chat(request)

@app.get("/conversations/{conversation_id}/history")
async def get_history(conversation_id: str):
    """Get conversation history"""
    state: AppState = app.state.state
    history = await state.langchain_integration.get_conversation_history(conversation_id)
    return {"conversation_id": conversation_id, "history": history}

@app.post("/documents")
async def upload_document(document: DocumentUpload):
    """Upload document for processing"""
    # In production, process document with LangChain document loaders
    return {
        "document_id": f"doc_{datetime.now().timestamp()}",
        "status": "processed",
        "chunks": len(document.content.split()) // 100 + 1
    }

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search documents"""
    # Simulated search results
    results = [
        {
            "id": f"result_{i}",
            "content": f"Relevant content about {request.query}",
            "score": 0.9 - (i * 0.1),
            "metadata": request.filters
        }
        for i in range(min(request.limit, 5))
    ]
    
    return {
        "query": request.query,
        "results": results,
        "total": len(results)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "LangChain Integration API"
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    state: AppState = app.state.state
    metrics = await state.langchain_integration.get_system_metrics()
    return metrics

# =========== WEBHOOK HANDLERS ===========

class WebhookHandler:
    """Handler for external webhooks"""
    
    @staticmethod
    async def handle_slack_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack webhook"""
        if "challenge" in payload:
            return {"challenge": payload["challenge"]}
        
        # Process Slack message
        event = payload.get("event", {})
        text = event.get("text", "")
        user = event.get("user", "")
        
        return {
            "response": f"Processed Slack message from {user}: {text[:50]}...",
            "status": "success"
        }
    
    @staticmethod
    async def handle_discord_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Discord webhook"""
        content = payload.get("content", "")
        author = payload.get("author", {}).get("username", "unknown")
        
        return {
            "response": f"Processed Discord message from {author}",
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

@app.post("/webhooks/slack")
async def slack_webhook(payload: Dict[str, Any]):
    """Slack webhook endpoint"""
    return await WebhookHandler.handle_slack_webhook(payload)

@app.post("/webhooks/discord")
async def discord_webhook(payload: Dict[str, Any]):
    """Discord webhook endpoint"""
    return await WebhookHandler.handle_discord_webhook(payload)

# =========== DEPLOYMENT CONFIGURATION ===========

class DeploymentConfig:
    """Deployment configuration"""
    
    @staticmethod
    def get_config():
        """Get deployment configuration"""
        return {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", 8000)),
            "reload": os.getenv("ENVIRONMENT", "development") == "development",
            "workers": int(os.getenv("WORKERS", 1)),
            "log_level": os.getenv("LOG_LEVEL", "info")
        }

# =========== DEMO AND TESTING ===========

def run_demo_api():
    """Run the demo API server"""
    config = DeploymentConfig.get_config()
    
    print(f"🌐 Starting demo API at http://{config['host']}:{config['port']}")
    print(f"📚 API Documentation: http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run(
        "langchain_integration:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"]
    )

async def test_integration():
    """Test the integration system"""
    print("🧪 Testing LangChain Integration...")
    
    integration = LangChainIntegration()
    
    # Test chat processing
    request = ChatRequest(
        message="What is LangChain?",
        user_id="test_user",
        conversation_id="test_conv"
    )
    
    try:
        response = await integration.process_chat(request)
        print(f"✅ Chat test successful:")
        print(f"   Response: {response.response[:100]}...")
        print(f"   Conversation ID: {response.conversation_id}")
        
        # Test metrics
        metrics = await integration.get_system_metrics()
        print(f"✅ Metrics test successful:")
        print(f"   Total requests: {metrics['analytics']['total_requests']}")
        print(f"   Average response time: {metrics['analytics']['average_response_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def demo_cli_interface():
    """Demo CLI interface for the integration"""
    print("=" * 60)
    print("🔗 LANGCHAIN INTEGRATION DEMO (Day 21)")
    print("=" * 60)
    
    print("\n🎯 Available Commands:")
    print("  1. Start API Server")
    print("  2. Test Integration")
    print("  3. Show Configuration")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        run_demo_api()
    elif choice == "2":
        asyncio.run(test_integration())
    elif choice == "3":
        config = DeploymentConfig.get_config()
        print(f"\n📋 Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_demo_api()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_integration())
    else:
        demo_cli_interface()