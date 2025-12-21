"""
FastAPI Basics - Advanced Features
"""
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator, EmailStr
from typing import Optional, List, Dict, Any
import time
import json
import asyncio

app = FastAPI(title="Advanced FastAPI Demo")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

class Order(BaseModel):
    items: List[Item]
    total_price: float = 0.0
    
    def calculate_total(self):
        self.total_price = sum(item.price + (item.tax or 0) for item in self.items)

# Dependencies
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency for token verification"""
    token = credentials.credentials
    # In production, validate against database or external service
    if token != "secret-token-123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return {"user_id": 1, "username": "demo_user"}

async def get_db_session():
    """Simulate database session"""
    print("Getting database session...")
    yield {"db": "postgresql://localhost/mydb"}
    print("Closing database session...")

# Routes with dependencies
@app.post("/items/", dependencies=[Depends(verify_token)])
async def create_item(item: Item, user: dict = Depends(verify_token)):
    """Create item with authentication"""
    return {
        "item": item,
        "created_by": user["username"],
        "timestamp": time.time()
    }

@app.post("/orders/")
async def create_order(order: Order, db: dict = Depends(get_db_session)):
    """Create order with database session"""
    order.calculate_total()
    # Simulate database save
    print(f"Saving order to {db['db']}")
    return {
        "order": order.dict(),
        "database": db["db"],
        "status": "created"
    }

# Path and Query parameters
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int,
    item_id: str,
    q: Optional[str] = None,
    short: bool = False
):
    """Demonstrate path and query parameters"""
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({
            "description": "This is a long description that appears only when short=False"
        })
    return item

# Custom response
@app.get("/custom-response/")
async def custom_response():
    """Return custom JSON response"""
    content = {"message": "Custom response", "status": "success"}
    return JSONResponse(
        status_code=200,
        content=content,
        headers={"X-Custom-Header": "custom-value"}
    )

# Streaming response
@app.get("/stream-data/")
async def stream_data():
    """Stream data as JSON lines"""
    async def generate():
        for i in range(10):
            data = {"id": i, "timestamp": time.time(), "data": f"chunk_{i}"}
            yield json.dumps(data) + "\n"
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"}
    )

# Error handling
@app.get("/error-demo/{error_type}")
async def error_demo(error_type: str):
    """Demonstrate different error types"""
    if error_type == "not_found":
        raise HTTPException(status_code=404, detail="Item not found")
    elif error_type == "validation":
        raise HTTPException(
            status_code=422,
            detail="Validation error",
            headers={"X-Error": "Validation failed"}
        )
    elif error_type == "server":
        # Simulate server error
        raise ValueError("Internal server error")
    return {"status": "ok"}

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

# Async endpoints
@app.get("/async-task/")
async def async_task(delay: int = 1):
    """Demonstrate async operations"""
    start_time = time.time()
    await asyncio.sleep(delay)  # Simulate async operation
    processing_time = time.time() - start_time
    
    return {
        "task": "async_task",
        "delay": delay,
        "processing_time": round(processing_time, 3),
        "status": "completed"
    }

# Multiple methods on same path
@app.api_route("/multi-method/", methods=["GET", "POST", "PUT", "DELETE"])
async def multi_method(request_data: Optional[dict] = None):
    """Handle multiple HTTP methods"""
    if request_data:
        return {"method": "POST/PUT", "data": request_data}
    return {"method": "GET/DELETE", "action": "retrieve/delete"}

# Background tasks (FastAPI 0.104+)
from fastapi import BackgroundTasks

def write_log(message: str):
    """Background task to write log"""
    with open("api_log.txt", "a") as f:
        f.write(f"{time.time()}: {message}\n")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    message: str,
    background_tasks: BackgroundTasks
):
    """Send notification with background task"""
    background_tasks.add_task(write_log, f"Notification sent to {email}: {message}")
    
    return {
        "message": "Notification queued",
        "email": email,
        "queued_at": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Advanced FastAPI server...")
    print("📚 OpenAPI docs: http://localhost:8000/docs")
    print("🔐 Try authenticated endpoint: POST /items/ with Authorization: Bearer secret-token-123")
    uvicorn.run(app, host="0.0.0.0", port=8000)