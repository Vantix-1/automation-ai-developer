# phase_2_complete/week_7_8_production/day_40_42_ragapi/api_security.py
"""
API Security Implementation
JWT authentication, rate limiting, input validation, and security headers
"""
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field, validator, EmailStr
import jwt
from jwt.exceptions import InvalidTokenError
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import bcrypt

app = FastAPI(
    title="Secure API Demo",
    description="API with comprehensive security features",
    version="1.0.0"
)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# In production, uncomment:
# app.add_middleware(HTTPSRedirectMiddleware)
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

# Redis for rate limiting and session management (fallback to in-memory)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available, using in-memory rate limiting")

# In-memory storage (use database in production)
users_db = {}
blacklisted_tokens = set()

# Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?`~' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserLogin(BaseModel):
    username: str = Field(...)
    password: str = Field(...)

class Token(BaseModel):
    access_token: str = Field(...)
    refresh_token: str = Field(...)
    token_type: str = Field("bearer")

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None

class SensitiveData(BaseModel):
    data: str = Field(..., description="Sensitive data to store")
    classification: str = Field("confidential", description="Data classification")

# Security utilities
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        # Check if token is blacklisted
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash in blacklisted_tokens:
            raise InvalidTokenError("Token has been revoked")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

def rate_limit_key(request: Request) -> str:
    """Create rate limit key based on user and endpoint"""
    user_id = "anonymous"
    if request.user:
        user_id = request.user.get("user_id", "anonymous")
    
    endpoint = request.url.path
    
    return f"{user_id}:{endpoint}"

# Dependencies
class JWTBearer(HTTPBearer):
    """Custom JWT bearer dependency"""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            payload = verify_token(credentials.credentials)
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid token type"
                )
            
            request.state.user = payload
            return payload
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code"
            )

# Input validation and sanitization
def sanitize_input(input_str: str) -> str:
    """Sanitize input to prevent XSS and injection attacks"""
    import html
    
    # HTML escape
    sanitized = html.escape(input_str)
    
    # Remove potential SQL injection patterns (simplified)
    dangerous_patterns = [
        "--", ";", "'", "\"", "/*", "*/", "@@", "@", "char(", "nchar(",
        "varchar(", "nvarchar(", "alter ", "create ", "drop ", "exec ",
        "execute ", "insert ", "update ", "delete ", "select ", "union "
    ]
    
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, "")
    
    return sanitized.strip()

def validate_file_upload(filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate file upload"""
    allowed_extensions = ['.pdf', '.txt', '.jpg', '.png', '.docx']
    allowed_mime_types = [
        'application/pdf',
        'text/plain',
        'image/jpeg',
        'image/png',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    
    # Check extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return False
    
    # Check MIME type
    if content_type not in allowed_mime_types:
        return False
    
    # Check file size (max_size in bytes)
    # This would be checked when reading the file
    
    return True

# Security headers middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Be specific in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # Remove server header
    if "server" in response.headers:
        del response.headers["server"]
    
    return response

# Routes
@app.post("/auth/register")
@limiter.limit("5/minute")
async def register_user(
    request: Request,
    user: UserCreate
):
    """Register new user"""
    # Check if username or email already exists
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    for existing_user in users_db.values():
        if existing_user["email"] == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Create user
    user_id = str(len(users_db) + 1)
    users_db[user.username] = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.now().isoformat(),
        "is_active": True,
        "failed_login_attempts": 0,
        "last_login": None
    }
    
    # Log registration (in production, use proper logging)
    print(f"User registered: {user.username}")
    
    return {
        "message": "User registered successfully",
        "user_id": user_id,
        "username": user.username
    }

@app.post("/auth/login")
@limiter.limit("10/minute")
async def login_user(
    request: Request,
    user: UserLogin
):
    """Login user and return tokens"""
    # Check if user exists
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_data = users_db[user.username]
    
    # Check if account is locked
    if user_data.get("failed_login_attempts", 0) >= 5:
        raise HTTPException(
            status_code=403,
            detail="Account locked due to too many failed attempts"
        )
    
    # Verify password
    if not verify_password(user.password, user_data["hashed_password"]):
        # Increment failed attempts
        user_data["failed_login_attempts"] = user_data.get("failed_login_attempts", 0) + 1
        
        # Lock account after 5 failed attempts
        if user_data["failed_login_attempts"] >= 5:
            user_data["locked_until"] = (datetime.now() + timedelta(hours=1)).isoformat()
        
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Reset failed attempts on successful login
    user_data["failed_login_attempts"] = 0
    user_data["last_login"] = datetime.now().isoformat()
    
    # Create tokens
    token_data = {"sub": user.username, "user_id": user_data["id"]}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    # Store refresh token (in production, use database)
    if REDIS_AVAILABLE:
        redis_client.setex(
            f"refresh_token:{user.username}",
            REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
            refresh_token
        )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

@app.post("/auth/refresh")
async def refresh_token(
    refresh_token: str = Form(...)
):
    """Refresh access token using refresh token"""
    try:
        payload = verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid token type")
        
        username = payload.get("sub")
        
        # Verify refresh token (in production, check against database)
        if REDIS_AVAILABLE:
            stored_token = redis_client.get(f"refresh_token:{username}")
            if stored_token != refresh_token:
                raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new access token
        token_data = {"sub": username, "user_id": payload.get("user_id")}
        new_access_token = create_access_token(token_data)
        
        return {"access_token": new_access_token, "token_type": "bearer"}
        
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.post("/auth/logout")
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    request: Request = None
):
    """Logout user and blacklist token"""
    token = credentials.credentials
    
    # Blacklist the token
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    blacklisted_tokens.add(token_hash)
    
    # In production, also remove refresh token from database
    if request and hasattr(request.state, 'user'):
        username = request.state.user.get("sub")
        if REDIS_AVAILABLE and username:
            redis_client.delete(f"refresh_token:{username}")
    
    return {"message": "Successfully logged out"}

@app.get("/secure/data")
@limiter.limit("100/minute")
async def get_secure_data(
    request: Request,
    user: dict = Depends(JWTBearer())
):
    """Get secure data (requires authentication)"""
    return {
        "message": "This is secure data",
        "user": user.get("sub"),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/secure/data")
@limiter.limit("60/minute")
async def post_secure_data(
    request: Request,
    data: SensitiveData,
    user: dict = Depends(JWTBearer())
):
    """Post sensitive data with validation"""
    # Sanitize input
    sanitized_data = sanitize_input(data.data)
    
    # Log the action (in production, use audit logging)
    print(f"User {user.get('sub')} posted data: {sanitized_data[:50]}...")
    
    return {
        "message": "Data received securely",
        "data_length": len(sanitized_data),
        "classification": data.classification,
        "processed_at": datetime.now().isoformat()
    }

@app.get("/public/info")
@limiter.limit("500/hour")
async def get_public_info():
    """Public endpoint with rate limiting"""
    return {
        "message": "Public information",
        "timestamp": datetime.now().isoformat(),
        "rate_limit": "500 requests per hour"
    }

# Admin endpoints
@app.get("/admin/users")
async def list_users(
    user: dict = Depends(JWTBearer())
):
    """Admin endpoint to list users (requires admin role)"""
    # Check if user is admin (simplified)
    if user.get("sub") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Return user list without passwords
    user_list = []
    for username, user_data in users_db.items():
        user_list.append({
            "username": username,
            "email": user_data["email"],
            "full_name": user_data["full_name"],
            "created_at": user_data["created_at"],
            "is_active": user_data["is_active"],
            "last_login": user_data.get("last_login")
        })
    
    return {"users": user_list}

# Security monitoring
class SecurityMonitor:
    """Monitor security events"""
    
    def __init__(self):
        self.events = []
    
    def log_event(self, event_type: str, details: dict):
        """Log security event"""
        event = {
            "type": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "ip": details.get("ip", "unknown")
        }
        self.events.append(event)
        
        # Keep only last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        
        # Alert on suspicious activity
        if event_type in ["failed_login", "rate_limit_exceeded"]:
            self.alert_suspicious_activity(event)
    
    def alert_suspicious_activity(self, event: dict):
        """Alert on suspicious activity"""
        print(f"⚠️  SECURITY ALERT: {event['type']}")
        print(f"   Details: {event['details']}")
        print(f"   Timestamp: {event['timestamp']}")
    
    def get_events(self, event_type: Optional[str] = None):
        """Get security events"""
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events

security_monitor = SecurityMonitor()

# Security middleware for monitoring
@app.middleware("http")
async def security_monitoring_middleware(request: Request, call_next):
    """Monitor requests for security events"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Log successful requests for sensitive endpoints
        if "/secure/" in request.url.path:
            security_monitor.log_event("secure_access", {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "processing_time": time.time() - start_time,
                "ip": request.client.host
            })
        
        return response
        
    except HTTPException as e:
        # Log failed requests
        if e.status_code == 401:
            security_monitor.log_event("failed_login", {
                "endpoint": request.url.path,
                "ip": request.client.host,
                "detail": e.detail
            })
        elif e.status_code == 429:
            security_monitor.log_event("rate_limit_exceeded", {
                "endpoint": request.url.path,
                "ip": request.client.host
            })
        
        raise e
    
    except Exception as e:
        # Log unexpected errors
        security_monitor.log_event("server_error", {
            "endpoint": request.url.path,
            "ip": request.client.host,
            "error": str(e)
        })
        raise e

@app.get("/admin/security/events")
async def get_security_events(
    event_type: Optional[str] = None,
    limit: int = 100,
    user: dict = Depends(JWTBearer())
):
    """Get security events (admin only)"""
    if user.get("sub") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    events = security_monitor.get_events(event_type)
    return {
        "count": len(events),
        "events": events[-limit:]
    }

# Health check with security info
@app.get("/health/security")
async def health_security():
    """Health check with security information"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "security": {
            "jwt_enabled": True,
            "rate_limiting": True,
            "password_hashing": "bcrypt",
            "input_sanitization": True,
            "security_headers": True,
            "blacklisted_tokens": len(blacklisted_tokens),
            "registered_users": len(users_db),
            "security_events": len(security_monitor.events)
        }
    }

# Initialize with admin user
@app.on_event("startup")
async def startup_event():
    """Initialize security system on startup"""
    # Create admin user if not exists
    if "admin" not in users_db:
        users_db["admin"] = {
            "id": "0",
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "System Administrator",
            "hashed_password": hash_password("Admin@123!"),  # Change in production!
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "failed_login_attempts": 0,
            "last_login": None
        }
    
    print("🔒 Security API initialized")
    print("👤 Admin user: admin / Admin@123!")
    print("🔑 Use /auth/login to get JWT tokens")

if __name__ == "__main__":
    import uvicorn
    
    print("🔒 Starting Secure API...")
    print("📚 OpenAPI docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=None,  # Add paths for HTTPS
        ssl_certfile=None
    )