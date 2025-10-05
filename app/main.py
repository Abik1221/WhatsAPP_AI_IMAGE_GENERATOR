"""
Main FastAPI application for WhatsApp + Gemini AI Bot
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.utils.logger import setup_logging
from app.routes import webhook, health

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting WhatsApp + Gemini AI Bot")
    logger.info(f"📱 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔧 Debug Mode: {settings.DEBUG}")
    logger.info(f"📊 Log Level: {settings.LOG_LEVEL}")
    logger.info(f"🌐 Base URL: {settings.BASE_URL}")
    
    # Validate critical configuration
    try:
        if not settings.WHATSAPP_TOKEN:
            raise ValueError("WHATSAPP_TOKEN is required")
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        logger.info("✅ All required configuration validated")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down WhatsApp + Gemini AI Bot")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="WhatsApp Gemini AI Bot",
    description="AI-powered dress visualization bot using WhatsApp and Gemini AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Rate limiter setup
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute", "10 per second"]
)
app.state.limiter = limiter

# Add slowapi middleware for rate limiting
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [settings.BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with structured logging
    """
    logger.warning(
        f"HTTP {exc.status_code} error for {request.method} {request.url}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "method": request.method,
            "url": str(request.url),
            "client": get_remote_address(request)
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors
    """
    logger.warning(
        f"Validation error for {request.method} {request.url}: {exc.errors()}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": get_remote_address(request)
        }
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Invalid request data",
            "details": exc.errors(),
            "status_code": 422
        }
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """
    Handle rate limit exceeded errors
    """
    logger.warning(
        f"Rate limit exceeded for {get_remote_address(request)}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": get_remote_address(request)
        }
    )
    return await _rate_limit_exceeded_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all other exceptions
    """
    logger.error(
        f"Unhandled exception for {request.method} {request.url}: {str(exc)}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": get_remote_address(request),
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests
    """
    start_time = time.time()
    
    # Skip logging for health checks in production
    if settings.ENVIRONMENT == "production" and request.url.path == "/health":
        response = await call_next(request)
        return response
    
    logger.info(
        f"📥 Incoming request: {request.method} {request.url}",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": get_remote_address(request),
            "user_agent": request.headers.get("user-agent", "")
        }
    )
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"📤 Response: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.2f}s",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": process_time,
            "client": get_remote_address(request)
        }
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(webhook.router, prefix="/webhook", tags=["webhook"])

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """
    Root endpoint with basic API information
    """
    return {
        "message": "WhatsApp Gemini AI Bot is running!",
        "status": "operational",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else None
    }

@app.get("/info")
@limiter.limit("20/minute")
async def api_info(request: Request):
    """
    API information endpoint
    """
    return {
        "service": "WhatsApp Gemini AI Bot",
        "description": "AI-powered dress visualization using WhatsApp and Gemini AI",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "features": [
            "WhatsApp webhook integration",
            "Gemini AI image processing",
            "Rate limiting",
            "Production-ready deployment"
        ],
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "docs": "/docs" if settings.DEBUG else "disabled in production"
        }
    }

# Health check endpoint at root level for load balancers
@app.get("/healthz")
@app.head("/healthz")
async def healthz():
    """
    Simple health check for load balancers and deployment services
    """
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False  # We have our own logging middleware
    )