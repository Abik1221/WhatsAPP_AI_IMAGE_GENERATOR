"""
Health check endpoints for monitoring and load balancers
"""

import time
import psutil
import logging
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.utils.logger import get_logger

# Initialize router and logger
router = APIRouter()
logger = get_logger(__name__)

# Rate limiter for health endpoints
limiter = Limiter(key_func=get_remote_address)
router.state.limiter = limiter

# Store startup time for uptime calculation
STARTUP_TIME = datetime.utcnow()

class HealthCheck:
    """Health check service class"""
    
    def __init__(self):
        self.checks = {
            "system": self.check_system_resources,
            "memory": self.check_memory,
            "disk": self.check_disk_space,
            "whatsapp": self.check_whatsapp_config,
            "gemini": self.check_gemini_config,
        }
    
    def check_system_resources(self) -> dict:
        """Check CPU and system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return {
                "status": "healthy" if cpu_percent < 90 else "degraded",
                "cpu_percent": round(cpu_percent, 2),
                "load_avg": load_avg,
                "active_processes": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"System resources check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_memory(self) -> dict:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "status": "healthy" if memory.percent < 85 else "degraded",
                "memory_used_percent": round(memory.percent, 2),
                "memory_used_gb": round(memory.used / (1024 ** 3), 2),
                "memory_total_gb": round(memory.total / (1024 ** 3), 2),
                "swap_used_percent": round(swap.percent, 2) if swap else 0
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_disk_space(self) -> dict:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                "status": "healthy" if disk.percent < 90 else "degraded",
                "disk_used_percent": round(disk.percent, 2),
                "disk_used_gb": round(disk.used / (1024 ** 3), 2),
                "disk_total_gb": round(disk.total / (1024 ** 3), 2),
                "disk_read_mb": round(disk_io.read_bytes / (1024 ** 2), 2) if disk_io else 0,
                "disk_write_mb": round(disk_io.write_bytes / (1024 ** 2), 2) if disk_io else 0
            }
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_whatsapp_config(self) -> dict:
        """Check WhatsApp configuration"""
        try:
            # Basic configuration validation
            required_configs = [
                'WHATSAPP_TOKEN',
                'WHATSAPP_PHONE_NUMBER_ID', 
                'VERIFY_TOKEN'
            ]
            
            missing_configs = []
            for config in required_configs:
                if not getattr(settings, config, None):
                    missing_configs.append(config)
            
            status = "healthy" if not missing_configs else "unhealthy"
            
            return {
                "status": status,
                "config_loaded": len(missing_configs) == 0,
                "missing_configs": missing_configs,
                "phone_number_id": settings.WHATSAPP_PHONE_NUMBER_ID if status == "healthy" else None
            }
        except Exception as e:
            logger.error(f"WhatsApp config check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_gemini_config(self) -> dict:
        """Check Gemini AI configuration"""
        try:
            has_api_key = bool(settings.GEMINI_API_KEY)
            has_model = bool(settings.GEMINI_MODEL)
            
            return {
                "status": "healthy" if has_api_key and has_model else "unhealthy",
                "api_key_configured": has_api_key,
                "model_configured": has_model,
                "model_name": settings.GEMINI_MODEL if has_model else None
            }
        except Exception as e:
            logger.error(f"Gemini config check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def run_all_checks(self) -> dict:
        """Run all health checks"""
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                # If any check is unhealthy, overall status is degraded
                if result.get("status") == "unhealthy":
                    overall_status = "unhealthy"
                elif result.get("status") == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                results[check_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": results
        }

# Initialize health check service
health_checker = HealthCheck()

@router.get("/")
@limiter.limit("30/minute")
async def health_root(request: Request):
    """
    Basic health check endpoint
    """
    uptime = (datetime.utcnow() - STARTUP_TIME).total_seconds()
    
    return {
        "status": "healthy",
        "service": "whatsapp-gemini-bot",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

@router.get("/detailed")
@limiter.limit("10/minute")
async def health_detailed(request: Request):
    """
    Detailed health check with system metrics
    """
    try:
        health_data = health_checker.run_all_checks()
        uptime = (datetime.utcnow() - STARTUP_TIME).total_seconds()
        
        response = {
            **health_data,
            "uptime_seconds": round(uptime, 2),
            "uptime_human": str(datetime.utcnow() - STARTUP_TIME).split('.')[0],
            "service": "whatsapp-gemini-bot",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT
        }
        
        # Log health check if not healthy
        if health_data["status"] != "healthy":
            logger.warning(
                f"Health check status: {health_data['status']}",
                extra={"health_status": health_data["status"], "checks": health_data["checks"]}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/ready")
@limiter.limit("20/minute")
async def readiness_probe(request: Request):
    """
    Readiness probe for Kubernetes and load balancers
    """
    try:
        health_data = health_checker.run_all_checks()
        
        # For readiness, we require core services to be healthy
        core_services = ["whatsapp", "gemini"]
        core_healthy = all(
            health_data["checks"].get(service, {}).get("status") == "healthy"
            for service in core_services
        )
        
        is_ready = health_data["status"] != "unhealthy" and core_healthy
        
        if not is_ready:
            logger.warning(
                "Readiness probe failed",
                extra={"health_status": health_data["status"], "core_healthy": core_healthy}
            )
            raise HTTPException(status_code=503, detail="Service not ready")
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Readiness check failed")

@router.get("/live")
@limiter.limit("30/minute")
async def liveness_probe(request: Request):
    """
    Liveness probe for Kubernetes
    """
    try:
        # Basic liveness check - just see if the application is responsive
        uptime = (datetime.utcnow() - STARTUP_TIME).total_seconds()
        
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": round(uptime, 2)
        }
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")

@router.get("/config")
@limiter.limit("5/minute")
async def config_check(request: Request):
    """
    Configuration check endpoint (limited for security)
    """
    try:
        # Only show non-sensitive config in development
        if settings.ENVIRONMENT != "development":
            return {
                "message": "Configuration endpoint only available in development mode",
                "environment": settings.ENVIRONMENT
            }
        
        config_summary = {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "log_level": settings.LOG_LEVEL,
            "max_images_per_user": settings.MAX_IMAGES_PER_USER,
            "request_timeout": settings.REQUEST_TIMEOUT,
            "base_url": settings.BASE_URL,
            "whatsapp_phone_number_id": settings.WHATSAPP_PHONE_NUMBER_ID,
            "gemini_model": settings.GEMINI_MODEL,
            "allowed_mime_types": settings.ALLOWED_MIME_TYPES
        }
        
        return config_summary
        
    except Exception as e:
        logger.error(f"Config check failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration check failed")

# Exception handler for rate limiting
@router.exception_handler(RateLimitExceeded)
async def health_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limiting for health endpoints"""
    logger.warning(
        f"Health endpoint rate limit exceeded for {get_remote_address(request)}"
    )
    return await _rate_limit_exceeded_handler(request, exc)