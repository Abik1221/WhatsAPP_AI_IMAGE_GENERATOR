"""
Rate limiting and user session management for WhatsApp bot
"""

import time
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import UserSession, RateLimitResponse
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class SessionData:
    """Internal session data structure"""
    user_id: str
    images_processed: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    session_start: datetime = field(default_factory=datetime.utcnow)
    requests_count: int = 0
    last_request_time: datetime = field(default_factory=datetime.utcnow)


class RateLimiter:
    """
    Rate limiter for user sessions with configurable limits
    """
    
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.cleanup_interval = 3600  # Clean up every hour
        self.session_timeout = 7200  # 2 hours session timeout
        self.request_limit_per_minute = 30
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for user_id, session in self.sessions.items():
                time_since_activity = (current_time - session.last_activity).total_seconds()
                if time_since_activity > self.session_timeout:
                    expired_sessions.append(user_id)
            
            for user_id in expired_sessions:
                del self.sessions[user_id]
                logger.debug(
                    f"Cleaned up expired session for user {user_id}",
                    extra={"user_id": user_id}
                )
            
            if expired_sessions:
                logger.info(
                    f"Cleaned up {len(expired_sessions)} expired sessions",
                    extra={"expired_count": len(expired_sessions)}
                )
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    async def get_user_session(self, user_id: str) -> UserSession:
        """
        Get or create user session
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            UserSession: User session data
        """
        try:
            current_time = datetime.utcnow()
            
            if user_id not in self.sessions:
                # Create new session
                self.sessions[user_id] = SessionData(user_id=user_id)
                logger.info(
                    f"🆕 Created new session for user {user_id}",
                    extra={"user_id": user_id}
                )
            else:
                # Update last activity for existing session
                self.sessions[user_id].last_activity = current_time
            
            session_data = self.sessions[user_id]
            
            return UserSession(
                user_id=session_data.user_id,
                images_processed=session_data.images_processed,
                last_activity=session_data.last_activity,
                session_start=session_data.session_start
            )
            
        except Exception as e:
            logger.error(f"Failed to get user session for {user_id}: {e}")
            # Return a default session on error
            return UserSession(
                user_id=user_id,
                images_processed=0,
                last_activity=datetime.utcnow(),
                session_start=datetime.utcnow()
            )
    
    async def check_limit(self, user_id: str) -> bool:
        """
        Check if user is within rate limits
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            bool: True if within limits, False if rate limited
        """
        try:
            session = await self.get_user_session(user_id)
            current_time = datetime.utcnow()
            
            # Check image processing limit
            if session.images_processed >= settings.MAX_IMAGES_PER_USER:
                logger.warning(
                    f"🚫 Image limit exceeded for user {user_id}",
                    extra={
                        "user_id": user_id,
                        "images_processed": session.images_processed,
                        "max_limit": settings.MAX_IMAGES_PER_USER
                    }
                )
                return False
            
            # Check request rate limit (per minute)
            internal_session = self.sessions.get(user_id)
            if internal_session:
                time_since_last_request = (current_time - internal_session.last_request_time).total_seconds()
                
                if time_since_last_request < 60:  # Within same minute
                    if internal_session.requests_count >= self.request_limit_per_minute:
                        logger.warning(
                            f"🚫 Request rate limit exceeded for user {user_id}",
                            extra={
                                "user_id": user_id,
                                "requests_count": internal_session.requests_count,
                                "limit": self.request_limit_per_minute
                            }
                        )
                        return False
                else:
                    # Reset counter for new minute
                    internal_session.requests_count = 0
                    internal_session.last_request_time = current_time
                
                # Increment request count
                internal_session.requests_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {user_id}: {e}")
            # Allow on error to avoid blocking users
            return True
    
    async def increment_processed_count(self, user_id: str) -> bool:
        """
        Increment processed image count for user
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            bool: True if successful
        """
        try:
            if user_id in self.sessions:
                self.sessions[user_id].images_processed += 1
                self.sessions[user_id].last_activity = datetime.utcnow()
                
                logger.debug(
                    f"Incremented processed count for user {user_id}",
                    extra={
                        "user_id": user_id,
                        "new_count": self.sessions[user_id].images_processed
                    }
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to increment processed count for {user_id}: {e}")
            return False
    
    async def update_user_activity(self, user_id: str) -> bool:
        """
        Update user's last activity timestamp
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            bool: True if successful
        """
        try:
            if user_id in self.sessions:
                self.sessions[user_id].last_activity = datetime.utcnow()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update activity for {user_id}: {e}")
            return False
    
    async def get_remaining_time(self, user_id: str) -> int:
        """
        Get remaining time until session reset (in minutes)
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            int: Remaining minutes until reset
        """
        try:
            if user_id in self.sessions:
                session = self.sessions[user_id]
                time_since_activity = (datetime.utcnow() - session.last_activity).total_seconds()
                remaining_seconds = self.session_timeout - time_since_activity
                
                if remaining_seconds > 0:
                    return max(1, int(remaining_seconds / 60))  # Return at least 1 minute
                
            # Default to 1 hour if no session or expired
            return 60
            
        except Exception as e:
            logger.error(f"Failed to get remaining time for {user_id}: {e}")
            return 60
    
    async def reset_user_session(self, user_id: str) -> bool:
        """
        Reset user session (for testing or manual intervention)
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            bool: True if successful
        """
        try:
            if user_id in self.sessions:
                del self.sessions[user_id]
                logger.info(
                    f"🔄 Reset session for user {user_id}",
                    extra={"user_id": user_id}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to reset session for {user_id}: {e}")
            return False
    
    async def get_session_stats(self) -> Dict[str, any]:
        """
        Get overall session statistics
        
        Returns:
            Dict[str, any]: Session statistics
        """
        try:
            total_sessions = len(self.sessions)
            active_sessions = 0
            total_images_processed = 0
            
            current_time = datetime.utcnow()
            for session in self.sessions.values():
                time_since_activity = (current_time - session.last_activity).total_seconds()
                if time_since_activity < 300:  # Active in last 5 minutes
                    active_sessions += 1
                total_images_processed += session.images_processed
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_images_processed": total_images_processed,
                "max_images_per_user": settings.MAX_IMAGES_PER_USER,
                "session_timeout_minutes": self.session_timeout // 60
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
    
    async def create_rate_limit_response(self, user_id: str) -> RateLimitResponse:
        """
        Create structured rate limit response
        
        Args:
            user_id: User WhatsApp ID
            
        Returns:
            RateLimitResponse: Rate limit response object
        """
        try:
            session = await self.get_user_session(user_id)
            remaining_time = await self.get_remaining_time(user_id)
            
            return RateLimitResponse(
                error=True,
                message=f"Rate limit exceeded. Please wait {remaining_time} minutes.",
                retry_after=remaining_time * 60,  # Convert to seconds
                limit=settings.MAX_IMAGES_PER_USER,
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(minutes=remaining_time)
            )
            
        except Exception as e:
            logger.error(f"Failed to create rate limit response for {user_id}: {e}")
            return RateLimitResponse(
                error=True,
                message="Rate limit exceeded. Please try again later.",
                limit=settings.MAX_IMAGES_PER_USER,
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
    
    def get_session_count(self) -> int:
        """
        Get current number of active sessions
        
        Returns:
            int: Number of sessions
        """
        return len(self.sessions)
    
    async def shutdown(self):
        """Clean shutdown of rate limiter"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Rate limiter shutdown completed")


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitExceededException(HTTPException):
    """Custom exception for rate limiting"""
    
    def __init__(self, user_id: str, detail: str = None):
        super().__init__(
            status_code=429,
            detail=detail or "Rate limit exceeded"
        )
        self.user_id = user_id


async def get_rate_limiter() -> RateLimiter:
    """
    Dependency function for FastAPI to get rate limiter instance
    """
    return rate_limiter


# Utility functions for session management
async def validate_user_session(user_id: str) -> bool:
    """
    Validate user session and check rate limits
    
    Args:
        user_id: User WhatsApp ID
        
    Returns:
        bool: True if session is valid and within limits
    """
    return await rate_limiter.check_limit(user_id)


async def track_image_processing(user_id: str) -> bool:
    """
    Track image processing for user
    
    Args:
        user_id: User WhatsApp ID
        
    Returns:
        bool: True if tracking successful
    """
    return await rate_limiter.increment_processed_count(user_id)


async def get_user_remaining_images(user_id: str) -> int:
    """
    Get remaining images user can process
    
    Args:
        user_id: User WhatsApp ID
        
    Returns:
        int: Number of remaining images
    """
    try:
        session = await rate_limiter.get_user_session(user_id)
        remaining = settings.MAX_IMAGES_PER_USER - session.images_processed
        return max(0, remaining)
    except Exception:
        return settings.MAX_IMAGES_PER_USER


async def get_session_duration(user_id: str) -> int:
    """
    Get session duration in minutes
    
    Args:
        user_id: User WhatsApp ID
        
    Returns:
        int: Session duration in minutes
    """
    try:
        session = await rate_limiter.get_user_session(user_id)
        duration = datetime.utcnow() - session.session_start
        return int(duration.total_seconds() / 60)
    except Exception:
        return 0