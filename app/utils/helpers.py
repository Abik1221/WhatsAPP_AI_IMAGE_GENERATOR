"""
Utility helper functions for the WhatsApp + Gemini AI Bot
"""

import os
import re
import uuid
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from fastapi import HTTPException

from app.core.config import settings
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class HelperUtils:
    """Collection of utility helper functions"""
    
    @staticmethod
    def generate_unique_id() -> str:
        """
        Generate a unique ID for requests and tracking
        
        Returns:
            str: Unique identifier
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to remove unsafe characters
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Remove directory path components
        filename = Path(filename).name
        
        # Replace unsafe characters
        filename = re.sub(r'[^\w\s\-_.]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        
        return filename
    
    @staticmethod
    def validate_phone_number(phone_number: str) -> bool:
        """
        Validate WhatsApp phone number format
        
        Args:
            phone_number: Phone number to validate
            
        Returns:
            bool: True if valid WhatsApp number format
        """
        # WhatsApp numbers are typically in international format without '+'
        cleaned = re.sub(r'[\s\-+()]', '', phone_number)
        return cleaned.isdigit() and 10 <= len(cleaned) <= 15
    
    @staticmethod
    def format_phone_number(phone_number: str) -> str:
        """
        Format phone number to standard WhatsApp format
        
        Args:
            phone_number: Raw phone number
            
        Returns:
            str: Formatted phone number
        """
        # Remove all non-digit characters
        cleaned = re.sub(r'[^\d]', '', phone_number)
        
        # Remove leading zeros and country code if needed
        if cleaned.startswith('0'):
            cleaned = cleaned[1:]
        
        return cleaned
    
    @staticmethod
    def calculate_file_hash(data: bytes, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of file data
        
        Args:
            data: File data bytes
            algorithm: Hash algorithm to use
            
        Returns:
            str: Hexadecimal hash string
        """
        hash_func = hashlib.new(algorithm)
        hash_func.update(data)
        return hash_func.hexdigest()
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Formatted file size
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """
        Format timestamp for display
        
        Args:
            timestamp: Datetime object
            
        Returns:
            str: Formatted timestamp string
        """
        return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    @staticmethod
    def get_time_ago(timestamp: datetime) -> str:
        """
        Get human-readable time ago string
        
        Args:
            timestamp: Past datetime
            
        Returns:
            str: Time ago string
        """
        now = datetime.utcnow()
        diff = now - timestamp
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add when truncated
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def safe_json_parse(json_string: str, default: Any = None) -> Any:
        """
        Safely parse JSON string with error handling
        
        Args:
            json_string: JSON string to parse
            default: Default value if parsing fails
            
        Returns:
            Any: Parsed JSON object or default
        """
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return default
    
    @staticmethod
    def safe_json_stringify(obj: Any, default: str = "{}") -> str:
        """
        Safely stringify object to JSON with error handling
        
        Args:
            obj: Object to stringify
            default: Default string if serialization fails
            
        Returns:
            str: JSON string or default
        """
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return default


class RetryManager:
    """Manager for retry operations with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def execute_with_retry(self, async_func, *args, **kwargs):
        """
        Execute async function with retry logic
        
        Args:
            async_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: Last exception after all retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} after {delay}s")
                    await asyncio.sleep(delay)
                
                return await async_func(*args, **kwargs)
                
            except HTTPException as e:
                # Don't retry on HTTP exceptions (client errors)
                raise e
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries}
                )
                
                if attempt == self.max_retries:
                    break
        
        logger.error(
            f"All {self.max_retries + 1} attempts failed",
            extra={"last_error": str(last_exception)}
        )
        raise last_exception


class ValidationHelpers:
    """Validation helper methods"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if valid email format
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if valid URL format
        """
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_image_dimensions(width: int, height: int, min_dim: int = 100, max_dim: int = 4096) -> bool:
        """
        Validate image dimensions
        
        Args:
            width: Image width
            height: Image height
            min_dim: Minimum dimension
            max_dim: Maximum dimension
            
        Returns:
            bool: True if dimensions are valid
        """
        return (min_dim <= width <= max_dim and 
                min_dim <= height <= max_dim and
                width * height <= 16_000_000)  # 16MP limit


class PerformanceUtils:
    """Performance monitoring and optimization utilities"""
    
    @staticmethod
    async def measure_execution_time(async_func, *args, **kwargs) -> tuple:
        """
        Measure execution time of async function
        
        Args:
            async_func: Async function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            tuple: (result, execution_time_seconds)
        """
        start_time = datetime.utcnow()
        result = await async_func(*args, **kwargs)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds()
        return result, execution_time
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}


# Global instances
helper_utils = HelperUtils()
retry_manager = RetryManager()
validation_helpers = ValidationHelpers()
performance_utils = PerformanceUtils()


async def async_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for async function retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Base delay in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = RetryManager(max_retries, delay)
            return await manager.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        List[str]: List of missing field names
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] in [None, ""]:
            missing_fields.append(field)
    return missing_fields