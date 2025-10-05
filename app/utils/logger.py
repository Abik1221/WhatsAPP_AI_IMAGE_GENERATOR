"""
Professional logging configuration for WhatsApp + Gemini AI Bot
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from typing import Any, Dict

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """Color formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors for console
        """
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        
        # Create formatted message
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = f"{log_color}{record.levelname:8}{self.COLORS['RESET']}"
        logger_name = f"{record.name}:{record.lineno}" if record.name != "root" else "app"
        
        message = super().format(record)
        
        return f"{formatted_time} | {level_name} | {logger_name:30} | {message}"


def setup_logging() -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("./storage/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main application logger
    logger = logging.getLogger("whatsapp_gemini_bot")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.DEBUG:
        console_handler.setFormatter(ColorFormatter())
    else:
        console_handler.setFormatter(JSONFormatter())
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    logger.addHandler(console_handler)
    
    # File handler for persistent logs (JSON format)
    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Error file handler for errors only
    error_handler = RotatingFileHandler(
        filename=log_dir / "error.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setFormatter(JSONFormatter())
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # Set logging level for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Prevent log propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance with proper configuration
    
    Args:
        name (str): Logger name, typically __name__
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(f"whatsapp_gemini_bot.{name}")


# Context manager for timing blocks of code
class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, operation: str, logger: logging.Logger = None):
        self.operation = operation
        self.logger = logger or get_logger("timer")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"🕒 Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"❌ Failed: {self.operation} - Duration: {duration:.2f}s - Error: {exc_val}",
                extra={"operation": self.operation, "duration": duration, "error": str(exc_val)}
            )
        else:
            self.logger.info(
                f"✅ Completed: {self.operation} - Duration: {duration:.2f}s",
                extra={"operation": self.operation, "duration": duration}
            )


# Utility function for structured logging
def log_operation(operation: str, **extra_fields):
    """
    Decorator for logging function execution
    
    Args:
        operation (str): Description of the operation
        **extra_fields: Additional fields to include in log
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(
                f"🚀 Starting operation: {operation}",
                extra={"operation": operation, **extra_fields}
            )
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"✅ Completed operation: {operation}",
                    extra={"operation": operation, **extra_fields}
                )
                return result
            except Exception as e:
                logger.error(
                    f"❌ Failed operation: {operation} - Error: {str(e)}",
                    extra={"operation": operation, "error": str(e), **extra_fields},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


# Initialize root logger
root_logger = setup_logging()