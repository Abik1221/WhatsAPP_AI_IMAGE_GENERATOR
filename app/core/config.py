import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import logging

class Settings(BaseSettings):
    """Application settings with environment variable validation"""
    
    # WhatsApp API Configuration
    WHATSAPP_TOKEN: str = Field(..., description="WhatsApp Business API Permanent Token")
    WHATSAPP_PHONE_NUMBER_ID: str = Field(..., description="WhatsApp Business Phone Number ID")
    WHATSAPP_APP_ID: str = Field(..., description="WhatsApp Business App ID")
    WHATSAPP_APP_SECRET: str = Field(..., description="WhatsApp Business App Secret")
    VERIFY_TOKEN: str = Field(..., description="Webhook verification token")
    
    # Gemini AI Configuration
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-exp", description="Gemini model to use")
    
    # Prompt Configuration
    IMAGE_PROMPT: str = Field(
        default="Replace the mannequin with a realistic human model while keeping the background exactly the same. The dress should fit naturally on the model. Show appropriate slit and flow. Make it look professional and realistic.",
        description="Preset prompt for image transformation"
    )
    
    # Application Configuration
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="production", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Rate Limiting
    MAX_IMAGES_PER_USER: int = Field(default=5, ge=1, le=20, description="Max images per user session")
    REQUEST_TIMEOUT: int = Field(default=30, ge=10, le=120, description="API timeout in seconds")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Server port")
    BASE_URL: str = Field(..., description="Base URL for webhook (e.g., https://yourapp.render.com)")
    
    # Temporary Storage
    TEMP_DIR: str = Field(default="./storage/temp", description="Temporary file directory")
    MAX_FILE_SIZE: int = Field(default=10_485_760, description="Max file size in bytes (10MB)")
    
    # Allowed MIME Types
    ALLOWED_MIME_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="Allowed image MIME types"
    )
    
    @validator("BASE_URL")
    def validate_base_url(cls, v):
        """Validate base URL format"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("BASE_URL must start with http:// or https://")
        return v.rstrip('/')
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()
    
    @property
    def whatsapp_api_url(self) -> str:
        """Get WhatsApp API base URL"""
        return f"https://graph.facebook.com/v18.0/{self.WHATSAPP_PHONE_NUMBER_ID}"
    
    @property
    def gemini_api_url(self) -> str:
        """Get Gemini API base URL"""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.GEMINI_MODEL}:generateContent"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

# Global settings instance
settings = Settings()

def create_temp_directories():
    """Create necessary temporary directories"""
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs("./storage/logs", exist_ok=True)

# Create directories on import
create_temp_directories()