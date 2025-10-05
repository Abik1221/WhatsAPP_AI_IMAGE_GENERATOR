"""
Pydantic models for request/response validation and data serialization
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl, conlist
from datetime import datetime
from enum import Enum


class WebhookMode(str, Enum):
    """Webhook mode enumeration"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class WebhookObject(str, Enum):
    """Webhook object type"""
    WHATSAPP_BUSINESS_ACCOUNT = "whatsapp_business_account"


class MessageType(str, Enum):
    """WhatsApp message types"""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACTS = "contacts"
    INTERACTIVE = "interactive"
    UNSUPPORTED = "unsupported"


class WebhookVerificationRequest(BaseModel):
    """Model for webhook verification request"""
    hub_mode: str = Field(..., alias="hub.mode", description="Webhook mode")
    hub_verify_token: str = Field(..., alias="hub.verify_token", description="Verification token")
    hub_challenge: str = Field(..., alias="hub.challenge", description="Challenge string")

    class Config:
        allow_population_by_field_name = True


class WhatsAppProfile(BaseModel):
    """WhatsApp user profile information"""
    name: str = Field(..., description="User's display name")


class WhatsAppContact(BaseModel):
    """WhatsApp contact information"""
    wa_id: str = Field(..., description="WhatsApp ID of the user")
    profile: WhatsAppProfile = Field(..., description="User profile information")


class Media(BaseModel):
    """Media information for images/files"""
    id: str = Field(..., description="Media ID for downloading")
    mime_type: str = Field(..., description="MIME type of the media")
    sha256: str = Field(..., description="SHA256 hash of the media")
    caption: Optional[str] = Field(None, description="Media caption if provided")


class TextMessage(BaseModel):
    """Text message content"""
    body: str = Field(..., description="Text message body")


class ImageMessage(BaseModel):
    """Image message content"""
    id: str = Field(..., description="Image ID")
    mime_type: str = Field(..., description="Image MIME type")
    sha256: str = Field(..., description="Image SHA256 hash")
    caption: Optional[str] = Field(None, description="Image caption")


class Message(BaseModel):
    """WhatsApp message object"""
    id: str = Field(..., description="Message ID")
    from_: str = Field(..., alias="from", description="Sender WhatsApp ID")
    timestamp: str = Field(..., description="Message timestamp")
    type: MessageType = Field(..., description="Message type")
    
    # Message content (conditional based on type)
    text: Optional[TextMessage] = Field(None, description="Text message content")
    image: Optional[ImageMessage] = Field(None, description="Image message content")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            # WhatsApp timestamps are Unix timestamps
            int(v)
            return v
        except ValueError:
            raise ValueError("Timestamp must be a valid Unix timestamp")
    
    class Config:
        allow_population_by_field_name = True


class Value(BaseModel):
    """Webhook value object containing messages"""
    messaging_product: str = Field(..., description="Messaging product (always 'whatsapp')")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the message")
    contacts: List[WhatsAppContact] = Field(..., description="Contact information")
    messages: List[Message] = Field(..., description="List of messages")


class Change(BaseModel):
    """Webhook change object"""
    value: Value = Field(..., description="Change value")
    field: str = Field(..., description="Field that changed")


class Entry(BaseModel):
    """Webhook entry object"""
    id: str = Field(..., description="Business account ID")
    changes: List[Change] = Field(..., description="List of changes")


class WebhookPayload(BaseModel):
    """Main webhook payload from WhatsApp"""
    object: WebhookObject = Field(..., description="Webhook object type")
    entry: List[Entry] = Field(..., description="List of webhook entries")
    
    @validator('entry')
    def validate_entry_length(cls, v):
        """Validate that entry list has exactly one item"""
        if len(v) != 1:
            raise ValueError("Webhook entry must contain exactly one item")
        return v


class GeminiContentPart(BaseModel):
    """Gemini API content part"""
    inline_data: Dict[str, Any] = Field(..., description="Inline data for image")


class GeminiContent(BaseModel):
    """Gemini API content"""
    parts: List[GeminiContentPart] = Field(..., description="Content parts")
    role: Optional[str] = Field(None, description="Content role")


class GeminiRequest(BaseModel):
    """Gemini API request model"""
    contents: List[GeminiContent] = Field(..., description="Request contents")
    generation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generation configuration"
    )
    safety_settings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Safety settings"
    )


class GeminiCandidate(BaseModel):
    """Gemini API response candidate"""
    content: GeminiContent = Field(..., description="Candidate content")
    finish_reason: str = Field(..., description="Finish reason")
    index: int = Field(..., description="Candidate index")
    avg_logprobs: Optional[float] = Field(None, description="Average log probabilities")


class GeminiUsageMetadata(BaseModel):
    """Gemini API usage metadata"""
    prompt_token_count: int = Field(..., description="Prompt token count")
    candidates_token_count: int = Field(..., description="Candidates token count")
    total_token_count: int = Field(..., description="Total token count")


class GeminiResponse(BaseModel):
    """Gemini API response model"""
    candidates: List[GeminiCandidate] = Field(..., description="Response candidates")
    usage_metadata: Optional[GeminiUsageMetadata] = Field(None, description="Usage metadata")


class WhatsAppMessageRequest(BaseModel):
    """Model for sending messages via WhatsApp API"""
    messaging_product: str = Field(default="whatsapp", description="Messaging product")
    recipient_type: str = Field(default="individual", description="Recipient type")
    to: str = Field(..., description="Recipient WhatsApp ID")
    type: MessageType = Field(..., description="Message type")
    
    # Message content (text or image)
    text: Optional[Dict[str, str]] = Field(None, description="Text message content")
    image: Optional[Dict[str, str]] = Field(None, description="Image message content")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: bool = Field(default=True, description="Error indicator")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Standard success response model"""
    success: bool = Field(default=True, description="Success indicator")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ImageProcessingRequest(BaseModel):
    """Model for image processing requests"""
    image_url: Optional[HttpUrl] = Field(None, description="Image URL")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    prompt: Optional[str] = Field(None, description="Custom prompt for image processing")
    
    @validator('prompt')
    def validate_prompt_length(cls, v):
        """Validate prompt length"""
        if v and len(v) > 1000:
            raise ValueError("Prompt must be less than 1000 characters")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_data_here",
                "prompt": "Replace mannequin with realistic model"
            }
        }


class ImageProcessingResponse(BaseModel):
    """Model for image processing responses"""
    success: bool = Field(..., description="Processing success indicator")
    processed_image_url: Optional[str] = Field(None, description="URL to processed image")
    processed_image_data: Optional[str] = Field(None, description="Base64 encoded processed image")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="AI model used for processing")
    prompt_tokens: Optional[int] = Field(None, description="Number of prompt tokens used")


class UserSession(BaseModel):
    """Model for tracking user sessions and rate limiting"""
    user_id: str = Field(..., description="User WhatsApp ID")
    images_processed: int = Field(default=0, description="Number of images processed in current session")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last user activity timestamp")
    session_start: datetime = Field(default_factory=datetime.utcnow, description="Session start timestamp")
    
    @validator('images_processed')
    def validate_images_processed(cls, v):
        """Validate images processed count"""
        if v < 0:
            raise ValueError("Images processed count cannot be negative")
        return v


class RateLimitResponse(BaseModel):
    """Model for rate limit exceeded responses"""
    error: bool = Field(default=True, description="Error indicator")
    message: str = Field(..., description="Rate limit message")
    retry_after: Optional[int] = Field(None, description="Seconds until retry")
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Rate limit reset time")


class WebhookResponse(BaseModel):
    """Model for webhook responses"""
    success: bool = Field(..., description="Webhook processing success")
    message: str = Field(..., description="Processing message")
    user_id: Optional[str] = Field(None, description="User WhatsApp ID")
    message_id: Optional[str] = Field(None, description="Message ID")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


# Response type aliases for better type hints
WebhookVerificationResponse = str  # Just the challenge string
HealthCheckResponse = Dict[str, Any]
WhatsAppAPIResponse = Dict[str, Any]
GeminiAPIResponse = Dict[str, Any]


class ValidationUtils:
    """Utility methods for data validation"""
    
    @staticmethod
    def validate_phone_number(phone_number: str) -> bool:
        """
        Validate WhatsApp phone number format
        
        Args:
            phone_number: Phone number to validate
            
        Returns:
            bool: True if valid
        """
        # WhatsApp numbers are typically in international format without '+'
        return phone_number.replace(' ', '').isdigit() and len(phone_number) >= 10
    
    @staticmethod
    def validate_image_mime_type(mime_type: str) -> bool:
        """
        Validate image MIME type
        
        Args:
            mime_type: MIME type to validate
            
        Returns:
            bool: True if valid image type
        """
        valid_types = [
            "image/jpeg",
            "image/jpg", 
            "image/png",
            "image/webp"
        ]
        return mime_type.lower() in valid_types
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int = 10 * 1024 * 1024) -> bool:
        """
        Validate file size
        
        Args:
            file_size: File size in bytes
            max_size: Maximum allowed size in bytes
            
        Returns:
            bool: True if within limits
        """
        return 0 < file_size <= max_size