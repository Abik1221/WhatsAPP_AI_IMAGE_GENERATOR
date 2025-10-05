"""
WhatsApp webhook-specific data models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from app.utils.logger import get_logger

logger = get_logger(__name__)


class WebhookChangeField(str, Enum):
    """Webhook change field types"""
    MESSAGES = "messages"
    MESSAGE_STATUS = "message_status"
    MESSAGE_TEMPLATE_STATUS = "message_template_status"


class WebhookMessagingProduct(str, Enum):
    """Messaging product types"""
    WHATSAPP = "whatsapp"


class WebhookMetadata(BaseModel):
    """Webhook metadata"""
    display_phone_number: str = Field(..., description="Display phone number")
    phone_number_id: str = Field(..., description="Phone number ID")


class WebhookProfile(BaseModel):
    """User profile in webhook"""
    name: str = Field(..., description="User's display name")


class WebhookContact(BaseModel):
    """Contact information in webhook"""
    wa_id: str = Field(..., description="WhatsApp ID")
    profile: WebhookProfile = Field(..., description="User profile")


class WebhookContext(BaseModel):
    """Message context for replied messages"""
    from_: str = Field(..., alias="from", description="Original message sender")
    id: str = Field(..., description="Original message ID")
    
    class Config:
        allow_population_by_field_name = True


class WebhookError(BaseModel):
    """Webhook error information"""
    code: int = Field(..., description="Error code")
    details: str = Field(..., description="Error details")
    title: str = Field(..., description="Error title")


class WebhookSystem(BaseModel):
    """System message in webhook"""
    body: str = Field(..., description="System message body")
    identity: str = Field(..., description="System identity")
    wa_id: str = Field(..., description="WhatsApp ID")
    type: str = Field(..., description="System message type")
    customer: str = Field(..., description="Customer identifier")


class WebhookButton(BaseModel):
    """Interactive button in webhook"""
    payload: Optional[str] = Field(None, description="Button payload")
    text: str = Field(..., description="Button text")


class WebhookInteractive(BaseModel):
    """Interactive message in webhook"""
    type: str = Field(..., description="Interactive type")
    button_reply: Optional[WebhookButton] = Field(None, description="Button reply")
    list_reply: Optional[Dict[str, Any]] = Field(None, description="List reply")


class WebhookLocation(BaseModel):
    """Location message in webhook"""
    address: Optional[str] = Field(None, description="Location address")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    name: Optional[str] = Field(None, description="Location name")
    url: Optional[str] = Field(None, description="Location URL")


class WebhookOrder(BaseModel):
    """Order message in webhook"""
    catalog_id: str = Field(..., description="Catalog ID")
    product_items: List[Dict[str, Any]] = Field(..., description="Product items")
    text: Optional[str] = Field(None, description="Order text")


class WebhookReaction(BaseModel):
    """Message reaction in webhook"""
    message_id: str = Field(..., description="Message ID")
    emoji: Optional[str] = Field(None, description="Reaction emoji")


class WebhookReferral(BaseModel):
    """Message referral in webhook"""
    source_url: Optional[str] = Field(None, description="Source URL")
    source_type: str = Field(..., description="Source type")
    source_id: Optional[str] = Field(None, description="Source ID")
    headline: Optional[str] = Field(None, description="Headline")
    body: Optional[str] = Field(None, description="Body text")
    media_type: Optional[str] = Field(None, description="Media type")
    image_url: Optional[str] = Field(None, description="Image URL")
    video_url: Optional[str] = Field(None, description="Video URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    ctwa_clid: Optional[str] = Field(None, description="CTWA click ID")


class WebhookStatus(BaseModel):
    """Message status in webhook"""
    id: str = Field(..., description="Message ID")
    status: str = Field(..., description="Message status")
    timestamp: str = Field(..., description="Status timestamp")
    recipient_id: str = Field(..., description="Recipient ID")
    conversation: Optional[Dict[str, Any]] = Field(None, description="Conversation data")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing data")
    errors: Optional[List[WebhookError]] = Field(None, description="Error list")


class WebhookMessage(BaseModel):
    """Complete webhook message structure"""
    audio: Optional[Dict[str, Any]] = Field(None, description="Audio message")
    button: Optional[Dict[str, Any]] = Field(None, description="Button message")
    context: Optional[WebhookContext] = Field(None, description="Message context")
    document: Optional[Dict[str, Any]] = Field(None, description="Document message")
    errors: Optional[List[WebhookError]] = Field(None, description="Error list")
    from_: str = Field(..., alias="from", description="Sender WhatsApp ID")
    id: str = Field(..., description="Message ID")
    identity: Optional[Dict[str, Any]] = Field(None, description="Identity data")
    image: Optional[Dict[str, Any]] = Field(None, description="Image message")
    interactive: Optional[WebhookInteractive] = Field(None, description="Interactive message")
    location: Optional[WebhookLocation] = Field(None, description="Location message")
    order: Optional[WebhookOrder] = Field(None, description="Order message")
    reaction: Optional[WebhookReaction] = Field(None, description="Reaction message")
    referral: Optional[WebhookReferral] = Field(None, description="Referral message")
    sticker: Optional[Dict[str, Any]] = Field(None, description="Sticker message")
    system: Optional[WebhookSystem] = Field(None, description="System message")
    text: Optional[Dict[str, Any]] = Field(None, description="Text message")
    timestamp: str = Field(..., description="Message timestamp")
    type: str = Field(..., description="Message type")
    video: Optional[Dict[str, Any]] = Field(None, description="Video message")
    
    class Config:
        allow_population_by_field_name = True


class WebhookValue(BaseModel):
    """Webhook value object containing the actual data"""
    messaging_product: WebhookMessagingProduct = Field(..., description="Messaging product")
    metadata: WebhookMetadata = Field(..., description="Message metadata")
    contacts: Optional[List[WebhookContact]] = Field(None, description="Contact list")
    messages: Optional[List[WebhookMessage]] = Field(None, description="Message list")
    statuses: Optional[List[WebhookStatus]] = Field(None, description="Status list")
    errors: Optional[List[WebhookError]] = Field(None, description="Error list")


class WebhookChange(BaseModel):
    """Webhook change object"""
    field: WebhookChangeField = Field(..., description="Changed field")
    value: WebhookValue = Field(..., description="Change value")


class WebhookEntry(BaseModel):
    """Webhook entry object"""
    id: str = Field(..., description="Business account ID")
    changes: List[WebhookChange] = Field(..., description="List of changes")


class WebhookObjectType(str, Enum):
    """Webhook object types"""
    WHATSAPP_BUSINESS_ACCOUNT = "whatsapp_business_account"


class WebhookPayload(BaseModel):
    """Complete webhook payload from WhatsApp"""
    object: WebhookObjectType = Field(..., description="Webhook object type")
    entry: List[WebhookEntry] = Field(..., description="List of entries")
    
    @validator('entry')
    def validate_entry_not_empty(cls, v):
        """Validate that entry list is not empty"""
        if not v:
            raise ValueError("Webhook entry list cannot be empty")
        return v


class WebhookProcessingResult(BaseModel):
    """Result of webhook processing"""
    success: bool = Field(..., description="Processing success")
    message: str = Field(..., description="Processing message")
    user_id: Optional[str] = Field(None, description="User WhatsApp ID")
    message_id: Optional[str] = Field(None, description="Message ID")
    message_type: Optional[str] = Field(None, description="Message type")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebhookVerificationRequest(BaseModel):
    """Webhook verification request model"""
    hub_mode: str = Field(..., alias="hub.mode", description="Webhook mode")
    hub_verify_token: str = Field(..., alias="hub.verify_token", description="Verification token")
    hub_challenge: str = Field(..., alias="hub.challenge", description="Challenge string")
    
    class Config:
        allow_population_by_field_name = True


class WebhookResponse(BaseModel):
    """Standard webhook response"""
    success: bool = Field(..., description="Success indicator")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }