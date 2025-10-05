"""
WhatsApp webhook routes for handling incoming messages and media
"""

import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.rate_limiter import RateLimiter, UserSession
from app.models.schemas import (
    WebhookVerificationRequest,
    WebhookPayload,
    MessageType,
    WebhookResponse,
    ErrorResponse,
    RateLimitResponse
)
from app.services.whatsapp_service import WhatsAppService, get_whatsapp_service
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.image_service import ImageService, get_image_service
from app.utils.logger import get_logger, Timer

# Initialize router and logger
router = APIRouter()
logger = get_logger(__name__)

# Rate limiter for webhook endpoints
limiter = Limiter(key_func=get_remote_address)
router.state.limiter = limiter

# Initialize rate limiter
rate_limiter = RateLimiter()


@router.get("/")
@limiter.limit("30/minute")
async def verify_webhook(
    request: Request,
    hub_mode: str,
    hub_verify_token: str,
    hub_challenge: str
):
    """
    Verify WhatsApp webhook subscription
    
    This endpoint is called by Meta during webhook setup to verify ownership.
    """
    try:
        logger.info(
            "🔐 Webhook verification request",
            extra={
                "hub_mode": hub_mode,
                "hub_verify_token": hub_verify_token[:10] + "..." if hub_verify_token else None,
                "hub_challenge": hub_challenge[:20] + "..." if hub_challenge else None
            }
        )
        
        verification_request = WebhookVerificationRequest(
            hub_mode=hub_mode,
            hub_verify_token=hub_verify_token,
            hub_challenge=hub_challenge
        )
        
        async with WhatsAppService() as whatsapp_service:
            challenge = await whatsapp_service.verify_webhook(
                verification_request.hub_mode,
                verification_request.hub_verify_token,
                verification_request.hub_challenge
            )
        
        logger.info("✅ Webhook verification successful")
        return challenge
        
    except HTTPException as e:
        logger.error(f"❌ Webhook verification failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"❌ Webhook verification error: {e}")
        raise HTTPException(status_code=500, detail="Webhook verification failed")


@router.post("/")
@limiter.limit("100/minute")  # Higher limit for webhook POST
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    whatsapp_service: WhatsAppService = Depends(get_whatsapp_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    image_service: ImageService = Depends(get_image_service)
):
    """
    Handle incoming WhatsApp webhook events
    
    This endpoint receives all messages and events from WhatsApp.
    """
    start_time = time.time()
    
    try:
        # Parse webhook payload
        payload_data = await request.json()
        logger.debug(
            "📨 Received webhook payload",
            extra={"payload": payload_data}
        )
        
        # Validate webhook payload structure
        try:
            webhook_payload = WebhookPayload(**payload_data)
        except Exception as e:
            logger.warning(f"Invalid webhook payload: {e}")
            return {"status": "ignored", "reason": "invalid_payload"}
        
        # Process webhook in background to avoid timeout
        background_tasks.add_task(
            process_webhook_background,
            webhook_payload,
            whatsapp_service,
            gemini_service,
            image_service
        )
        
        processing_time = time.time() - start_time
        logger.info(
            "✅ Webhook accepted for background processing",
            extra={"processing_time": processing_time}
        )
        
        return WebhookResponse(
            success=True,
            message="Webhook accepted for processing",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Webhook handling error: {e}")
        processing_time = time.time() - start_time
        return WebhookResponse(
            success=False,
            message="Webhook processing failed",
            processing_time=processing_time
        )


async def process_webhook_background(
    webhook_payload: WebhookPayload,
    whatsapp_service: WhatsAppService,
    gemini_service: GeminiService,
    image_service: ImageService
):
    """
    Process webhook payload in background
    
    Args:
        webhook_payload: Validated webhook payload
        whatsapp_service: WhatsApp service instance
        gemini_service: Gemini service instance  
        image_service: Image service instance
    """
    try:
        with Timer("Webhook background processing", logger):
            # Extract messages from webhook payload
            for entry in webhook_payload.entry:
                for change in entry.changes:
                    await process_webhook_change(
                        change, 
                        whatsapp_service, 
                        gemini_service,
                        image_service
                    )
                    
    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}")


async def process_webhook_change(
    change: Any,
    whatsapp_service: WhatsAppService,
    gemini_service: GeminiService,
    image_service: ImageService
):
    """
    Process individual webhook change
    
    Args:
        change: Webhook change object
        whatsapp_service: WhatsApp service instance
        gemini_service: Gemini service instance
        image_service: Image service instance
    """
    try:
        value = change.value
        
        # Only process messages
        if not value.messages:
            logger.debug("No messages in webhook change, skipping")
            return
        
        # Process each message
        for message in value.messages:
            await process_message(
                message,
                value.contacts[0] if value.contacts else None,
                whatsapp_service,
                gemini_service,
                image_service
            )
            
    except Exception as e:
        logger.error(f"Change processing failed: {e}")


async def process_message(
    message: Any,
    contact: Optional[Any],
    whatsapp_service: WhatsAppService,
    gemini_service: GeminiService,
    image_service: ImageService
):
    """
    Process individual WhatsApp message
    
    Args:
        message: WhatsApp message object
        contact: Contact information
        whatsapp_service: WhatsApp service instance
        gemini_service: Gemini service instance
        image_service: Image service instance
    """
    user_id = message.from_
    message_id = message.id
    
    try:
        logger.info(
            f"📩 Processing message from {user_id}",
            extra={
                "user_id": user_id,
                "message_id": message_id,
                "message_type": message.type,
                "user_name": contact.profile.name if contact else "Unknown"
            }
        )
        
        # Check rate limiting
        if not await rate_limiter.check_limit(user_id):
            await handle_rate_limit_exceeded(user_id, message_id, whatsapp_service)
            return
        
        # Mark message as read
        await whatsapp_service.mark_message_as_read(message_id)
        
        # Handle different message types
        if message.type == MessageType.IMAGE:
            await handle_image_message(
                message, 
                user_id, 
                message_id,
                whatsapp_service,
                gemini_service,
                image_service
            )
        elif message.type == MessageType.TEXT:
            await handle_text_message(
                message,
                user_id,
                message_id,
                whatsapp_service
            )
        else:
            await handle_unsupported_message(
                message.type,
                user_id,
                message_id,
                whatsapp_service
            )
            
        # Update user activity
        await rate_limiter.update_user_activity(user_id)
        
    except Exception as e:
        logger.error(
            f"❌ Message processing failed for {user_id}: {e}",
            extra={"user_id": user_id, "message_id": message_id}
        )
        await whatsapp_service.handle_error_response(user_id, e, message_id)


async def handle_image_message(
    message: Any,
    user_id: str,
    message_id: str,
    whatsapp_service: WhatsAppService,
    gemini_service: GeminiService,
    image_service: ImageService
):
    """
    Handle incoming image message - the core business logic
    
    Args:
        message: Image message object
        user_id: User WhatsApp ID
        message_id: Message ID
        whatsapp_service: WhatsApp service instance
        gemini_service: Gemini service instance
        image_service: Image service instance
    """
    try:
        with Timer(f"Process image message {message_id}", logger):
            # Send initial acknowledgment
            await whatsapp_service.send_text_message(
                user_id, 
                "🔄 Processing your dress image... This may take a moment."
            )
            
            # Download the image
            image_data, mime_type = await whatsapp_service.download_media(
                message.image.id
            )
            
            logger.info(
                f"📥 Image downloaded successfully",
                extra={
                    "user_id": user_id,
                    "image_size": len(image_data),
                    "mime_type": mime_type
                }
            )
            
            # Validate image
            if not await image_service.validate_image_data(image_data, mime_type):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format or size"
                )
            
            # Process image with Gemini AI
            processing_result = await gemini_service.process_image(
                image_data, 
                mime_type
            )
            
            logger.info(
                f"🎨 Image processed by Gemini AI",
                extra={
                    "user_id": user_id,
                    "output_size": len(processing_result.processed_image_data),
                    "prompt_tokens": processing_result.prompt_tokens
                }
            )
            
            # Send processed image back to user
            await whatsapp_service.send_image_message(
                user_id,
                processing_result.processed_image_data,
                "image/jpeg",  # Gemini returns JPEG
                "✨ Here's your dress on a model! Hope you like it!"
            )
            
            # Update processed image count
            await rate_limiter.increment_processed_count(user_id)
            
            logger.info(
                f"✅ Image processing completed successfully",
                extra={
                    "user_id": user_id,
                    "message_id": message_id,
                    "processing_time": processing_result.processing_time
                }
            )
            
    except HTTPException as e:
        logger.warning(
            f"Image processing failed for {user_id}: {e.detail}",
            extra={"user_id": user_id, "status_code": e.status_code}
        )
        await whatsapp_service.handle_error_response(user_id, e, message_id)
    except Exception as e:
        logger.error(
            f"Unexpected error processing image for {user_id}: {e}",
            extra={"user_id": user_id, "message_id": message_id}
        )
        await whatsapp_service.handle_error_response(user_id, e, message_id)


async def handle_text_message(
    message: Any,
    user_id: str,
    message_id: str,
    whatsapp_service: WhatsAppService
):
    """
    Handle incoming text message
    
    Args:
        message: Text message object
        user_id: User WhatsApp ID
        message_id: Message ID
        whatsapp_service: WhatsApp service instance
    """
    try:
        text_content = message.text.body.lower().strip()
        
        logger.info(
            f"💬 Processing text message from {user_id}",
            extra={"text_content": text_content}
        )
        
        # Handle different text commands
        if text_content in ["hello", "hi", "hey", "start"]:
            response = """👋 Hello! I'm your Dress Visualization Assistant!

Send me a photo of a dress on a mannequin, and I'll show you how it looks on a real model!

📸 Just take a clear photo of the dress and send it here."""
            
        elif text_content in ["help", "info"]:
            response = """🤖 **How to use this bot:**

1. 📸 Take a clear photo of a dress on a mannequin
2. 🖼️ Send the photo to this chat
3. ⏳ Wait a moment while I process it
4. 👗 Receive the transformed image with a real model

💡 **Tips:**
- Ensure good lighting
- Capture the entire dress
- Avoid blurry photos
- Max 5 images per session

Ready to try? Send me a dress photo!"""
            
        elif text_content in ["status", "limit"]:
            user_session = await rate_limiter.get_user_session(user_id)
            remaining = settings.MAX_IMAGES_PER_USER - user_session.images_processed
            response = f"""📊 **Your Session Status:**

🖼️ Images processed: {user_session.images_processed}/{settings.MAX_IMAGES_PER_USER}
🔄 Remaining: {remaining} images
⏰ Session started: {user_session.session_start.strftime('%H:%M:%S')}

Send another dress photo to continue!"""
            
        else:
            response = """🤔 I'm here to help you visualize dresses on models!

Just send me a photo of a dress on a mannequin, and I'll work my magic.

Type 'help' for instructions or send a photo to get started!"""
        
        await whatsapp_service.send_text_message(user_id, response)
        
        logger.info(
            f"✅ Text message handled successfully",
            extra={"user_id": user_id, "response_type": "text_command"}
        )
        
    except Exception as e:
        logger.error(f"Text message handling failed for {user_id}: {e}")
        await whatsapp_service.handle_error_response(user_id, e, message_id)


async def handle_unsupported_message(
    message_type: MessageType,
    user_id: str,
    message_id: str,
    whatsapp_service: WhatsAppService
):
    """
    Handle unsupported message types
    
    Args:
        message_type: Type of unsupported message
        user_id: User WhatsApp ID
        message_id: Message ID
        whatsapp_service: WhatsApp service instance
    """
    try:
        logger.info(
            f"❓ Unsupported message type from {user_id}",
            extra={"message_type": message_type}
        )
        
        response = f"""❌ I can only process images at the moment.

Received: {message_type.value.title()} message

Please send a photo of a dress on a mannequin to get started!"""
        
        await whatsapp_service.send_text_message(user_id, response)
        
    except Exception as e:
        logger.error(f"Unsupported message handling failed for {user_id}: {e}")
        await whatsapp_service.handle_error_response(user_id, e, message_id)


async def handle_rate_limit_exceeded(
    user_id: str,
    message_id: str,
    whatsapp_service: WhatsAppService
):
    """
    Handle rate limit exceeded scenario
    
    Args:
        user_id: User WhatsApp ID
        message_id: Message ID
        whatsapp_service: WhatsApp service instance
    """
    try:
        user_session = await rate_limiter.get_user_session(user_id)
        remaining_time = await rate_limiter.get_remaining_time(user_id)
        
        logger.warning(
            f"🚫 Rate limit exceeded for {user_id}",
            extra={
                "user_id": user_id,
                "images_processed": user_session.images_processed,
                "limit": settings.MAX_IMAGES_PER_USER
            }
        )
        
        response = f"""🚫 **Rate Limit Reached**

You've processed {user_session.images_processed} images in this session.

⏰ Please wait {remaining_time} minutes before sending more images, or start a new session later.

Thank you for using our service! 🙏"""
        
        await whatsapp_service.send_text_message(user_id, response)
        await whatsapp_service.mark_message_as_read(message_id)
        
    except Exception as e:
        logger.error(f"Rate limit handling failed for {user_id}: {e}")


@router.get("/test")
@limiter.limit("10/minute")
async def test_webhook(request: Request):
    """
    Test webhook endpoint for debugging
    """
    return {
        "status": "active",
        "service": "whatsapp-webhook",
        "timestamp": time.time(),
        "environment": settings.ENVIRONMENT
    }


@router.get("/session/{user_id}")
@limiter.limit("20/minute")
async def get_user_session(
    request: Request,
    user_id: str
):
    """
    Get user session information (for debugging)
    """
    try:
        session = await rate_limiter.get_user_session(user_id)
        return {
            "user_id": user_id,
            "session": session.dict() if session else None,
            "rate_limited": not await rate_limiter.check_limit(user_id),
            "remaining_time": await rate_limiter.get_remaining_time(user_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Exception handler for rate limiting
@router.exception_handler(RateLimitExceeded)
async def webhook_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limiting for webhook endpoints"""
    logger.warning(
        f"Webhook rate limit exceeded for {get_remote_address(request)}"
    )
    return await _rate_limit_exceeded_handler(request, exc)