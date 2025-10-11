"""
WhatsApp Business API service for handling webhooks, messages, and media
"""

import httpx
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
from pathlib import Path
import aiofiles
import magic
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import (
    WhatsAppMessageRequest,
    MessageType,
    Media,
    ErrorResponse,
    WebhookResponse
)
from app.utils.logger import get_logger, Timer

# Initialize logger
logger = get_logger(__name__)


class WhatsAppService:
    """Service for WhatsApp Business API interactions"""
    
    def __init__(self):
        self.base_url = settings.whatsapp_api_url
        self.headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        self.media_headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        }
        self.client = None
        self.media_client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_clients()
    
    async def initialize_clients(self):
        """Initialize HTTP clients"""
        timeout = httpx.Timeout(settings.REQUEST_TIMEOUT)
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
            follow_redirects=True
        )
        
        self.media_client = httpx.AsyncClient(
            headers=self.media_headers,
            timeout=timeout,
            follow_redirects=True
        )
        
        logger.info("✅ WhatsApp HTTP clients initialized")
    
    async def close_clients(self):
        """Close HTTP clients"""
        if self.client:
            await self.client.aclose()
        if self.media_client:
            await self.media_client.aclose()
        logger.info("✅ WhatsApp HTTP clients closed")
    
    async def verify_webhook(self, mode: str, token: str, challenge: str) -> str:
        """
        Verify WhatsApp webhook subscription
        
        Args:
            mode: hub.mode from request
            token: hub.verify_token from request  
            challenge: hub.challenge from request
            
        Returns:
            str: Challenge string if verification successful
            
        Raises:
            HTTPException: If verification fails
        """
        try:
            if mode != "subscribe":
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid hub.mode"
                )
            
            if token != settings.VERIFY_TOKEN:
                logger.warning(
                    f"Webhook verification failed: invalid token",
                    extra={"received_token": token, "expected_token": settings.VERIFY_TOKEN}
                )
                raise HTTPException(
                    status_code=403,
                    detail="Invalid verification token"
                )
            
            logger.info("✅ Webhook verification successful")
            return challenge
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Webhook verification error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Webhook verification failed"
            )
    
    async def download_media(self, media_id: str) -> Tuple[bytes, str]:
        """
        Download media from WhatsApp servers
        
        Args:
            media_id: WhatsApp media ID
            
        Returns:
            Tuple[bytes, str]: Media content and MIME type
            
        Raises:
            HTTPException: If download fails
        """
        try:
            with Timer(f"Download media {media_id}", logger):
                # Get media URL
                media_url = f"{self.base_url}/media/{media_id}"
                
                logger.info(f"📥 Downloading media: {media_id}")
                
                response = await self.client.get(media_url)
                response.raise_for_status()
                
                media_data = response.json()
                download_url = media_data.get("url")
                mime_type = media_data.get("mime_type")
                
                if not download_url:
                    raise HTTPException(
                        status_code=404,
                        detail="Media URL not found"
                    )
                
                # Validate MIME type
                if not self._validate_mime_type(mime_type):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported media type: {mime_type}"
                    )
                
                # Download actual media content
                download_response = await self.media_client.get(download_url)
                download_response.raise_for_status()
                
                media_content = download_response.content
                
                # Validate file size
                if len(media_content) > settings.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large: {len(media_content)} bytes"
                    )
                
                # Verify MIME type with magic
                detected_mime = magic.from_buffer(media_content, mime=True)
                if detected_mime != mime_type:
                    logger.warning(
                        f"MIME type mismatch: declared={mime_type}, detected={detected_mime}",
                        extra={"media_id": media_id}
                    )
                    mime_type = detected_mime
                
                logger.info(
                    f"✅ Media downloaded successfully",
                    extra={
                        "media_id": media_id,
                        "mime_type": mime_type,
                        "size_bytes": len(media_content)
                    }
                )
                
                return media_content, mime_type
                
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error downloading media {media_id}: {e.response.status_code}",
                extra={"status_code": e.response.status_code, "media_id": media_id}
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to download media: {e.response.text}"
            )
        except httpx.TimeoutException:
            logger.error(f"Timeout downloading media {media_id}")
            raise HTTPException(
                status_code=408,
                detail="Media download timeout"
            )
        except Exception as e:
            logger.error(f"Unexpected error downloading media {media_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Media download failed: {str(e)}"
            )
    
    async def send_text_message(self, to: str, message: str) -> Dict[str, Any]:
        """
        Send text message via WhatsApp
        
        Args:
            to: Recipient WhatsApp ID
            message: Text message content
            
        Returns:
            Dict[str, Any]: WhatsApp API response
            
        Raises:
            HTTPException: If message sending fails
        """
        try:
            with Timer(f"Send text message to {to}", logger):
                payload = WhatsAppMessageRequest(
                    to=to,
                    type=MessageType.TEXT,
                    text={"body": message}
                ).dict(exclude_none=True)
                
                logger.info(f"💬 Sending text message to {to}")
                
                response = await self.client.post("/messages", json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                logger.info(
                    f"✅ Text message sent successfully",
                    extra={
                        "to": to,
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "message_preview": message[:50] + "..." if len(message) > 50 else message
                    }
                )
                
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error sending text to {to}: {e.response.status_code}",
                extra={"status_code": e.response.status_code, "to": to}
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to send message: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Unexpected error sending text to {to}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Message sending failed: {str(e)}"
            )
    
    async def send_image_message(self, to: str, image_data: bytes, mime_type: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """
        Send image message via WhatsApp
        
        Args:
            to: Recipient WhatsApp ID
            image_data: Image binary data
            mime_type: Image MIME type
            caption: Optional image caption
            
        Returns:
            Dict[str, Any]: WhatsApp API response
            
        Raises:
            HTTPException: If message sending fails
        """
        try:
            with Timer(f"Send image message to {to}", logger):
                # First upload media to get media ID
                media_id = await self._upload_media(image_data, mime_type)
                
                # Send message with media ID
                image_payload = {"id": media_id}
                if caption:
                    image_payload["caption"] = caption
                
                payload = WhatsAppMessageRequest(
                    to=to,
                    type=MessageType.IMAGE,
                    image=image_payload
                ).dict(exclude_none=True)
                
                logger.info(f"🖼️ Sending image message to {to}")
                
                response = await self.client.post("/messages", json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                logger.info(
                    f"✅ Image message sent successfully",
                    extra={
                        "to": to,
                        "media_id": media_id,
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "caption": caption
                    }
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Error sending image to {to}: {e}")
            raise
    
    async def _upload_media(self, media_data: bytes, mime_type: str) -> str:
        """
        Upload media to WhatsApp servers and get media ID
        
        Args:
            media_data: Media binary data
            mime_type: Media MIME type
            
        Returns:
            str: WhatsApp media ID
            
        Raises:
            HTTPException: If upload fails
        """
        try:
            # Determine file extension from MIME type
            extension_map = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg", 
                "image/png": ".png",
                "image/webp": ".webp"
            }
            extension = extension_map.get(mime_type, ".bin")
            
            # Prepare form data
            files = {
                "file": (f"media{extension}", media_data, mime_type),
                "type": (None, mime_type),
                "messaging_product": (None, "whatsapp")
            }
            
            logger.info(f"📤 Uploading media to WhatsApp")
            
            upload_url = f"{self.base_url}/media"
            response = await self.client.post(upload_url, files=files)
            response.raise_for_status()
            
            result = response.json()
            media_id = result.get("id")
            
            if not media_id:
                raise HTTPException(
                    status_code=500,
                    detail="No media ID received from WhatsApp"
                )
            
            logger.info(
                f"✅ Media uploaded successfully",
                extra={"media_id": media_id, "mime_type": mime_type}
            )
            
            return media_id
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error uploading media: {e.response.status_code}",
                extra={"status_code": e.response.status_code}
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Media upload failed: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Unexpected error uploading media: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Media upload failed: {str(e)}"
            )
    
    def _validate_mime_type(self, mime_type: str) -> bool:
        """
        Validate that MIME type is supported
        
        Args:
            mime_type: MIME type to validate
            
        Returns:
            bool: True if valid
        """
        return mime_type.lower() in settings.ALLOWED_MIME_TYPES
    
    async def mark_message_as_read(self, message_id: str) -> bool:
        """
        Mark message as read
        
        Args:
            message_id: WhatsApp message ID
            
        Returns:
            bool: True if successful
        """
        try:
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            
            response = await self.client.post("/messages", json=payload)
            response.raise_for_status()
            
            logger.debug(f"✅ Message {message_id} marked as read")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as read: {e}")
            return False
    
    async def get_media_info(self, media_id: str) -> Optional[Dict[str, Any]]:
        """
        Get media information
        
        Args:
            media_id: WhatsApp media ID
            
        Returns:
            Optional[Dict[str, Any]]: Media information or None
        """
        try:
            response = await self.client.get(f"/media/{media_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get media info for {media_id}: {e}")
            return None
    
    async def handle_error_response(self, user_id: str, error: Exception, message_id: str = None):
        """
        Send appropriate error message to user
        
        Args:
            user_id: User WhatsApp ID
            error: Exception that occurred
            message_id: Original message ID (for context)
        """
        try:
            error_messages = {
                400: "Sorry, I couldn't process that image. Please make sure it's a clear photo of a dress on a mannequin.",
                403: "Authentication error. Please contact support.",
                404: "The media was not found. Please try sending the image again.",
                408: "The request timed out. Please try again in a moment.",
                413: "The image is too large. Please send a smaller image (max 10MB).",
                415: "Unsupported image format. Please send JPEG, PNG, or WebP.",
                429: "Too many requests. Please wait a moment before trying again.",
                500: "Service temporarily unavailable. Please try again shortly.",
                503: "Service busy. Please try again in a moment."
            }
            
            status_code = getattr(error, 'status_code', 500)
            user_message = error_messages.get(status_code, "An unexpected error occurred. Please try again.")
            
            # Add troubleshooting tips for common issues
            if status_code == 400:
                user_message += "\n\n💡 Tip: Make sure the image shows a dress clearly and the mannequin is visible."
            
            await self.send_text_message(user_id, user_message)
            
            # Mark original message as read if we have message_id
            if message_id:
                await self.mark_message_as_read(message_id)
                
        except Exception as e:
            logger.error(f"Failed to send error message to {user_id}: {e}")


# Global service instance
whatsapp_service = WhatsAppService()


async def get_whatsapp_service() -> AsyncGenerator[WhatsAppService, None]:
    """
    Dependency function for FastAPI to get WhatsApp service instance
    """
    async with WhatsAppService() as service:
        yield service