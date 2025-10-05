"""
Gemini AI service for image processing and transformation
"""

import base64
import httpx
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
import json
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import (
    GeminiRequest,
    GeminiResponse,
    GeminiContent,
    GeminiContentPart,
    ImageProcessingResponse,
    ErrorResponse
)
from app.utils.logger import get_logger, Timer

# Initialize logger
logger = get_logger(__name__)


class GeminiService:
    """Service for Google Gemini AI API interactions"""
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.base_url = settings.gemini_api_url
        self.default_prompt = settings.IMAGE_PROMPT
        
        # Generation configuration for consistent results
        self.generation_config = {
            "temperature": 0.4,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 2048,
            "responseMimeType": "image/jpeg"
        }
        
        # Safety settings to avoid blocked content
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_client()
    
    async def initialize_client(self):
        """Initialize HTTP client"""
        timeout = httpx.Timeout(settings.REQUEST_TIMEOUT * 2)  # Longer timeout for AI processing
        
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True
        )
        
        logger.info("✅ Gemini AI client initialized")
    
    async def close_client(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
        logger.info("✅ Gemini AI client closed")
    
    async def process_image(
        self, 
        image_data: bytes, 
        mime_type: str,
        custom_prompt: Optional[str] = None
    ) -> ImageProcessingResponse:
        """
        Process image through Gemini AI to transform mannequin to model
        
        Args:
            image_data: Raw image bytes
            mime_type: Image MIME type
            custom_prompt: Optional custom prompt for processing
            
        Returns:
            ImageProcessingResponse: Processing results including transformed image
            
        Raises:
            HTTPException: If processing fails
        """
        try:
            with Timer("Gemini AI image processing", logger):
                # Prepare the request payload
                prompt = custom_prompt or self.default_prompt
                payload = await self._prepare_payload(image_data, mime_type, prompt)
                
                logger.info(
                    f"🧠 Sending image to Gemini AI for processing",
                    extra={
                        "model": self.model_name,
                        "image_size": len(image_data),
                        "mime_type": mime_type,
                        "prompt_length": len(prompt)
                    }
                )
                
                # Make API request
                response = await self._make_gemini_request(payload)
                
                # Process response and extract image
                processed_image_data, usage_metadata = await self._process_gemini_response(response)
                
                logger.info(
                    f"✅ Image processing completed successfully",
                    extra={
                        "prompt_tokens": usage_metadata.get("prompt_token_count", 0),
                        "total_tokens": usage_metadata.get("total_token_count", 0),
                        "output_image_size": len(processed_image_data)
                    }
                )
                
                return ImageProcessingResponse(
                    success=True,
                    processed_image_data=processed_image_data,
                    processing_time=0.0,  # Will be set by timer context
                    model_used=self.model_name,
                    prompt_tokens=usage_metadata.get("prompt_token_count")
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in image processing: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Image processing failed: {str(e)}"
            )
    
    async def _prepare_payload(
        self, 
        image_data: bytes, 
        mime_type: str, 
        prompt: str
    ) -> Dict[str, Any]:
        """
        Prepare Gemini API payload with image and prompt
        
        Args:
            image_data: Raw image bytes
            mime_type: Image MIME type
            prompt: Processing prompt
            
        Returns:
            Dict[str, Any]: Gemini API payload
        """
        try:
            # Encode image to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create content parts
            content_parts = [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_image
                    }
                },
                {
                    "text": prompt
                }
            ]
            
            # Build payload
            payload = {
                "contents": [
                    {
                        "parts": content_parts
                    }
                ],
                "generationConfig": self.generation_config,
                "safetySettings": self.safety_settings
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Error preparing Gemini payload: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to prepare image for processing"
            )
    
    async def _make_gemini_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make request to Gemini API
        
        Args:
            payload: Gemini API payload
            
        Returns:
            Dict[str, Any]: Gemini API response
            
        Raises:
            HTTPException: If API request fails
        """
        try:
            url = f"{self.base_url}?key={self.api_key}"
            
            logger.debug(
                f"Making Gemini API request",
                extra={
                    "url": url,
                    "payload_keys": list(payload.keys()),
                    "content_parts": len(payload.get("contents", [])[0].get("parts", []))
                }
            )
            
            response = await self.client.post(url, json=payload)
            
            if response.status_code != 200:
                await self._handle_gemini_error(response)
            
            return response.json()
            
        except httpx.TimeoutException:
            logger.error("Gemini API request timeout")
            raise HTTPException(
                status_code=408,
                detail="AI processing timeout - please try again"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code}")
            await self._handle_gemini_error(e.response)
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"AI service unavailable: {str(e)}"
            )
    
    async def _handle_gemini_error(self, response: httpx.Response):
        """
        Handle Gemini API error responses
        
        Args:
            response: HTTP response from Gemini API
            
        Raises:
            HTTPException: Appropriate error based on response
        """
        try:
            error_data = response.json()
            error_message = "Unknown Gemini API error"
            
            # Extract error message from Gemini response
            if "error" in error_data:
                gemini_error = error_data["error"]
                error_message = gemini_error.get("message", "Unknown error")
                status_code = gemini_error.get("code", response.status_code)
            else:
                status_code = response.status_code
                error_message = error_data.get("message", str(error_data))
            
            # Map Gemini errors to user-friendly messages
            user_friendly_errors = {
                400: "Invalid image or prompt. Please try with a different image.",
                401: "AI service authentication failed.",
                403: "AI service access denied.",
                429: "AI service rate limit exceeded. Please wait a moment.",
                500: "AI service internal error. Please try again.",
                503: "AI service temporarily unavailable."
            }
            
            user_message = user_friendly_errors.get(status_code, "AI processing failed.")
            
            logger.error(
                f"Gemini API error: {status_code} - {error_message}",
                extra={
                    "status_code": status_code,
                    "gemini_error": error_message,
                    "response_body": error_data
                }
            )
            
            raise HTTPException(
                status_code=status_code,
                detail=user_message
            )
            
        except ValueError:
            # If response is not JSON
            logger.error(f"Gemini API non-JSON error: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"AI service error: {response.text}"
            )
    
    async def _process_gemini_response(self, response_data: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Process Gemini API response and extract generated image
        
        Args:
            response_data: Gemini API response
            
        Returns:
            Tuple[bytes, Dict[str, Any]]: Processed image data and usage metadata
            
        Raises:
            HTTPException: If response processing fails
        """
        try:
            # Validate response structure
            if not response_data.get("candidates"):
                raise HTTPException(
                    status_code=500,
                    detail="No response generated from AI"
                )
            
            candidate = response_data["candidates"][0]
            
            # Check for safety blocks
            if candidate.get("finishReason") == "SAFETY":
                logger.warning("Gemini blocked content for safety reasons")
                raise HTTPException(
                    status_code=400,
                    detail="The image was blocked for safety reasons. Please try a different image."
                )
            
            if candidate.get("finishReason") == "RECITATION":
                logger.warning("Gemini detected recitation/content issue")
                raise HTTPException(
                    status_code=400,
                    detail="The image contains content that cannot be processed."
                )
            
            # Extract content parts
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise HTTPException(
                    status_code=500,
                    detail="No content generated by AI"
                )
            
            # Look for image data in response parts
            processed_image_data = None
            for part in parts:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    if inline_data.get("mime_type", "").startswith("image/"):
                        processed_image_data = base64.b64decode(inline_data["data"])
                        break
            
            if not processed_image_data:
                # Check if we got text instead of image (error case)
                for part in parts:
                    if "text" in part:
                        error_text = part["text"]
                        logger.warning(f"Gemini returned text instead of image: {error_text}")
                        raise HTTPException(
                            status_code=500,
                            detail="AI returned unexpected response format"
                        )
                
                raise HTTPException(
                    status_code=500,
                    detail="No processed image received from AI"
                )
            
            # Extract usage metadata
            usage_metadata = response_data.get("usageMetadata", {})
            
            return processed_image_data, usage_metadata
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing Gemini response: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process AI response"
            )
    
    async def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that prompt is appropriate and safe
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            bool: True if prompt is valid
        """
        try:
            # Check prompt length
            if len(prompt) > 1000:
                return False
            
            # Check for blocked terms (basic safety)
            blocked_terms = [
                "nude", "naked", "explicit", "porn", "sexual", 
                "violence", "harm", "illegal", "offensive"
            ]
            
            prompt_lower = prompt.lower()
            for term in blocked_terms:
                if term in prompt_lower:
                    logger.warning(f"Prompt contains blocked term: {term}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Prompt validation error: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Gemini model
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}?key={self.api_key}"
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {"name": self.model_name, "error": str(e)}
    
    async def test_connection(self) -> bool:
        """
        Test connection to Gemini API
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Simple text generation test
            test_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Say 'OK' if you're working."}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 10
                }
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = await self.client.post(url, json=test_payload)
            
            if response.status_code == 200:
                logger.info("✅ Gemini API connection test successful")
                return True
            else:
                logger.warning(f"Gemini API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False


# Global service instance
gemini_service = GeminiService()


async def get_gemini_service() -> GeminiService:
    """
    Dependency function for FastAPI to get Gemini service instance
    """
    async with GeminiService() as service:
        yield service


class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    async def validate_image_data(image_data: bytes, mime_type: str) -> bool:
        """
        Validate image data before processing
        
        Args:
            image_data: Image bytes
            mime_type: Image MIME type
            
        Returns:
            bool: True if image is valid
        """
        try:
            # Check file size
            if len(image_data) > settings.MAX_FILE_SIZE:
                return False
            
            # Check MIME type
            if mime_type not in settings.ALLOWED_MIME_TYPES:
                return False
            
            # Basic image validation (could be extended with PIL)
            if len(image_data) < 100:  # Too small to be a valid image
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    async def convert_image_format(
        image_data: bytes, 
        target_mime_type: str = "image/jpeg"
    ) -> bytes:
        """
        Convert image to target format (basic implementation)
        
        Args:
            image_data: Source image bytes
            target_mime_type: Target MIME type
            
        Returns:
            bytes: Converted image data
        """
        # Note: For production, you might want to use Pillow for proper conversion
        # This is a basic implementation that returns original if formats match
        
        try:
            # For now, return original if target is same as source
            # In production, implement proper conversion with Pillow
            return image_data
            
        except Exception as e:
            logger.warning(f"Image conversion failed, using original: {e}")
            return image_data