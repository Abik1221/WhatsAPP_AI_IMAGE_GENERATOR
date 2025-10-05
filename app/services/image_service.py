"""
Image service for validation, processing, and utility operations
"""

import os
import base64
import magic
import hashlib
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import aiofiles
from fastapi import HTTPException

from app.core.config import settings
from app.utils.logger import get_logger, Timer

# Initialize logger
logger = get_logger(__name__)


class ImageService:
    """Service for image validation, processing, and utility operations"""
    
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_mime_types = settings.ALLOWED_MIME_TYPES
        
        # Initialize magic MIME type detector
        try:
            self.mime_detector = magic.Magic(mime=True)
        except Exception as e:
            logger.warning(f"Failed to initialize magic MIME detector: {e}")
            self.mime_detector = None
    
    async def validate_image_data(self, image_data: bytes, declared_mime_type: str = None) -> bool:
        """
        Comprehensive image validation
        
        Args:
            image_data: Raw image bytes
            declared_mime_type: MIME type declared by uploader
            
        Returns:
            bool: True if image is valid
        """
        try:
            with Timer("Image validation", logger):
                # Check file size
                if not await self._validate_file_size(image_data):
                    logger.warning(
                        "Image validation failed: file size exceeded",
                        extra={"file_size": len(image_data), "max_size": self.max_file_size}
                    )
                    return False
                
                # Detect actual MIME type
                actual_mime_type = await self._detect_mime_type(image_data)
                if not actual_mime_type:
                    logger.warning("Image validation failed: unable to detect MIME type")
                    return False
                
                # Validate MIME type
                if not await self._validate_mime_type(actual_mime_type):
                    logger.warning(
                        "Image validation failed: unsupported MIME type",
                        extra={"mime_type": actual_mime_type}
                    )
                    return False
                
                # Check for MIME type mismatch
                if declared_mime_type and declared_mime_type != actual_mime_type:
                    logger.warning(
                        "MIME type mismatch",
                        extra={
                            "declared_mime": declared_mime_type,
                            "actual_mime": actual_mime_type
                        }
                    )
                    # We'll use the actual detected type, but log the discrepancy
                
                # Basic image content validation
                if not await self._validate_image_content(image_data):
                    logger.warning("Image validation failed: invalid image content")
                    return False
                
                logger.debug(
                    "✅ Image validation successful",
                    extra={
                        "file_size": len(image_data),
                        "mime_type": actual_mime_type,
                        "sha256_hash": await self._calculate_hash(image_data)
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    async def _validate_file_size(self, image_data: bytes) -> bool:
        """
        Validate image file size
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            bool: True if within size limits
        """
        file_size = len(image_data)
        return 0 < file_size <= self.max_file_size
    
    async def _detect_mime_type(self, image_data: bytes) -> Optional[str]:
        """
        Detect MIME type of image data
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Optional[str]: Detected MIME type or None
        """
        try:
            if self.mime_detector:
                return self.mime_detector.from_buffer(image_data)
            else:
                # Fallback MIME type detection
                return await self._fallback_mime_detection(image_data)
        except Exception as e:
            logger.error(f"MIME type detection failed: {e}")
            return None
    
    async def _fallback_mime_detection(self, image_data: bytes) -> Optional[str]:
        """
        Fallback MIME type detection using magic bytes
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Optional[str]: Detected MIME type or None
        """
        try:
            # Check common image format magic bytes
            if image_data.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                return "image/png"
            elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':
                return "image/webp"
            elif image_data.startswith(b'GIF8'):
                return "image/gif"
            elif image_data.startswith(b'BM'):
                return "image/bmp"
            else:
                return None
        except Exception as e:
            logger.error(f"Fallback MIME detection failed: {e}")
            return None
    
    async def _validate_mime_type(self, mime_type: str) -> bool:
        """
        Validate MIME type against allowed types
        
        Args:
            mime_type: MIME type to validate
            
        Returns:
            bool: True if MIME type is allowed
        """
        return mime_type.lower() in self.allowed_mime_types
    
    async def _validate_image_content(self, image_data: bytes) -> bool:
        """
        Basic image content validation
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            bool: True if image content appears valid
        """
        try:
            # Check minimum size for valid image
            if len(image_data) < 100:  # Too small to be a valid image
                return False
            
            # Check for common image corruption patterns
            # This is a basic check - in production you might use PIL for more validation
            if await self._has_null_bytes(image_data):
                logger.warning("Image contains null bytes - possible corruption")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image content validation failed: {e}")
            return False
    
    async def _has_null_bytes(self, data: bytes) -> bool:
        """
        Check for null bytes in image data (common corruption indicator)
        
        Args:
            data: Bytes to check
            
        Returns:
            bool: True if null bytes found
        """
        return b'\x00\x00\x00' in data  # Look for sequences of null bytes
    
    async def _calculate_hash(self, image_data: bytes) -> str:
        """
        Calculate SHA256 hash of image data
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            str: SHA256 hash
        """
        return hashlib.sha256(image_data).hexdigest()
    
    async def save_temp_image(self, image_data: bytes, filename: str = None) -> str:
        """
        Save image to temporary storage
        
        Args:
            image_data: Raw image bytes
            filename: Optional filename (will generate if not provided)
            
        Returns:
            str: Path to saved file
            
        Raises:
            HTTPException: If save fails
        """
        try:
            # Ensure temp directory exists
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                file_hash = await self._calculate_hash(image_data)
                mime_type = await self._detect_mime_type(image_data)
                extension = await self._get_file_extension(mime_type) if mime_type else ".bin"
                filename = f"{file_hash}{extension}"
            
            file_path = self.temp_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(image_data)
            
            logger.debug(
                "💾 Image saved to temporary storage",
                extra={
                    "file_path": str(file_path),
                    "file_size": len(image_data)
                }
            )
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save temp image: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save image"
            )
    
    async def load_temp_image(self, file_path: str) -> bytes:
        """
        Load image from temporary storage
        
        Args:
            file_path: Path to image file
            
        Returns:
            bytes: Image data
            
        Raises:
            HTTPException: If load fails
        """
        try:
            path = Path(file_path)
            
            # Security check: ensure file is within temp directory
            if not path.is_relative_to(self.temp_dir):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file path"
                )
            
            async with aiofiles.open(path, 'rb') as f:
                image_data = await f.read()
            
            logger.debug(
                "📤 Image loaded from temporary storage",
                extra={
                    "file_path": file_path,
                    "file_size": len(image_data)
                }
            )
            
            return image_data
            
        except FileNotFoundError:
            logger.warning(f"Temp image not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail="Image file not found"
            )
        except Exception as e:
            logger.error(f"Failed to load temp image {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load image"
            )
    
    async def delete_temp_image(self, file_path: str) -> bool:
        """
        Delete image from temporary storage
        
        Args:
            file_path: Path to image file
            
        Returns:
            bool: True if deletion successful
        """
        try:
            path = Path(file_path)
            
            # Security check: ensure file is within temp directory
            if not path.is_relative_to(self.temp_dir):
                logger.warning(f"Attempt to delete file outside temp directory: {file_path}")
                return False
            
            if path.exists():
                path.unlink()
                logger.debug(f"🗑️ Deleted temp image: {file_path}")
                return True
            else:
                logger.warning(f"Temp image not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete temp image {file_path}: {e}")
            return False
    
    async def cleanup_old_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age of files in hours
            
        Returns:
            int: Number of files deleted
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            deleted_count = 0
            
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(
                    f"🧹 Cleaned up {deleted_count} old temp files",
                    extra={"deleted_count": deleted_count, "max_age_hours": max_age_hours}
                )
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
            return 0
    
    async def get_image_info(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get detailed information about image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict[str, Any]: Image information
        """
        try:
            mime_type = await self._detect_mime_type(image_data)
            file_size = len(image_data)
            file_hash = await self._calculate_hash(image_data)
            
            info = {
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "mime_type": mime_type,
                "sha256_hash": file_hash,
                "is_valid": await self.validate_image_data(image_data, mime_type)
            }
            
            # Try to get dimensions if possible
            dimensions = await self._get_image_dimensions(image_data, mime_type)
            if dimensions:
                info.update({
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "aspect_ratio": round(dimensions[0] / dimensions[1], 2) if dimensions[1] > 0 else 0
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {
                "size_bytes": len(image_data),
                "mime_type": "unknown",
                "is_valid": False,
                "error": str(e)
            }
    
    async def _get_image_dimensions(self, image_data: bytes, mime_type: str) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions (basic implementation)
        
        Args:
            image_data: Raw image bytes
            mime_type: Image MIME type
            
        Returns:
            Optional[Tuple[int, int]]: (width, height) or None
        """
        try:
            # For production, you might want to use PIL for accurate dimensions
            # This is a basic implementation that works for common formats
            
            if mime_type == "image/jpeg":
                # JPEG: SOF markers contain dimension information
                return await self._get_jpeg_dimensions(image_data)
            elif mime_type == "image/png":
                # PNG: IHDR chunk contains dimensions
                return await self._get_png_dimensions(image_data)
            else:
                # For other formats, return None or implement specific parsers
                return None
                
        except Exception:
            return None
    
    async def _get_jpeg_dimensions(self, image_data: bytes) -> Optional[Tuple[int, int]]:
        """Get dimensions from JPEG data"""
        try:
            # JPEG SOF markers: 0xFFC0, 0xFFC1, 0xFFC2, etc.
            for i in range(len(image_data) - 1):
                if image_data[i] == 0xFF and image_data[i+1] in [0xC0, 0xC1, 0xC2]:
                    height = (image_data[i+5] << 8) | image_data[i+6]
                    width = (image_data[i+7] << 8) | image_data[i+8]
                    return width, height
            return None
        except Exception:
            return None
    
    async def _get_png_dimensions(self, image_data: bytes) -> Optional[Tuple[int, int]]:
        """Get dimensions from PNG data"""
        try:
            # PNG IHDR chunk starts at byte 16
            if len(image_data) >= 24:
                width = int.from_bytes(image_data[16:20], byteorder='big')
                height = int.from_bytes(image_data[20:24], byteorder='big')
                return width, height
            return None
        except Exception:
            return None
    
    async def _get_file_extension(self, mime_type: str) -> str:
        """
        Get file extension from MIME type
        
        Args:
            mime_type: MIME type
            
        Returns:
            str: File extension with dot
        """
        extension_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp"
        }
        return extension_map.get(mime_type, ".bin")
    
    async def convert_to_base64(self, image_data: bytes, mime_type: str = None) -> str:
        """
        Convert image data to base64 string
        
        Args:
            image_data: Raw image bytes
            mime_type: Image MIME type
            
        Returns:
            str: Base64 encoded image with data URI
        """
        try:
            if not mime_type:
                mime_type = await self._detect_mime_type(image_data) or "application/octet-stream"
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"Base64 conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to encode image"
            )
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get temporary storage statistics
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "temp_directory": str(self.temp_dir),
                "max_file_size_bytes": self.max_file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}


# Global service instance
image_service = ImageService()


async def get_image_service() -> ImageService:
    """
    Dependency function for FastAPI to get image service instance
    """
    return image_service


class ImageValidationError(Exception):
    """Custom exception for image validation errors"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


async def validate_and_get_image_info(image_data: bytes, mime_type: str = None) -> Dict[str, Any]:
    """
    Utility function to validate image and get information
    
    Args:
        image_data: Raw image bytes
        mime_type: Declared MIME type
        
    Returns:
        Dict[str, Any]: Validation results and image info
    """
    try:
        is_valid = await image_service.validate_image_data(image_data, mime_type)
        image_info = await image_service.get_image_info(image_data)
        
        return {
            "valid": is_valid,
            "info": image_info,
            "errors": [] if is_valid else ["Image validation failed"]
        }
        
    except Exception as e:
        return {
            "valid": False,
            "info": {"size_bytes": len(image_data)},
            "errors": [str(e)]
        }