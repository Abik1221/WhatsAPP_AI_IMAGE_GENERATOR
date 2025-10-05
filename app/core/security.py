"""
Security utilities and middleware for the WhatsApp + Gemini AI Bot
"""

import secrets
import hashlib
import hmac
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi.util import get_remote_address

from app.core.config import settings
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class SecurityUtils:
    """Security utility functions"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        Generate a cryptographically secure random token
        
        Args:
            length: Token length in bytes
            
        Returns:
            str: Secure random token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def verify_webhook_token(received_token: str, expected_token: str) -> bool:
        """
        Verify webhook verification token securely
        
        Args:
            received_token: Token received from webhook
            expected_token: Expected token from configuration
            
        Returns:
            bool: True if tokens match
        """
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(received_token, expected_token)
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """
        Hash data using specified algorithm
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm
            
        Returns:
            str: Hexadecimal hash
        """
        hash_func = hashlib.new(algorithm)
        hash_func.update(data.encode('utf-8'))
        return hash_func.hexdigest()
    
    @staticmethod
    def verify_hmac_signature(data: bytes, signature: str, secret: str) -> bool:
        """
        Verify HMAC signature for webhook integrity
        
        Args:
            data: Raw data bytes
            signature: Received signature
            secret: Secret key for HMAC
            
        Returns:
            bool: True if signature is valid
        """
        try:
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                data,
                hashlib.sha256
            ).hexdigest()
            
            return secrets.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"HMAC verification failed: {e}")
            return False
    
    @staticmethod
    def sanitize_user_input(input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            input_string: User input string
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized input
        """
        if not input_string:
            return ""
        
        # Truncate to maximum length
        sanitized = input_string[:max_length]
        
        # Remove potentially dangerous characters
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('/', '&#x2F;')
        
        return sanitized
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
        """
        Validate file extension against allowed list
        
        Args:
            filename: File name to validate
            allowed_extensions: List of allowed extensions
            
        Returns:
            bool: True if extension is allowed
        """
        if not filename:
            return False
        
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        return file_extension in allowed_extensions


class RateLimitSecurity:
    """Security measures for rate limiting"""
    
    @staticmethod
    def calculate_client_fingerprint(request: Request) -> str:
        """
        Calculate client fingerprint for rate limiting
        
        Args:
            request: FastAPI request object
            
        Returns:
            str: Client fingerprint hash
        """
        client_ip = get_remote_address(request)
        user_agent = request.headers.get('user-agent', '')
        
        fingerprint_data = f"{client_ip}:{user_agent}"
        return SecurityUtils.hash_data(fingerprint_data)
    
    @staticmethod
    def should_block_request(client_fingerprint: str, violation_count: int) -> bool:
        """
        Determine if request should be blocked based on violation history
        
        Args:
            client_fingerprint: Client fingerprint
            violation_count: Number of previous violations
            
        Returns:
            bool: True if request should be blocked
        """
        # Block if more than 10 violations in short period
        return violation_count > 10


class APITokenSecurity(HTTPBearer):
    """API Token security for internal endpoints"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        """
        Validate API token for protected endpoints
        """
        credentials = await super().__call__(request)
        
        if credentials:
            token = credentials.credentials
            if not self.verify_api_token(token):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid API token"
                )
        
        return credentials
    
    def verify_api_token(self, token: str) -> bool:
        """
        Verify API token (for internal endpoints)
        
        Args:
            token: API token to verify
            
        Returns:
            bool: True if token is valid
        """
        # In production, you might want to use a proper token validation
        # For now, we'll use a simple approach
        expected_tokens = [
            SecurityUtils.hash_data("whatsapp-bot-internal-token"),
            # Add other valid tokens as needed
        ]
        
        token_hash = SecurityUtils.hash_data(token)
        return token_hash in expected_tokens


class RequestSecurity:
    """Request security validation"""
    
    @staticmethod
    def validate_request_origin(request: Request) -> bool:
        """
        Validate request origin for security
        
        Args:
            request: FastAPI request object
            
        Returns:
            bool: True if origin is valid
        """
        # Check if request comes from known WhatsApp IP ranges
        # Note: WhatsApp doesn't publish static IPs, so this is basic validation
        client_ip = get_remote_address(request)
        
        # Basic IP validation (extend with known WhatsApp IP ranges)
        if client_ip in ['127.0.0.1', 'localhost'] and settings.ENVIRONMENT == 'development':
            return True
        
        # In production, you might want to implement more sophisticated checks
        # For now, we'll allow all origins and rely on token verification
        return True
    
    @staticmethod
    def validate_content_length(request: Request) -> bool:
        """
        Validate content length to prevent DoS attacks
        
        Args:
            request: FastAPI request object
            
        Returns:
            bool: True if content length is acceptable
        """
        content_length = request.headers.get('content-length')
        
        if content_length:
            try:
                length = int(content_length)
                # Maximum 10MB for webhook payloads
                return length <= 10 * 1024 * 1024
            except ValueError:
                return False
        
        return True
    
    @staticmethod
    def validate_user_agent(request: Request) -> bool:
        """
        Validate User-Agent header
        
        Args:
            request: FastAPI request object
            
        Returns:
            bool: True if User-Agent is valid
        """
        user_agent = request.headers.get('user-agent', '')
        
        # Block obviously malicious User-Agents
        malicious_patterns = [
            'sqlmap',
            'nikto',
            'metasploit',
            'nmap',
            'wget',
            'curl'
        ]
        
        user_agent_lower = user_agent.lower()
        return not any(pattern in user_agent_lower for pattern in malicious_patterns)


class SecurityMiddleware:
    """Security middleware for additional protection"""
    
    def __init__(self):
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    async def add_security_headers(self, request: Request, call_next):
        """
        Add security headers to all responses
        
        Args:
            request: FastAPI request object
            call_next: Next middleware callable
            
        Returns:
            Response: Response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response
    
    async def validate_request_security(self, request: Request) -> Dict[str, Any]:
        """
        Comprehensive request security validation
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'origin_valid': RequestSecurity.validate_request_origin(request),
            'content_length_valid': RequestSecurity.validate_content_length(request),
            'user_agent_valid': RequestSecurity.validate_user_agent(request),
            'client_fingerprint': RateLimitSecurity.calculate_client_fingerprint(request)
        }
        
        # Log security warnings
        if not all(validation_results.values()):
            logger.warning(
                "Request security validation failed",
                extra={
                    'client_ip': get_remote_address(request),
                    'user_agent': request.headers.get('user-agent'),
                    'validation_results': validation_results
                }
            )
        
        return validation_results


# Global instances
security_utils = SecurityUtils()
rate_limit_security = RateLimitSecurity()
request_security = RequestSecurity()
security_middleware = SecurityMiddleware()
api_token_security = APITokenSecurity()


async def get_security_utils() -> SecurityUtils:
    """
    Dependency function for FastAPI to get security utils
    """
    return security_utils


async def validate_webhook_security(request: Request) -> bool:
    """
    Comprehensive webhook security validation
    
    Args:
        request: FastAPI request object
        
    Returns:
        bool: True if webhook request is secure
    """
    try:
        # Validate basic request security
        security_validation = await security_middleware.validate_request_security(request)
        
        if not all([
            security_validation['origin_valid'],
            security_validation['content_length_valid'],
            security_validation['user_agent_valid']
        ]):
            return False
        
        # Additional webhook-specific security checks can be added here
        return True
        
    except Exception as e:
        logger.error(f"Webhook security validation failed: {e}")
        return False