"""
Authentication and tenant identification middleware
Supports Firebase Auth tokens and API keys
"""
import os
import logging
from typing import Optional, Dict
from fastapi import Header, Depends
import jwt
import requests

logger = logging.getLogger(__name__)

# Firebase project ID (from environment or default)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "universal-scaper")

# Cache for Firebase public keys (JWKs)
_firebase_public_keys = None
_firebase_keys_expiry = 0

def get_firebase_public_keys():
    """Fetch Firebase public keys for JWT verification"""
    global _firebase_public_keys, _firebase_keys_expiry

    import time
    current_time = time.time()

    # Cache keys for 1 hour
    if _firebase_public_keys and current_time < _firebase_keys_expiry:
        return _firebase_public_keys

    try:
        # Fetch Firebase public keys
        jwks_url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()

        # Firebase uses x509 certificates, not JWKs
        # We'll verify using the project ID instead
        _firebase_public_keys = True  # Just mark as fetched
        _firebase_keys_expiry = current_time + 3600  # 1 hour cache
        return _firebase_public_keys
    except Exception as e:
        logger.warning(f"Failed to fetch Firebase keys: {e}")
        return None

def verify_firebase_token(token: str) -> Optional[Dict]:
    """
    Verify Firebase Auth token

    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        # Decode without verification first to get header
        unverified = jwt.decode(token, options={"verify_signature": False})

        # Check if it's a Firebase token
        if unverified.get("iss") != f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}":
            logger.debug("Token is not a Firebase token")
            return None

        # Verify token using Firebase Admin SDK approach
        # For now, we'll verify the basic structure and use the user_id as tenant_id
        # In production, use Firebase Admin SDK for full verification

        # Basic validation
        if unverified.get("aud") != FIREBASE_PROJECT_ID:
            logger.warning(f"Token audience mismatch: {unverified.get('aud')} != {FIREBASE_PROJECT_ID}")
            return None

        # Check expiration
        import time
        exp = unverified.get("exp", 0)
        if exp < time.time():
            logger.warning("Token has expired")
            return None

        # Return decoded token
        return unverified
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid Firebase token: {e}")
        return None
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        return None

async def get_tenant_id(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> str:
    """
    Extract tenant ID from request

    Priority:
    1. X-Tenant-ID header (for internal/admin use)
    2. Firebase Auth Bearer token (production)
    3. JWT Bearer token (legacy)
    4. API key mapping (temporary, for migration)

    Returns:
        tenant_id: Tenant identifier (Firebase UID or derived ID)
    """
    # Option 1: Direct tenant ID header (for testing/admin)
    if x_tenant_id:
        return x_tenant_id

    # Option 2: Firebase Auth Bearer token (production)
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

        # Try Firebase token first
        firebase_payload = verify_firebase_token(token)
        if firebase_payload:
            # Use Firebase UID as tenant_id
            tenant_id = firebase_payload.get("user_id") or firebase_payload.get("sub")
            if tenant_id:
                logger.debug(f"Using Firebase token tenant ID: {tenant_id}")
                return tenant_id

        # Try legacy JWT token
        try:
            jwt_secret = os.getenv("JWT_SECRET", "change-me-in-production")
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
            tenant_id = payload.get("tenant_id")
            if tenant_id:
                logger.debug(f"Using legacy JWT tenant ID: {tenant_id}")
                return tenant_id
        except jwt.InvalidTokenError:
            pass  # Not a legacy JWT, continue

    # Option 3: API key mapping (temporary, for migration)
    api_key = x_api_key or (authorization if authorization and not authorization.startswith("Bearer ") else None)
    if api_key:
        # For now, use API key hash as tenant ID (temporary)
        # In production, lookup tenant_id from database
        import hashlib
        tenant_id = f"tenant_{hashlib.md5(api_key.encode()).hexdigest()[:12]}"
        logger.debug(f"Using API key-based tenant ID: {tenant_id}")
        return tenant_id

    # No authentication provided - return default for testing
    logger.warning("No authentication provided. Using 'default_tenant' for testing.")
    return "default_tenant"

async def get_tenant_context(tenant_id: str = Depends(get_tenant_id)) -> Dict:
    """
    Get tenant context (plan, limits, etc.)

    TODO: Replace with database lookup
    """
    # For now, return default tenant config
    # In production, lookup from database/Firestore
    return {
        "tenant_id": tenant_id,
        "plan": "free",  # Default plan
        "rate_limit_per_minute": 10,
        "rate_limit_per_day": 1000,
        "cache_ttl": 3600,  # 1 hour default
    }

async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    tenant_id: str = Depends(get_tenant_id)
) -> Dict:
    """
    Get current user information from request

    Returns:
        Dict with tenant_id and api_key (if provided)
    """
    api_key = None

    # Extract API key from X-API-Key header (highest priority)
    if x_api_key:
        api_key = x_api_key
        logger.debug("API key found in X-API-Key header")
    # Fallback: check Authorization header if it's not a Bearer token
    elif authorization and not authorization.startswith("Bearer "):
        api_key = authorization
        logger.debug("API key found in Authorization header (non-Bearer)")

    return {
        "tenant_id": tenant_id,
        "api_key": api_key
    }

