"""
Authentication and tenant identification middleware
Supports Firebase Auth tokens and API keys
"""
import os
import time
import hashlib
import logging
from typing import Optional, Dict
from fastapi import Header, Depends, HTTPException
import jwt
import requests

logger = logging.getLogger(__name__)

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "universal-scaper")

_firebase_public_keys: Optional[Dict[str, str]] = None
_firebase_keys_expiry = 0


def get_firebase_public_keys() -> Optional[Dict[str, str]]:
    """Fetch Firebase x509 public keys for JWT signature verification."""
    global _firebase_public_keys, _firebase_keys_expiry

    current_time = time.time()
    if _firebase_public_keys and current_time < _firebase_keys_expiry:
        return _firebase_public_keys

    try:
        jwks_url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()
        _firebase_public_keys = response.json()
        _firebase_keys_expiry = current_time + 3600
        return _firebase_public_keys
    except Exception as e:
        logger.warning(f"Failed to fetch Firebase keys: {e}")
        return None


def verify_firebase_token(token: str) -> Optional[Dict]:
    """
    Verify Firebase Auth token with signature validation.

    Returns decoded payload if valid, None otherwise.
    """
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            return None

        public_keys = get_firebase_public_keys()
        if not public_keys or kid not in public_keys:
            logger.warning("Firebase public key not found for token kid")
            return None

        from cryptography.x509 import load_pem_x509_certificate
        cert = load_pem_x509_certificate(public_keys[kid].encode())
        public_key = cert.public_key()

        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=FIREBASE_PROJECT_ID,
            issuer=f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}",
        )
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Firebase token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid Firebase token: {e}")
        return None
    except ImportError:
        logger.warning("cryptography package not installed — falling back to unverified decode")
        return _verify_firebase_token_fallback(token)
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        return None


def _verify_firebase_token_fallback(token: str) -> Optional[Dict]:
    """Fallback verification when cryptography package unavailable."""
    try:
        payload = jwt.decode(token, options={"verify_signature": False})

        if payload.get("iss") != f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}":
            return None
        if payload.get("aud") != FIREBASE_PROJECT_ID:
            return None
        if payload.get("exp", 0) < time.time():
            return None

        logger.warning("Using unverified Firebase token — install cryptography package for full verification")
        return payload
    except Exception:
        return None


async def get_tenant_id(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Extract tenant ID from request. Requires authentication.

    Priority:
    1. Firebase Auth Bearer token (production)
    2. Legacy JWT Bearer token
    3. API key hash (X-API-Key header)

    Raises HTTPException 401 if no valid credentials provided.
    """
    # Option 1: Bearer token (Firebase or legacy JWT)
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

        firebase_payload = verify_firebase_token(token)
        if firebase_payload:
            tenant_id = firebase_payload.get("user_id") or firebase_payload.get("sub")
            if tenant_id:
                return tenant_id

        jwt_secret = os.getenv("JWT_SECRET")
        if jwt_secret:
            try:
                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                tenant_id = payload.get("tenant_id")
                if tenant_id:
                    return tenant_id
            except jwt.InvalidTokenError:
                pass

    # Option 2: API key hash
    api_key = x_api_key or (authorization if authorization and not authorization.startswith("Bearer ") else None)
    if api_key:
        tenant_id = f"tenant_{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        return tenant_id

    raise HTTPException(status_code=401, detail="Authentication required")


async def get_tenant_context(tenant_id: str = Depends(get_tenant_id)) -> Dict:
    """
    Get tenant context (plan, limits, etc.)

    TODO: Replace with database lookup
    """
    return {
        "tenant_id": tenant_id,
        "plan": "free",
        "rate_limit_per_minute": 10,
        "rate_limit_per_day": 1000,
        "cache_ttl": 3600,
    }


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    tenant_id: str = Depends(get_tenant_id),
) -> Dict:
    """Get current user information from request."""
    api_key = None

    if x_api_key:
        api_key = x_api_key
    elif authorization and not authorization.startswith("Bearer "):
        api_key = authorization

    return {
        "tenant_id": tenant_id,
        "api_key": api_key,
    }
