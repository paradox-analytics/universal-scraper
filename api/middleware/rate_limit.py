"""
Rate limiting middleware for multi-tenant SaaS
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from fastapi import HTTPException
from universal_scraper.core.redis_cache import RedisCache

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Per-tenant rate limiting using Redis

    Tracks:
    - Requests per minute
    - Requests per day
    - Requests per hour (optional)
    """

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis = redis_cache or RedisCache()

    async def check_rate_limit(
        self,
        tenant_id: str,
        tenant_config: Dict,
        url: Optional[str] = None
    ) -> bool:
        """
        Check if tenant has exceeded rate limits

        Args:
            tenant_id: Tenant identifier
            tenant_config: Tenant configuration (plan, limits, etc.)
            url: Optional URL for per-domain rate limiting

        Returns:
            True if within limits, False if exceeded

        Raises:
            HTTPException(429) if rate limit exceeded
        """
        now = datetime.utcnow()

        # Per-minute limit
        minute_key = f"rate:{tenant_id}:minute:{now.strftime('%Y%m%d%H%M')}"
        minute_count = await self.redis.increment(minute_key)

        # Set expiry if first request in this minute
        if minute_count == 1:
            await self.redis.set_with_expiry(minute_key, minute_count, 60)

        rate_limit_per_minute = tenant_config.get("rate_limit_per_minute", 10)
        if minute_count > rate_limit_per_minute:
            logger.warning(
                f"Rate limit exceeded for {tenant_id}: "
                f"{minute_count}/{rate_limit_per_minute} per minute"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {minute_count}/{rate_limit_per_minute} requests per minute. "
                       f"Please upgrade your plan or wait before retrying."
            )

        # Per-day limit
        day_key = f"rate:{tenant_id}:day:{now.strftime('%Y%m%d')}"
        day_count = await self.redis.increment(day_key)

        # Set expiry if first request today
        if day_count == 1:
            await self.redis.set_with_expiry(day_key, day_count, 86400)  # 24 hours

        rate_limit_per_day = tenant_config.get("rate_limit_per_day", 1000)
        if day_count > rate_limit_per_day:
            logger.warning(
                f"Daily rate limit exceeded for {tenant_id}: "
                f"{day_count}/{rate_limit_per_day} per day"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Daily rate limit exceeded: {day_count}/{rate_limit_per_day} requests per day. "
                       f"Please upgrade your plan or try again tomorrow."
            )

        return True

    async def get_rate_limit_status(self, tenant_id: str) -> Dict:
        """Get current rate limit status for tenant"""
        now = datetime.utcnow()

        minute_key = f"rate:{tenant_id}:minute:{now.strftime('%Y%m%d%H%M')}"
        day_key = f"rate:{tenant_id}:day:{now.strftime('%Y%m%d')}"

        minute_count = await self.redis.get(minute_key) or 0
        day_count = await self.redis.get(day_key) or 0

        return {
            "requests_this_minute": int(minute_count) if isinstance(minute_count, (int, str)) else 0,
            "requests_today": int(day_count) if isinstance(day_count, (int, str)) else 0,
        }




