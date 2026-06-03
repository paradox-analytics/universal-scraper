"""
Adaptive Rate Limiter
Manages per-domain request rates and concurrency to prevent blocking.
Adapts dynamically based on response codes (429, 503) and blocking detection.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class AdaptiveRateLimiter:
    """
    Manages rate limiting for scraping targets.

    Features:
    - Per-domain rate limits
    - Token bucket algorithm (conceptually)
    - Dynamic adjustment (backoff on 429/Block)
    - Distributed support (via Redis, optional)
    """

    # Default settings
    DEFAULT_DELAY = 2.0  # Seconds between requests
    MIN_DELAY = 1.0      # Minimum delay
    MAX_DELAY = 60.0     # Maximum delay

    def __init__(self, redis_client: Optional[Any] = None):
        """
        Initialize Rate Limiter

        Args:
            redis_client: Optional Redis client for distributed limiting
        """
        self.redis_client = redis_client

        # Local state (used if Redis is not available)
        # domain -> last_request_timestamp
        self.last_request_times: Dict[str, float] = {}

        # domain -> current_delay_seconds
        self.domain_delays: Dict[str, float] = {}

        # domain -> consecutive_blocks
        self.block_counts: Dict[str, int] = {}

        logger.info(f"⏱️ Adaptive Rate Limiter initialized (Redis: {'✅' if redis_client else '❌'})")

    async def wait_for_token(self, url: str) -> None:
        """
        Wait until it's safe to send a request to the given URL.
        """
        domain = self._get_domain(url)

        # Get current required delay for this domain
        delay = self._get_domain_delay(domain)

        # Get last request time
        last_time = await self._get_last_request_time(domain)
        now = time.time()

        # Calculate time to wait
        elapsed = now - last_time
        wait_time = max(0, delay - elapsed)

        if wait_time > 0:
            logger.info(f"⏳ Rate limit: Waiting {wait_time:.2f}s for {domain} (Delay: {delay}s)")
            await asyncio.sleep(wait_time)

        # Update last request time (optimistic, assuming request will be sent)
        await self._set_last_request_time(domain, time.time())

    def report_result(self, url: str, status_code: int, is_blocked: bool = False):
        """
        Report the result of a request to adjust rate limits dynamically.

        Args:
            url: The URL requested
            status_code: HTTP status code
            is_blocked: Whether the request was blocked (detected by content analysis)
        """
        domain = self._get_domain(url)

        # Check for rate limiting signals
        is_rate_limited = status_code == 429 or status_code == 503

        if is_rate_limited or is_blocked:
            # INCREASE DELAY (Backoff)
            self._increase_delay(domain, is_hard_block=is_rate_limited)
        elif status_code == 200:
            # DECREASE DELAY (Recovery)
            self._decrease_delay(domain)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return url

    def _get_domain_delay(self, domain: str) -> float:
        """Get current delay setting for domain"""
        # TODO: Fetch from Redis if available
        return self.domain_delays.get(domain, self.DEFAULT_DELAY)

    async def _get_last_request_time(self, domain: str) -> float:
        """Get timestamp of last request"""
        if self.redis_client:
            try:
                val = self.redis_client.get(f"ratelimit:last_req:{domain}")
                return float(val) if val else 0.0
            except Exception as e:
                logger.warning(f"Redis error getting rate limit: {e}")

        return self.last_request_times.get(domain, 0.0)

    async def _set_last_request_time(self, domain: str, timestamp: float):
        """Set timestamp of last request"""
        if self.redis_client:
            try:
                # Set with expiry (1 hour) to keep DB clean
                self.redis_client.setex(f"ratelimit:last_req:{domain}", 3600, str(timestamp))
            except Exception as e:
                logger.warning(f"Redis error setting rate limit: {e}")

        self.last_request_times[domain] = timestamp

    def _increase_delay(self, domain: str, is_hard_block: bool):
        """Exponential backoff"""
        current = self._get_domain_delay(domain)

        # Hard blocks (429) get bigger penalty than soft blocks
        multiplier = 2.0 if is_hard_block else 1.5

        new_delay = min(current * multiplier, self.MAX_DELAY)
        self.domain_delays[domain] = new_delay

        logger.warning(f"⚠️ Rate limit increased for {domain}: {current:.1f}s -> {new_delay:.1f}s")

    def _decrease_delay(self, domain: str):
        """Linear recovery"""
        current = self._get_domain_delay(domain)

        if current > self.DEFAULT_DELAY:
            # Slowly reduce delay
            new_delay = max(current * 0.9, self.DEFAULT_DELAY)
            self.domain_delays[domain] = new_delay
            # Only log significant changes
            if current - new_delay > 0.5:
                logger.info(f"📉 Rate limit relaxing for {domain}: {current:.1f}s -> {new_delay:.1f}s")
