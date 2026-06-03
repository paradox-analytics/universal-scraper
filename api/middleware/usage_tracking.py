"""
Usage tracking for billing and analytics
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from universal_scraper.core.redis_cache import RedisCache

logger = logging.getLogger(__name__)

class UsageTracker:
    """
    Track usage per tenant for billing and analytics

    Metrics tracked:
    - Requests per tenant
    - Items extracted
    - Cache hit rate
    - Execution time
    - API costs (LLM calls)
    """

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis = redis_cache or RedisCache()
        # In production, also write to database/Firestore

    async def track_request(
        self,
        tenant_id: str,
        endpoint: str,
        url: str,
        items_extracted: int = 0,
        cache_hit: bool = False,
        execution_time_ms: int = 0,
        llm_calls: int = 0,
        llm_cost_usd: float = 0.0
    ):
        """
        Track a request (async, non-blocking)

        Args:
            tenant_id: Tenant identifier
            endpoint: API endpoint called
            url: URL scraped
            items_extracted: Number of items extracted
            cache_hit: Whether cache was hit
            execution_time_ms: Execution time in milliseconds
            llm_calls: Number of LLM API calls made
            llm_cost_usd: Cost of LLM calls in USD
        """
        try:
            usage_log = {
                "tenant_id": tenant_id,
                "endpoint": endpoint,
                "url": url,
                "items_extracted": items_extracted,
                "cache_hit": cache_hit,
                "execution_time_ms": execution_time_ms,
                "llm_calls": llm_calls,
                "llm_cost_usd": llm_cost_usd,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Update Redis counters (for real-time metrics)
            await self._update_counters(tenant_id, usage_log)

            # TODO: Write to database/Firestore (async, non-blocking)
            # asyncio.create_task(self._write_to_database(usage_log))

        except Exception as e:
            logger.error(f"Usage tracking failed: {e}", exc_info=True)

    async def _update_counters(self, tenant_id: str, usage_log: Dict):
        """Update Redis counters for real-time metrics"""
        now = datetime.utcnow()
        day_key = f"usage:{tenant_id}:{now.strftime('%Y%m%d')}"

        # Increment counters
        await self.redis.increment(f"{day_key}:requests")
        await self.redis.increment(f"{day_key}:items", usage_log.get("items_extracted", 0))

        if usage_log.get("cache_hit"):
            await self.redis.increment(f"{day_key}:cache_hits")

        # Track LLM costs
        if usage_log.get("llm_cost_usd", 0) > 0:
            current_cost = await self.redis.get(f"{day_key}:llm_cost") or "0"
            new_cost = float(current_cost) + usage_log["llm_cost_usd"]
            await self.redis.set(f"{day_key}:llm_cost", str(new_cost))

    async def get_usage_stats(self, tenant_id: str, date: Optional[str] = None) -> Dict:
        """Get usage statistics for tenant"""
        if not date:
            date = datetime.utcnow().strftime('%Y%m%d')

        day_key = f"usage:{tenant_id}:{date}"

        requests = await self.redis.get(f"{day_key}:requests") or 0
        items = await self.redis.get(f"{day_key}:items") or 0
        cache_hits = await self.redis.get(f"{day_key}:cache_hits") or 0
        llm_cost = await self.redis.get(f"{day_key}:llm_cost") or "0"

        cache_hit_rate = (int(cache_hits) / int(requests) * 100) if int(requests) > 0 else 0

        return {
            "date": date,
            "requests": int(requests),
            "items_extracted": int(items),
            "cache_hits": int(cache_hits),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "llm_cost_usd": float(llm_cost),
        }




