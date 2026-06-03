"""
Redis-based distributed cache for multi-tenant SaaS
Replaces local filesystem cache with Redis for Cloud Run compatibility
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisCache:
    """
    Redis cache backend for distributed, multi-tenant caching

    Cache Strategy:
    - Shared code cache: All tenants benefit (domain + structure hash)
    - Tenant-specific execution cache: Isolated per tenant
    - Rate limiting: Per-tenant counters
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL (e.g., redis://host:6379)
                      If not provided, reads from REDIS_URL env var
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool = None
        self._in_memory_cache: Dict[str, str] = {}  # Fallback for when Redis is unavailable

        # Try to connect (will fail gracefully if Redis not available)
        try:
            self._connect()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None

    def _connect(self):
        """Connect to Redis"""
        if not self.redis_url:
            raise ValueError("Redis URL not provided")

        # Parse Redis URL for password/auth
        if "@" in self.redis_url:
            # Format: redis://:password@host:port
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=50,
                retry_on_timeout=True
            )
        else:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=50
            )

        self.redis_client = redis.Redis(connection_pool=self._connection_pool)
        # NOTE: We don't ping here because it's an async client and we might not have an event loop yet

    async def get(self, key: str) -> Optional[Dict]:
        """Get value from Redis or in-memory fallback"""
        if not self.redis_client:
            data = self._in_memory_cache.get(key)
            if data:
                return json.loads(data)
            return None

        try:
            data = await self.redis_client.get(key)
            if data:
                if isinstance(data, str):
                    return json.loads(data)
                return data
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Dict, ttl: int = 3600) -> bool:
        """Set value in Redis or in-memory fallback"""
        data = json.dumps(value) if not isinstance(value, str) else value

        if not self.redis_client:
            self._in_memory_cache[key] = data
            return True

        try:
            await self.redis_client.setex(key, ttl, data)
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis or in-memory fallback"""
        if not self.redis_client:
            if key in self._in_memory_cache:
                del self._in_memory_cache[key]
                return True
            return False

        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis_client:
            return False

        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis exists check failed: {e}")
            return False

    async def get_multi(self, keys: List[str]) -> Dict[str, Optional[Dict]]:
        """Get multiple keys at once (pipeline)"""
        if not self.redis_client or not keys:
            return {key: None for key in keys}

        try:
            pipeline = self.redis_client.pipeline()
            for key in keys:
                pipeline.get(key)
            results = await pipeline.execute()

            return {
                key: json.loads(result) if result else None
                for key, result in zip(keys, results)
            }
        except Exception as e:
            logger.warning(f"Redis get_multi failed: {e}")
            return {key: None for key in keys}

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter (for rate limiting)"""
        if not self.redis_client:
            return 0

        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Redis increment failed: {e}")
            return 0

    async def set_with_expiry(self, key: str, value: Any, ttl: int) -> bool:
        """Set value with expiry (for rate limiting windows)"""
        if not self.redis_client:
            return False

        try:
            data = json.dumps(value) if not isinstance(value, str) else value
            await self.redis_client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Redis set_with_expiry failed: {e}")
            return False

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern (Redis or in-memory)"""
        if not self.redis_client:
            import fnmatch
            return [k for k in self._in_memory_cache.keys() if fnmatch.fnmatch(k, pattern)]

        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                # Ensure key is a string (Redis returns bytes)
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                keys.append(key)
            return keys
        except Exception as e:
            logger.warning(f"Redis list_keys failed: {e}")
            return []

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()

