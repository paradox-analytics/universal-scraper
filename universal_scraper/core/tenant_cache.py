"""
Tenant-aware cache wrapper for multi-tenant SaaS
"""
import logging
from typing import Optional, Dict, List
from urllib.parse import urlparse
import hashlib
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

class TenantCache:
    """
    Multi-tenant cache with isolation

    Cache Strategy:
    - Shared code cache: All tenants benefit (domain + structure hash)
    - Tenant-specific execution cache: Isolated per tenant (tenant_id + url + fields)
    - Rate limiting: Per-tenant counters
    """

    def __init__(self, tenant_id: str, redis_cache: Optional[RedisCache] = None):
        """
        Initialize tenant-aware cache

        Args:
            tenant_id: Tenant identifier
            redis_cache: Redis cache instance (optional, creates new if not provided)
        """
        self.tenant_id = tenant_id
        self.redis = redis_cache or RedisCache()

    # ========== Shared Code Cache (All Tenants Benefit) ==========

    async def get_code_cache(self, structure_hash: str, domain: Optional[str] = None) -> Optional[Dict]:
        """
        Get shared code cache (all tenants benefit)

        Args:
            structure_hash: HTML structure hash
            domain: Optional domain name

        Returns:
            Cached code dict or None
        """
        if domain:
            # Domain-based cache key (reusable across pages on same domain)
            cache_key = f"code:{domain}:{structure_hash[:16]}"
        else:
            cache_key = f"code:{structure_hash[:16]}"

        return await self.redis.get(cache_key)

    async def set_code_cache(
        self,
        structure_hash: str,
        code: str,
        metadata: Dict,
        domain: Optional[str] = None,
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """
        Set shared code cache

        Args:
            structure_hash: HTML structure hash
            code: Generated extraction code
            metadata: Code metadata (fields, model, etc.)
            domain: Optional domain name
            ttl: Time to live in seconds
        """
        if domain:
            cache_key = f"code:{domain}:{structure_hash[:16]}"
        else:
            cache_key = f"code:{structure_hash[:16]}"

        cache_entry = {
            "code": code,
            "metadata": metadata,
            "structure_hash": structure_hash,
            "domain": domain,
        }

        return await self.redis.set(cache_key, cache_entry, ttl)

    # ========== Tenant-Specific Execution Cache ==========

    async def get_execution_cache(self, url: str, fields: List[str]) -> Optional[Dict]:
        """
        Get tenant-specific execution cache

        Args:
            url: Scraped URL
            fields: Fields to extract

        Returns:
            Cached execution result or None
        """
        cache_key = self._generate_execution_key(url, fields)
        key = f"exec:{self.tenant_id}:{cache_key}"
        return await self.redis.get(key)

    async def set_execution_cache(
        self,
        url: str,
        fields: List[str],
        result: Dict,
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """
        Set tenant-specific execution cache

        Args:
            url: Scraped URL
            fields: Fields extracted
            result: Execution result
            ttl: Time to live in seconds
        """
        cache_key = self._generate_execution_key(url, fields)
        key = f"exec:{self.tenant_id}:{cache_key}"

        cache_entry = {
            "data": result.get("data", []),
            "metadata": result.get("metadata", {}),
            "url": url,
            "fields": fields,
            "tenant_id": self.tenant_id,
        }

        return await self.redis.set(key, cache_entry, ttl)

    def _generate_execution_key(self, url: str, fields: List[str]) -> str:
        """Generate cache key for execution results"""
        fields_str = ','.join(sorted(fields))
        key_data = f"{url}:{fields_str}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    # ========== Pattern Cache (Domain-Level, Shared) ==========

    async def get_pattern_cache(
        self,
        domain: str,
        fields: List[str],
        embedding_hash: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get cached extraction pattern (shared across tenants for same domain)

        Args:
            domain: Domain name (e.g., 'metacritic.com')
            fields: Fields to extract
            embedding_hash: Optional embedding hash

        Returns:
            Cached pattern or None
        """
        fields_str = ','.join(sorted(fields))
        fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]

        if embedding_hash:
            cache_key = f"pattern:{domain}:{embedding_hash}:{fields_hash}"
        else:
            cache_key = f"pattern:{domain}:{fields_hash}"

        return await self.redis.get(cache_key)

    async def set_pattern_cache(
        self,
        domain: str,
        fields: List[str],
        pattern: Dict,
        embedding_hash: Optional[str] = None,
        ttl: int = 604800  # 7 days
    ) -> bool:
        """
        Set cached extraction pattern (shared across tenants)

        Args:
            domain: Domain name
            fields: Fields extracted
            pattern: Pattern data
            embedding_hash: Optional embedding hash
            ttl: Time to live in seconds
        """
        fields_str = ','.join(sorted(fields))
        fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]

        if embedding_hash:
            cache_key = f"pattern:{domain}:{embedding_hash}:{fields_hash}"
        else:
            cache_key = f"pattern:{domain}:{fields_hash}"

        cache_entry = {
            "pattern": pattern,
            "domain": domain,
            "fields": fields,
            "embedding_hash": embedding_hash,
        }

        return await self.redis.set(cache_key, cache_entry, ttl)

    # ========== Direct LLM Cache (Tenant-Specific) ==========

    async def get_direct_llm_cache(
        self,
        url: str,
        fields: List[str],
        structure_hash: Optional[str] = None
    ) -> Optional[Dict]:
        """Get cached Direct LLM extraction result (tenant-specific)"""
        cache_key = self._generate_direct_llm_key(url, fields, structure_hash)
        key = f"direct_llm:{self.tenant_id}:{cache_key}"
        return await self.redis.get(key)

    async def set_direct_llm_cache(
        self,
        url: str,
        fields: List[str],
        items: List[Dict],
        structure_hash: Optional[str] = None,
        ttl: int = 3600  # 1 hour
    ) -> bool:
        """Set cached Direct LLM extraction result (tenant-specific)"""
        cache_key = self._generate_direct_llm_key(url, fields, structure_hash)
        key = f"direct_llm:{self.tenant_id}:{cache_key}"

        # Extract domain for metadata
        domain = None
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
        except Exception:
            pass

        cache_entry = {
            "items": items,
            "url": url,
            "domain": domain,
            "fields": fields,
            "structure_hash": structure_hash,
            "tenant_id": self.tenant_id,
        }

        return await self.redis.set(key, cache_entry, ttl)

    def _generate_direct_llm_key(
        self,
        url: str,
        fields: List[str],
        structure_hash: Optional[str] = None
    ) -> str:
        """Generate cache key for Direct LLM results"""
        fields_str = ','.join(sorted(fields))
        fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]

        # Extract domain for domain-based caching
        domain = None
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('.', '_')
        except Exception:
            pass

        if domain:
            # Domain-based key (reusable across pages on same domain)
            return f"{domain}_{fields_hash}"
        elif structure_hash:
            # Structure-based key (page-specific)
            return f"{structure_hash[:16]}_{fields_hash}"
        else:
            # URL-based key (fallback)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            return f"{url_hash}_{fields_hash}"




