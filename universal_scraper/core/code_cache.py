"""
Code Cache System
Stores and retrieves generated extraction code for reuse
"""

import json
import logging
import time
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import diskcache

logger = logging.getLogger(__name__)


def generate_cache_key(structure_hash: str, fields: List[str], domain: Optional[str] = None) -> str:
    """
    Generate a cache key that includes structure hash, field names, and optionally domain.

    Domain-based caching allows reusing extraction patterns across pages on the same domain,
    even if the structure hash differs slightly (e.g., different pagination, ads, etc.).

    Args:
        structure_hash: HTML structure hash
        fields: List of field names to extract
        domain: Optional domain name (e.g., 'reddit.com') for domain-based caching

    Returns:
        Combined cache key string
    """
    # Sort fields for consistency (order shouldn't matter)
    sorted_fields = sorted(fields)
    fields_str = ','.join(sorted_fields)

    # Create a hash of the field names
    fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]

    # If domain provided, use domain-based key (allows reuse across pages on same domain)
    if domain:
        # Domain + fields = reusable pattern across pages
        domain_key = f"{domain}:{fields_hash}"
        return domain_key

    # Otherwise, use structure-based key (page-specific)
    return f"{structure_hash}:{fields_hash}"


class CodeCache:
    """Manages caching of generated extraction code"""

    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl: int = 86400,  # 24 hours default
        enable_cache: bool = True,
        redis_cache=None  # Optional Redis cache for multi-tenant SaaS
    ):
        """
        Initialize Code Cache

        Args:
            cache_dir: Directory for cache storage (fallback if Redis not available)
            ttl: Time to live in seconds
            enable_cache: Enable/disable caching
            redis_cache: Optional Redis cache instance (for Cloud Run/multi-tenant)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enable_cache = enable_cache
        self.redis_cache = redis_cache

        # Use Redis if available, otherwise fall back to diskcache
        if self.enable_cache:
            if self.redis_cache:
                logger.info(" Code Cache initialized: Redis (multi-tenant)")
                self.cache = None  # Use Redis instead
            else:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.cache = diskcache.Cache(str(self.cache_dir))
                logger.info(f" Code Cache initialized: Local filesystem ({self.cache_dir})")
        else:
            self.cache = None
            logger.info(" Cache disabled")

    def get(self, structure_hash: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached code by structure hash

        Args:
            structure_hash: Structural hash of the page
            domain: Optional domain name for domain-based caching

        Returns:
            Cached code dict or None if not found/expired
        """
        if not self.enable_cache:
            return None

        # Use Redis if available (sync wrapper for async Redis)
        if self.redis_cache:
            try:
                # Try to get from Redis (async operation in sync context)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # In async context - return None, caller should use async_get
                        # This happens when called from async scrape() method
                        logger.debug(f" Redis cache get skipped (async context): {structure_hash[:16]}...")
                    else:
                        # Sync context - can run async
                        cached_data = loop.run_until_complete(
                            self.redis_cache.get(f"code:{structure_hash[:16]}")
                        )
                        if cached_data:
                            logger.info(f" Cache hit (Redis): {structure_hash[:16]}...")
                            return cached_data
                except RuntimeError:
                    # No event loop - create one
                    cached_data = asyncio.run(
                        self.redis_cache.get(f"code:{structure_hash[:16]}")
                    )
                    if cached_data:
                        logger.info(f" Cache hit (Redis): {structure_hash[:16]}...")
                        return cached_data
            except Exception as e:
                logger.warning(f" Redis cache get error: {e}, falling back to diskcache")
                # Fall through to diskcache

        # Fallback to diskcache
        if not self.cache:
            return None

        try:
            cache_key = f"code:{structure_hash}"
            cached_data = self.cache.get(cache_key)

            if cached_data:
                # Check if expired
                if self._is_expired(cached_data):
                    logger.info(f" Cache entry expired: {structure_hash[:16]}...")
                    self.delete(structure_hash)
                    return None

                logger.info(f" Cache hit (local): {structure_hash[:16]}...")
                return cached_data
            else:
                logger.info(f" Cache miss (local): {structure_hash[:16]}...")
                return None

        except Exception as e:
            logger.error(f" Cache get error: {str(e)}")
            return None

    def set(
        self,
        structure_hash: str,
        code: str,
        metadata: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> bool:
        """
        Store code in cache

        Args:
            structure_hash: Structural hash of the page
            code: Generated extraction code
            metadata: Optional metadata about the code
            domain: Optional domain name for domain-based caching

        Returns:
            True if stored successfully
        """
        if not self.enable_cache:
            return False

        cached_data = {
            'code': code,
            'metadata': metadata or {},
            'created_at': time.time(),
            'ttl': self.ttl,
            'structure_hash': structure_hash,
            'domain': domain
        }

        # Use Redis if available (sync wrapper for async Redis)
        if self.redis_cache:
            try:
                cache_key = f"code:{structure_hash[:16]}"
                # Try to set in Redis (async operation in sync context)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # In async context - return False, caller should use async_set
                        # This happens when called from async scrape() method
                        logger.debug(f" Redis cache set skipped (async context): {structure_hash[:16]}...")
                    else:
                        # Sync context - can run async
                        success = loop.run_until_complete(
                            self.redis_cache.set(cache_key, cached_data, ttl=self.ttl)
                        )
                        if success:
                            logger.info(f" Cached code (Redis): {structure_hash[:16]}... (TTL: {self.ttl}s)")
                            return True
                except RuntimeError:
                    # No event loop - create one
                    success = asyncio.run(
                        self.redis_cache.set(cache_key, cached_data, ttl=self.ttl)
                    )
                    if success:
                        logger.info(f" Cached code (Redis): {structure_hash[:16]}... (TTL: {self.ttl}s)")
                        return True
            except Exception as e:
                logger.warning(f" Redis cache set error: {e}, falling back to diskcache")
                # Fall through to diskcache

        # Fallback to diskcache
        if not self.cache:
            return False

        try:
            cache_key = f"code:{structure_hash}"
            self.cache.set(cache_key, cached_data, expire=self.ttl)

            logger.info(f" Cached code (local): {structure_hash[:16]}... (TTL: {self.ttl}s)")
            return True

        except Exception as e:
            logger.error(f" Cache set error: {str(e)}")
            return False

    async def async_get(self, cache_key: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of get() for use in async contexts (scraper.scrape())

        Args:
            cache_key: Cache key (structure hash or domain-based key)
            domain: Optional domain name

        Returns:
            Cached code dict or None if not found/expired
        """
        if not self.enable_cache:
            return None

        # Extract structure_hash from cache_key if it's a full key
        structure_hash = cache_key.split(':')[-1] if ':' in cache_key else cache_key

        # Use Redis if available
        if self.redis_cache:
            try:
                redis_key = f"code:{structure_hash[:16]}"
                cached_data = await self.redis_cache.get(redis_key)
                if cached_data:
                    logger.info(f" Cache hit (Redis): {structure_hash[:16]}...")
                    return cached_data
                else:
                    logger.info(f" Cache miss (Redis): {structure_hash[:16]}...")
                    return None
            except Exception as e:
                logger.warning(f" Redis cache get error: {e}, falling back to diskcache")
                # Fall through to diskcache

        # Fallback to diskcache
        if not self.cache:
            return None

        try:
            cached_data = self.cache.get(cache_key)

            if cached_data:
                # Check if expired
                if self._is_expired(cached_data):
                    logger.info(f" Cache entry expired: {structure_hash[:16]}...")
                    self.delete(structure_hash)
                    return None

                logger.info(f" Cache hit (local): {structure_hash[:16]}...")
                return cached_data
            else:
                logger.info(f" Cache miss (local): {structure_hash[:16]}...")
                return None

        except Exception as e:
            logger.error(f" Cache get error: {str(e)}")
            return None

    async def async_set(
        self,
        cache_key: str,
        code: str,
        metadata: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> bool:
        """
        Async version of set() for use in async contexts (scraper.scrape())

        Args:
            cache_key: Cache key (structure hash or domain-based key)
            code: Generated extraction code
            metadata: Optional metadata about the code
            domain: Optional domain name

        Returns:
            True if stored successfully
        """
        if not self.enable_cache:
            return False

        # Extract structure_hash from cache_key if it's a full key
        structure_hash = cache_key.split(':')[-1] if ':' in cache_key else cache_key

        cached_data = {
            'code': code,
            'metadata': metadata or {},
            'created_at': time.time(),
            'ttl': self.ttl,
            'structure_hash': structure_hash,
            'domain': domain
        }

        # Use Redis if available
        if self.redis_cache:
            try:
                redis_key = f"code:{structure_hash[:16]}"
                success = await self.redis_cache.set(redis_key, cached_data, ttl=self.ttl)
                if success:
                    logger.info(f" Cached code (Redis): {structure_hash[:16]}... (TTL: {self.ttl}s)")
                    return True
            except Exception as e:
                logger.warning(f" Redis cache set error: {e}, falling back to diskcache")
                # Fall through to diskcache

        # Fallback to diskcache
        if not self.cache:
            return False

        try:
            self.cache.set(cache_key, cached_data, expire=self.ttl)

            logger.info(f" Cached code (local): {structure_hash[:16]}... (TTL: {self.ttl}s)")
            return True

        except Exception as e:
            logger.error(f" Cache set error: {str(e)}")
            return False

    def delete(self, structure_hash: str) -> bool:
        """
        Delete cached code

        Args:
            structure_hash: Structural hash to delete

        Returns:
            True if deleted successfully
        """
        if not self.enable_cache or not self.cache:
            return False

        try:
            cache_key = f"code:{structure_hash}"
            deleted = self.cache.delete(cache_key)

            if deleted:
                logger.info(f" Deleted cache: {structure_hash[:16]}...")

            return deleted

        except Exception as e:
            logger.error(f" Cache delete error: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear entire cache

        Returns:
            True if cleared successfully
        """
        if not self.enable_cache or not self.cache:
            return False

        try:
            self.cache.clear()
            logger.info(" Cache cleared")
            return True

        except Exception as e:
            logger.error(f" Cache clear error: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache statistics
        """
        if not self.enable_cache or not self.cache:
            return {'enabled': False}

        try:
            stats = {
                'enabled': True,
                'size': len(self.cache),
                'volume': self.cache.volume(),
                'directory': str(self.cache_dir),
                'ttl': self.ttl
            }

            return stats

        except Exception as e:
            logger.error(f" Cache stats error: {str(e)}")
            return {'enabled': True, 'error': str(e)}

    def _is_expired(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is expired"""
        if 'created_at' not in cached_data or 'ttl' not in cached_data:
            return True

        age = time.time() - cached_data['created_at']
        return age > cached_data['ttl']

    def get_or_generate(
        self,
        structure_hash: str,
        generator_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get from cache or generate new (convenience method)

        Args:
            structure_hash: Structural hash
            generator_func: Function to call if cache miss
            *args, **kwargs: Arguments to pass to generator_func

        Returns:
            Dict with 'code', 'cached', 'metadata' keys
        """
        # Try to get from cache
        cached_data = self.get(structure_hash)

        if cached_data:
            return {
                'code': cached_data['code'],
                'cached': True,
                'metadata': cached_data.get('metadata', {})
            }

        # Generate new
        logger.info(" Cache miss, generating new code...")
        result = generator_func(*args, **kwargs)

        # Store in cache
        if isinstance(result, dict) and 'code' in result:
            self.set(
                structure_hash,
                result['code'],
                result.get('metadata', {})
            )
            return {
                'code': result['code'],
                'cached': False,
                'metadata': result.get('metadata', {})
            }
        else:
            # Assume result is just the code string
            self.set(structure_hash, result)
            return {
                'code': result,
                'cached': False,
                'metadata': {}
            }

    def list_cached_hashes(self) -> list:
        """
        List all cached structure hashes

        Returns:
            List of structure hashes in cache
        """
        if not self.enable_cache or not self.cache:
            return []

        try:
            keys = list(self.cache.iterkeys())
            hashes = [key.replace('code:', '') for key in keys if key.startswith('code:')]
            return hashes
        except Exception as e:
            logger.error(f" Error listing cache: {str(e)}")
            return []

    def list_by_domain(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List cached patterns by domain

        Args:
            domain: Optional domain filter (e.g., 'reddit.com')

        Returns:
            List of cached entries with domain metadata
        """
        if not self.enable_cache or not self.cache:
            return []

        try:
            results = []
            for key in self.cache.iterkeys():
                if not key.startswith('code:'):
                    continue

                cached_data = self.cache.get(key)
                if not cached_data:
                    continue

                metadata = cached_data.get('metadata', {})
                cached_domain = metadata.get('domain')

                # Filter by domain if specified
                if domain and cached_domain != domain:
                    continue

                cache_key = key.replace('code:', '')
                results.append({
                    'cache_key': cache_key,
                    'domain': cached_domain,
                    'fields': metadata.get('fields', []),
                    'url': metadata.get('url', ''),
                    'cache_type': metadata.get('cache_type', 'structure'),
                    'created_at': cached_data.get('created_at', 0),
                    'structure_hash': metadata.get('structure_hash', '')
                })

            return results
        except Exception as e:
            logger.error(f" Error listing cache by domain: {str(e)}")
            return []

    async def async_list_by_domain(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Async version of list_by_domain for Redis cache

        Args:
            domain: Optional domain filter

        Returns:
            List of cached entries with domain metadata
        """
        if not self.enable_cache:
            return []

        # Use Redis if available
        if self.redis_cache:
            try:
                results = []
                # Scan Redis for code cache keys
                async for key in self.redis_cache.redis_client.scan_iter(match="code:*"):
                    try:
                        cached_data = await self.redis_cache.get(key)
                        if not cached_data:
                            continue

                        metadata = cached_data.get('metadata', {})
                        cached_domain = metadata.get('domain')

                        # Filter by domain if specified
                        if domain and cached_domain != domain:
                            continue

                        cache_key = key.replace('code:', '')
                        results.append({
                            'cache_key': cache_key,
                            'domain': cached_domain,
                            'fields': metadata.get('fields', []),
                            'url': metadata.get('url', ''),
                            'cache_type': metadata.get('cache_type', 'structure'),
                            'created_at': cached_data.get('created_at', 0),
                            'structure_hash': metadata.get('structure_hash', '')
                        })
                    except Exception as e:
                        logger.warning(f"Error reading cache entry {key}: {e}")
                        continue

                logger.info(f"Found {len(results)} cached patterns in Redis")
                return results
            except Exception as e:
                logger.warning(f"Redis cache list error: {e}, falling back to diskcache")
                # Fall through to diskcache

        # Fallback to sync diskcache method
        return self.list_by_domain(domain)

    def get_domains(self) -> List[str]:
        """
        Get list of all domains in cache

        Returns:
            List of unique domain names
        """
        if not self.enable_cache or not self.cache:
            return []

        try:
            domains = set()
            for key in self.cache.iterkeys():
                if not key.startswith('code:'):
                    continue

                cached_data = self.cache.get(key)
                if cached_data:
                    metadata = cached_data.get('metadata', {})
                    domain = metadata.get('domain')
                    if domain:
                        domains.add(domain)

            return sorted(list(domains))
        except Exception as e:
            logger.error(f" Error getting domains: {str(e)}")
            return []

    def export_cache(self, export_path: str) -> bool:
        """
        Export cache to JSON file

        Args:
            export_path: Path to export file

        Returns:
            True if exported successfully
        """
        if not self.enable_cache or not self.cache:
            return False

        try:
            export_data = {}

            for key in self.cache.iterkeys():
                if key.startswith('code:'):
                    structure_hash = key.replace('code:', '')
                    cached_data = self.cache.get(key)
                    if cached_data:
                        export_data[structure_hash] = cached_data

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f" Exported {len(export_data)} cache entries to {export_path}")
            return True

        except Exception as e:
            logger.error(f" Export error: {str(e)}")
            return False

    def import_cache(self, import_path: str) -> bool:
        """
        Import cache from JSON file

        Args:
            import_path: Path to import file

        Returns:
            True if imported successfully
        """
        if not self.enable_cache or not self.cache:
            return False

        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            count = 0
            for structure_hash, cached_data in import_data.items():
                self.set(
                    structure_hash,
                    cached_data['code'],
                    cached_data.get('metadata', {})
                )
                count += 1

            logger.info(f" Imported {count} cache entries from {import_path}")
            return True

        except Exception as e:
            logger.error(f" Import error: {str(e)}")
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache:
            self.cache.close()

