"""
Multi-tenant Pattern Cache with Public/Private Visibility
Stores extraction patterns per tenant with sharing capabilities
"""
import logging
import time
import hashlib
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class CacheVisibility(str, Enum):
    PRIVATE = "private"  # Only visible to owner
    PUBLIC = "public"    # Visible to all users


class TenantPatternCache:
    """
    Multi-tenant cache for extraction patterns

    Features:
    - Per-tenant pattern storage
    - Public/private visibility
    - Pattern sharing between tenants
    - Usage tracking for shared patterns
    """

    def __init__(self, redis_cache: Optional[Any] = None):
        """
        Initialize tenant pattern cache

        Args:
            redis_cache: RedisCache instance for distributed storage
        """
        self.redis_cache = redis_cache
        self.prefix = "pattern:"

        if redis_cache and redis_cache.redis_client:
            logger.info("TenantPatternCache initialized with Redis backend")
        else:
            logger.warning("TenantPatternCache initialized without Redis - patterns won't persist")

    def _make_key(self, tenant_id: str, domain: str, fields_hash: str) -> str:
        """Generate cache key for tenant pattern"""
        return f"{self.prefix}{tenant_id}:{domain}:{fields_hash}"

    def _make_public_key(self, domain: str, fields_hash: str) -> str:
        """Generate cache key for public pattern"""
        return f"{self.prefix}public:{domain}:{fields_hash}"

    def _hash_fields(self, fields: List[str]) -> str:
        """Generate hash from fields list"""
        fields_str = ','.join(sorted(fields))
        return hashlib.md5(fields_str.encode()).hexdigest()[:8]

    async def store_pattern(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str],
        pattern_data: Dict[str, Any],
        visibility: CacheVisibility = CacheVisibility.PRIVATE,
        url: Optional[str] = None
    ) -> bool:
        """
        Store extraction pattern for tenant

        Args:
            tenant_id: Tenant identifier
            domain: Domain the pattern is for
            fields: Fields extracted
            pattern_data: Extraction pattern (code or Direct LLM result)
            visibility: PUBLIC or PRIVATE
            url: Original URL (for reference)

        Returns:
            True if stored successfully
        """
        if not self.redis_cache:
            logger.warning("Cannot store pattern - no Redis cache")
            return False

        try:
            fields_hash = self._hash_fields(fields)

            # Create pattern entry
            entry = {
                "tenant_id": tenant_id,
                "domain": domain,
                "fields": fields,
                "fields_hash": fields_hash,
                "pattern_data": pattern_data,
                "visibility": visibility.value,
                "url": url,
                "created_at": time.time(),
                "updated_at": time.time(),
                "usage_count": 0,
                "shared_from": None,  # Set if this is a copy of shared pattern
            }

            # Store tenant-specific pattern
            tenant_key = self._make_key(tenant_id, domain, fields_hash)
            success = await self.redis_cache.set(tenant_key, entry, ttl=86400 * 30)  # 30 days

            if not success:
                logger.error(f"Failed to store tenant pattern for {tenant_id}:{domain}")
                return False

            # If public, also store in public index
            if visibility == CacheVisibility.PUBLIC:
                public_key = self._make_public_key(domain, fields_hash)
                public_entry = {
                    **entry,
                    "original_tenant_id": tenant_id,
                }
                await self.redis_cache.set(public_key, public_entry, ttl=86400 * 30)

            logger.info(f"Stored pattern for {tenant_id}:{domain} ({visibility.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False

    async def get_pattern(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str],
        check_public: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get extraction pattern for tenant

        Args:
            tenant_id: Tenant identifier
            domain: Domain to look up
            fields: Fields to extract
            check_public: Also check public patterns if tenant pattern not found

        Returns:
            Pattern data or None
        """
        if not self.redis_cache:
            return None

        try:
            fields_hash = self._hash_fields(fields)

            # First check tenant-specific pattern
            tenant_key = self._make_key(tenant_id, domain, fields_hash)
            pattern = await self.redis_cache.get(tenant_key)

            if pattern:
                # Update usage count
                pattern["usage_count"] = pattern.get("usage_count", 0) + 1
                pattern["last_used"] = time.time()
                await self.redis_cache.set(tenant_key, pattern, ttl=86400 * 30)
                return pattern

            # Check public patterns if enabled
            if check_public:
                public_key = self._make_public_key(domain, fields_hash)
                public_pattern = await self.redis_cache.get(public_key)

                if public_pattern:
                    logger.info(f"Using public pattern for {domain}")
                    # Track that tenant used a public pattern
                    public_pattern["usage_count"] = public_pattern.get("usage_count", 0) + 1
                    await self.redis_cache.set(public_key, public_pattern, ttl=86400 * 30)
                    return public_pattern

            return None

        except Exception as e:
            logger.error(f"Failed to get pattern: {e}")
            return None

    async def list_tenant_patterns(
        self,
        tenant_id: str,
        domain: Optional[str] = None,
        visibility: Optional[CacheVisibility] = None
    ) -> List[Dict[str, Any]]:
        """
        List all patterns for a tenant

        Args:
            tenant_id: Tenant identifier
            domain: Optional domain filter
            visibility: Optional visibility filter

        Returns:
            List of pattern entries
        """
        if not self.redis_cache:
            return []

        try:
            patterns = []
            pattern_prefix = f"{self.prefix}{tenant_id}:"

            # Scan for tenant patterns
            keys = await self.redis_cache.list_keys(pattern=f"{pattern_prefix}*")

            for key in keys:
                try:
                    pattern = await self.redis_cache.get(key)
                    if pattern:
                        # Apply filters
                        if domain and pattern.get("domain") != domain:
                            continue
                        if visibility and pattern.get("visibility") != visibility.value:
                            continue

                        patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to get pattern {key}: {e}")

            return patterns

        except Exception as e:
            logger.error(f"Failed to list tenant patterns: {e}")
            return []

    async def list_public_patterns(
        self,
        domain: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List all public patterns

        Args:
            domain: Optional domain filter
            limit: Maximum number of patterns to return

        Returns:
            List of public pattern entries
        """
        if not self.redis_cache:
            return []

        try:
            patterns = []
            public_prefix = f"{self.prefix}public:"

            # Scan for public patterns
            keys = await self.redis_cache.list_keys(pattern=f"{public_prefix}*")

            for key in keys[:limit]:
                try:
                    pattern = await self.redis_cache.get(key)
                    if pattern:
                        # Apply domain filter
                        if domain and pattern.get("domain") != domain:
                            continue

                        patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to get public pattern {key}: {e}")

            # Sort by usage count (most popular first)
            patterns.sort(key=lambda p: p.get("usage_count", 0), reverse=True)

            return patterns

        except Exception as e:
            logger.error(f"Failed to list public patterns: {e}")
            return []

    async def update_visibility(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str],
        visibility: CacheVisibility
    ) -> bool:
        """
        Update pattern visibility (make public/private)

        Args:
            tenant_id: Tenant identifier
            domain: Domain
            fields: Fields
            visibility: New visibility setting

        Returns:
            True if updated successfully
        """
        if not self.redis_cache:
            return False

        try:
            fields_hash = self._hash_fields(fields)
            tenant_key = self._make_key(tenant_id, domain, fields_hash)
            public_key = self._make_public_key(domain, fields_hash)

            # Get existing pattern
            pattern = await self.redis_cache.get(tenant_key)
            if not pattern:
                return False

            old_visibility = pattern.get("visibility")
            pattern["visibility"] = visibility.value
            pattern["updated_at"] = time.time()

            # Update tenant pattern
            await self.redis_cache.set(tenant_key, pattern, ttl=86400 * 30)

            # Handle public index
            if visibility == CacheVisibility.PUBLIC and old_visibility != "public":
                # Add to public index
                public_entry = {**pattern, "original_tenant_id": tenant_id}
                await self.redis_cache.set(public_key, public_entry, ttl=86400 * 30)
                logger.info(f"Made pattern public: {domain}")
            elif visibility == CacheVisibility.PRIVATE and old_visibility == "public":
                # Remove from public index
                await self.redis_cache.delete(public_key)
                logger.info(f"Made pattern private: {domain}")

            return True

        except Exception as e:
            logger.error(f"Failed to update visibility: {e}")
            return False

    async def copy_public_pattern(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str]
    ) -> bool:
        """
        Copy a public pattern to tenant's private cache

        Args:
            tenant_id: Tenant to copy to
            domain: Domain
            fields: Fields

        Returns:
            True if copied successfully
        """
        if not self.redis_cache:
            return False

        try:
            fields_hash = self._hash_fields(fields)
            public_key = self._make_public_key(domain, fields_hash)

            # Get public pattern
            public_pattern = await self.redis_cache.get(public_key)
            if not public_pattern:
                return False

            # Create copy for tenant
            tenant_pattern = {
                **public_pattern,
                "tenant_id": tenant_id,
                "visibility": CacheVisibility.PRIVATE.value,
                "shared_from": public_pattern.get("original_tenant_id"),
                "created_at": time.time(),
                "usage_count": 0,
            }

            tenant_key = self._make_key(tenant_id, domain, fields_hash)
            await self.redis_cache.set(tenant_key, tenant_pattern, ttl=86400 * 30)

            logger.info(f"Copied public pattern to {tenant_id}: {domain}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy public pattern: {e}")
            return False

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain (remove www. prefix for consistency)"""
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    async def delete_pattern(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str]
    ) -> bool:
        """
        Delete a tenant's pattern

        Args:
            tenant_id: Tenant identifier
            domain: Domain
            fields: Fields

        Returns:
            True if deleted successfully
        """
        if not self.redis_cache:
            logger.warning("Cannot delete pattern - no Redis cache")
            return False

        try:
            # Normalize domain and fields
            normalized_domain = self._normalize_domain(domain)
            normalized_fields = [f.strip().lower() for f in fields]
            fields_hash = self._hash_fields(fields)

            logger.info(f"Attempting to delete pattern: domain={domain}, fields={fields}, hash={fields_hash}")

            # Try multiple key variations to find the pattern
            key_variations = [
                # Standard key with normalized domain
                self._make_key(tenant_id, normalized_domain, fields_hash),
                # Key with original domain (might have www.)
                self._make_key(tenant_id, domain, fields_hash),
                # Key with domain as-is (lowercase)
                self._make_key(tenant_id, domain.lower().strip(), fields_hash),
            ]

            # Also try with www. prefix if domain doesn't have it
            if not domain.startswith('www.'):
                key_variations.append(self._make_key(tenant_id, f"www.{domain}", fields_hash))

            # Try to find pattern by checking all key variations
            matching_key = None
            matching_pattern = None

            for key_variant in key_variations:
                pattern = await self.redis_cache.get(key_variant)
                if pattern:
                    # Verify fields match (order-independent)
                    pattern_fields = [f.strip().lower() for f in pattern.get("fields", [])]
                    if set(pattern_fields) == set(normalized_fields):
                        matching_key = key_variant
                        matching_pattern = pattern
                        logger.info(f"Found pattern with key: {key_variant}")
                        break

            # If not found by direct key lookup, search through all patterns
            if not matching_pattern:
                logger.info("Pattern not found by direct key lookup, searching all patterns...")
                patterns = await self.list_tenant_patterns(tenant_id)

                for pattern in patterns:
                    pattern_fields = [f.strip().lower() for f in pattern.get("fields", [])]
                    pattern_domain = pattern.get("domain", "").lower().strip()

                    # Check if fields and domain match (order-independent)
                    if (pattern_domain in [normalized_domain, domain.lower().strip(), f"www.{normalized_domain}", f"www.{domain.lower().strip()}"] and
                        set(pattern_fields) == set(normalized_fields)):
                        # Reconstruct the key using stored values
                        stored_fields_hash = pattern.get("fields_hash") or self._hash_fields(pattern.get("fields", []))
                        stored_domain = pattern.get("domain")
                        matching_key = self._make_key(tenant_id, stored_domain, stored_fields_hash)
                        matching_pattern = pattern
                        logger.info(f"Found pattern by searching: key={matching_key}")
                        break

            if not matching_pattern or not matching_key:
                logger.warning(f"Pattern not found after searching: domain={domain}, fields={fields}")
                # Log available patterns for debugging
                all_patterns = await self.list_tenant_patterns(tenant_id)
                logger.info(f"Available patterns for tenant {tenant_id}: {len(all_patterns)}")
                for p in all_patterns[:5]:  # Log first 5
                    logger.info(f"  - {p.get('domain')}: {p.get('fields')}")
                return False

            # Delete tenant pattern
            deleted = await self.redis_cache.delete(matching_key)

            if not deleted:
                logger.warning(f"Failed to delete pattern from Redis: {matching_key}")
                return False

            # If it was public, also remove from public index
            if matching_pattern.get("visibility") == "public":
                stored_domain = matching_pattern.get("domain")
                stored_fields_hash = matching_pattern.get("fields_hash") or self._hash_fields(matching_pattern.get("fields", []))
                public_key = self._make_public_key(stored_domain, stored_fields_hash)
                await self.redis_cache.delete(public_key)
                logger.info(f"Also deleted public pattern: {public_key}")

            logger.info(f"Successfully deleted pattern for {tenant_id}: {matching_pattern.get('domain')} (fields: {matching_pattern.get('fields')})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete pattern: {e}", exc_info=True)
            return False

    async def get_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Statistics dict
        """
        if not self.redis_cache:
            return {"error": "No Redis cache available"}

        try:
            if tenant_id:
                patterns = await self.list_tenant_patterns(tenant_id)
                private_count = len([p for p in patterns if p.get("visibility") == "private"])
                public_count = len([p for p in patterns if p.get("visibility") == "public"])

                return {
                    "tenant_id": tenant_id,
                    "total_patterns": len(patterns),
                    "private_patterns": private_count,
                    "public_patterns": public_count,
                    "domains": list(set(p.get("domain") for p in patterns if p.get("domain"))),
                }
            else:
                # Global stats
                public_patterns = await self.list_public_patterns()
                return {
                    "total_public_patterns": len(public_patterns),
                    "popular_domains": list(set(p.get("domain") for p in public_patterns[:20])),
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

