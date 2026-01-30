"""
DOM Digest Cache - Layer 2 Cache
Caches page type and template associations based on DOM fingerprints

Purpose: Detect "same layout as before" quickly (<10ms) without LLM calls.
This enables fast template matching before expensive LLM analysis.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .dom_digest import DOMDigestGenerator
from .unified_cache import UnifiedPatternCache

logger = logging.getLogger(__name__)


class DOMDigestCache:
    """
    Caches DOM digests with page type and template associations
    
    Architecture:
    - Key: domain + path_pattern + dom_digest
    - Value: page_type + template_id + version + success_rate
    - TTL: 24 hours (configurable)
    
    Usage:
    1. Generate digest from HTML
    2. Check cache for matching digest
    3. If match: return cached template_id (skip LLM)
    4. If miss: proceed to LLM analysis, then cache result
    """
    
    def __init__(
        self,
        ttl_hours: int = 24,
        enable_cache: bool = True
    ):
        """
        Initialize DOM digest cache
        
        Args:
            ttl_hours: Cache TTL in hours
            enable_cache: Enable caching (can disable for testing)
        """
        self.ttl_hours = ttl_hours
        self.enable_cache = enable_cache
        
        self.digest_generator = DOMDigestGenerator()
        
        # Initialize cache backend
        if self.enable_cache:
            try:
                self.cache = UnifiedPatternCache(force_local=False)
                logger.info(f" DOM Digest Cache enabled (TTL: {ttl_hours}h)")
            except Exception as e:
                logger.warning(f"  Failed to initialize cache: {e}, caching disabled")
                self.cache = None
                self.enable_cache = False
        else:
            self.cache = None
    
    async def get_template_for_digest(
        self,
        url: str,
        html: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached template for DOM digest
        
        Args:
            url: Source URL
            html: HTML content
            
        Returns:
            Cached template info or None if not found:
            {
                'template_id': str,
                'page_type': str,
                'version': int,
                'success_rate': float,
                'digest': str
            }
        """
        if not self.enable_cache or not self.cache:
            return None
        
        try:
            # Generate digest
            digest_result = self.digest_generator.generate_digest(html, url)
            digest = digest_result['digest']
            
            # Generate cache key
            cache_key = self.digest_generator.get_cache_key(url, digest)
            
            # Check cache
            cached_data = await self.cache.backend.get(cache_key)
            
            if cached_data:
                # Check TTL
                cached_time = datetime.fromisoformat(cached_data.get('cached_at', ''))
                age = datetime.now() - cached_time
                
                if age < timedelta(hours=self.ttl_hours):
                    logger.info(f" DOM digest cache hit: {digest[:16]}... (template: {cached_data.get('template_id', 'unknown')})")
                    return {
                        'template_id': cached_data.get('template_id'),
                        'page_type': cached_data.get('page_type', 'unknown'),
                        'version': cached_data.get('version', 1),
                        'success_rate': cached_data.get('success_rate', 0.0),
                        'digest': digest
                    }
                else:
                    logger.debug(f" DOM digest cache expired: {digest[:16]}... (age: {age})")
            
            return None
            
        except Exception as e:
            logger.warning(f"  DOM digest cache lookup failed: {e}")
            return None
    
    async def store_template_for_digest(
        self,
        url: str,
        html: str,
        template_id: str,
        page_type: Optional[str] = None,
        version: int = 1,
        success_rate: float = 1.0
    ) -> bool:
        """
        Store template association for DOM digest
        
        Args:
            url: Source URL
            html: HTML content
            template_id: Template identifier
            page_type: Page type (inferred if not provided)
            version: Template version
            success_rate: Success rate (0.0-1.0)
            
        Returns:
            True if stored successfully
        """
        if not self.enable_cache or not self.cache:
            return False
        
        try:
            # Generate digest
            digest_result = self.digest_generator.generate_digest(html, url)
            digest = digest_result['digest']
            
            # Use inferred page type if not provided
            if page_type is None:
                page_type = digest_result.get('page_type', 'unknown')
            
            # Generate cache key
            cache_key = self.digest_generator.get_cache_key(url, digest)
            
            # Prepare cache entry
            cache_entry = {
                'template_id': template_id,
                'page_type': page_type,
                'version': version,
                'success_rate': success_rate,
                'digest': digest,
                'cached_at': datetime.now().isoformat(),
                'url': url
            }
            
            # Store in cache
            await self.cache.backend.set(cache_key, cache_entry)
            
            logger.info(f" DOM digest cached: {digest[:16]}... (template: {template_id}, type: {page_type})")
            
            return True
            
        except Exception as e:
            logger.warning(f"  DOM digest cache store failed: {e}")
            return False
    
    async def find_similar_digests(
        self,
        url: str,
        html: str,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find similar digests (for clustering/fuzzy matching)
        
        Note: Current implementation uses exact matching.
        Future: Implement LSH/simhash for fuzzy matching.
        
        Args:
            url: Source URL
            html: HTML content
            similarity_threshold: Minimum similarity (not used in exact match)
            
        Returns:
            List of similar cached templates
        """
        # For now: return exact match only
        cached = await self.get_template_for_digest(url, html)
        
        if cached:
            return [cached]
        
        return []
    
    def generate_digest(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate DOM digest (convenience method)
        
        Args:
            html: HTML content
            url: Optional URL
            
        Returns:
            Digest result dict
        """
        return self.digest_generator.generate_digest(html, url)



