"""
API Discovery Cache
Stores discovered API endpoints for direct access (bypassing browser)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class APICache:
    """
    Caches discovered API endpoints for direct access
    This is the key to the JSON-forward architecture
    """
    
    def __init__(self, cache_dir: str = "./cache/apis"):
        """
        Initialize API Cache
        
        Args:
            cache_dir: Directory to store API cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "discovered_apis.json"
        self.cache = self._load_cache()
        
        logger.info(f" API Cache initialized: {self.cache_dir}")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f" Loaded {len(cache)} cached APIs")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load API cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API cache: {e}")
    
    def store_discovered_apis(
        self,
        domain: str,
        apis: Dict[str, Any],
        source_url: str
    ) -> None:
        """
        Store discovered APIs for a domain
        
        Args:
            domain: Domain name (e.g., 'www.leafly.com')
            apis: Discovered API endpoints
            source_url: URL where APIs were discovered
        """
        if domain not in self.cache:
            self.cache[domain] = {
                'apis': {},
                'discovered_at': time.time(),
                'source_url': source_url,
                'last_used': time.time(),
                'use_count': 0
            }
        
        # Merge new APIs
        self.cache[domain]['apis'].update(apis)
        self.cache[domain]['last_updated'] = time.time()
        
        self._save_cache()
        
        logger.info(f" Stored {len(apis)} APIs for {domain}")
    
    def get_apis(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get cached APIs for a domain
        
        Args:
            domain: Domain name
            
        Returns:
            Dict of APIs or None if not cached
        """
        if domain in self.cache:
            # Update usage stats
            self.cache[domain]['last_used'] = time.time()
            self.cache[domain]['use_count'] = self.cache[domain].get('use_count', 0) + 1
            self._save_cache()
            
            logger.info(f" Found {len(self.cache[domain]['apis'])} cached APIs for {domain}")
            return self.cache[domain]['apis']
        
        logger.debug(f" No cached APIs for {domain}")
        return None
    
    def has_api(self, domain: str, api_pattern: str) -> bool:
        """
        Check if a specific API pattern is cached
        
        Args:
            domain: Domain name
            api_pattern: API pattern (e.g., 'GET:/api/products')
            
        Returns:
            True if cached
        """
        apis = self.get_apis(domain)
        return apis is not None and api_pattern in apis
    
    def get_api(self, domain: str, api_pattern: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific API endpoint
        
        Args:
            domain: Domain name
            api_pattern: API pattern
            
        Returns:
            API details or None
        """
        apis = self.get_apis(domain)
        if apis:
            return apis.get(api_pattern)
        return None
    
    def clear(self, domain: Optional[str] = None) -> None:
        """
        Clear cache
        
        Args:
            domain: Optional domain to clear, or None to clear all
        """
        if domain:
            if domain in self.cache:
                del self.cache[domain]
                logger.info(f" Cleared API cache for {domain}")
        else:
            self.cache = {}
            logger.info(" Cleared all API cache")
        
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_domains = len(self.cache)
        total_apis = sum(len(domain_data['apis']) for domain_data in self.cache.values())
        
        return {
            'total_domains': total_domains,
            'total_apis': total_apis,
            'cache_file': str(self.cache_file),
            'domains': list(self.cache.keys())
        }


