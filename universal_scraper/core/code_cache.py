"""
Code Cache System
Stores and retrieves generated extraction code for reuse
"""

import os
import json
import logging
import time
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path
import diskcache

logger = logging.getLogger(__name__)


def generate_cache_key(structure_hash: str, fields: List[str]) -> str:
    """
    Generate a cache key that includes both structure hash and field names.
    
    This ensures that different field sets get their own cached code,
    preventing field mismatch issues (e.g., 'title' vs 'question_title').
    
    Args:
        structure_hash: HTML structure hash
        fields: List of field names to extract
        
    Returns:
        Combined cache key string
    """
    # Sort fields for consistency (order shouldn't matter)
    sorted_fields = sorted(fields)
    fields_str = ','.join(sorted_fields)
    
    # Create a hash of the field names
    fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]
    
    # Combine: structure_hash + fields_hash
    return f"{structure_hash}:{fields_hash}"


class CodeCache:
    """Manages caching of generated extraction code"""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl: int = 86400,  # 24 hours default
        enable_cache: bool = True
    ):
        """
        Initialize Code Cache
        
        Args:
            cache_dir: Directory for cache storage
            ttl: Time to live in seconds
            enable_cache: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enable_cache = enable_cache
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(self.cache_dir))
            logger.info(f" Cache initialized: {self.cache_dir}")
        else:
            self.cache = None
            logger.info(" Cache disabled")
    
    def get(self, structure_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached code by structure hash
        
        Args:
            structure_hash: Structural hash of the page
            
        Returns:
            Cached code dict or None if not found/expired
        """
        if not self.enable_cache or not self.cache:
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
                
                logger.info(f" Cache hit: {structure_hash[:16]}...")
                return cached_data
            else:
                logger.info(f" Cache miss: {structure_hash[:16]}...")
                return None
                
        except Exception as e:
            logger.error(f" Cache get error: {str(e)}")
            return None
    
    def set(
        self,
        structure_hash: str,
        code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store code in cache
        
        Args:
            structure_hash: Structural hash of the page
            code: Generated extraction code
            metadata: Optional metadata about the code
            
        Returns:
            True if stored successfully
        """
        if not self.enable_cache or not self.cache:
            return False
        
        try:
            cache_key = f"code:{structure_hash}"
            
            cached_data = {
                'code': code,
                'metadata': metadata or {},
                'created_at': time.time(),
                'ttl': self.ttl,
                'structure_hash': structure_hash
            }
            
            self.cache.set(cache_key, cached_data, expire=self.ttl)
            
            logger.info(f" Cached code: {structure_hash[:16]}... (TTL: {self.ttl}s)")
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

