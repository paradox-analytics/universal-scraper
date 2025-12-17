"""
Unified Cache Abstraction
Works both locally (file-based) and on Apify (KV Store)
Automatically detects environment and uses appropriate backend
"""
import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Dict) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix"""
        pass


class LocalFileCache(CacheBackend):
    """
    Local file-based cache for development
    Mimics Apify KV Store behavior
    """
    
    def __init__(self, cache_dir: str = "./local_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f" Using LOCAL file cache: {self.cache_dir}")
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key"""
        # Hash key to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get value from file cache"""
        path = self._get_path(key)
        
        if not path.exists():
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                logger.debug(f" Cache HIT (local): {key}")
                return data
        except Exception as e:
            logger.warning(f"  Failed to read cache: {e}")
            return None
    
    async def set(self, key: str, value: Dict) -> None:
        """Set value in file cache"""
        path = self._get_path(key)
        
        try:
            with open(path, 'w') as f:
                json.dump(value, f, indent=2)
            
            # Also save key mapping for debugging
            mapping_path = self.cache_dir / "key_mapping.json"
            mappings = {}
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
            
            mappings[key] = path.name
            with open(mapping_path, 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.debug(f" Cache SET (local): {key}")
        except Exception as e:
            logger.error(f" Failed to write cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            logger.debug(f"  Cache DELETE (local): {key}")
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys (requires key_mapping.json)"""
        mapping_path = self.cache_dir / "key_mapping.json"
        if not mapping_path.exists():
            return []
        
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        
        if prefix:
            return [k for k in mappings.keys() if k.startswith(prefix)]
        return list(mappings.keys())


class ApifyKVCache(CacheBackend):
    """
    Apify Key-Value Store cache for production
    Uses a named KV Store to persist across actor runs
    """
    
    def __init__(self, store_name: str = "direct-llm-cache"):
        # Import only when needed (not available locally)
        try:
            from apify import Actor
            self.Actor = Actor
            self.store_name = store_name
            # Use named store for persistence across runs
            # Named stores persist indefinitely, default store is scoped per run
            self._store = None
            logger.info(f"  Using APIFY KV Store cache (named store: {store_name})")
        except ImportError:
            raise RuntimeError("Apify SDK not available. Use LocalFileCache instead.")
    
    async def _get_store(self):
        """Get named KV Store (persists across runs)"""
        if not hasattr(self, '_store') or self._store is None:
            try:
                # Open or create named store (persists across runs)
                # Named stores don't expire, default store is scoped per run
                self._store = await self.Actor.open_key_value_store(name=self.store_name)
                if self._store:
                    logger.info(f" Opened named KV Store: {self.store_name} (persists across runs)")
                return self._store
            except Exception as e:
                logger.warning(f"  Could not open named store '{self.store_name}': {e}")
                logger.warning("  Falling back to default store (cache won't persist across runs)")
                self._store = None
                return None
        return self._store
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get value from Apify KV Store"""
        try:
            logger.info(f" Apify KV Store GET: {key[:50]}...")
            
            # Try named store first (persists across runs)
            store = await self._get_store()
            if store:
                try:
                    # Apify SDK uses get_value() method on KeyValueStore
                    value = await store.get_value(key)
                    if value:
                        # Parse if it's a string
                        if isinstance(value, str):
                            value = json.loads(value)
                        logger.info(f" Cache HIT (Apify named store '{self.store_name}'): {key[:50]}... (found {len(value.get('items', []))} items)")
                        return value
                    else:
                        logger.info(f" Cache MISS (Apify named store '{self.store_name}'): {key[:50]}... (key not found in named store)")
                        # Don't fall back to default store - named store is the source of truth
                        return None
                except Exception as e:
                    logger.warning(f"  Named store get failed: {e}, trying default store")
            
            # Fallback to default store only if named store failed to open
            value = await self.Actor.get_value(key)
            if value:
                logger.info(f" Cache HIT (Apify default store): {key[:50]}... (found {len(value.get('items', []))} items)")
            else:
                logger.info(f" Cache MISS (Apify default store): {key[:50]}... (key not found)")
            return value
        except Exception as e:
            logger.warning(f"  Failed to read Apify cache: {e}")
            return None
    
    async def set(self, key: str, value: Dict) -> None:
        """Set value in Apify KV Store"""
        try:
            logger.info(f" Apify KV Store SET: {key[:50]}... ({len(value.get('items', []))} items)")
            
            # Try named store first (persists across runs)
            store = await self._get_store()
            if store:
                try:
                    # Apify SDK uses set_value() method on KeyValueStore
                    await store.set_value(key, value)
                    logger.info(f" Cache SET (Apify named store '{self.store_name}'): {key[:50]}... (saved, will persist across runs)")
                    return
                except Exception as e:
                    logger.warning(f"  Named store set failed: {e}, using default store")
            
            # Fallback to default store (scoped per run)
            await self.Actor.set_value(key, value)
            logger.warning(f"  Cache SET (Apify default store): {key[:50]}... (saved, but WON'T persist across runs - using default store)")
        except Exception as e:
            logger.error(f" Failed to write Apify cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete value from Apify KV Store"""
        try:
            await self.Actor.set_value(key, None)
            logger.debug(f"  Cache DELETE (Apify): {key}")
        except Exception as e:
            logger.error(f" Failed to delete from Apify cache: {e}")
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys in Apify KV Store"""
        # Note: Apify doesn't have a native list_keys, so this is limited
        logger.warning("  list_keys not fully supported in Apify KV Store")
        return []


class UnifiedPatternCache:
    """
    Unified pattern cache that works locally and on Apify
    Automatically detects environment and uses appropriate backend
    """
    
    def __init__(self, force_local: bool = False):
        """
        Initialize cache with automatic backend selection
        
        Args:
            force_local: Force local cache even if running on Apify (for testing)
        """
        # Detect environment
        self.is_apify = self._detect_apify_environment()
        
        # Choose backend
        if force_local or not self.is_apify:
            self.backend = LocalFileCache()
            self.env = "LOCAL"
        else:
            try:
                self.backend = ApifyKVCache()
                self.env = "APIFY"
            except RuntimeError:
                logger.warning("  Apify not available, falling back to local cache")
                self.backend = LocalFileCache()
                self.env = "LOCAL (fallback)"
        
        # In-memory cache for current run (L1)
        self.memory_cache: Dict[str, Dict] = {}
        
        logger.info(f" Pattern Cache initialized: {self.env}")
    
    def _detect_apify_environment(self) -> bool:
        """Detect if running on Apify"""
        # Apify sets these environment variables
        return bool(
            os.environ.get('APIFY_IS_AT_HOME') or
            os.environ.get('APIFY_ACTOR_ID') or
            os.environ.get('APIFY_ACTOR_RUN_ID')
        )
    
    def _make_cache_key(self, embedding_hash: str, fields: List[str]) -> str:
        """Create cache key from embedding and fields"""
        fields_str = "_".join(sorted(fields))
        fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]
        return f"pattern_{embedding_hash}_{fields_hash}"
    
    async def get_pattern(
        self, 
        embedding_hash: str, 
        fields: List[str],
        domain: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get cached pattern
        
        Args:
            embedding_hash: Hash of structural embedding
            fields: List of fields to extract
            domain: Optional domain name for logging
            
        Returns:
            Cached pattern or None if not found
        """
        cache_key = self._make_cache_key(embedding_hash, fields)
        
        # Check L1 cache (memory - fastest)
        if cache_key in self.memory_cache:
            logger.info(f" L1 Cache HIT (memory): {cache_key}")
            return self.memory_cache[cache_key]
        
        # Check L2 cache (file/Apify KV - persistent)
        pattern = await self.backend.get(cache_key)
        
        if pattern:
            logger.info(f" L2 Cache HIT ({self.env}): {cache_key}")
            if domain:
                logger.info(f"   Domain: {domain}")
            logger.info(f"   Fields: {fields}")
            
            # Warm up L1 cache
            self.memory_cache[cache_key] = pattern
            
            # Update usage stats
            if 'metadata' in pattern:
                pattern['metadata']['usage_count'] = pattern['metadata'].get('usage_count', 0) + 1
                pattern['metadata']['last_used'] = str(Path.cwd())  # Timestamp placeholder
                await self.backend.set(cache_key, pattern)
            
            return pattern
        
        logger.info(f" Cache MISS: {cache_key}")
        if domain:
            logger.info(f"   Domain: {domain} (will learn new pattern)")
        return None
    
    async def save_pattern(
        self,
        embedding_hash: str,
        fields: List[str],
        pattern: Dict,
        domain: str,
        url: str
    ) -> str:
        """
        Save pattern to cache
        
        Args:
            embedding_hash: Hash of structural embedding
            fields: List of fields to extract
            pattern: Extraction pattern to cache
            domain: Domain name
            url: Example URL
            
        Returns:
            Cache key
        """
        cache_key = self._make_cache_key(embedding_hash, fields)
        
        # Add metadata
        cached_data = {
            "pattern": pattern,
            "metadata": {
                "domain": domain,
                "fields": fields,
                "example_url": url,
                "embedding_hash": embedding_hash,
                "created_at": "now",  # Placeholder
                "usage_count": 0
            }
        }
        
        # Save to L2 (persistent)
        await self.backend.set(cache_key, cached_data)
        
        # Save to L1 (memory)
        self.memory_cache[cache_key] = cached_data
        
        logger.info(f" Pattern SAVED ({self.env}): {cache_key}")
        logger.info(f"   Domain: {domain}")
        logger.info(f"   Fields: {fields}")
        logger.info(f"   Cache location: {self.backend.cache_dir if hasattr(self.backend, 'cache_dir') else 'Apify KV'}")
        
        return cache_key
    
    async def clear_cache(self, prefix: str = "pattern_") -> int:
        """
        Clear cache (useful for testing)
        
        Args:
            prefix: Only clear keys with this prefix
            
        Returns:
            Number of keys cleared
        """
        keys = await self.backend.list_keys(prefix)
        
        for key in keys:
            await self.backend.delete(key)
        
        # Clear memory cache
        self.memory_cache.clear()
        
        logger.info(f"  Cache CLEARED: {len(keys)} patterns removed")
        return len(keys)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        keys = await self.backend.list_keys("pattern_")
        
        total_patterns = len(keys)
        memory_patterns = len(self.memory_cache)
        
        # Calculate total usage if possible
        total_usage = 0
        if isinstance(self.backend, LocalFileCache):
            for key in keys:
                pattern = await self.backend.get(key)
                if pattern and 'metadata' in pattern:
                    total_usage += pattern['metadata'].get('usage_count', 0)
        
        return {
            "environment": self.env,
            "total_patterns": total_patterns,
            "memory_patterns": memory_patterns,
            "total_usage": total_usage,
            "backend": self.backend.__class__.__name__
        }




