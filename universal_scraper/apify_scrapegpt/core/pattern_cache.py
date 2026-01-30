"""
Pattern Cache with ChromaDB - Vector-based pattern storage and retrieval

This module stores semantic extraction patterns with their structural embeddings,
enabling pattern reuse across structurally similar websites.

Key Concept:
- When a new website is scraped, we generate its structural embedding
- We search the vector DB for similar websites (>0.85 similarity)
- If found, we reuse the cached pattern (NO LLM call needed)
- If not found, we generate a new pattern and cache it
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Pattern caching will be limited.")

logger = logging.getLogger(__name__)


class PatternCache:
    """
    Vector-based pattern cache using ChromaDB.
    
    Features:
    - Stores semantic patterns with structural embeddings
    - Enables similarity-based pattern retrieval
    - Tracks pattern success rates for quality feedback
    - Supports pattern versioning and updates
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache/patterns",
        collection_name: str = "semantic_patterns",
        similarity_threshold: float = 0.85
    ):
        """
        Initialize pattern cache.
        
        Args:
            cache_dir: Directory to store ChromaDB data
            collection_name: Name of ChromaDB collection
            similarity_threshold: Minimum similarity score for pattern reuse (0.85 = 85%)
        """
        self.cache_dir = cache_dir
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        if CHROMADB_AVAILABLE:
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(
                path=cache_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Semantic extraction patterns with structural embeddings"}
            )
            
            logger.info(f" Pattern Cache initialized with ChromaDB ({self.collection.count()} patterns)")
        else:
            # Fallback to simple dict-based cache
            self.client = None
            self.collection = None
            self._fallback_cache: Dict[str, Dict] = {}
            logger.warning(" Using fallback cache (ChromaDB not available)")
    
    def find_similar_pattern(
        self,
        embedding: np.ndarray,
        fields: List[str],
        min_similarity: Optional[float] = None
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        Find cached pattern for structurally similar website.
        
        Args:
            embedding: Structural embedding of target website
            fields: Fields to extract (must match cached pattern)
            min_similarity: Override default similarity threshold
            
        Returns:
            Tuple of (pattern_id, pattern_dict, similarity_score) or None if no match
        """
        threshold = min_similarity or self.similarity_threshold
        
        if self.collection is None:
            # Fallback mode
            logger.debug("Using fallback cache (no similarity search)")
            return None
        
        try:
            # Query ChromaDB for similar embeddings
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=5,  # Get top 5 matches
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                logger.info(" No similar patterns found in cache")
                return None
            
            # Check each result for field compatibility and similarity
            for i, pattern_id in enumerate(results['ids'][0]):
                similarity = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                metadata = results['metadatas'][0][i]
                pattern_json = results['documents'][0][i]
                
                # Check similarity threshold
                if similarity < threshold:
                    logger.debug(f"Pattern {pattern_id} similarity too low: {similarity:.3f} < {threshold}")
                    continue
                
                # Check field compatibility
                cached_fields = metadata.get('fields', '').split(',')
                if not all(field in cached_fields for field in fields):
                    logger.debug(f"Pattern {pattern_id} fields don't match")
                    continue
                
                # Parse pattern
                pattern = json.loads(pattern_json)
                
                logger.info(f" Found similar pattern: {pattern_id} (similarity={similarity:.3f})")
                logger.debug(f"   Pattern metadata: {metadata}")
                
                return (pattern_id, pattern, similarity)
            
            logger.info(f" No compatible patterns found (checked {len(results['ids'][0])} candidates)")
            return None
            
        except Exception as e:
            logger.error(f" Error searching for similar patterns: {e}")
            return None
    
    def save_pattern(
        self,
        pattern: Dict[str, Any],
        embedding: np.ndarray,
        domain: str,
        fields: List[str],
        success_rate: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save semantic pattern with embedding to cache.
        
        Args:
            pattern: Semantic extraction pattern
            embedding: Structural embedding of website
            domain: Website domain (e.g., "example.com")
            fields: List of extracted fields
            success_rate: Initial success rate (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        # Generate pattern ID
        pattern_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare metadata
        meta = {
            'domain': domain,
            'fields': ','.join(fields),
            'success_rate': success_rate,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        if metadata:
            meta.update(metadata)
        
        # Serialize pattern
        pattern_json = json.dumps(pattern, indent=2)
        
        if self.collection is not None:
            try:
                # Add to ChromaDB
                self.collection.add(
                    ids=[pattern_id],
                    embeddings=[embedding.tolist()],
                    documents=[pattern_json],
                    metadatas=[meta]
                )
                logger.info(f" Saved pattern: {pattern_id} (fields: {', '.join(fields)})")
                
            except Exception as e:
                logger.error(f" Error saving pattern to ChromaDB: {e}")
                # Fallback to dict cache
                self._fallback_cache[pattern_id] = {
                    'pattern': pattern,
                    'metadata': meta
                }
        else:
            # Fallback mode
            self._fallback_cache[pattern_id] = {
                'pattern': pattern,
                'metadata': meta
            }
            logger.info(f" Saved pattern to fallback cache: {pattern_id}")
        
        return pattern_id
    
    def update_success_rate(
        self,
        pattern_id: str,
        success: bool,
        weight: float = 0.1
    ):
        """
        Update pattern success rate based on extraction result.
        
        Uses exponential moving average to track long-term success.
        
        Args:
            pattern_id: Pattern ID to update
            success: Whether extraction was successful
            weight: Weight for new data point (0.1 = 10% influence)
        """
        if self.collection is None:
            logger.debug("Cannot update success rate (no ChromaDB)")
            return
        
        try:
            # Get current metadata
            result = self.collection.get(
                ids=[pattern_id],
                include=["metadatas"]
            )
            
            if not result['ids']:
                logger.warning(f"Pattern not found: {pattern_id}")
                return
            
            metadata = result['metadatas'][0]
            current_rate = float(metadata.get('success_rate', 1.0))
            
            # Calculate new success rate (exponential moving average)
            new_value = 1.0 if success else 0.0
            new_rate = current_rate * (1 - weight) + new_value * weight
            
            # Update metadata
            metadata['success_rate'] = new_rate
            metadata['last_used'] = datetime.now().isoformat()
            
            self.collection.update(
                ids=[pattern_id],
                metadatas=[metadata]
            )
            
            logger.debug(f"Updated success rate: {pattern_id} → {new_rate:.3f}")
            
        except Exception as e:
            logger.error(f" Error updating success rate: {e}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern dict or None if not found
        """
        if self.collection is not None:
            try:
                result = self.collection.get(
                    ids=[pattern_id],
                    include=["documents"]
                )
                
                if result['ids'] and result['documents']:
                    return json.loads(result['documents'][0])
                    
            except Exception as e:
                logger.error(f" Error retrieving pattern: {e}")
        
        # Fallback
        cached = self._fallback_cache.get(pattern_id)
        if cached:
            return cached['pattern']
        
        return None
    
    def delete_pattern(self, pattern_id: str):
        """
        Delete pattern from cache.
        
        Args:
            pattern_id: Pattern ID to delete
        """
        if self.collection is not None:
            try:
                self.collection.delete(ids=[pattern_id])
                logger.info(f" Deleted pattern: {pattern_id}")
            except Exception as e:
                logger.error(f" Error deleting pattern: {e}")
        
        # Also remove from fallback cache
        if pattern_id in self._fallback_cache:
            del self._fallback_cache[pattern_id]
    
    def list_patterns(
        self,
        domain: Optional[str] = None,
        min_success_rate: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        List cached patterns with optional filtering.
        
        Args:
            domain: Filter by domain (optional)
            min_success_rate: Filter by minimum success rate
            
        Returns:
            List of pattern metadata dicts
        """
        if self.collection is None:
            logger.debug("Cannot list patterns (no ChromaDB)")
            return []
        
        try:
            # Get all patterns
            results = self.collection.get(include=["metadatas"])
            
            patterns = []
            for i, pattern_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                
                # Apply filters
                if domain and metadata.get('domain') != domain:
                    continue
                
                if float(metadata.get('success_rate', 0)) < min_success_rate:
                    continue
                
                patterns.append({
                    'id': pattern_id,
                    **metadata
                })
            
            logger.info(f" Found {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f" Error listing patterns: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        stats = {
            'chromadb_available': CHROMADB_AVAILABLE,
            'total_patterns': 0,
            'avg_success_rate': 0.0,
            'domains': 0
        }
        
        if self.collection is not None:
            try:
                count = self.collection.count()
                stats['total_patterns'] = count
                
                if count > 0:
                    results = self.collection.get(include=["metadatas"])
                    success_rates = [float(m.get('success_rate', 0)) for m in results['metadatas']]
                    domains = set(m.get('domain', '') for m in results['metadatas'])
                    
                    stats['avg_success_rate'] = np.mean(success_rates)
                    stats['domains'] = len(domains)
                    
            except Exception as e:
                logger.error(f" Error getting stats: {e}")
        else:
            stats['total_patterns'] = len(self._fallback_cache)
        
        return stats
    
    def clear(self):
        """
        Clear all cached patterns.
        
        WARNING: This is destructive and cannot be undone!
        """
        if self.collection is not None:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Semantic extraction patterns with structural embeddings"}
                )
                logger.warning(" Cleared all patterns from ChromaDB")
            except Exception as e:
                logger.error(f" Error clearing patterns: {e}")
        
        self._fallback_cache.clear()
        logger.warning(" Cleared fallback cache")




