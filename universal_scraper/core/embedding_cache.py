"""
Embedding-Based Selector Cache

Uses semantic similarity to find matching selector patterns from previously scraped sites.
This allows the system to learn from successful extractions and apply them to similar websites.

Benefits:
- 50x faster than LLM calls (0.1s vs 5s)
- 50x cheaper than LLM calls ($0.00002 vs $0.001)
- Learns from every successful extraction
- Works on new sites through similarity matching
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import litellm
from bs4 import BeautifulSoup
from collections import defaultdict

logger = logging.getLogger(__name__)


class EmbeddingBasedSelectorCache:
    """
    Cache successful selectors and find them using embedding similarity.
    
    Instead of exact domain matching, uses HTML structure similarity to find
    selector patterns from similar websites.
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache/embedding_selectors",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.api_key = api_key
        
        # In-memory index for fast lookup
        self.embeddings_index = []  # List of (embedding, metadata)
        self.selectors_db = {}  # domain -> selector patterns
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f" Embedding cache initialized: {len(self.selectors_db)} sites cached")
    
    def _load_cache(self):
        """Load cached embeddings and selectors from disk"""
        index_file = self.cache_dir / "embeddings_index.json"
        selectors_file = self.cache_dir / "selectors_db.json"
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.embeddings_index = json.load(f)
                logger.debug(f"   Loaded {len(self.embeddings_index)} embeddings from cache")
            
            if selectors_file.exists():
                with open(selectors_file, 'r') as f:
                    self.selectors_db = json.load(f)
                logger.debug(f"   Loaded {len(self.selectors_db)} selector patterns from cache")
        except Exception as e:
            logger.warning(f"     Error loading cache: {e}")
            self.embeddings_index = []
            self.selectors_db = {}
    
    def _save_cache(self):
        """Save embeddings and selectors to disk"""
        try:
            with open(self.cache_dir / "embeddings_index.json", 'w') as f:
                json.dump(self.embeddings_index, f)
            
            with open(self.cache_dir / "selectors_db.json", 'w') as f:
                json.dump(self.selectors_db, f, indent=2)
            
            logger.debug(f"    Saved cache: {len(self.selectors_db)} sites")
        except Exception as e:
            logger.warning(f"     Error saving cache: {e}")
    
    def _extract_html_structure(self, html: str, max_length: int = 8000) -> str:
        """
        Extract structural HTML (tags, classes, hierarchy) without content.
        
        This creates a "fingerprint" of the website's structure that can be
        compared to other sites via embeddings.
        
        Example:
        Input: <div class="post"><h3 class="title">Hello</h3><p>World</p></div>
        Output: <div class="post"><h3 class="title"></h3><p></p></div>
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove all text content, keep only structure
            for element in soup.find_all(text=True):
                element.extract()
            
            # Remove script and style tags
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            
            structure_html = str(soup)
            
            # Truncate to max embedding size
            if len(structure_html) > max_length:
                structure_html = structure_html[:max_length]
            
            return structure_html
        except Exception as e:
            logger.debug(f"Error extracting structure: {e}")
            return html[:max_length]  # Fallback to raw HTML
    
    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI's embedding model.
        
        Cost: ~$0.00002 per call (vs $0.001 for LLM completion)
        """
        try:
            response = litellm.embedding(
                model=self.embedding_model,
                input=[text],
                api_key=self.api_key
            )
            return response.data[0]['embedding']
        except Exception as e:
            logger.warning(f"     Embedding failed: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def find_similar_selectors(
        self,
        html: str,
        domain: str,
        top_k: int = 3,
        min_similarity: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Find selector patterns from similar websites.
        
        Args:
            html: Page HTML
            domain: Domain being scraped (to avoid self-matching)
            top_k: Number of similar sites to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
        
        Returns:
            List of selector patterns from similar sites, sorted by similarity
        """
        if not self.embeddings_index:
            logger.debug("   No cached embeddings to search")
            return []
        
        logger.info(f" Searching embedding cache for similar sites...")
        
        # Extract structure and embed
        structure_html = self._extract_html_structure(html)
        query_embedding = self._embed(structure_html)
        
        if not query_embedding:
            logger.warning("     Failed to generate embedding")
            return []
        
        # Calculate similarities
        similarities = []
        for entry in self.embeddings_index:
            cached_domain = entry['domain']
            
            # Skip same domain (we're trying to find NEW patterns)
            if cached_domain == domain:
                continue
            
            similarity = self._cosine_similarity(query_embedding, entry['embedding'])
            
            if similarity >= min_similarity:
                similarities.append({
                    'domain': cached_domain,
                    'similarity': similarity,
                    'selectors': self.selectors_db.get(cached_domain, {}),
                    'success_rate': entry.get('success_rate', 0.9)
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = similarities[:top_k]
        
        if results:
            logger.info(f"    Found {len(results)} similar sites:")
            for i, r in enumerate(results, 1):
                logger.info(f"      {i}. {r['domain']} (similarity: {r['similarity']:.2f})")
        else:
            logger.info(f"     No similar sites found (min similarity: {min_similarity})")
        
        return results
    
    def store_success(
        self,
        html: str,
        domain: str,
        selectors: Dict[str, Any],
        quality: float
    ):
        """
        Store successful selector pattern for future use.
        
        Args:
            html: Page HTML (used to generate embedding)
            domain: Domain that was scraped
            selectors: Successful selector patterns
            quality: Extraction quality (0.0-1.0)
        """
        if quality < 0.8:
            logger.debug(f"   Skipping cache (quality {quality:.1%} < 80%)")
            return
        
        logger.info(f" Storing selector pattern for {domain} (quality: {quality:.1%})")
        
        # Extract structure and embed
        structure_html = self._extract_html_structure(html)
        embedding = self._embed(structure_html)
        
        if not embedding:
            logger.warning("     Failed to generate embedding, not caching")
            return
        
        # Check if domain already exists
        existing_idx = None
        for i, entry in enumerate(self.embeddings_index):
            if entry['domain'] == domain:
                existing_idx = i
                break
        
        # Create entry
        entry = {
            'domain': domain,
            'embedding': embedding,
            'success_rate': quality,
            'timestamp': str(Path(self.cache_dir / f"{domain}.json").stat().st_mtime if 
                            (self.cache_dir / f"{domain}.json").exists() else 0)
        }
        
        if existing_idx is not None:
            # Update existing
            self.embeddings_index[existing_idx] = entry
            logger.debug(f"   Updated existing cache entry for {domain}")
        else:
            # Add new
            self.embeddings_index.append(entry)
            logger.debug(f"   Added new cache entry for {domain}")
        
        # Store selectors
        self.selectors_db[domain] = {
            'container_selector': selectors.get('container_selector'),
            'field_selectors': selectors.get('field_selectors', {}),
            'extraction_strategy': selectors.get('extraction_strategy', 'nested_elements'),
            'quality': quality,
            'pattern_type': selectors.get('type', 'unknown')
        }
        
        # Save to disk
        self._save_cache()
        
        logger.info(f"    Cached {domain} - Total sites: {len(self.selectors_db)}")
    
    def get_selector_for_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get cached selectors for a specific domain (exact match).
        
        This is faster than similarity search if we've scraped this exact domain before.
        """
        return self.selectors_db.get(domain)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_sites': len(self.selectors_db),
            'total_embeddings': len(self.embeddings_index),
            'cache_dir': str(self.cache_dir),
            'embedding_model': self.embedding_model
        }






