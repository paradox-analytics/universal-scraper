"""
DOM Digest Generator - Layer 2 Cache
Generates stable fingerprints of page structure for fast template matching

This enables "same layout as before" detection without LLM calls.
Key optimization: Detect layout changes quickly (<10ms) vs LLM analysis (~2s).
"""

import hashlib
import logging
import re
from typing import Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DOMDigestGenerator:
    """
    Generates stable DOM fingerprints for cache matching
    
    Features:
    - Strips scripts/styles (content-agnostic)
    - Normalizes whitespace
    - Drops dynamic IDs/classes (or hashes them)
    - Creates "tag path histogram" or "DOM shape signature"
    - Fast: <10ms vs LLM analysis (~2s)
    """
    
    def __init__(self):
        """Initialize DOM digest generator"""
        self.hash_algorithm = 'sha256'
    
    def generate_digest(
        self,
        html: str,
        url: Optional[str] = None,
        normalize_dynamic: bool = True
    ) -> Dict[str, Any]:
        """
        Generate DOM digest (fingerprint) from HTML
        
        Args:
            html: HTML content (can be cleaned or raw)
            url: Optional URL for domain extraction
            normalize_dynamic: If True, normalize dynamic IDs/classes
            
        Returns:
            Dict with:
            - digest: Hash string (primary key)
            - fingerprint: Human-readable signature
            - features: Extracted features
            - page_type: Inferred page type (if possible)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structural features
        features = self._extract_features(soup, normalize_dynamic)
        
        # Generate fingerprint (human-readable signature)
        fingerprint = self._create_fingerprint(features)
        
        # Generate digest (hash)
        digest = self._hash_fingerprint(fingerprint)
        
        # Infer page type (heuristic, no LLM)
        page_type = self._infer_page_type(soup, url)
        
        logger.debug(f"   DOM digest: {digest[:16]}... (type: {page_type})")
        
        return {
            'digest': digest,
            'fingerprint': fingerprint,
            'features': features,
            'page_type': page_type,
            'url': url
        }
    
    def _extract_features(
        self,
        soup: BeautifulSoup,
        normalize_dynamic: bool = True
    ) -> Dict[str, Any]:
        """Extract structural features from DOM"""
        features = {}
        
        # 1. Tag path histogram (most important - structure shape)
        features['tag_paths'] = self._get_tag_path_histogram(soup)
        
        # 2. Normalized class patterns (structure indicators)
        features['class_patterns'] = self._get_normalized_class_patterns(
            soup, normalize_dynamic
        )
        
        # 3. Structural depth distribution
        features['depth_distribution'] = self._get_depth_distribution(soup)
        
        # 4. Container patterns (article, section, div, etc.)
        features['containers'] = self._get_container_patterns(soup)
        
        # 5. Semantic elements (header, nav, main, footer, article)
        features['semantic_elements'] = self._get_semantic_elements(soup)
        
        # 6. List patterns (ul, ol - often indicate repeating items)
        features['list_patterns'] = self._get_list_patterns(soup)
        
        return features
    
    def _get_tag_path_histogram(self, soup: BeautifulSoup, max_depth: int = 4) -> str:
        """
        Create histogram of tag paths (e.g., "div>div>article>h2")
        
        This captures the "shape" of the DOM structure.
        """
        paths = []
        
        def traverse(element, path: list = [], level: int = 0):
            if level >= max_depth:
                return
            
            if element.name:
                current_path = path + [element.name]
                path_str = '>'.join(current_path)
                paths.append(path_str)
                
                # Traverse children
                for child in element.find_all(recursive=False):
                    if child.name:
                        traverse(child, current_path, level + 1)
        
        # Start from body or root
        body = soup.find('body')
        if body:
            traverse(body)
        else:
            traverse(soup)
        
        # Create histogram (count occurrences)
        path_counter = Counter(paths)
        
        # Get top 30 most common paths (structure indicators)
        top_paths = path_counter.most_common(30)
        path_str = ','.join([f"{path}={count}" for path, count in top_paths])
        
        return path_str
    
    def _get_normalized_class_patterns(
        self,
        soup: BeautifulSoup,
        normalize_dynamic: bool = True
    ) -> str:
        """
        Extract class patterns, normalizing dynamic parts
        
        Examples:
        - "product-123" -> "product-N"
        - "item-abc-xyz" -> "item-N-N"
        """
        classes = []
        
        for tag in soup.find_all(True):
            tag_classes = tag.get('class', [])
            if tag_classes:
                for cls in tag_classes:
                    if normalize_dynamic:
                        # Normalize: replace numbers/IDs with N
                        normalized = re.sub(r'\d+', 'N', cls)
                        normalized = re.sub(r'[a-f0-9]{8,}', 'HASH', normalized)  # Hash-like
                        classes.append(normalized)
                    else:
                        classes.append(cls)
        
        # Get most common patterns
        class_counter = Counter(classes)
        top_classes = class_counter.most_common(20)
        
        # Create pattern string
        pattern = ','.join([f"{cls}={count}" for cls, count in top_classes])
        
        return pattern
    
    def _get_depth_distribution(self, soup: BeautifulSoup) -> str:
        """Get distribution of element depths"""
        depths = []
        
        def get_depth(element, current_depth: int = 0):
            if not element.find_all(recursive=False):
                depths.append(current_depth)
                return
            
            for child in element.find_all(recursive=False):
                if child.name:
                    get_depth(child, current_depth + 1)
        
        body = soup.find('body')
        if body:
            get_depth(body)
        else:
            get_depth(soup)
        
        # Create distribution (buckets)
        depth_counter = Counter(depths)
        max_depth = max(depths) if depths else 0
        
        # Create distribution string
        dist_str = ','.join([
            f"d{i}={depth_counter.get(i, 0)}"
            for i in range(min(max_depth + 1, 10))
        ])
        
        return dist_str
    
    def _get_container_patterns(self, soup: BeautifulSoup) -> str:
        """Detect container patterns (article, section, div, etc.)"""
        container_tags = ['article', 'section', 'div', 'main', 'ul', 'ol', 'table']
        
        patterns = []
        for tag_name in container_tags:
            containers = soup.find_all(tag_name)
            if containers:
                # Get average children count (structure indicator)
                children_counts = [
                    len(list(container.find_all(recursive=False)))
                    for container in containers[:10]  # Sample first 10
                ]
                avg_children = sum(children_counts) / len(children_counts) if children_counts else 0
                patterns.append(f"{tag_name}={len(containers)}:avg={avg_children:.1f}")
        
        return ','.join(patterns)
    
    def _get_semantic_elements(self, soup: BeautifulSoup) -> str:
        """Detect semantic HTML5 elements"""
        semantic_tags = ['header', 'nav', 'main', 'footer', 'article', 'aside', 'section']
        
        elements = []
        for tag_name in semantic_tags:
            count = len(soup.find_all(tag_name))
            if count > 0:
                elements.append(f"{tag_name}={count}")
        
        return ','.join(elements)
    
    def _get_list_patterns(self, soup: BeautifulSoup) -> str:
        """Detect list patterns (often indicate repeating items)"""
        lists = soup.find_all(['ul', 'ol'])
        
        patterns = []
        for list_elem in lists[:10]:  # Sample first 10
            items = list_elem.find_all('li', recursive=False)
            if items:
                patterns.append(f"li={len(items)}")
        
        if patterns:
            # Get most common pattern
            pattern_counter = Counter(patterns)
            top_pattern = pattern_counter.most_common(1)[0]
            return f"{top_pattern[0]}:count={top_pattern[1]}"
        
        return ""
    
    def _create_fingerprint(self, features: Dict[str, Any]) -> str:
        """Create human-readable fingerprint from features"""
        parts = []
        
        parts.append(f"paths:{features.get('tag_paths', '')[:200]}")  # Truncate
        parts.append(f"classes:{features.get('class_patterns', '')[:200]}")
        parts.append(f"depth:{features.get('depth_distribution', '')}")
        parts.append(f"containers:{features.get('containers', '')}")
        parts.append(f"semantic:{features.get('semantic_elements', '')}")
        parts.append(f"lists:{features.get('list_patterns', '')}")
        
        return '|'.join(parts)
    
    def _hash_fingerprint(self, fingerprint: str) -> str:
        """Hash fingerprint to create digest"""
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(fingerprint.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _infer_page_type(
        self,
        soup: BeautifulSoup,
        url: Optional[str] = None
    ) -> str:
        """
        Infer page type heuristically (no LLM)
        
        Returns:
            Page type: 'listing', 'detail', 'search', 'unknown'
        """
        # Check for common listing patterns
        if soup.find_all(['ul', 'ol']):
            # Check if lists contain many items (likely listing page)
            lists = soup.find_all(['ul', 'ol'])
            total_items = sum(len(list_elem.find_all('li')) for list_elem in lists)
            if total_items > 5:
                return 'listing'
        
        # Check for article/detail patterns
        if soup.find_all('article') or soup.find_all('main'):
            articles = soup.find_all('article')
            if len(articles) == 1:  # Single article = detail page
                return 'detail'
            elif len(articles) > 1:  # Multiple articles = listing
                return 'listing'
        
        # Check for search patterns
        if soup.find_all('form') and soup.find_all('input', {'type': 'search'}):
            return 'search'
        
        # Check URL patterns
        if url:
            url_lower = url.lower()
            if any(keyword in url_lower for keyword in ['/product/', '/item/', '/detail/']):
                return 'detail'
            elif any(keyword in url_lower for keyword in ['/search', '/results', '/list']):
                return 'listing'
        
        return 'unknown'
    
    def compare_digests(
        self,
        digest1: str,
        digest2: str,
        threshold: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Compare two digests (exact match by default)
        
        Args:
            digest1: First digest
            digest2: Second digest
            threshold: Similarity threshold (1.0 = exact match)
            
        Returns:
            (is_match, similarity_score)
        """
        if digest1 == digest2:
            return (True, 1.0)
        
        # For future: could implement fuzzy matching using fingerprint similarity
        # For now: exact match only
        return (False, 0.0)
    
    def get_cache_key(
        self,
        url: str,
        digest: str
    ) -> str:
        """
        Generate cache key for DOM digest cache
        
        Format: dom_digest_{domain}_{digest}
        """
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '').replace('.', '_')
        
        return f"dom_digest_{domain}_{digest[:16]}"



