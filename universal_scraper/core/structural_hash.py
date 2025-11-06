"""
Structural Hash Generator
Creates fingerprint of page structure for intelligent caching
"""

import hashlib
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup
from collections import Counter

logger = logging.getLogger(__name__)


class StructuralHashGenerator:
    """Generates structural fingerprint of HTML for cache matching"""
    
    def __init__(self):
        self.hash_algorithm = 'sha256'
    
    def generate_hash(self, html: str) -> Dict[str, Any]:
        """
        Generate structural hash from HTML
        
        Args:
            html: Cleaned HTML content
            
        Returns:
            Dict with 'hash', 'signature', 'metrics' keys
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structural features
        structure_signature = self._extract_structure_signature(soup)
        
        # Generate hash
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(structure_signature.encode('utf-8'))
        structure_hash = hash_obj.hexdigest()
        
        logger.info(f"🔑 Generated hash: {structure_hash[:16]}...")
        
        return {
            'hash': structure_hash,
            'signature': structure_signature,
            'metrics': self._calculate_metrics(soup)
        }
    
    def _extract_structure_signature(self, soup: BeautifulSoup) -> str:
        """
        Extract structural signature (tag hierarchy and patterns)
        
        This creates a signature that is:
        - Similar for pages with same structure
        - Different for pages with different structure
        - Ignores content, focuses on structure
        """
        features = []
        
        # 1. Tag hierarchy (depth and patterns)
        tag_hierarchy = self._get_tag_hierarchy(soup)
        features.append(f"hierarchy:{tag_hierarchy}")
        
        # 2. Tag frequency (normalized)
        tag_counts = self._get_tag_counts(soup)
        features.append(f"tags:{tag_counts}")
        
        # 3. Class patterns (structure indicators)
        class_patterns = self._get_class_patterns(soup)
        features.append(f"classes:{class_patterns}")
        
        # 4. ID patterns
        id_patterns = self._get_id_patterns(soup)
        features.append(f"ids:{id_patterns}")
        
        # 5. Structural depth
        max_depth = self._get_max_depth(soup)
        features.append(f"depth:{max_depth}")
        
        # 6. Content container patterns
        container_patterns = self._get_container_patterns(soup)
        features.append(f"containers:{container_patterns}")
        
        signature = '|'.join(features)
        
        logger.debug(f"   Signature length: {len(signature)} chars")
        
        return signature
    
    def _get_tag_hierarchy(self, soup: BeautifulSoup, max_levels: int = 5) -> str:
        """Get tag hierarchy representation"""
        def traverse(element, level=0):
            if level >= max_levels:
                return []
            
            hierarchy = []
            for child in element.find_all(recursive=False):
                if child.name:
                    hierarchy.append(f"{level}:{child.name}")
                    hierarchy.extend(traverse(child, level + 1))
            
            return hierarchy
        
        body = soup.find('body')
        if not body:
            body = soup
        
        hierarchy = traverse(body)
        
        # Count occurrences and create pattern
        hierarchy_counter = Counter(hierarchy)
        hierarchy_str = ','.join([f"{k}={v}" for k, v in sorted(hierarchy_counter.most_common(20))])
        
        return hierarchy_str
    
    def _get_tag_counts(self, soup: BeautifulSoup) -> str:
        """Get normalized tag counts"""
        tags = [tag.name for tag in soup.find_all(True)]
        tag_counter = Counter(tags)
        
        # Get top 15 most common tags
        top_tags = tag_counter.most_common(15)
        tag_str = ','.join([f"{k}={v}" for k, v in top_tags])
        
        return tag_str
    
    def _get_class_patterns(self, soup: BeautifulSoup) -> str:
        """Get class name patterns (structure indicators)"""
        classes = []
        for tag in soup.find_all(True):
            tag_classes = tag.get('class', [])
            if tag_classes:
                classes.extend(tag_classes)
        
        # Get most common classes
        class_counter = Counter(classes)
        top_classes = class_counter.most_common(15)
        
        # Create pattern (just counts, not actual names for privacy)
        pattern = ','.join([str(v) for _, v in top_classes])
        
        return pattern
    
    def _get_id_patterns(self, soup: BeautifulSoup) -> str:
        """Get ID patterns"""
        ids = []
        for tag in soup.find_all(True):
            tag_id = tag.get('id')
            if tag_id:
                # Extract pattern (e.g., "product-123" -> "product-N")
                pattern = ''.join([c if not c.isdigit() else 'N' for c in tag_id])
                ids.append(pattern)
        
        id_counter = Counter(ids)
        top_ids = id_counter.most_common(10)
        
        pattern = ','.join([f"{k}={v}" for k, v in top_ids])
        
        return pattern
    
    def _get_max_depth(self, soup: BeautifulSoup) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(element):
            if not element.find_all(recursive=False):
                return 1
            
            max_child_depth = 0
            for child in element.find_all(recursive=False):
                if child.name:
                    max_child_depth = max(max_child_depth, get_depth(child))
            
            return 1 + max_child_depth
        
        body = soup.find('body')
        if not body:
            body = soup
        
        return get_depth(body)
    
    def _get_container_patterns(self, soup: BeautifulSoup) -> str:
        """Detect common content container patterns"""
        container_tags = ['article', 'section', 'div', 'main', 'ul', 'ol']
        
        patterns = []
        for tag_name in container_tags:
            containers = soup.find_all(tag_name)
            if containers:
                patterns.append(f"{tag_name}={len(containers)}")
        
        return ','.join(patterns)
    
    def _calculate_metrics(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Calculate structural metrics"""
        return {
            'total_tags': len(soup.find_all(True)),
            'unique_tags': len(set(tag.name for tag in soup.find_all(True))),
            'max_depth': self._get_max_depth(soup),
            'links': len(soup.find_all('a')),
            'images': len(soup.find_all('img')),
            'forms': len(soup.find_all('form')),
            'lists': len(soup.find_all(['ul', 'ol'])),
        }
    
    def compare_hashes(self, hash1: str, hash2: str) -> bool:
        """
        Compare two structural hashes
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            True if hashes match (same structure)
        """
        return hash1 == hash2
    
    def get_hash_from_url(self, url: str, html: str) -> str:
        """
        Convenience method to get hash from URL and HTML
        
        Args:
            url: Source URL
            html: HTML content
            
        Returns:
            Structural hash string
        """
        result = self.generate_hash(html)
        return result['hash']

