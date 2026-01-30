"""
Attribute Extractor - Direct extraction from HTML attributes
For modern sites using custom elements or data-* attributes
"""

import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class AttributeExtractor:
    """
    Extracts data directly from HTML attributes
    Fast, reliable, no AI needed for attribute-based sites
    """
    
    def extract(
        self,
        html: str,
        fields: List[str],
        pattern_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract data from HTML attributes
        
        Args:
            html: Raw HTML
            fields: Fields to extract (used for field name matching)
            pattern_details: Pattern details from PatternDetector
                - element_name: Name of element to find
                - key_attributes: List of attributes to extract
        
        Returns:
            List of extracted items
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        element_name = pattern_details.get('element_name')
        key_attributes = pattern_details.get('key_attributes', [])
        
        logger.info(f" Extracting from attributes: element={element_name}, attrs={key_attributes[:5]}")
        
        # Find all matching elements
        if element_name:
            # Specific element name
            elements = soup.find_all(element_name)
        else:
            # Find any custom elements
            elements = soup.find_all(lambda tag: '-' in tag.name)
        
        if not elements:
            logger.warning(" No elements found for attribute extraction")
            return []
        
        logger.info(f" Found {len(elements)} elements")
        
        items = []
        for elem in elements:
            item = {}
            
            # Strategy 1: If we have field names, try to map them to attributes
            if fields:
                item = self._extract_by_field_names(elem, fields)
            else:
                # Strategy 2: Extract all attributes
                item = self._extract_all_attributes(elem, key_attributes)
            
            # Only add if we got some data
            if item and any(v is not None for v in item.values()):
                items.append(item)
        
        logger.info(f" Extracted {len(items)} items from attributes")
        return items
    
    def _extract_by_field_names(
        self,
        elem: Any,
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        Extract by matching field names to attribute names
        
        Examples:
        - field='title' → try: 'post-title', 'data-title', 'title'
        - field='author' → try: 'author', 'data-author', 'user'
        """
        item = {}
        
        for field in fields:
            # Try multiple variations of the attribute name
            variations = self._generate_attribute_variations(field)
            
            value = None
            for attr_name in variations:
                value = elem.get(attr_name)
                if value:
                    break
            
            item[field] = value
        
        return item
    
    def _extract_all_attributes(
        self,
        elem: Any,
        key_attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Extract all interesting attributes from element
        """
        item = {}
        
        # Get all attributes
        for attr_name, attr_value in elem.attrs.items():
            # Skip internal/styling attributes
            if attr_name in ['class', 'style', 'id']:
                continue
            
            # Clean up attribute name for field name
            field_name = attr_name.replace('data-', '').replace('-', '_')
            item[field_name] = attr_value
        
        return item
    
    def _generate_attribute_variations(self, field: str) -> List[str]:
        """
        Generate possible attribute names for a field
        
        Examples:
            'title' → ['title', 'post-title', 'data-title', 'post_title']
            'upvotes' → ['upvotes', 'score', 'votes', 'data-score', 'data-upvotes']
        """
        variations = [field]  # Original field name
        
        # Common patterns
        variations.append(f'data-{field}')
        variations.append(f'post-{field}')
        variations.append(f'item-{field}')
        variations.append(field.replace('_', '-'))
        
        # Common field aliases
        aliases = {
            'title': ['post-title', 'item-title', 'name', 'heading'],
            'author': ['user', 'username', 'data-author', 'posted-by'],
            'upvotes': ['score', 'votes', 'points', 'data-score', 'rating'],
            'comments': ['comment-count', 'comments-count', 'num-comments', 'data-comments'],
            'comments_count': ['comment-count', 'comments-count', 'num-comments', 'data-comments'],
            'price': ['data-price', 'product-price', 'cost'],
            'url': ['href', 'link', 'permalink', 'data-url', 'content-href'],
            'link': ['href', 'permalink', 'url', 'data-url', 'content-href'],
        }
        
        if field in aliases:
            variations.extend(aliases[field])
        
        # Remove duplicates, preserve order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return unique_variations







