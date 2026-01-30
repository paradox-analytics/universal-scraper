"""
Heuristic Prefilter - Reduce LLM Context Size
Prefilters candidate nodes before sending to LLM

Purpose: Reduce LLM context size by 50-80% through intelligent prefiltering.
This makes LLM calls faster and cheaper while maintaining extraction quality.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup, Tag, NavigableString

logger = logging.getLogger(__name__)


class HeuristicPrefilter:
    """
    Prefilters HTML to find candidate nodes for each field
    
    Strategies:
    1. Price nodes contain currency symbols
    2. Titles are largest headings near top
    3. Availability has known phrases
    4. Images have src attributes
    5. URLs are in href attributes
    """
    
    # Common currency symbols
    CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹', '₽', '₩', '₪', '₦', '₨', '₫', '₭', '₮', '₯', '₰', '₱', '₲', '₳', '₴', '₵', '₶', '₷', '₸', '₹', '₺', '₻', '₼', '₽', '₾', '₿']
    
    # Common availability phrases
    AVAILABILITY_PHRASES = [
        'in stock', 'available', 'out of stock', 'sold out', 'pre-order',
        'backorder', 'discontinued', 'limited', 'on sale', 'sale',
        'buy now', 'add to cart', 'add to bag', 'purchase'
    ]
    
    # Common price patterns
    PRICE_PATTERNS = [
        r'\$\d+\.?\d*',  # $10.99
        r'€\d+\.?\d*',   # €10.99
        r'£\d+\.?\d*',   # £10.99
        r'\d+\.?\d*\s*(USD|EUR|GBP|JPY|CNY)',  # 10.99 USD
    ]
    
    def __init__(self):
        """Initialize heuristic prefilter"""
        self.max_candidates_per_field = 20  # Limit candidates to reduce context
    
    def filter_candidates(
        self,
        html: str,
        fields: List[str],
        max_candidates: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter candidate nodes for each field
        
        Args:
            html: HTML content
            fields: List of fields to extract
            max_candidates: Max candidates per field (default: 20)
            
        Returns:
            Dict mapping field name to list of candidate nodes:
            {
                'field_name': [
                    {
                        'element': Tag,
                        'selector': str,
                        'text': str,
                        'confidence': float,
                        'reason': str
                    },
                    ...
                ]
            }
        """
        if max_candidates is None:
            max_candidates = self.max_candidates_per_field
        
        soup = BeautifulSoup(html, 'html.parser')
        candidates = {}
        
        for field in fields:
            field_candidates = self._find_candidates_for_field(
                soup, field, max_candidates
            )
            candidates[field] = field_candidates
        
        # Log summary
        total_candidates = sum(len(c) for c in candidates.values())
        logger.debug(f"   Heuristic prefilter: {total_candidates} candidates for {len(fields)} fields")
        
        return candidates
    
    def _find_candidates_for_field(
        self,
        soup: BeautifulSoup,
        field: str,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find candidate nodes for a specific field"""
        field_lower = field.lower()
        candidates = []
        
        # Field-specific strategies
        if any(keyword in field_lower for keyword in ['price', 'cost', 'amount', 'fee']):
            candidates.extend(self._find_price_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['title', 'name', 'heading', 'headline']):
            candidates.extend(self._find_title_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['availability', 'stock', 'in_stock', 'available']):
            candidates.extend(self._find_availability_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['image', 'img', 'photo', 'picture', 'thumbnail']):
            candidates.extend(self._find_image_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['url', 'link', 'href', 'permalink']):
            candidates.extend(self._find_url_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['description', 'desc', 'summary', 'details']):
            candidates.extend(self._find_description_candidates(soup, max_candidates))
        
        elif any(keyword in field_lower for keyword in ['rating', 'score', 'review', 'stars']):
            candidates.extend(self._find_rating_candidates(soup, max_candidates))
        
        else:
            # Generic strategy: look for semantic elements
            candidates.extend(self._find_generic_candidates(soup, field, max_candidates))
        
        # Sort by confidence and limit
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates[:max_candidates]
    
    def _find_price_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find price candidates (contain currency symbols or price patterns)"""
        candidates = []
        
        # Find all text nodes and elements
        for element in soup.find_all(True):
            text = element.get_text(strip=True)
            
            if not text:
                continue
            
            # Check for currency symbols
            has_currency = any(symbol in text for symbol in self.CURRENCY_SYMBOLS)
            
            # Check for price patterns
            has_price_pattern = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in self.PRICE_PATTERNS
            )
            
            if has_currency or has_price_pattern:
                # Extract numeric value
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    confidence = 0.8 if has_currency else 0.6
                    candidates.append({
                        'element': element,
                        'selector': self._generate_selector(element),
                        'text': text[:200],  # Truncate
                        'confidence': confidence,
                        'reason': 'contains currency symbol' if has_currency else 'matches price pattern',
                        'value': numbers[0]  # First number found
                    })
        
        return candidates
    
    def _find_title_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find title candidates (headings, large text near top)"""
        candidates = []
        
        # Strategy 1: Headings (h1-h6)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings[:max_candidates]:
            text = heading.get_text(strip=True)
            if text and len(text) > 5:  # Minimum length
                # Higher confidence for h1, lower for h6
                level = int(heading.name[1])
                confidence = 1.0 - (level - 1) * 0.1
                
                candidates.append({
                    'element': heading,
                    'selector': self._generate_selector(heading),
                    'text': text[:200],
                    'confidence': confidence,
                    'reason': f'{heading.name} heading'
                })
        
        # Strategy 2: Large text near top (in first 1000 chars)
        body = soup.find('body')
        if body:
            # Find elements with significant text in first part of body
            body_text = str(body)[:1000]
            body_soup = BeautifulSoup(body_text, 'html.parser')
            
            for element in body_soup.find_all(['p', 'div', 'span', 'a']):
                text = element.get_text(strip=True)
                if text and 10 < len(text) < 200:  # Reasonable title length
                    candidates.append({
                        'element': element,
                        'selector': self._generate_selector(element),
                        'text': text[:200],
                        'confidence': 0.5,
                        'reason': 'text near top'
                    })
        
        return candidates
    
    def _find_availability_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find availability candidates (contain availability phrases)"""
        candidates = []
        
        for element in soup.find_all(True):
            text = element.get_text(strip=True).lower()
            
            if not text:
                continue
            
            # Check for availability phrases
            matches = [
                phrase for phrase in self.AVAILABILITY_PHRASES
                if phrase in text
            ]
            
            if matches:
                confidence = 0.8 if len(matches) > 1 else 0.6
                candidates.append({
                    'element': element,
                    'selector': self._generate_selector(element),
                    'text': text[:200],
                    'confidence': confidence,
                    'reason': f'matches availability phrases: {", ".join(matches[:2])}'
                })
        
        return candidates
    
    def _find_image_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find image candidates (img tags with src)"""
        candidates = []
        
        images = soup.find_all('img')
        for img in images[:max_candidates]:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            
            if src:
                # Filter out tracking pixels and icons
                if any(skip in src.lower() for skip in ['pixel', 'tracking', 'icon', 'logo', 'spacer']):
                    continue
                
                # Higher confidence for larger images (check dimensions)
                width = img.get('width')
                height = img.get('height')
                confidence = 0.7
                
                if width and height:
                    try:
                        w, h = int(width), int(height)
                        if w > 100 and h > 100:  # Reasonable size
                            confidence = 0.9
                    except ValueError:
                        pass
                
                candidates.append({
                    'element': img,
                    'selector': self._generate_selector(img),
                    'text': src[:200],
                    'confidence': confidence,
                    'reason': 'img tag with src',
                    'src': src
                })
        
        return candidates
    
    def _find_url_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find URL candidates (a tags with href)"""
        candidates = []
        
        links = soup.find_all('a', href=True)
        for link in links[:max_candidates]:
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and href.startswith(('http', '/', '#')):
                # Filter out common non-content links
                if any(skip in href.lower() for skip in ['javascript:', 'mailto:', '#top', '#bottom']):
                    continue
                
                confidence = 0.8 if text else 0.6
                candidates.append({
                    'element': link,
                    'selector': self._generate_selector(link),
                    'text': text[:200] if text else href[:200],
                    'confidence': confidence,
                    'reason': 'a tag with href',
                    'href': href
                })
        
        return candidates
    
    def _find_description_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find description candidates (longer text blocks)"""
        candidates = []
        
        # Look for paragraphs and divs with substantial text
        for element in soup.find_all(['p', 'div', 'span', 'article', 'section']):
            text = element.get_text(strip=True)
            
            if text and 50 < len(text) < 1000:  # Reasonable description length
                candidates.append({
                    'element': element,
                    'selector': self._generate_selector(element),
                    'text': text[:500],  # Truncate for context
                    'confidence': 0.6,
                    'reason': 'longer text block'
                })
        
        return candidates
    
    def _find_rating_candidates(
        self,
        soup: BeautifulSoup,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find rating candidates (numbers 0-5 or 0-10, stars)"""
        candidates = []
        
        # Look for star elements or numeric ratings
        stars = soup.find_all(['span', 'div', 'i'], class_=re.compile(r'star|rating', re.I))
        for star in stars[:max_candidates]:
            text = star.get_text(strip=True)
            if text:
                candidates.append({
                    'element': star,
                    'selector': self._generate_selector(star),
                    'text': text[:200],
                    'confidence': 0.8,
                    'reason': 'star/rating element'
                })
        
        # Look for numeric ratings (0-5 or 0-10)
        for element in soup.find_all(True):
            text = element.get_text(strip=True)
            if text:
                # Check for rating patterns
                rating_match = re.search(r'(\d+\.?\d*)\s*(out of|/|stars?|rating)', text, re.I)
                if rating_match:
                    rating = float(rating_match.group(1))
                    if 0 <= rating <= 10:
                        candidates.append({
                            'element': element,
                            'selector': self._generate_selector(element),
                            'text': text[:200],
                            'confidence': 0.7,
                            'reason': 'numeric rating',
                            'value': rating
                        })
        
        return candidates
    
    def _find_generic_candidates(
        self,
        soup: BeautifulSoup,
        field: str,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find generic candidates (semantic elements, data attributes)"""
        candidates = []
        
        # Look for data attributes matching field name
        field_normalized = field.lower().replace('_', '-')
        
        for element in soup.find_all(True):
            # Check data attributes
            for attr_name, attr_value in element.attrs.items():
                if field_normalized in attr_name.lower() or field.lower() in attr_name.lower():
                    text = element.get_text(strip=True) or str(attr_value)
                    if text:
                        candidates.append({
                            'element': element,
                            'selector': self._generate_selector(element),
                            'text': text[:200],
                            'confidence': 0.7,
                            'reason': f'data attribute: {attr_name}'
                        })
                        break
        
        # Look for semantic elements
        semantic_tags = ['article', 'section', 'main', 'header', 'footer']
        for tag in semantic_tags:
            elements = soup.find_all(tag)
            for element in elements[:5]:  # Limit per tag
                text = element.get_text(strip=True)
                if text and len(text) > 10:
                    candidates.append({
                        'element': element,
                        'selector': self._generate_selector(element),
                        'text': text[:200],
                        'confidence': 0.5,
                        'reason': f'semantic element: {tag}'
                    })
        
        return candidates
    
    def _generate_selector(self, element: Tag) -> str:
        """Generate CSS selector for element"""
        # Simple selector generation (can be enhanced)
        parts = []
        
        if element.name:
            parts.append(element.name)
        
        # Add ID if present
        if element.get('id'):
            parts.append(f"#{element.get('id')}")
        
        # Add class if present (first class only)
        classes = element.get('class', [])
        if classes:
            parts.append(f".{classes[0]}")
        
        return ' '.join(parts) if parts else element.name or 'unknown'
    
    def compress_html_with_candidates(
        self,
        html: str,
        fields: List[str],
        include_context: bool = True
    ) -> str:
        """
        Compress HTML by keeping only candidate nodes + minimal context
        
        Args:
            html: Original HTML
            fields: Fields to extract
            include_context: Include parent/neighbor context
            
        Returns:
            Compressed HTML string (50-80% smaller)
        """
        soup = BeautifulSoup(html, 'html.parser')
        candidates = self.filter_candidates(html, fields)
        
        # Collect all candidate elements
        candidate_elements = set()
        for field_candidates in candidates.values():
            for candidate in field_candidates:
                candidate_elements.add(candidate['element'])
        
        # Build compressed HTML
        compressed_parts = []
        
        for element in candidate_elements:
            # Include element + parent context if requested
            if include_context:
                # Include parent
                parent = element.parent
                if parent and parent.name:
                    compressed_parts.append(str(parent))
                else:
                    compressed_parts.append(str(element))
            else:
                compressed_parts.append(str(element))
        
        compressed_html = '\n'.join(compressed_parts)
        
        reduction = (1 - len(compressed_html) / len(html)) * 100
        logger.debug(f"   HTML compression: {len(html):,} → {len(compressed_html):,} bytes ({reduction:.1f}% reduction)")
        
        return compressed_html



