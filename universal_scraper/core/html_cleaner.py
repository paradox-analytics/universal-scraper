"""
Smart HTML Cleaner - Reduces HTML size by ~98% while preserving structure
Removes unnecessary elements and detects repeating patterns
"""

import re
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup, Comment
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class SmartHTMLCleaner:
    """Intelligently cleans HTML while preserving structure"""
    
    # Tags to remove completely
    REMOVE_TAGS = [
        'script', 'style', 'noscript', 'iframe', 'embed', 'object',
        'link', 'meta'  # Keep meta for now, might contain useful data
    ]
    
    # Tags that are typically navigation/non-content
    NAVIGATION_TAGS = [
        'nav', 'header', 'footer', 'aside', 'menu'
    ]
    
    # Attributes to keep (minimal set for structure)
    KEEP_ATTRIBUTES = [
        'id', 'class', 'href', 'src', 'alt', 'title',
        'data-*'  # Keep data attributes, might be useful
    ]
    
    # Classes/IDs that indicate ads/tracking
    AD_PATTERNS = [
        'ad', 'ads', 'advertisement', 'banner', 'sponsor',
        'tracking', 'analytics', 'cookie', 'gdpr', 'consent',
        'popup', 'modal', 'overlay'
    ]
    
    def __init__(self, keep_samples: int = 2):
        """
        Initialize HTML Cleaner
        
        Args:
            keep_samples: Number of repeating elements to keep for pattern detection
        """
        self.keep_samples = keep_samples
        self.original_size = 0
        self.cleaned_size = 0
    
    def clean(self, html: str) -> Dict[str, Any]:
        """
        Clean HTML and return cleaned version with metadata
        
        Args:
            html: Raw HTML content
            
        Returns:
            Dict with 'html', 'original_size', 'cleaned_size', 'reduction_percent' keys
        """
        self.original_size = len(html)
        
        logger.info(f"🧹 Cleaning HTML ({self.original_size:,} bytes)")
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Step 1: Remove scripts, styles, etc.
        self._remove_unwanted_tags(soup)
        
        # Step 2: Remove ads and analytics
        self._remove_ads_and_tracking(soup)
        
        # Step 3: Remove inline SVGs (can be huge)
        self._remove_inline_svgs(soup)
        
        # Step 4: Replace URLs with placeholders
        self._replace_urls_with_placeholders(soup)
        
        # Step 5: Remove non-essential attributes
        self._remove_non_essential_attributes(soup)
        
        # Step 6: Remove navigation elements
        self._remove_navigation(soup)
        
        # Step 7: Detect and sample repeating structures
        self._sample_repeating_structures(soup)
        
        # Step 8: Remove empty elements
        self._remove_empty_elements(soup)
        
        # Step 9: Remove comments
        self._remove_comments(soup)
        
        # Get cleaned HTML
        cleaned_html = str(soup)
        self.cleaned_size = len(cleaned_html)
        
        reduction_percent = ((self.original_size - self.cleaned_size) / self.original_size * 100) if self.original_size > 0 else 0
        
        logger.info(f"✅ Cleaned: {self.cleaned_size:,} bytes ({reduction_percent:.1f}% reduction)")
        
        return {
            'html': cleaned_html,
            'original_size': self.original_size,
            'cleaned_size': self.cleaned_size,
            'reduction_percent': reduction_percent
        }
    
    def _remove_unwanted_tags(self, soup: BeautifulSoup) -> None:
        """Remove script, style, and other unwanted tags"""
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        logger.debug(f"   Removed {len(self.REMOVE_TAGS)} tag types")
    
    def _remove_ads_and_tracking(self, soup: BeautifulSoup) -> None:
        """Remove elements that look like ads or tracking"""
        removed_count = 0
        
        for tag in soup.find_all(True):  # Find all tags
            # Check class and id for ad patterns
            classes = tag.get('class', [])
            id_attr = tag.get('id', '')
            
            # Convert to strings for checking
            class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
            combined = (class_str + ' ' + id_attr).lower()
            
            # Check if matches ad patterns
            if any(pattern in combined for pattern in self.AD_PATTERNS):
                tag.decompose()
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"   Removed {removed_count} ad/tracking elements")
    
    def _remove_inline_svgs(self, soup: BeautifulSoup) -> None:
        """Remove inline SVG images (can be very large)"""
        svg_count = 0
        for svg in soup.find_all('svg'):
            svg.decompose()
            svg_count += 1
        
        if svg_count > 0:
            logger.debug(f"   Removed {svg_count} inline SVGs")
    
    def _replace_urls_with_placeholders(self, soup: BeautifulSoup) -> None:
        """Replace long URLs with placeholders to reduce size"""
        url_pattern = re.compile(r'https?://[^\s<>"\']+')
        
        # Replace in text content
        for tag in soup.find_all(text=True):
            if url_pattern.search(str(tag)):
                new_text = url_pattern.sub('[URL]', str(tag))
                tag.replace_with(new_text)
        
        # Shorten src/href attributes
        for tag in soup.find_all(['img', 'a', 'link']):
            if tag.get('src'):
                # Keep filename but remove long paths
                src = tag['src']
                if len(src) > 50:
                    tag['src'] = '[SRC]'
            if tag.get('href'):
                href = tag['href']
                if len(href) > 50 and not href.startswith('#'):
                    tag['href'] = '[HREF]'
        
        logger.debug("   Replaced URLs with placeholders")
    
    def _remove_non_essential_attributes(self, soup: BeautifulSoup) -> None:
        """Remove non-essential HTML attributes"""
        for tag in soup.find_all(True):
            # Get all attributes
            attrs = dict(tag.attrs)
            
            # Keep only essential attributes
            for attr in list(attrs.keys()):
                should_keep = False
                
                # Check if in keep list
                if attr in self.KEEP_ATTRIBUTES:
                    should_keep = True
                
                # Keep data-* attributes
                if attr.startswith('data-'):
                    should_keep = True
                
                # Remove if not needed
                if not should_keep:
                    del tag.attrs[attr]
        
        logger.debug("   Removed non-essential attributes")
    
    def _remove_navigation(self, soup: BeautifulSoup) -> None:
        """Remove navigation elements"""
        removed_count = 0
        for tag_name in self.NAVIGATION_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"   Removed {removed_count} navigation elements")
    
    def _sample_repeating_structures(self, soup: BeautifulSoup) -> None:
        """Detect repeating structures and keep only samples"""
        # Find potential list containers
        list_containers = soup.find_all(['ul', 'ol', 'div', 'section', 'article'])
        
        removed_count = 0
        for container in list_containers:
            # Get direct children with same tag name
            children = list(container.find_all(recursive=False))
            
            if len(children) <= self.keep_samples:
                continue
            
            # Group by tag name and structure
            structure_groups = defaultdict(list)
            for child in children:
                # Create structure signature
                signature = self._get_structure_signature(child)
                structure_groups[signature].append(child)
            
            # For each group, keep only samples
            for signature, group in structure_groups.items():
                if len(group) > self.keep_samples:
                    # Keep first N samples, remove rest
                    for item in group[self.keep_samples:]:
                        item.decompose()
                        removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"   Sampled repeating structures (removed {removed_count} duplicates)")
    
    def _get_structure_signature(self, tag) -> str:
        """Get structural signature of a tag for comparison"""
        if not tag.name:
            return ''
        
        # Create signature from tag name and immediate children
        children_tags = [child.name for child in tag.find_all(recursive=False) if child.name]
        
        signature = f"{tag.name}:{','.join(sorted(children_tags))}"
        
        # Add class signature if present
        classes = tag.get('class', [])
        if classes:
            signature += f":{','.join(sorted(classes))}"
        
        return signature
    
    def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove empty divs and containers"""
        removed_count = 0
        
        # Multiple passes to handle nested empty elements
        for _ in range(3):
            for tag in soup.find_all(['div', 'span', 'p', 'section', 'article']):
                # Check if empty (no text, no children with content)
                if not tag.get_text(strip=True) and not tag.find_all(['img', 'video', 'audio']):
                    tag.decompose()
                    removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"   Removed {removed_count} empty elements")
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments"""
        comments = soup.find_all(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        if len(comments) > 0:
            logger.debug(f"   Removed {len(comments)} comments")
    
    def get_reduction_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return {
            'original_size': self.original_size,
            'cleaned_size': self.cleaned_size,
            'reduction_percent': ((self.original_size - self.cleaned_size) / self.original_size * 100) if self.original_size > 0 else 0,
            'reduction_ratio': f"{self.original_size / self.cleaned_size:.1f}x" if self.cleaned_size > 0 else "N/A"
        }

