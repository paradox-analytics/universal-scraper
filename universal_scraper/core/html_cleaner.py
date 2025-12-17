"""
Smart HTML Cleaner - Aggressively removes noise while preserving data content
Phase 1 Enhancements: Remove UI elements, widgets, social, forms
Based on ScrapeGraphAI principles + optimization for complex dynamic sites
"""

import re
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)


class SmartHTMLCleaner:
    """
    Intelligently cleans HTML - aggressive noise removal for better LLM extraction
    
    Philosophy (inspired by ScrapeGraphAI + optimized for quality):
    - Remove true noise: scripts, styles, comments
    - Remove UI elements: forms, buttons, SVGs (Phase 1 NEW)
    - Remove widgets: social, newsletters, CTAs (Phase 1 NEW)
    - Remove structural noise: nav, header, footer, aside
    - Minify whitespace without losing content boundaries
    - Result: Fewer chunks, better LLM context, higher quality extraction
    """
    
    # Tags to remove completely (true noise + navigation + UI)
    REMOVE_TAGS = [
        # Core noise
        'script',    # JavaScript code
        'style',     # CSS styles
        'noscript',  # Noscript fallbacks
        'iframe',    # Embedded frames
        'embed',     # Embedded objects
        'object',    # Object embeds
        
        # Structural noise
        'nav',       # Navigation menus
        'header',    # Page headers
        'footer',    # Page footers
        'aside',     # Sidebar content
        
        # UI elements (Phase 1 additions)
        'svg',       # Icons and graphics (not data-bearing)
        'form',      # Forms (search, login) - rarely contain list data
        'button',    # Buttons - pure UI elements
        'select',    # Dropdown menus - UI controls
        'input',     # Input fields - UI controls
        'textarea',  # Text inputs - UI controls
        'label',     # Form labels - UI text
    ]
    
    # Classes/IDs that indicate noise (Phase 1 expansion)
    NOISE_PATTERNS = [
        # Ads & tracking
        'advertisement', 'ad-container', 'ad-banner', 'google-ad',
        'sponsored-content', 'sponsored', 'promoted',
        'cookie-consent', 'gdpr-notice', 'privacy-notice',
        
        # Social & sharing
        'social-share', 'share-button', 'social-links', 'social-media',
        'share-tools', 'sharing', 'social-icons',
        
        # Newsletter & CTA
        'newsletter', 'email-signup', 'subscribe', 'subscription',
        'call-to-action', 'cta', 'signup', 'sign-up',
        
        # Related content & widgets
        'related-posts', 'related-content', 'related-articles',
        'you-may-like', 'recommended', 'suggestions',
        'sidebar-widget', 'widget-area', 'widget',
        'trending', 'popular-posts',
        
        # Navigation & breadcrumbs
        'breadcrumb', 'breadcrumbs', 'pagination',
        'mobile-menu', 'mobile-nav', 'menu-toggle',
        
        # Author & meta
        'author-bio', 'author-info', 'author-box', 'byline',
        'meta-info', 'post-meta', 'entry-meta',
        
        # Comments & interaction
        'comment-form', 'comments-section', 'comments',
        'discussion', 'replies',
    ]
    
    def __init__(self):
        """Initialize HTML Cleaner"""
        self.original_size = 0
        self.cleaned_size = 0
    
    def clean(self, html: str) -> Dict[str, Any]:
        """
        Clean HTML - Remove noise, keep content (ScrapeGraphAI approach)
        
        Args:
            html: Raw HTML content
            
        Returns:
            Dict with 'html', 'original_size', 'cleaned_size', 'reduction_percent' keys
        """
        self.original_size = len(html)
        
        logger.info(f" Cleaning HTML ({self.original_size:,} bytes)")
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Step 1: Remove noise tags (scripts, styles, nav, forms, buttons, SVGs)
        self._remove_noise_tags(soup)
        
        # Step 2: Remove HTML comments
        self._remove_comments(soup)
        
        # Step 3: Remove noise by class/ID (ads, widgets, social, CTAs)
        self._remove_obvious_ads(soup)
        
        # Step 4: Minify whitespace
        cleaned_html = self._minify_html(str(soup))
        
        self.cleaned_size = len(cleaned_html)
        
        reduction_percent = ((self.original_size - self.cleaned_size) / self.original_size * 100) if self.original_size > 0 else 0
        
        logger.info(f" Cleaned: {self.cleaned_size:,} bytes ({reduction_percent:.1f}% reduction)")
        
        return {
            'html': cleaned_html,
            'original_size': self.original_size,
            'cleaned_size': self.cleaned_size,
            'reduction_percent': reduction_percent
        }
    
    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        """Remove ONLY noise tags (scripts, styles, etc.)"""
        removed_count = 0
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"   Removed {removed_count} noise tags")
    
    def _remove_obvious_ads(self, soup: BeautifulSoup) -> None:
        """
        Remove noise elements (ads, widgets, UI, social, etc.)
        Conservative approach: only remove when class/ID matches known patterns
        """
        removed_count = 0
        
        for tag in soup.find_all(True):
            try:
                # Check class and id for noise pattern matches
                classes = tag.get('class', [])
                id_attr = tag.get('id', '')
                
                class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                combined = (class_str + ' ' + id_attr).lower()
                
                # Check if class/ID contains any noise patterns
                if any(pattern in combined for pattern in self.NOISE_PATTERNS):
                    tag.decompose()
                    removed_count += 1
            except (AttributeError, TypeError):
                continue
        
        if removed_count > 0:
            logger.debug(f"   Removed {removed_count} noise elements")
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments"""
        comments = soup.find_all(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        if len(comments) > 0:
            logger.debug(f"   Removed {len(comments)} comments")
    
    def _minify_html(self, html: str) -> str:
        """
        Minify HTML by removing excessive whitespace
        Based on ScrapeGraphAI's approach
        """
        # Remove HTML comments (<!-- ... -->)
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Remove whitespace between tags
        html = re.sub(r'>\s+<', '><', html)
        
        # Remove leading/trailing whitespace from lines
        html = re.sub(r'^\s+', '', html, flags=re.MULTILINE)
        html = re.sub(r'\s+$', '', html, flags=re.MULTILINE)
        
        # Collapse multiple spaces to single space
        html = re.sub(r'\s+', ' ', html)
        
        return html.strip()
    
    def get_reduction_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return {
            'original_size': self.original_size,
            'cleaned_size': self.cleaned_size,
            'reduction_percent': ((self.original_size - self.cleaned_size) / self.original_size * 100) if self.original_size > 0 else 0,
            'reduction_ratio': f"{self.original_size / self.cleaned_size:.1f}x" if self.cleaned_size > 0 else "N/A"
        }
