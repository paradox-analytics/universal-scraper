"""
Page Classifier - Detects page types for intelligent crawling
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PageType(Enum):
 """
 Types of pages encountered during crawling (Universal)

 These types apply across ALL websites:
 - E-commerce: product listings vs product details
 - News: article archives vs individual articles
 - Forums: thread lists vs individual threads
 - Directories: business listings vs business details
 - Social media: feeds vs individual posts
 - Government: record listings vs record details
 """
 LISTING = "listing" # Multiple items, links to details
 DETAIL = "detail" # Single item with full details
 NAVIGATION = "navigation" # Hub page with category links
 SEARCH_REQUIRED = "search_required" # Requires search to see results
 PAGINATED = "paginated" # Part of paginated sequence
 UNKNOWN = "unknown"


class PageClassifier:
 """
 Classifies page types for intelligent crawling (Universal)

 Works on ANY website type:
 - E-commerce (product listings → products)
 - News sites (article archives → articles)
 - Forums (thread lists → threads)
 - Directories (listings → details)
 - Government databases (search results → records)
 - Social media (feeds → posts)
 - Documentation (index → pages)

 Uses multiple signals:
 - URL patterns (universal keywords)
 - HTML structure (repeated elements)
 - Content analysis (item counts)
 - DOM patterns (semantic HTML)
 """

 def __init__(self):
 # Universal URL patterns (not site-specific)
 self.listing_patterns = [
 '/search', '/category', '/categories', '/listings', '/browse',
 '/list', '/index', '/archive', '/results', '/directory',
 '/catalog', '/all', '/feed', '/posts', '/articles', '/items'
 ]

 self.detail_patterns = [
 '/detail', '/view', '/show', '/info', '/profile',
 '/post/', '/article/', '/item/', '/record/', '/page/',
 '/-info/', '/id/', '/thread/'
 ]

 self.search_patterns = [
 '/search', '/find', '/lookup', '/query', '/discover'
 ]

 logger.debug(" Page Classifier initialized (Universal)")

 def classify(self, url: str, html: Optional[str] = None) -> PageType:
 """
 Classify a page type

 Args:
 url: Page URL
 html: Optional HTML content for deeper analysis

 Returns:
 PageType enum value
 """

 # Quick URL-based classification
 url_lower = url.lower()

 # Check for detail pages
 if any(pattern in url_lower for pattern in self.detail_patterns):
 return PageType.DETAIL

 # Check for listing pages
 if any(pattern in url_lower for pattern in self.listing_patterns):
 return PageType.LISTING

 # Check for search pages
 if any(pattern in url_lower for pattern in self.search_patterns):
 # Need to determine if it's search-required
 # For now, assume search URLs with no query params are search-required
 if '?' not in url:
 return PageType.SEARCH_REQUIRED
 return PageType.LISTING

 # If HTML provided, do deeper analysis
 if html:
 return self._classify_from_html(html)

 return PageType.UNKNOWN

 def _classify_from_html(self, html: str) -> PageType:
 """
 Classify based on HTML content (Universal patterns)

 Uses generic patterns that work across all website types
 """

 html_lower = html.lower()

 # Check for search forms (search-required indicator)
 if '<form' in html_lower:
 # Look for search-related inputs
 if any(keyword in html_lower for keyword in ['type="search"', 'name="search"', 'name="q"', 'name="query"']):
 # If form exists but no results shown, likely search-required
 if html_lower.count('<a ') < 10: # Few links = empty results page
 return PageType.SEARCH_REQUIRED

 # Check for multiple similar elements (universal listing indicator)
 # Look for ANY repeated structural patterns

 # Common list/grid patterns (universal)
 patterns_to_check = [
 'class="item', 'class="card', 'class="entry', 'class="post',
 'class="article', 'class="row', 'class="result', 'class="listing',
 'class="product', 'class="record', 'class="thread', 'class="tile',
 '<article', '<li', 'data-item', 'data-id'
 ]

 total_items = sum(html_lower.count(pattern) for pattern in patterns_to_check)

 # Heuristic: 10+ similar elements = listing page
 if total_items > 10:
 return PageType.LISTING

 # 1-5 similar elements = detail page (some internal structure)
 elif total_items > 0:
 return PageType.DETAIL

 # Check for pagination indicators (listing page signal)
 pagination_indicators = [
 'pagination', 'pager', 'page-nav', 'next-page',
 'prev-page', 'page=', 'page-number'
 ]
 if any(indicator in html_lower for indicator in pagination_indicators):
 return PageType.LISTING

 return PageType.UNKNOWN

