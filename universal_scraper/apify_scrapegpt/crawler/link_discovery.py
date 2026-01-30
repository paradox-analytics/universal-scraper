"""
Link Discovery - Traditional HTML link extraction
"""

import logging
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class LinkDiscoverer:
 """
 Discovers links on web pages

 Strategy: Extract <a> tags and filter for valid URLs
 Universal: Works on any website, any content type
 """

 def __init__(self, fetcher=None):
 """
 Initialize link discoverer

 Args:
 fetcher: Optional fetcher instance (HTMLFetcher or HybridFetcher)
 If None, will lazy-load when needed
 """
 self.fetcher = fetcher
 self.excluded_extensions = [
 '.pdf', '.jpg', '.png', '.gif', '.css', '.js',
 '.zip', '.tar', '.gz', '.mp4', '.mp3', '.wav',
 '.avi', '.mov', '.doc', '.docx', '.xls', '.xlsx'
 ]
 logger.debug(" Link Discoverer initialized")

 async def discover(self, url: str, html: str = None) -> List[str]:
 """
 Discover links on a page (Universal - works on any website)

 Args:
 url: Page URL
 html: HTML content (if None, will fetch)

 Returns:
 List of discovered URLs
 """

 if html is None:
 # Lazy-load fetcher if needed
 if self.fetcher is None:
 self.fetcher = self._get_fetcher()

 # Fetch HTML
 try:
 logger.debug(f" Fetching HTML for link discovery: {url}")
 result = await self.fetcher.fetch(url)
 html = result.get('html', '')
 if not html:
 logger.warning(f" No HTML returned for {url}")
 return []
 except Exception as e:
 logger.error(f" Failed to fetch {url}: {e}")
 return []

 try:
 soup = BeautifulSoup(html, 'html.parser')
 links = []

 # Extract all <a> tags
 for a_tag in soup.find_all('a', href=True):
 href = a_tag['href']

 # Convert to absolute URL
 absolute_url = urljoin(url, href)

 # Filter and validate
 if self._is_valid_url(absolute_url):
 links.append(absolute_url)

 logger.debug(f" Discovered {len(links)} links on {url}")
 return list(set(links)) # Deduplicate

 except Exception as e:
 logger.error(f" Error discovering links on {url}: {e}")
 return []

 def _is_valid_url(self, url: str) -> bool:
 """Check if URL should be followed"""

 # Parse URL
 parsed = urlparse(url)

 # Must have scheme and netloc
 if not parsed.scheme or not parsed.netloc:
 return False

 # Skip non-http(s) schemes
 if parsed.scheme not in ['http', 'https']:
 return False

 # Skip file extensions
 if any(url.lower().endswith(ext) for ext in self.excluded_extensions):
 return False

 # Skip fragments
 if parsed.fragment and not parsed.query:
 return False

 return True

 def _get_fetcher(self):
 """
 Lazy-load fetcher (universal - uses HybridFetcher for flexibility)

 HybridFetcher auto-detects if JavaScript is needed, making it
 universal across static and dynamic sites.
 """
 try:
 from ..core.hybrid_fetcher import HybridFetcher
 logger.debug(" Using HybridFetcher (universal static + JS support)")
 return HybridFetcher(
 enable_cache=True,
 enable_warming=False # Don't warm session for crawling
 )
 except ImportError:
 # Fallback to HTMLFetcher if hybrid not available
 try:
 from ..core.html_fetcher import HTMLFetcher
 logger.debug(" Using HTMLFetcher (static HTML only)")
 return HTMLFetcher(enable_warming=False)
 except ImportError:
 logger.error(" No fetcher available!")
 raise ImportError("Cannot import HTMLFetcher or HybridFetcher")

