"""
Universal Crawler - Main orchestration for URL discovery

Coordinates multiple discovery strategies:
- Link-based discovery (traditional crawling)
- API-based discovery (network interception)
- Search-based discovery (form enumeration)
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .link_discovery import LinkDiscoverer
from .api_discovery import APIDiscoverer
from .search_discovery import SearchDiscoverer
from .page_classifier import PageClassifier, PageType
from .pagination_handler import PaginationHandler

logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
 """Configuration for crawling behavior"""
 mode: str = "smart" # 'smart', 'links_only', 'api_only', 'search_only'
 max_depth: int = 3
 max_pages: int = 1000
 max_items: int = 10000
 follow_patterns: List[str] = field(default_factory=list)
 ignore_patterns: List[str] = field(default_factory=list)
 handle_pagination: bool = True
 discover_apis: bool = True
 enable_search_discovery: bool = True
 rate_limit: str = "10/minute"
 timeout_minutes: int = 60
 respect_robots_txt: bool = True


@dataclass
class CrawledURL:
 """Represents a discovered URL"""
 url: str
 depth: int
 parent_url: Optional[str]
 page_type: Optional[PageType] = None
 data_type: Optional[str] = None
 discovered_via: str = "link" # 'link', 'api', 'search', 'pagination'
 metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlResult:
 """Result of crawling operation"""
 urls: List[CrawledURL]
 start_urls: List[str]
 total_discovered: int
 total_crawled: int
 apis_discovered: Dict[str, Any]
 search_queries_used: List[Dict[str, Any]]
 crawl_tree: Dict[str, int]
 duration_seconds: float
 metadata: Dict[str, Any]


class UniversalCrawler:
 """
 Universal crawler that discovers URLs using multiple strategies

 Strategies:
 1. Link-based: Traditional HTML link extraction
 2. API-based: Network request interception
 3. Search-based: Form enumeration and query permutation

 Example:
 config = CrawlConfig(
 mode='smart',
 max_depth=3,
 max_pages=1000
 )

 crawler = UniversalCrawler(config)
 result = crawler.crawl(['https://example.com'])
 """

 def __init__(self, config: Optional[CrawlConfig] = None, fetcher=None):
 self.config = config or CrawlConfig()
 self.fetcher = fetcher

 # Initialize sub-modules with shared fetcher
 self.link_discoverer = LinkDiscoverer(fetcher=fetcher)
 self.api_discoverer = APIDiscoverer()
 self.search_discoverer = SearchDiscoverer()
 self.page_classifier = PageClassifier()
 self.pagination_handler = PaginationHandler(fetcher=fetcher)

 # State tracking
 self.discovered_urls: Set[str] = set()
 self.crawled_urls: Set[str] = set()
 self.queue: deque = deque()
 self.results: List[CrawledURL] = []
 self.apis_discovered = {}
 self.search_queries = []

 # Statistics
 self.stats = {
 'urls_by_depth': {},
 'urls_by_type': {},
 'urls_by_discovery': {}
 }

 logger.info(f" Universal Crawler initialized")
 logger.info(f" Mode: {self.config.mode}")
 logger.info(f" Max depth: {self.config.max_depth}")
 logger.info(f" Max pages: {self.config.max_pages}")

 async def crawl(self, start_urls: List[str]) -> CrawlResult:
 """
 Crawl starting from given URLs

 Args:
 start_urls: List of URLs to start crawling from

 Returns:
 CrawlResult with all discovered URLs and metadata
 """
 start_time = datetime.now()

 logger.info(f" Starting crawl from {len(start_urls)} URLs")

 # Initialize queue with start URLs
 for url in start_urls:
 self._add_to_queue(url, depth=0, parent=None)

 # Main crawl loop
 while self.queue and len(self.crawled_urls) < self.config.max_pages:
 url_info = self.queue.popleft()
 await self._crawl_url(url_info)

 # Calculate duration
 duration = (datetime.now() - start_time).total_seconds()

 logger.info(f" Crawl complete!")
 logger.info(f" URLs discovered: {len(self.discovered_urls)}")
 logger.info(f" URLs crawled: {len(self.crawled_urls)}")
 logger.info(f" Duration: {duration:.2f}s")

 return CrawlResult(
 urls=self.results,
 start_urls=start_urls,
 total_discovered=len(self.discovered_urls),
 total_crawled=len(self.crawled_urls),
 apis_discovered=self.apis_discovered,
 search_queries_used=self.search_queries,
 crawl_tree=self.stats['urls_by_depth'],
 duration_seconds=duration,
 metadata=self._get_crawl_metadata()
 )

 async def _crawl_url(self, url_info: Dict[str, Any]) -> None:
 """Crawl a single URL"""
 url = url_info['url']
 depth = url_info['depth']

 if url in self.crawled_urls:
 return

 logger.info(f" Crawling [{depth}]: {url}")
 self.crawled_urls.add(url)

 try:
 # Classify page type
 page_type = self.page_classifier.classify(url)

 # Determine strategy based on page type and config
 if page_type == PageType.SEARCH_REQUIRED:
 self._handle_search_page(url, depth)
 elif self.config.mode == 'smart' or self.config.mode == 'links_only':
 await self._handle_standard_page(url, depth, page_type)

 # Record result
 crawled_url = CrawledURL(
 url=url,
 depth=depth,
 parent_url=url_info.get('parent'),
 page_type=page_type,
 discovered_via=url_info.get('discovered_via', 'unknown')
 )
 self.results.append(crawled_url)

 # Update stats
 self._update_stats(depth, page_type, url_info.get('discovered_via'))

 except Exception as e:
 logger.error(f" Error crawling {url}: {e}")

 async def _handle_standard_page(
 self,
 url: str,
 depth: int,
 page_type: PageType
 ) -> None:
 """Handle standard pages with link discovery"""

 # Discover APIs if enabled
 if self.config.discover_apis:
 apis = self.api_discoverer.discover(url)
 if apis:
 self.apis_discovered[url] = apis
 self._process_api_links(apis, depth)

 # Discover links
 links = await self.link_discoverer.discover(url)

 # Handle pagination if it's a listing page
 if page_type == PageType.LISTING and self.config.handle_pagination:
 pagination_urls = self.pagination_handler.discover_pages(url)
 links.extend(pagination_urls)

 # Add links to queue
 for link in links:
 if self._should_follow(link, depth):
 self._add_to_queue(
 link,
 depth=depth + 1,
 parent=url,
 discovered_via='link'
 )

 def _handle_search_page(self, url: str, depth: int) -> None:
 """Handle search-required pages"""
 logger.info(f" Detected search-required page: {url}")

 if not self.config.enable_search_discovery:
 logger.warning(" Search discovery disabled, skipping")
 return

 # Run search enumeration
 search_results = self.search_discoverer.enumerate(url)

 self.search_queries.extend(search_results['queries_used'])

 # Add discovered URLs to queue
 for result_url in search_results['urls']:
 if self._should_follow(result_url, depth):
 self._add_to_queue(
 result_url,
 depth=depth + 1,
 parent=url,
 discovered_via='search'
 )

 def _process_api_links(self, apis: Dict[str, Any], depth: int) -> None:
 """Process URLs discovered from API responses"""
 # Extract URLs from API responses
 for api_endpoint, api_data in apis.items():
 if 'links' in api_data:
 for link in api_data['links']:
 if self._should_follow(link, depth):
 self._add_to_queue(
 link,
 depth=depth + 1,
 parent=api_endpoint,
 discovered_via='api'
 )

 def _should_follow(self, url: str, current_depth: int) -> bool:
 """Determine if URL should be followed"""

 # Check depth limit
 if current_depth >= self.config.max_depth:
 return False

 # Check if already discovered
 if url in self.discovered_urls:
 return False

 # Check follow patterns
 if self.config.follow_patterns:
 if not any(pattern in url for pattern in self.config.follow_patterns):
 return False

 # Check ignore patterns
 if self.config.ignore_patterns:
 if any(pattern in url for pattern in self.config.ignore_patterns):
 return False

 return True

 def _add_to_queue(
 self,
 url: str,
 depth: int,
 parent: Optional[str],
 discovered_via: str = 'link'
 ) -> None:
 """Add URL to crawl queue"""
 if url not in self.discovered_urls:
 self.discovered_urls.add(url)
 self.queue.append({
 'url': url,
 'depth': depth,
 'parent': parent,
 'discovered_via': discovered_via
 })

 def _update_stats(
 self,
 depth: int,
 page_type: PageType,
 discovered_via: str
 ) -> None:
 """Update crawl statistics"""
 self.stats['urls_by_depth'][depth] = \
 self.stats['urls_by_depth'].get(depth, 0) + 1

 if page_type:
 type_name = page_type.value
 self.stats['urls_by_type'][type_name] = \
 self.stats['urls_by_type'].get(type_name, 0) + 1

 if discovered_via:
 self.stats['urls_by_discovery'][discovered_via] = \
 self.stats['urls_by_discovery'].get(discovered_via, 0) + 1

 def _get_crawl_metadata(self) -> Dict[str, Any]:
 """Get crawl metadata"""
 return {
 'mode': self.config.mode,
 'max_depth': self.config.max_depth,
 'urls_by_depth': self.stats['urls_by_depth'],
 'urls_by_type': self.stats['urls_by_type'],
 'urls_by_discovery': self.stats['urls_by_discovery'],
 'apis_discovered_count': len(self.apis_discovered),
 'search_queries_count': len(self.search_queries)
 }

