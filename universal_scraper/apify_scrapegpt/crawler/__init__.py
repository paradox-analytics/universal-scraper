"""
Universal Crawler Module

Discovers URLs across websites using multiple strategies.
"""

from .crawler import UniversalCrawler, CrawlConfig, CrawledURL, CrawlResult
from .link_discovery import LinkDiscoverer
from .api_discovery import APIDiscoverer
from .search_discovery import SearchDiscoverer, SearchStrategy
from .page_classifier import PageClassifier, PageType
from .pagination_handler import PaginationHandler

__all__ = [
 'UniversalCrawler',
 'CrawlConfig',
 'CrawledURL',
 'CrawlResult',
 'LinkDiscoverer',
 'APIDiscoverer',
 'SearchDiscoverer',
 'SearchStrategy',
 'PageClassifier',
 'PageType',
 'PaginationHandler'
]

