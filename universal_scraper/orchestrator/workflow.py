"""
Universal Workflow Orchestrator

Coordinates crawler and scraper modules for unified data extraction workflows.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..crawler import UniversalCrawler, CrawlConfig
from ..core.scraper import UniversalScraper
from ..core.schema_manager import SchemaDefinition

logger = logging.getLogger(__name__)


class WorkflowMode(Enum):
 """Workflow execution modes"""
 CRAWL_ONLY = "crawl_only" # Only discover URLs
 SCRAPE_ONLY = "scrape_only" # Only scrape provided URLs
 CRAWL_THEN_SCRAPE = "crawl_then_scrape" # Sequential: crawl all, then scrape all
 STREAM_SCRAPE = "stream_scrape" # Streaming: scrape as URLs discovered
 FULL_AUTO = "full_auto" # Intelligent: auto-detect and execute


@dataclass
class WorkflowConfig:
 """Complete workflow configuration"""
 mode: WorkflowMode = WorkflowMode.FULL_AUTO

 # Crawl configuration
 crawl_config: Optional[CrawlConfig] = None

 # Scrape configuration
 schema: Optional[SchemaDefinition] = None
 fields: List[str] = None
 strict_schema: bool = False

 # Workflow behavior
 max_total_items: int = 10000
 save_intermediate: bool = False
 continue_on_error: bool = True


class UniversalWorkflow:
    """
    Orchestrates complete workflows combining crawling and scraping

    Modes:
    1. CRAWL_ONLY: Just discover URLs
    2. SCRAPE_ONLY: Scrape provided URLs
    3. CRAWL_THEN_SCRAPE: Discover all URLs, then scrape all
    4. STREAM_SCRAPE: Scrape URLs as they're discovered
    5. FULL_AUTO: Detect page type and execute appropriate workflow

    Example:
    # Auto mode
    workflow = UniversalWorkflow()
    result = workflow.execute(
        start_urls=['https://leafly.com/dispensaries/nevada'],
        fields=['name', 'address', 'rating']
    )

    # Manual mode
    config = WorkflowConfig(
        mode=WorkflowMode.CRAWL_THEN_SCRAPE,
        crawl_config=CrawlConfig(max_depth=3),
        schema=my_schema
    )
    workflow = UniversalWorkflow(config)
    result = workflow.execute(start_urls=[...])
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        api_key: Optional[str] = None,
        use_camoufox: bool = True,
        headless: bool = True,
        proxy_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or WorkflowConfig()
        self.api_key = api_key
        self.use_camoufox = use_camoufox
        self.headless = headless
        self.proxy_config = proxy_config

        # Initialize modules (lazy)
        self._crawler = None
        self._scraper = None

        logger.info(" Universal Workflow initialized")
        logger.info(f" Mode: {self.config.mode.value}")
        logger.info(f" Browser: {'Camoufox' if use_camoufox else 'Playwright'}")
        if proxy_config:
            logger.info(f" Proxy: Enabled")

    @property
    def crawler(self) -> UniversalCrawler:
        """Lazy initialization of crawler"""
        if self._crawler is None:
            crawl_config = self.config.crawl_config or CrawlConfig()
            self._crawler = UniversalCrawler(crawl_config)
        return self._crawler

    @property
    def scraper(self) -> UniversalScraper:
        """Lazy initialization of scraper"""
        if self._scraper is None:
            # Determine if auto-pagination should be enabled
            enable_auto_pagination = True
            if self.config.crawl_config:
                enable_auto_pagination = self.config.crawl_config.handle_pagination

            print(f" WORKFLOW: Initializing UniversalScraper (Camoufox={self.use_camoufox}, Proxy={bool(self.proxy_config)}, AutoPagination={enable_auto_pagination})", flush=True)
            self._scraper = UniversalScraper(
                api_key=self.api_key,
                schema=self.config.schema,
                strict_schema=self.config.strict_schema,
                use_camoufox=self.use_camoufox,
                headless=self.headless,
                proxy_config=self.proxy_config,
                enable_auto_pagination=enable_auto_pagination  # Pass pagination flag
            )
            print(" WORKFLOW: UniversalScraper initialized successfully", flush=True)
        return self._scraper

    async def execute(
        self,
        start_urls: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow

        Args:
            start_urls: URLs to start crawling from (for crawl modes)
            urls: URLs to scrape (for scrape-only mode)
            fields: Fields to extract

        Returns:
            Complete workflow result
        """

        print(f" WORKFLOW.execute(): Starting (mode={self.config.mode.value})", flush=True)
        start_time = datetime.now()

        logger.info(" Starting workflow execution")

        # Use fields from config if not provided
        if fields is None:
            fields = self.config.fields or []

        print(f" WORKFLOW.execute(): Mode check - {self.config.mode.value}", flush=True)

        # Execute based on mode
        if self.config.mode == WorkflowMode.CRAWL_ONLY:
            print(" WORKFLOW.execute(): Executing CRAWL_ONLY", flush=True)
            result = await self._execute_crawl_only(start_urls)

        elif self.config.mode == WorkflowMode.SCRAPE_ONLY:
            print(f" WORKFLOW.execute(): Executing SCRAPE_ONLY (urls={len(urls) if urls else 0})", flush=True)
            result = await self._execute_scrape_only(urls, fields)

        elif self.config.mode == WorkflowMode.CRAWL_THEN_SCRAPE:
            result = await self._execute_crawl_then_scrape(start_urls, fields)

        elif self.config.mode == WorkflowMode.STREAM_SCRAPE:
            result = await self._execute_stream_scrape(start_urls, fields)

        elif self.config.mode == WorkflowMode.FULL_AUTO:
            result = await self._execute_auto(start_urls, urls, fields)

        else:
            raise ValueError(f"Unknown workflow mode: {self.config.mode}")

        # Add execution metadata
        duration = (datetime.now() - start_time).total_seconds()
        result['workflow_metadata'] = {
            'mode': self.config.mode.value,
            'duration_seconds': duration,
            'start_time': start_time.isoformat()
        }

        logger.info(f" Workflow complete in {duration:.2f}s")

        return result

    async def _execute_crawl_only(self, start_urls: List[str]) -> Dict[str, Any]:
        """Execute crawl-only workflow"""
        logger.info(" Executing CRAWL_ONLY workflow")

        crawl_result = await self.crawler.crawl(start_urls)

        return {
            'mode': 'crawl_only',
            'urls_discovered': [u.url for u in crawl_result.urls],
            'total_urls': len(crawl_result.urls),
            'crawl_metadata': {
                'total_discovered': crawl_result.total_discovered,
                'total_crawled': crawl_result.total_crawled,
                'crawl_tree': crawl_result.crawl_tree
            }
        }

    async def _execute_scrape_only(
        self,
        urls: List[str],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Execute scrape-only workflow"""
        print(f" WORKFLOW._execute_scrape_only(): Starting (urls={len(urls)})", flush=True)
        logger.info(f" Executing SCRAPE_ONLY workflow for {len(urls)} URLs")

        all_data = []
        successful = 0
        failed = 0

        print(f" WORKFLOW._execute_scrape_only(): Starting loop over {len(urls)} URLs", flush=True)

        for i, url in enumerate(urls, 1):
            print(f" WORKFLOW._execute_scrape_only(): Scraping {i}/{len(urls)}: {url}", flush=True)
            logger.info(f" Scraping {i}/{len(urls)}: {url}")

            try:
                print(f" WORKFLOW._execute_scrape_only(): Accessing self.scraper", flush=True)
                print(f" WORKFLOW._execute_scrape_only(): Calling scraper.scrape()", flush=True)
                result = await self.scraper.scrape(url, fields)
                print(f" WORKFLOW._execute_scrape_only(): scraper.scrape() returned {len(result.get('data', []))} items", flush=True)
                all_data.extend(result['data'])
                successful += 1
            except Exception as e:
                logger.error(f" Failed: {e}")
                failed += 1
                if not self.config.continue_on_error:
                    raise

        return {
            'mode': 'scrape_only',
            'data': all_data,
            'total_items': len(all_data),
            'scrape_metadata': {
                'urls_scraped': len(urls),
                'successful': successful,
                'failed': failed
            }
        }

    async def _execute_crawl_then_scrape(
        self,
        start_urls: List[str],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Execute sequential crawl then scrape workflow"""
        logger.info(" Executing CRAWL_THEN_SCRAPE workflow")

        # Phase 1: Crawl
        logger.info(" Phase 1: Crawling...")
        crawl_result = await self.crawler.crawl(start_urls)
        discovered_urls = [u.url for u in crawl_result.urls]

        logger.info(f" Discovered {len(discovered_urls)} URLs")

        # Phase 2: Scrape
        logger.info(" Phase 2: Scraping...")
        scrape_result = await self._execute_scrape_only(discovered_urls, fields)

        # Combine results
        return {
            'mode': 'crawl_then_scrape',
            'data': scrape_result['data'],
            'total_items': len(scrape_result['data']),
            'urls_discovered': discovered_urls,
            'crawl_metadata': {
                'total_discovered': crawl_result.total_discovered,
                'total_crawled': crawl_result.total_crawled
            },
            'scrape_metadata': scrape_result['scrape_metadata']
        }

    async def _execute_stream_scrape(
        self,
        start_urls: List[str],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Execute streaming scrape workflow (scrape as URLs discovered)"""
        logger.info(" Executing STREAM_SCRAPE workflow")

        # This would require generator-based crawling
        # For now, fallback to crawl_then_scrape
        logger.warning(" STREAM_SCRAPE not yet implemented, using CRAWL_THEN_SCRAPE")
        return await self._execute_crawl_then_scrape(start_urls, fields)

    async def _execute_auto(
        self,
        start_urls: Optional[List[str]],
        urls: Optional[List[str]],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Execute auto-detect workflow"""
        logger.info(" Executing FULL_AUTO workflow")

        # Determine mode based on inputs
        if urls and not start_urls:
            # User provided specific URLs → scrape only
            return await self._execute_scrape_only(urls, fields)

        elif start_urls and not urls:
            # User provided start URLs → crawl and scrape
            return await self._execute_crawl_then_scrape(start_urls, fields)

        else:
            raise ValueError("Provide either start_urls (for crawling) or urls (for scraping)")

    def close(self):
        """Clean up resources"""
        if self._scraper:
            self._scraper.close()
        logger.info(" Workflow closed")

