"""
Universal Scraper - Main orchestration class
Coordinates all components following the architecture diagram
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
import litellm

from .html_fetcher import HTMLFetcher
from .hybrid_fetcher import HybridFetcher
from .json_detector import JSONDetector
from .json_quality_validator import JSONQualityValidator
from .html_cleaner import SmartHTMLCleaner
from .structural_hash import StructuralHashGenerator
from .code_cache import CodeCache, generate_cache_key
from .ai_generator import AICodeGenerator
from .schema_manager import SchemaManager, SchemaDefinition
from .pagination_analyzer import LLMPaginationAnalyzer
from .pagination_executor import PaginationExecutor
from .pagination_detector import FastPaginationDetector
from .context_manager import ContextManager, ExtractionContext
from .data_validator import LLMDataValidator
from .json_analyzer import LLMJsonAnalyzer
from .html_structure_analyzer import HTMLStructureAnalyzer
from .json_structure_analyzer import JSONStructureAnalyzer
from .field_mapper import UniversalFieldMapper  # NEW: Semantic field mapping
from .smart_sampler import SmartHTMLSampler  # NEW: Dynamic HTML sampling
from .adaptive_dom_detector import AdaptiveDOMDetector  # NEW: Reinforcement-style iteration
from .embedding_cache import EmbeddingBasedSelectorCache  # NEW: ML-based learning
from .direct_llm_extractor import DirectLLMExtractor  # NEW: Direct LLM extraction (like ScrapeGraphAI)

logger = logging.getLogger(__name__)


class UniversalScraper:
    """
    Universal Web Scraper with JSON-first architecture
    
    Architecture Flow:
    1. Fetch HTML (with CloudScraper + proxies)
    2. Detect JSON sources (priority)
    3. Clean HTML (98% reduction)
    4. Generate structural hash
    5. Check code cache
    6. Generate extraction code (if cache miss)
    7. Execute code and extract data
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        proxy_config: Optional[Dict[str, str]] = None,
        cache_dir: str = "./cache",
        cache_ttl: int = 86400,
        enable_cache: bool = True,
        enable_warming: bool = True,
        fetch_mode: str = "hybrid",  # 'hybrid', 'static', or 'browser'
        headless: bool = True,
        browser_timeout: int = 60000,  # Browser navigation timeout in milliseconds
        use_camoufox: bool = True,  # NEW: Use Camoufox for better anti-detection (recommended)
        schema: Optional[SchemaDefinition] = None,  # Schema for stable output
        strict_schema: bool = False,  # Fail on schema validation errors
        enable_llm_pagination: bool = True,  # Enable LLM-based pagination analysis
        enable_auto_pagination: bool = True,  # Enable automatic pagination (scrape all pages)
        extraction_context: Optional[Any] = None,  # NEW: User's extraction goal (context-driven scraping)
        enable_context_validation: bool = True,  # NEW: Enable LLM validation of extracted data
        use_direct_llm: bool = True,  # NEW: Use Direct LLM extraction as primary method (like ScrapeGraphAI)
        quality_mode: str = "balanced",  # NEW: Quality mode for DirectLLM ('conservative', 'balanced', 'aggressive')
        web_unblocker_api_key: Optional[str] = None,  # NEW: Bright Data Web Unblocker API key (fallback for Kasada)
        web_unblocker_zone: str = "web_unlocker1",  # NEW: Web Unblocker zone name
        log_level: int = logging.INFO
    ):
        """
        Initialize Universal Scraper
        
        Args:
            api_key: API key for AI provider
            model_name: AI model to use (auto-detected if None)
            proxy_config: Proxy configuration dict
            cache_dir: Directory for code cache
            cache_ttl: Cache TTL in seconds
            enable_cache: Enable/disable caching
            enable_warming: Enable session warming
            fetch_mode: Fetching mode - 'hybrid' (auto-detect), 'static' (fast), or 'browser' (JS support)
            headless: Run browser in headless mode (if browser mode used)
            browser_timeout: Browser navigation timeout in milliseconds (increase for slow proxies)
            schema: Optional schema definition for stable output
            strict_schema: Fail on schema validation errors
            enable_llm_pagination: Enable LLM-based universal pagination detection (recommended)
            extraction_context: User's extraction goal - string or dict (enables context-driven scraping)
                               Example: "Extract product listings with name, price, rating"
            enable_context_validation: Enable LLM validation of extracted data (prevents false positives)
            use_direct_llm: Use Direct LLM extraction as primary method (recommended, like ScrapeGraphAI)
            quality_mode: Quality mode for Direct LLM extraction:
                         - 'conservative': Like ScrapeGraphAI (≥70% fields, fewer items, highest quality)
                         - 'balanced': Default (≥50% fields, good compromise)
                         - 'aggressive': Maximum extraction (≥30% fields, most items)
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Auto-detect API key from environment if not provided
        import os
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                logger.debug(" Using API key from environment variable")
        
        self.api_key = api_key
        self.model_name = model_name
        self.fetch_mode = fetch_mode
        
        # NEW: Create ProxyManager for per-request rotation (Oxylabs approach)
        proxy_manager = None
        if proxy_config:
            from .proxy_manager import ProxyManager
            proxy_manager = ProxyManager(
                proxy_config=proxy_config,
                provider='apify',  # Will detect Apify vs local automatically
                rotation_strategy='per_request'  # Rotate on every request
            )
            logger.info(" ProxyManager created: Per-request rotation enabled")
        
        # Initialize fetcher based on mode
        if fetch_mode == "hybrid":
            # Hybrid fetcher (recommended) - auto-detects and uses best method
            self.html_fetcher = HybridFetcher(
                proxy_config=proxy_config,
                proxy_manager=proxy_manager,  # NEW: Pass ProxyManager
                enable_cache=enable_cache,
                enable_warming=enable_warming,
                cache_dir=cache_dir,
                headless=headless,
                browser_timeout=browser_timeout,
                use_camoufox=use_camoufox,  # NEW: Pass Camoufox preference
                web_unblocker_api_key=web_unblocker_api_key,  # NEW: Web Unblocker fallback
                web_unblocker_zone=web_unblocker_zone  # NEW: Web Unblocker zone
            )
        elif fetch_mode == "static":
            # Static HTML only (fast but no JS support)
            self.html_fetcher = HTMLFetcher(
                proxy_config=proxy_config,
                proxy_manager=proxy_manager,  # NEW: Pass ProxyManager
                enable_warming=enable_warming
            )
        elif fetch_mode == "browser":
            # Force browser mode (slow but full JS support)
            self.html_fetcher = HybridFetcher(
                proxy_config=proxy_config,
                proxy_manager=proxy_manager,  # NEW: Pass ProxyManager
                enable_cache=enable_cache,
                enable_warming=enable_warming,
                cache_dir=cache_dir,
                headless=headless,
                browser_timeout=browser_timeout,
                force_mode="browser",
                use_camoufox=use_camoufox,  # NEW: Pass Camoufox preference
                web_unblocker_api_key=web_unblocker_api_key,  # NEW: Web Unblocker fallback
                web_unblocker_zone=web_unblocker_zone  # NEW: Web Unblocker zone
            )
        else:
            raise ValueError(f"Invalid fetch_mode: {fetch_mode}. Use 'hybrid', 'static', or 'browser'")
        
        self.json_detector = JSONDetector()
        self.json_quality_validator = JSONQualityValidator()
        self.html_cleaner = SmartHTMLCleaner()
        self.hash_generator = StructuralHashGenerator()
        
        self.code_cache = CodeCache(
            cache_dir=cache_dir,
            ttl=cache_ttl,
            enable_cache=enable_cache
        )
        
        self.ai_generator = AICodeGenerator(
            api_key=api_key,
            model_name=model_name
        )
        
        # Initialize Direct LLM Extractor (NEW: Like ScrapeGraphAI)
        self.direct_llm_extractor = None
        self.use_direct_llm = use_direct_llm
        if use_direct_llm and api_key:
            self.direct_llm_extractor = DirectLLMExtractor(
                api_key=api_key,
                model_name=model_name or "gpt-4o-mini",
                quality_mode=quality_mode
            )
            logger.info(f" Direct LLM Extractor enabled (quality={quality_mode})")
        elif use_direct_llm and not api_key:
            logger.warning(" Direct LLM extraction requested but no API key provided")
            self.use_direct_llm = False
        
        # Initialize schema manager if schema provided
        self.schema_manager = None
        if schema:
            self.schema_manager = SchemaManager(
                schema=schema,
                ai_generator=self.ai_generator,
                strict_mode=strict_schema,
                enable_ai_mapping=True
            )
        
        # Initialize pagination detection (fast patterns + LLM fallback)
        self.fast_pagination_detector = FastPaginationDetector()
        self.pagination_analyzer = None
        self.pagination_executor = None
        self.enable_llm_pagination = enable_llm_pagination
        self.enable_auto_pagination = enable_auto_pagination  # NEW: Control auto-pagination behavior
        
        if enable_llm_pagination and api_key:
            try:
                self.pagination_analyzer = LLMPaginationAnalyzer(
                    openai_api_key=api_key,
                    cache_dir=f"{cache_dir}/pagination"
                )
                
                # Get the actual browser fetcher from HybridFetcher or use the fetcher directly
                browser_fetcher = None
                if hasattr(self.html_fetcher, 'browser_fetcher'):
                    # HybridFetcher case - extract the internal browser_fetcher
                    browser_fetcher = self.html_fetcher
                elif hasattr(self.html_fetcher, 'page'):
                    # Direct BrowserFetcher case
                    browser_fetcher = self.html_fetcher
                
                self.pagination_executor = PaginationExecutor(
                    browser_fetcher=browser_fetcher
                )
                logger.info(" Hybrid Pagination Detection enabled (fast patterns + LLM fallback)")
            except Exception as e:
                logger.warning(f" Failed to initialize LLM Pagination Analyzer: {e}")
                logger.warning("   Using fast pattern detection only")
                self.enable_llm_pagination = False
        elif not api_key:
            logger.info("ℹ  Using fast pattern detection only (no LLM)")
            self.enable_llm_pagination = False
        
        # Initialize context-driven components
        self.context_manager = None
        self.json_analyzer = None
        self.data_validator = None
        self.html_structure_analyzer = None  # NEW: From ScrapeGraphAI
        self.json_structure_analyzer = None  # NEW: JSON structure analysis (like HTML)
        self.enable_context_validation = enable_context_validation
        
        if extraction_context and api_key:
            # Parse and enrich user's extraction context
            self.context_manager = ContextManager(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                enable_cache=enable_cache,
                cache_dir=f"{cache_dir}/context"
            )
            
            # Parse context immediately (can also be done per-scrape)
            try:
                self.context_manager.parse_context(extraction_context)
                logger.info(f" Extraction Context: {self.context_manager.context}")
            except Exception as e:
                logger.warning(f" Failed to parse extraction context: {e}")
                self.context_manager = None
        
        if api_key and enable_context_validation:
            # JSON source analyzer (ranks multiple JSON sources)
            self.json_analyzer = LLMJsonAnalyzer(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                enable_cache=enable_cache
            )
            
            # Data validator (validates extraction matches user's goal)
            self.data_validator = LLMDataValidator(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                enable_cache=enable_cache
            )
            
            logger.info(" Context-driven validation enabled (JSON ranking + data validation)")
        
        # Initialize HTML Structure Analyzer (NEW: From ScrapeGraphAI)
        if api_key:
            self.html_structure_analyzer = HTMLStructureAnalyzer(
                api_key=api_key,
                model=model_name or "gpt-4o-mini"
            )
            logger.info(" HTML Structure Analyzer enabled (improves code generation)")
            
            # Initialize JSON Structure Analyzer (NEW: Like HTML but for JSON)
            self.json_structure_analyzer = JSONStructureAnalyzer(
                api_key=api_key,
                model=model_name or "gpt-4o-mini"
            )
            logger.info(" JSON Structure Analyzer enabled (improves JSON extraction)")
            
            # Initialize Adaptive DOM Detector (NEW: Reinforcement iteration)
            self.adaptive_dom_detector = AdaptiveDOMDetector(
                api_key=api_key,
                model_name=model_name or "gpt-4o-mini",
                max_passes=3
            )
            logger.info(" Adaptive DOM Detector enabled (reinforcement-style iteration)")
        
        # Initialize Universal Field Mapper (NEW: Semantic field understanding)
        self.field_mapper = None
        if api_key:
            self.field_mapper = UniversalFieldMapper(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                cache_dir=f"{cache_dir}/field_mappings",
                enable_cache=enable_cache
            )
            logger.info("  Universal Field Mapper enabled (semantic field understanding)")
        
        # Initialize Smart HTML Sampler (universal, no LLM required)
        self.smart_sampler = SmartHTMLSampler()
        logger.info(" Smart HTML Sampler enabled (dynamic sizing per website)")
        
        # Initialize Embedding-Based Selector Cache (ML learning from successes)
        self.embedding_cache = None
        if api_key and enable_cache:
            self.embedding_cache = EmbeddingBasedSelectorCache(
                cache_dir=f"{cache_dir}/embedding_selectors",
                api_key=api_key
            )
            stats = self.embedding_cache.get_stats()
            logger.info(f" Embedding Cache enabled ({stats['total_sites']} sites learned)")
        
        logger.info(" Universal Scraper initialized")
        logger.info(f"   Fetch Mode: {fetch_mode}")
        logger.info(f"   AI Model: {self.ai_generator.model_name}")
        logger.info(f"   Cache: {'Enabled' if enable_cache else 'Disabled'}")
        logger.info(f"   Proxy: {'Enabled' if proxy_config else 'Disabled'}")
        logger.info(f"   Browser: {' Headless' if headless else 'Headed'} (if needed)")
        logger.info(f"   LLM Pagination: {'Enabled' if self.enable_llm_pagination else 'Disabled'}")
        logger.info(f"   Context Validation: {'Enabled' if enable_context_validation and api_key else 'Disabled'}")
        if schema:
            logger.info(f"   Schema: {schema.name} v{schema.version} (strict={strict_schema})")
    
    @staticmethod
    async def generate_fields_from_prompt(
        prompt: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        return_descriptions: bool = False
    ) -> Union[List[str], Dict[str, str]]:
        """
        Generate field names from natural language prompt.
        
        Universal feature inspired by Oxylabs AI Scraper.
        Converts "I want product names and prices" → ['product_name', 'price']
        
        Args:
            prompt: Natural language description
            url: Optional URL for domain context
            api_key: OpenAI API key (required)
            return_descriptions: Return {field: description} dict
            
        Returns:
            List of field names or Dict of {field: description}
            
        Example:
            fields = await UniversalScraper.generate_fields_from_prompt(
                prompt="I want game titles, developers, platforms, and genres",
                url="https://example.com/games",
                api_key="sk-..."
            )
            # Returns: ['title', 'developer', 'platform', 'genre']
        """
        if not api_key:
            raise ValueError("api_key is required for field generation")
        
        from .field_generator import NaturalLanguageFieldGenerator
        
        generator = NaturalLanguageFieldGenerator(api_key=api_key)
        return await generator.generate_fields(
            prompt=prompt,
            url=url,
            return_descriptions=return_descriptions
        )
    
    @staticmethod
    async def scrape_from_prompt(
        url: str,
        prompt: str,
        api_key: str,
        **scraper_kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method: Scrape directly from natural language prompt.
        
        Combines field generation + scraping in one call.
        Universal feature inspired by Oxylabs AI Scraper.
        
        Args:
            url: Target URL
            prompt: Natural language description (e.g., "I want product names and prices")
            api_key: OpenAI API key
            **scraper_kwargs: Additional UniversalScraper init arguments
            
        Returns:
            Scraping result dict with 'data', 'quality', etc.
            
        Example:
            result = await UniversalScraper.scrape_from_prompt(
                url="https://example.com/products",
                prompt="I want product names, prices, and ratings",
                api_key="sk-..."
            )
            # Automatically generates fields ['product_name', 'price', 'rating']
            # and scrapes the page
        """
        # Step 1: Generate fields from prompt
        logger.info(f" Generating fields from prompt...")
        fields = await UniversalScraper.generate_fields_from_prompt(
            prompt=prompt,
            url=url,
            api_key=api_key
        )
        logger.info(f"    Generated fields: {', '.join(fields)}")
        
        # Step 2: Scrape with generated fields
        scraper = UniversalScraper(api_key=api_key, **scraper_kwargs)
        result = await scraper.scrape(url=url, fields=fields)
        await scraper.close()
        
        return result
    
    async def scrape(
        self,
        url: str,
        fields: List[str],
        force_html: bool = False,
        force_generate: bool = False,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None,
        wait_for_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scrape data from URL
        
        Args:
            url: Target URL
            fields: Fields to extract (empty list = auto-extraction)
            force_html: Skip JSON detection, use HTML parsing
            force_generate: Skip cache, generate new extraction code
            scroll_to_bottom: Scroll to bottom for infinite scroll pagination
            click_load_more: CSS selector for "Load More" button
            wait_for_selector: Wait for specific element before scraping
            
        Returns:
            Dict with 'data', 'metadata', 'source' keys
        """
        start_time = time.time()
        
        logger.info(f" Scraping: {url}")
        logger.info(f" Fields: {', '.join(fields) if fields else 'AUTO-EXTRACT ALL'}")
        
        if scroll_to_bottom:
            logger.info(" Pagination: Infinite scroll enabled")
        if click_load_more:
            logger.info(f" Pagination: Load More button '{click_load_more}'")
        
        # OPTIMIZATION: Pre-warm cache check (before expensive fetch)
        # For scale (thousands of pages/day): Check if we have cached field mappings
        # This allows skipping LLM analysis for pages 2-N in pagination scenarios
        pre_warmed_mappings = None
        if self.json_structure_analyzer and fields:
            try:
                pre_warmed_mappings = self.json_structure_analyzer.get_cached_mappings(url, fields)
                if pre_warmed_mappings:
                    logger.info(" Pre-warmed cache hit: Will skip JSON structure analysis LLM call")
            except Exception as e:
                logger.debug(f"Pre-warm cache check failed: {e}")
        
        # Step 1: Fetch HTML (with pagination support)
        logger.info(" Step 1: Fetching HTML...")
        fetch_result = await self.html_fetcher.fetch(
            url,
            wait_for_selector=wait_for_selector,
            scroll_to_bottom=scroll_to_bottom,
            click_load_more=click_load_more
        )
        html = fetch_result['html']
        original_html = html  # NEW: Store original HTML for Direct LLM supplementation
        captured_json = fetch_result.get('captured_json', [])  # Get captured JSON blobs
        
        # Step 1.5: Smart Pagination Detection (fast patterns + LLM fallback)
        # Try to detect and handle pagination to get ALL items from listing pages
        pagination_strategy = None
        
        try:
            logger.info(" Step 1.5: Detecting pagination pattern...")
            
            # FAST PATH: Try pattern-based detection first (instant, 90% of cases)
            # Get current item count for context
            json_results_preview = self.json_detector.detect_and_extract(html, url, captured_json=captured_json)
            current_items = len(json_results_preview.get('data', []))
            
            pagination_strategy = self.fast_pagination_detector.detect(url, html, current_items)
            
            if pagination_strategy:
                logger.info(f" Fast detection: {pagination_strategy['type']} (confidence: {pagination_strategy['confidence']})")
                logger.info(f" Reasoning: {pagination_strategy['reasoning']}")
                
                # Handle URL-based pagination (most common) - generate URLs and scrape them
                if pagination_strategy['type'] in ['url_param', 'path_based'] and pagination_strategy.get('max_page'):
                    # Check if there's a max_pages limit set (from Apify input or other source)
                    max_pages_limit = getattr(self, '_max_pages_limit', None)
                    detected_max_page = pagination_strategy['max_page']
                    
                    # Apply limit if set
                    if max_pages_limit and max_pages_limit > 0:
                        max_page = min(detected_max_page, max_pages_limit)
                        if max_page < detected_max_page:
                            logger.info(f" Limiting pagination: {detected_max_page} pages detected, limiting to {max_page} pages")
                    else:
                        max_page = detected_max_page
                    
                    logger.info(f" Generating URLs for {max_page} pages...")
                    
                    # Generate all page URLs
                    base_url = pagination_strategy.get('base_url', url.split('?')[0])
                    param_name = pagination_strategy.get('param_name', 'page')
                    other_params = pagination_strategy.get('other_params', '')
                    
                    page_urls = []
                    for page_num in range(1, max_page + 1):
                        if other_params:
                            page_url = f"{base_url}?{other_params}&{param_name}={page_num}"
                        else:
                            page_url = f"{base_url}?{param_name}={page_num}"
                        page_urls.append(page_url)
                    
                    logger.info(f" Generated {len(page_urls)} URLs for parallel scraping")
                    pagination_strategy['generated_urls'] = page_urls
                    
                    # AUTO-PAGINATION: Automatically scrape all pages (if enabled)
                    if self.enable_auto_pagination:
                        logger.info(f" Auto-pagination enabled: scraping all {len(page_urls)} pages...")
                        all_page_data, first_page_html, first_page_json_data = await self._scrape_all_pages(
                            page_urls, 
                            fields, 
                            wait_for_selector=wait_for_selector,
                            pre_warmed_mappings=pre_warmed_mappings  # NEW: Pass pre-warmed mappings for optimization
                        )
                    else:
                        logger.info(f" Auto-pagination disabled: returning single page only (detected {len(page_urls)} total pages)")
                        all_page_data = None
                    
                    # Check for missing fields and supplement with Direct LLM if needed
                    if all_page_data and fields:
                        # Check which requested fields are present in extracted items
                        found_fields = set()
                        for item in all_page_data[:10]:  # Check first 10 items
                            found_fields.update(item.keys())
                        
                        missing_fields = [f for f in fields if f not in found_fields]
                        
                        # If fields are missing, try Direct LLM supplementation
                        if missing_fields and self.use_direct_llm and self.direct_llm_extractor:
                            logger.info(f" Missing fields in pagination extraction: {missing_fields}")
                            logger.info(f" Trying Direct LLM to supplement missing fields...")
                            
                            try:
                                # NEW: Since pagination uses JSON extraction, prefer JSON data over HTML
                                # JSON data contains the actual product information, HTML might not
                                # IMPROVED: Use extracted items (which have product data) instead of raw JSON blob
                                if first_page_json_data or (all_page_data and len(all_page_data) > 0):
                                    logger.info("    Using JSON data from pagination (data is in JSON, not HTML)")
                                    import json
                                    
                                    # PREFER: Use extracted items (they have the actual product data)
                                    # This is better than raw JSON blob because items are already structured
                                    if all_page_data and len(all_page_data) > 0:
                                        # Use extracted items - they contain product data we need
                                        # Show first 20 items to give LLM enough context
                                        items_sample = all_page_data[:20] if len(all_page_data) > 20 else all_page_data
                                        json_str = json.dumps(items_sample, indent=2, default=str)
                                        logger.info(f"    Using extracted items ({len(items_sample)} items) for Direct LLM supplementation")
                                    elif first_page_json_data:
                                        # Fallback: Use raw JSON data if items not available
                                        json_str = json.dumps(first_page_json_data, indent=2, default=str)
                                        # Limit size to avoid token limits (show first 50KB)
                                        json_str = json_str[:50000] if len(json_str) > 50000 else json_str
                                        logger.info(f"    Using raw JSON data ({len(json_str):,} chars)")
                                    
                                    # Create readable format with context about what we're looking for
                                    html_for_llm = f"""# Product Data from {url}

The following JSON contains product information. Extract the "{', '.join(missing_fields)}" field(s) for each product.

```json
{json_str}
```

Note: Look for product names/titles in fields like 'name', 'title', 'productName', 'product_name', or similar fields."""
                                    logger.info(f"    Converted to readable format ({len(json_str):,} chars)")
                                elif original_html and len(original_html) > 1000:  # Ensure HTML is substantial
                                    logger.info("    Using HTML from original fetch (JSON data not available)")
                                    html_for_llm = original_html
                                elif first_page_html and len(first_page_html) > 1000:
                                    logger.info("    Using HTML from pagination fetch (JSON data not available)")
                                    html_for_llm = first_page_html
                                else:
                                    logger.warning("    Neither JSON data nor HTML available, falling back to fetching original URL...")
                                    # Use original URL, not page_urls[0] (which might be ?pageNumber=1)
                                    first_page_fetch = await self.html_fetcher.fetch(
                                        url,  # Use original URL, not paginated URL
                                        wait_for_selector=wait_for_selector,
                                        scroll_to_bottom=False,
                                        click_load_more=None
                                    )
                                    html_for_llm = first_page_fetch['html']
                                
                                # Clean HTML before Direct LLM (only if it's actual HTML, not JSON)
                                if not html_for_llm.startswith("# JSON Data"):
                                    clean_result = self.html_cleaner.clean(html_for_llm)
                                    cleaned_html = clean_result['html']
                                else:
                                    cleaned_html = html_for_llm  # JSON is already clean
                                
                                # Get context if available
                                context_str = None
                                if hasattr(self, 'context_manager') and self.context_manager and hasattr(self.context_manager, 'context') and self.context_manager.context:
                                    context_str = self.context_manager.context.goal
                                
                                # Direct LLM extraction for missing fields only
                                import asyncio
                                if asyncio.iscoroutinefunction(self.direct_llm_extractor.extract):
                                    direct_llm_items = await self.direct_llm_extractor.extract(
                                        html=cleaned_html,
                                        fields=missing_fields,  # Only extract missing fields
                                        context=context_str,
                                        url=url  # Use original URL, not paginated URL
                                    )
                                else:
                                    direct_llm_items = self.direct_llm_extractor.extract(
                                        html=cleaned_html,
                                        fields=missing_fields,
                                        context=context_str,
                                        url=url  # Use original URL, not paginated URL
                                    )
                                
                                # Merge Direct LLM results with pagination results
                                if direct_llm_items and len(direct_llm_items) > 0:
                                    logger.info(f" Direct LLM extracted {len(direct_llm_items)} items with missing fields")
                                    
                                    # Merge by position (match items by index)
                                    merged_data = []
                                    for i, pagination_item in enumerate(all_page_data):
                                        merged_item = pagination_item.copy()
                                        if i < len(direct_llm_items):
                                            llm_item = direct_llm_items[i]
                                            # Add missing fields from Direct LLM
                                            for field in missing_fields:
                                                if field in llm_item and llm_item.get(field):
                                                    merged_item[field] = llm_item[field]
                                        merged_data.append(merged_item)
                                    
                                    all_page_data = merged_data
                                    logger.info(f" Merged pagination + Direct LLM results: {len(merged_data)} items")
                            except Exception as e:
                                logger.warning(f" Direct LLM supplementation failed: {e}")
                                logger.info("   Continuing with pagination results only...")
                    
                    # Return pagination data
                    if all_page_data:
                        logger.info(f" Auto-pagination complete: collected {len(all_page_data)} items from {len(page_urls)} pages")
                        
                        end_time = time.time()
                        elapsed = end_time - start_time
                        
                        logger.info(f" Extraction complete: {len(all_page_data)} items in {elapsed:.2f}s")
                        
                        return {
                            'data': all_page_data,
                            'metadata': {
                                'url': url,
                                'fields': fields,
                                'items_extracted': len(all_page_data),
                                'execution_time': elapsed,
                                'extraction_source': 'auto_pagination',
                                'code_cached': None,
                                'schema_quality': 'high',
                                'pagination_detected': True,
                                'total_pages_scraped': len(page_urls),
                                'auto_pagination': True,
                                'timestamp': time.time()
                            },
                            'source': 'json'
                        }
            
            # FALLBACK: Use LLM if fast detection failed and LLM is enabled
            elif self.enable_llm_pagination and self.pagination_analyzer:
                logger.info(" Fast detection failed, trying LLM analysis...")
                
                # Prepare user hints for LLM
                user_hints = {}
                if scroll_to_bottom:
                    user_hints['scroll_to_bottom'] = True
                if click_load_more:
                    user_hints['click_load_more'] = click_load_more
                if wait_for_selector:
                    user_hints['wait_for_selector'] = wait_for_selector
                
                # Analyze pagination strategy (cached per domain)
                pagination_strategy = await self.pagination_analyzer.analyze_pagination_strategy(
                    url, html, user_hints
                )
                
                if pagination_strategy and self.pagination_executor:
                    # Execute strategy to get ALL items
                    all_items = await self.pagination_executor.execute_strategy(
                        pagination_strategy,
                        url,
                        html,
                        fetch_result
                    )
                    
                    if all_items:
                        logger.info(f" LLM Pagination extracted {len(all_items)} total items")
                        
                        # If we got items, add them to captured_json for normal processing
                        if isinstance(all_items, list):
                            captured_json.extend(all_items)
                        
                        # Update fetch_result with enhanced captured_json
                        fetch_result['captured_json'] = captured_json
            else:
                logger.info("ℹ  No pagination detected - treating as single page")
                    
        except Exception as e:
            logger.warning(f" Pagination detection failed: {e}")
            logger.warning("   Continuing with single-page extraction...")
        
        # Step 2: Context-driven JSON extraction
        json_data = []
        extraction_source = 'html'
        extraction_metadata = {}
        
        # Get context (if available)
        context = self.context_manager.context if self.context_manager else None
        
        # Infer fields from context if not provided
        if not fields and context and context.fields:
            fields = context.fields
            logger.info(f" Inferred fields from context: {fields}")
        
        if not force_html:
            logger.info(" Step 2: Detecting JSON sources...")
            # Pass captured JSON blobs (from APIs, embedded JSON, etc.)
            json_results = self.json_detector.detect_and_extract(html, url, captured_json=captured_json)
            
            if json_results['json_found']:
                logger.info(f" Found {len(json_results.get('sources', []))} JSON source(s)")
                
                # CONTEXT-DRIVEN APPROACH: Rank and validate JSON sources
                if context and self.json_analyzer and self.data_validator:
                    logger.info(" Using context-driven JSON analysis...")
                    
                    # Prepare sources for ranking
                    # json_results has parallel lists: 'sources' (names) and 'data' (actual data)
                    json_sources_dict = {}
                    sources_list = json_results.get('sources', [])
                    data_list = json_results.get('data', [])
                    
                    if sources_list and data_list:
                        # Zip sources (names) with data
                        for source_name, source_data in zip(sources_list, data_list):
                            json_sources_dict[source_name] = source_data
                    elif data_list:
                        # No source names, just number them
                        for i, source_data in enumerate(data_list):
                            json_sources_dict[f'source_{i}'] = source_data
                    else:
                        # Fallback: treat entire data as single source
                        json_sources_dict = {'json_data': json_results.get('data', {})}
                    
                    # SELECT BEST SOURCE (Simplified Approach)
                    try:
                        best_source = self.json_analyzer.select_best_source(
                            json_sources=json_sources_dict,
                            url=url,
                            context=context
                        )
                        
                        if best_source and best_source in json_sources_dict:
                            logger.info(f" Selected source: {best_source}")
                            
                            # Extract from this ONE source
                            source_data = json_sources_dict[best_source]
                            
                            # NEW: Analyze JSON structure first (like HTML structure analysis)
                            # OPTIMIZATION: Use pre-warmed mappings if available (skip LLM call)
                            json_structure_analysis = None
                            if pre_warmed_mappings:
                                json_structure_analysis = pre_warmed_mappings
                                logger.info(" Using pre-warmed field mappings (skipped LLM analysis)")
                            elif self.json_structure_analyzer:
                                try:
                                    logger.info(" Analyzing JSON structure before extraction...")
                                    # Build enriched context with website-specific field meanings
                                    enriched_context = context.goal if context else None
                                    if enriched_context:
                                        # Add field-specific context hint
                                        enriched_context += " | Extract FULL product names for 'title' field, not short labels."
                                    
                                    json_structure_analysis = self.json_structure_analyzer.analyze(
                                        json_data=source_data,
                                        url=url,
                                        fields=fields,
                                        context=enriched_context
                                    )
                                    
                                    if json_structure_analysis.get('field_mappings'):
                                        logger.info(f"    JSON structure analyzed (confidence: {json_structure_analysis.get('confidence', 0):.2f})")
                                        logger.info(f"    Found {len(json_structure_analysis['field_mappings'])} field mappings")
                                        if json_structure_analysis.get('extraction_path'):
                                            logger.info(f"    Extraction path: {json_structure_analysis['extraction_path']}")
                                except Exception as e:
                                    logger.warning(f" JSON structure analysis failed: {e}")
                                    logger.info("   Continuing with extraction...")
                            
                            try:
                                # NEW: Pass LLM-discovered field mappings to extraction (universal fallback)
                                llm_field_mappings = None
                                if json_structure_analysis and json_structure_analysis.get('field_mappings'):
                                    llm_field_mappings = json_structure_analysis['field_mappings']
                                    logger.info("    Using LLM-discovered field mappings for universal extraction")
                                
                                items = self.json_detector.extract_from_json(source_data, fields, llm_field_mappings=llm_field_mappings)
                                
                                # Use structure analysis to improve extraction if available (post-processing)
                                if json_structure_analysis and json_structure_analysis.get('field_mappings') and items:
                                    logger.info("    Applying JSON structure analysis to improve extraction...")
                                    items = self._apply_json_structure_analysis(items, json_structure_analysis, fields)
                                
                                if items:
                                    logger.info(f"    Extracted {len(items)} items")
                                    
                                    # QUALITY VALIDATION (fast, universal, no LLM)
                                    is_valid, reason, quality_score = self.json_quality_validator.validate(
                                        extracted_items=items,
                                        requested_fields=fields,
                                        extraction_context=context.goal if context else None
                                    )
                                    
                                    if not is_valid or quality_score < 0.3:
                                        # CRITICAL FIX: If JSON quality is very low (<30%), fall back to Direct LLM
                                        if quality_score < 0.3:
                                            logger.warning(f"    JSON quality too low ({quality_score:.1%} < 30%)")
                                            logger.info(f"    Falling back to Direct LLM extraction (JSON likely navigation/filter data)")
                                            # Don't return, fall through to Direct LLM extraction
                                        else:
                                            logger.warning(f"    JSON quality check failed: {reason}")
                                            logger.info(f"    {self.json_quality_validator.suggest_fallback(reason)}")
                                            logger.info("    JSON data is unhealthy - will fall back to HTML extraction")
                                            # Don't return, fall through to HTML extraction
                                    else:
                                        logger.info(f"    JSON quality validated (score: {quality_score:.2f})")
                                        
                                        # Calculate field coverage using quality calculator (distinguishes required/optional)
                                        from .quality_calculator import QualityCalculator
                                        quality_calc = QualityCalculator()
                                        field_coverage = quality_calc.calculate_field_coverage(items, fields)
                                        quality_score_improved = quality_calc.calculate_quality_score(items, fields)
                                        missing_fields = quality_calc.get_missing_fields(items, fields, required_only=False)
                                        missing_required_fields = quality_calc.get_missing_fields(items, fields, required_only=True)
                                        
                                        logger.info(f"    Field coverage: {field_coverage}")
                                        logger.info(f"    Quality score (improved): {quality_score_improved:.1f}%")
                                        
                                        # LLM VALIDATION (expensive, context-driven)
                                        validation = self.data_validator.validate_extraction(
                                            items=items,
                                            url=url,
                                            context=context
                                        )
                                        
                                        logger.info(f"    LLM Validation: {' PASS' if validation['is_target_data'] else ' FAIL'} (confidence: {validation['confidence']:.2f})")
                                        logger.info(f"   Reasoning: {validation['reasoning']}")
                                        
                                        if missing_fields:
                                            logger.warning(f"    Missing fields in JSON extraction: {missing_fields}")
                                            if missing_required_fields:
                                                logger.warning(f"    Missing REQUIRED fields: {missing_required_fields}")
                                            logger.info(f"    Will try Direct LLM extraction to fill missing fields...")
                                        
                                        # UNIVERSAL FLOW: Accept JSON if BOTH validations pass (JSON is healthy)
                                        # Only fall back to HTML if JSON is truly unhealthy
                                        if validation['is_target_data'] and validation['confidence'] > 0.5:
                                            logger.info(f" JSON data is HEALTHY - using JSON extraction from {best_source}!")
                                            logger.info(f"    JSON structure analysis cached for future requests")
                                            json_data = items
                                            extraction_source = 'json'
                                            extraction_metadata = {
                                                'json_source': best_source,
                                                'validation': validation,
                                                'context_used': str(context),
                                                'selection_method': 'llm_select_best',
                                                'missing_fields': missing_fields if missing_fields else [],
                                                'missing_required_fields': missing_required_fields if missing_required_fields else [],
                                                'field_coverage': field_coverage,
                                                'json_structure_analyzed': True,
                                                'quality_score': quality_score_improved  # Use improved score
                                            }
                                            
                                            # PHASE 1 OPTIMIZATION: Early exit if JSON is high quality and complete
                                            # Only exit if no required fields are missing and quality is high
                                            if not missing_required_fields and quality_score_improved >= 70.0:
                                                # High quality JSON with all or most fields - early exit!
                                                logger.info(f" PHASE 1 OPTIMIZATION: Early exit - JSON extraction complete and high quality")
                                                logger.info(f"   Skipping Direct LLM and HTML extraction (saved ~30-50% time)")
                                                
                                                # Calculate execution time
                                                execution_time = time.time() - start_time
                                                
                                                logger.info(f" Extraction complete: {len(json_data)} items in {execution_time:.2f}s")
                                                
                                                return {
                                                    'data': json_data,
                                                    'metadata': {
                                                        'url': url,
                                                        'fields': fields,
                                                        'items_extracted': len(json_data),
                                                        'execution_time': execution_time,
                                                        'extraction_source': extraction_source,
                                                        'code_cached': None,
                                                        'schema_quality': None,
                                                        'pagination_detected': pagination_strategy['type'] if pagination_strategy else None,
                                                        'total_pages_scraped': 1,
                                                        'auto_pagination': False,
                                                        'early_exit': True,  # NEW: Flag for early exit
                                                        'timestamp': time.time()
                                                    },
                                                    'source': extraction_source
                                                }
                                            
                                            # If critical fields missing, still try Direct LLM as supplement (not fallback)
                                            if missing_fields and 'title' in missing_fields:
                                                logger.info(f"    Title field missing, will try Direct LLM to supplement (not replace) JSON data...")
                                        else:
                                            logger.warning(f"    JSON data validation failed (confidence too low)")
                                            logger.info("    JSON data is unhealthy - will fall back to HTML extraction")
                                else:
                                    logger.warning("    No items extracted from selected source, falling back to HTML...")
                            
                            except Exception as e:
                                logger.warning(f"    Extraction from {best_source} failed: {e}")
                                logger.warning("   Falling back to HTML...")
                        else:
                            logger.warning(" No suitable JSON source found - falling back to HTML...")
                    
                    except Exception as e:
                        logger.error(f" JSON source ranking failed: {e}")
                        logger.info("   Falling back to traditional JSON detection...")
                        # Fall through to traditional detection
                
                # FALLBACK: Traditional approach (if no context or context features disabled)
                if not json_data and not context:
                    if self.json_detector.is_json_sufficient(json_results, fields):
                        logger.info(" JSON sources sufficient (traditional check), extracting...")
                        
                        # NEW: Analyze JSON structure first (like HTML structure analysis)
                        json_structure_analysis = None
                        if self.json_structure_analyzer and json_results['data']:
                            try:
                                logger.info(" Analyzing JSON structure before extraction...")
                                context_str = None
                                if hasattr(self, 'context_manager') and self.context_manager and hasattr(self.context_manager, 'context') and self.context_manager.context:
                                    context_str = self.context_manager.context.goal
                                
                                # Analyze the first JSON source
                                first_source_data = json_results['data'][0] if json_results['data'] else None
                                if first_source_data:
                                    json_structure_analysis = self.json_structure_analyzer.analyze(
                                        json_data=first_source_data,
                                        url=url,
                                        fields=fields,
                                        context=context_str
                                    )
                                    
                                    if json_structure_analysis.get('field_mappings'):
                                        logger.info(f"    JSON structure analyzed (confidence: {json_structure_analysis.get('confidence', 0):.2f})")
                                        logger.info(f"    Found {len(json_structure_analysis['field_mappings'])} field mappings")
                            except Exception as e:
                                logger.warning(f" JSON structure analysis failed: {e}")
                                logger.info("   Continuing with traditional extraction...")
                        
                        # NEW: Pass LLM-discovered field mappings to extraction (universal fallback)
                        llm_field_mappings = None
                        if json_structure_analysis and json_structure_analysis.get('field_mappings'):
                            llm_field_mappings = json_structure_analysis['field_mappings']
                            logger.info("    Using LLM-discovered field mappings for universal extraction")
                        
                        json_data = self.json_detector.extract_from_json(
                            json_results['data'],
                            fields,
                            llm_field_mappings=llm_field_mappings
                        )
                        
                        # Use structure analysis to improve extraction if available (post-processing)
                        if json_structure_analysis and json_structure_analysis.get('field_mappings') and json_data:
                            logger.info("    Applying JSON structure analysis to improve extraction...")
                            json_data = self._apply_json_structure_analysis(json_data, json_structure_analysis, fields)
                        
                        # VALIDATE JSON QUALITY (even without context)
                        if json_data:
                            is_valid, reason, quality_score = self.json_quality_validator.validate(
                                extracted_items=json_data,
                                requested_fields=fields,
                                extraction_context=None  # No context available
                            )
                            
                            if not is_valid or quality_score < 0.3:
                                # CRITICAL FIX: If JSON quality is very low (<30%), fall back to Direct LLM
                                if quality_score < 0.3:
                                    logger.warning(f"    JSON quality too low ({quality_score:.1%} < 30%)")
                                    logger.info(f"    Falling back to Direct LLM extraction (JSON likely navigation/filter data)")
                                    json_data = None  # Clear and fall through to Direct LLM
                                else:
                                    logger.warning(f"    JSON quality check failed: {reason}")
                                    logger.info(f"    {self.json_quality_validator.suggest_fallback(reason)}")
                                    logger.info("    JSON data is unhealthy - will fall back to HTML extraction")
                                    json_data = None  # Clear and fall through to HTML
                            else:
                                logger.info(f"    JSON quality validated (score: {quality_score:.2f})")
                                logger.info("    JSON structure analysis cached for future requests")
                                
                                # Calculate missing fields using quality calculator (distinguishes required/optional)
                                from .quality_calculator import QualityCalculator
                                quality_calc = QualityCalculator()
                                
                                if json_data and fields:
                                    # Calculate field coverage and quality score
                                    field_coverage = quality_calc.calculate_field_coverage(json_data, fields)
                                    quality_score_improved = quality_calc.calculate_quality_score(json_data, fields)
                                    missing_fields = quality_calc.get_missing_fields(json_data, fields, required_only=False)
                                    missing_required_fields = quality_calc.get_missing_fields(json_data, fields, required_only=True)
                                    
                                    logger.info(f"    Field coverage: {field_coverage}")
                                    logger.info(f"    Quality score (improved): {quality_score_improved:.1f}%")
                                    
                                    if missing_fields:
                                        logger.info(f"    Missing fields: {missing_fields}")
                                        if missing_required_fields:
                                            logger.warning(f"    Missing REQUIRED fields: {missing_required_fields}")
                                    
                                    extraction_metadata = {
                                        'json_source': 'traditional',
                                        'missing_fields': missing_fields,
                                        'missing_required_fields': missing_required_fields,
                                        'field_coverage': field_coverage,
                                        'validation': {'confidence': quality_score},
                                        'quality_score': quality_score_improved  # Use improved score
                                    }
                                    
                                    # PHASE 1 OPTIMIZATION: Early exit if JSON is high quality and complete
                                    # Only exit if no required fields are missing
                                    if not missing_required_fields and quality_score_improved >= 70.0:
                                        logger.info(f" PHASE 1 OPTIMIZATION: Early exit - JSON extraction complete and high quality")
                                        logger.info(f"   Skipping Direct LLM and HTML extraction (saved ~30-50% time)")
                                        
                                        # Calculate execution time
                                        execution_time = time.time() - start_time
                                        
                                        logger.info(f" Extraction complete: {len(json_data)} items in {execution_time:.2f}s")
                                        
                                        return {
                                            'data': json_data,
                                            'metadata': {
                                                'url': url,
                                                'fields': fields,
                                                'items_extracted': len(json_data),
                                                'execution_time': execution_time,
                                                'extraction_source': 'json',
                                                'code_cached': None,
                                                'schema_quality': None,
                                                'pagination_detected': pagination_strategy['type'] if pagination_strategy else None,
                                                'total_pages_scraped': 1,
                                                'auto_pagination': False,
                                                'early_exit': True,  # NEW: Flag for early exit
                                                'quality_score': quality_score_improved,
                                                'field_coverage': field_coverage,
                                                'timestamp': time.time()
                                            },
                                            'source': 'json'
                                        }
                                else:
                                    extraction_metadata = {
                                        'json_source': 'traditional',
                                        'missing_fields': fields if fields else [],
                                        'validation': {'confidence': quality_score},
                                        'quality_score': quality_score
                                    }
                        extraction_source = 'json'
                    else:
                        logger.info(" JSON sources insufficient, falling back to HTML...")
        
        # UNIVERSAL FLOW: JSON-first with LLM analysis, cache results, fallback to HTML only if unhealthy
        # Step 2.5: Direct LLM Extraction (fallback/supplement)
        # Only use HTML/Direct LLM if:
        #   1. No JSON data found, OR
        #   2. JSON data is unhealthy (failed validation), OR  
        #   3. JSON data is healthy but missing some fields (supplement, don't replace)
        
        # Calculate missing fields if not already set
        # json_data should already be semantically mapped, so check exact keys
        if json_data and fields and not extraction_metadata.get('missing_fields'):
            # Check which requested fields are present in the semantically extracted items
            found_fields = set()
            for item in json_data[:10]:  # Check first 10 items
                found_fields.update(item.keys())
            
            missing_fields = [f for f in fields if f not in found_fields]
            
            if missing_fields:
                extraction_metadata['missing_fields'] = missing_fields
                logger.info(f"    Calculated missing fields: {missing_fields}")
        
        # UNIVERSAL RULE: Only fall back to HTML if JSON is unhealthy or missing
        # If JSON is healthy, use it (even if some fields are missing - we'll supplement)
        json_is_healthy = (
            json_data and 
            len(json_data) > 0 and 
            extraction_metadata.get('quality_score', 0) >= 0.3  # Quality validated
        )
        
        # Try direct LLM extraction if:
        #   1. No JSON data (fallback to HTML)
        #   2. JSON data is unhealthy (fallback to HTML)
        #   3. JSON data is healthy but missing fields (supplement JSON, don't replace)
        should_try_direct_llm = (
            self.use_direct_llm and 
            self.direct_llm_extractor and 
            (not json_is_healthy or (json_is_healthy and extraction_metadata.get('missing_fields')))
        )
        
        if should_try_direct_llm:
            if not json_is_healthy:
                logger.info(" Step 2.5: JSON data is unhealthy - falling back to Direct LLM Extraction (HTML)...")
            elif json_is_healthy and extraction_metadata.get('missing_fields'):
                logger.info(f" Step 2.5: JSON data is healthy but missing fields - supplementing with Direct LLM...")
                logger.info(f"   Missing fields: {extraction_metadata['missing_fields']}")
                logger.info("   (Will merge Direct LLM results with JSON data, not replace)")
            else:
                logger.info(" Step 2.5: Trying Direct LLM Extraction (no JSON found)...")
            
            # OPTIMIZATION: Clean HTML BEFORE chunking (ScrapeGraphAI approach)
            # This dramatically reduces chunk count and processing time
            # For 6.8MB HTML → cleaned → ~800KB → 80 chunks instead of 646 chunks
            clean_result = self.html_cleaner.clean(html)
            cleaned_html = clean_result['html']
            logger.info(f"   HTML reduced: {clean_result['reduction_percent']:.1f}% ({len(html):,} → {len(cleaned_html):,} bytes)")
            
            # Get extraction context
            context_str = None
            if hasattr(self, 'context_manager') and self.context_manager and hasattr(self.context_manager, 'context') and self.context_manager.context:
                context_str = self.context_manager.context.goal
            
            try:
                # Direct LLM extraction (async)
                # Use CLEANED HTML to reduce chunk count (10x faster for large pages)
                # HybridMarkdownExtractor will do additional cleaning if needed
                import asyncio
                if asyncio.iscoroutinefunction(self.direct_llm_extractor.extract):
                    direct_llm_items = await self.direct_llm_extractor.extract(
                        html=cleaned_html,  # CLEANED HTML - reduces chunks dramatically
                        fields=fields,
                        context=context_str,
                        url=url  # NEW: Pass URL for cache key generation
                    )
                else:
                    direct_llm_items = self.direct_llm_extractor.extract(
                        html=cleaned_html,  # CLEANED HTML - reduces chunks dramatically
                        fields=fields,
                        context=context_str,
                        url=url  # NEW: Pass URL for cache key generation
                    )
                
                if direct_llm_items and len(direct_llm_items) > 0:
                    logger.info(f" Direct LLM extracted {len(direct_llm_items)} items")
                    
                    # UNIVERSAL RULE: If JSON is healthy, supplement it (don't replace)
                    # If JSON is unhealthy, use Direct LLM as primary
                    if json_is_healthy and json_data and len(json_data) > 0:
                        logger.info(f"    JSON data is healthy - supplementing missing fields with Direct LLM...")
                        # Match items by position or merge fields
                        merged_items = []
                        for i, json_item in enumerate(json_data):
                            merged_item = json_item.copy()
                            # Try to find matching Direct LLM item (by position or by matching fields)
                            if i < len(direct_llm_items):
                                llm_item = direct_llm_items[i]
                                # Add missing fields from Direct LLM (only if missing in JSON)
                                for field in extraction_metadata.get('missing_fields', []):
                                    if field not in merged_item or not merged_item.get(field):
                                        if field in llm_item and llm_item.get(field):
                                            merged_item[field] = llm_item[field]
                            merged_items.append(merged_item)
                        json_data = merged_items
                        extraction_source = 'json+direct_llm'
                        extraction_metadata['supplemented_with_direct_llm'] = True
                        logger.info(f"    Supplemented {len(merged_items)} JSON items with Direct LLM fields")
                        logger.info("    JSON remains primary source (Direct LLM only filled gaps)")
                    else:
                        # No JSON data or JSON is unhealthy, use Direct LLM as primary
                        # Calculate quality using quality calculator (distinguishes required/optional)
                        from .quality_calculator import QualityCalculator
                        quality_calc = QualityCalculator()
                        
                        if fields:
                            field_coverage = quality_calc.calculate_field_coverage(direct_llm_items, fields)
                            quality = quality_calc.calculate_quality_score(direct_llm_items, fields)
                            missing_fields = quality_calc.get_missing_fields(direct_llm_items, fields, required_only=False)
                            missing_required_fields = quality_calc.get_missing_fields(direct_llm_items, fields, required_only=True)
                            
                            logger.info(f"    Field coverage: {field_coverage}")
                            logger.info(f"    Quality: {quality:.1f}% (required/optional weighted)")
                            if missing_fields:
                                logger.info(f"    Missing fields: {missing_fields}")
                                if missing_required_fields:
                                    logger.warning(f"    Missing REQUIRED fields: {missing_required_fields}")
                        else:
                            # Fallback to simple calculation if no fields specified
                            quality = 100.0 if direct_llm_items else 0.0
                        
                        # Accept if quality is reasonable (≥40% for aggressive, higher for others)
                        quality_threshold = 40.0  # Lower threshold since we have quality modes
                        if quality >= quality_threshold:
                            logger.info(f" Direct LLM quality acceptable ({quality:.1f}% >= {quality_threshold:.1f}%)")
                            json_data = direct_llm_items
                            extraction_source = 'direct_llm'
                            
                            # PHASE 1 OPTIMIZATION: Early exit if Direct LLM quality is high
                            if quality >= 60.0:  # High quality threshold
                                logger.info(f" PHASE 1 OPTIMIZATION: Early exit - Direct LLM extraction high quality")
                                logger.info(f"   Skipping HTML extraction (saved ~20-30% time)")
                                
                                # Calculate execution time
                                execution_time = time.time() - start_time
                                
                                logger.info(f" Extraction complete: {len(json_data)} items in {execution_time:.2f}s")
                                
                                return {
                                    'data': json_data,
                                    'metadata': {
                                        'url': url,
                                        'fields': fields,
                                        'items_extracted': len(json_data),
                                        'execution_time': execution_time,
                                        'extraction_source': extraction_source,
                                        'code_cached': None,
                                        'schema_quality': None,
                                        'pagination_detected': pagination_strategy['type'] if pagination_strategy else None,
                                        'total_pages_scraped': 1,
                                        'auto_pagination': False,
                                        'early_exit': True,  # NEW: Flag for early exit
                                        'direct_llm_quality': quality,
                                        'field_coverage': field_coverage if fields else {},
                                        'missing_fields': missing_fields if fields else [],
                                        'missing_required_fields': missing_required_fields if fields else [],
                                        'timestamp': time.time()
                                    },
                                    'source': extraction_source
                                }
                            
                            # Skip HTML extraction
                            logger.info("   Skipping pattern-based extraction (Direct LLM succeeded)")
                        else:
                            logger.warning(f"    Direct LLM quality too low ({quality:.1f}% < {quality_threshold:.1f}%)")
                            logger.info("   Falling back to pattern-based extraction...")
                else:
                    logger.warning("    Direct LLM extracted 0 items, falling back to pattern-based...")
            
            except Exception as e:
                logger.error(f"    Direct LLM extraction failed: {e}")
                logger.info("   Falling back to pattern-based extraction...")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Step 3: If JSON and Direct LLM not sufficient, use HTML extraction
        if not json_data:
            logger.info(" Step 3: Cleaning HTML for pattern-based extraction...")
            
            # Clean HTML if not already done
            if 'cleaned_html' not in locals():
                clean_result = self.html_cleaner.clean(html)
                cleaned_html = clean_result['html']
            else:
                # Already cleaned in Direct LLM step
                pass
            
            logger.info(f"   Reduction: {clean_result['reduction_percent']:.1f}%")
            
            # Step 4: Generate structural hash
            logger.info(" Step 4: Generating structural hash...")
            hash_result = self.hash_generator.generate_hash(cleaned_html)
            structure_hash = hash_result['hash']
            
            # Step 5: Check cache (unless forced to generate)
            extraction_code = None
            code_cached = False
            
            if not force_generate:
                logger.info(" Step 5: Checking code cache...")
                # Generate field-aware cache key (fixes field mismatch issues)
                cache_key = generate_cache_key(structure_hash, fields)
                cached_entry = self.code_cache.get(cache_key)
                
                if cached_entry:
                    extraction_code = cached_entry['code']
                    code_cached = True
                    logger.info(" Using cached extraction code")
            
            # Step 5.5: Analyze HTML structure (NEW: From ScrapeGraphAI)
            structure_analysis = None
            if self.html_structure_analyzer and not code_cached:
                logger.info(" Step 5.5: Analyzing HTML structure...")
                try:
                    context_str = None
                    if hasattr(self, 'context_manager') and self.context_manager and hasattr(self.context_manager, 'context') and self.context_manager.context:
                        context_str = self.context_manager.context.goal
                    
                    structure_analysis = self.html_structure_analyzer.analyze(
                        cleaned_html,
                        url,
                        context=context_str
                    )
                except Exception as e:
                    logger.warning(f" Structure analysis failed: {e}")
            
            # Step 5.7: Map fields semantically (NEW: Semantic field understanding)
            field_hints = None
            if self.field_mapper and fields and not code_cached:
                logger.info("  Step 5.7: Mapping fields semantically...")
                try:
                    # Extract a dynamically-sized HTML sample (adapts to website structure)
                    html_sample = self._extract_smart_html_sample(
                        html=cleaned_html,
                        structure_analysis=structure_analysis,
                        fields=fields,
                        url=url
                    )
                    
                    field_hints = self.field_mapper.map_fields(
                        fields=fields,
                        url=url,
                        html_sample=html_sample,
                        structure_analysis=structure_analysis
                    )
                    logger.info(f"    Mapped {len(field_hints)} fields with semantic understanding")
                except Exception as e:
                    logger.warning(f"    Field mapping failed: {e}, continuing without semantic hints")
            
            # Step 6: Generate code if needed
            if not extraction_code:
                logger.info(" Step 6: Generating extraction code with AI...")
                # Pass extraction context to improve code generation
                context_str = None
                if hasattr(self, 'context_manager') and self.context_manager and hasattr(self.context_manager, 'context') and self.context_manager.context:
                    context_str = self.context_manager.context.goal
                gen_result = self.ai_generator.generate_extraction_code(
                    cleaned_html,
                    fields,
                    url,
                    extraction_context=context_str,
                    structure_analysis=structure_analysis,  # NEW!
                    max_iterations=3,  # NEW: Multi-iteration refinement
                    field_hints=field_hints  # NEW: Semantic field mappings
                )
                extraction_code = gen_result['code']
                
                # Cache the generated code with field-aware key
                cache_key = generate_cache_key(structure_hash, fields)
                self.code_cache.set(
                    cache_key,
                    extraction_code,
                    {
                        'model': gen_result['model_used'],
                        'fields': fields,
                        'url': url
                    }
                )
            
            # Step 7: Execute extraction code with REINFORCEMENT LOOP
            logger.info(" Step 7: Executing extraction code with adaptive iteration...")
            
            # Multi-pass extraction with quality-based retry
            best_result = []
            best_quality = 0.0
            current_structure = structure_analysis
            
            for pass_number in range(1, 4):  # Up to 3 passes
                if pass_number > 1:
                    # Only retry if adaptive detector is available
                    if not hasattr(self, 'adaptive_dom_detector') or not self.adaptive_dom_detector:
                        logger.warning(f" Adaptive DOM detector not available - cannot retry")
                        json_data = best_result
                        break
                    
                    logger.info(f" Pass {pass_number}: Retrying with improved selectors...")
                    
                    # Get improved pattern from adaptive DOM detector
                    current_structure = self.adaptive_dom_detector.detect_with_reinforcement(
                        html=html,
                        fields=fields,
                        initial_pattern=structure_analysis,
                        extraction_result={'items': best_result, 'quality': best_quality},
                        pass_number=pass_number
                    )
                    
                    # Regenerate code with improved pattern
                    gen_result = self.ai_generator.generate_extraction_code(
                        html,
                        fields,
                        url,
                        extraction_context=context_str,
                        structure_analysis=current_structure,  # Use improved pattern
                        max_iterations=3,
                        field_hints=field_hints
                    )
                    extraction_code = gen_result['code']
                
                # Save generated code for debugging
                try:
                    debug_code_path = f"{self.code_cache.cache_dir}/last_generated_code_pass{pass_number}.py"
                    with open(debug_code_path, 'w') as f:
                        f.write(f"# Generated for: {url}\n")
                        f.write(f"# Pass: {pass_number}\n")
                        f.write(f"# Fields: {fields}\n")
                        if current_structure:
                            f.write(f"# Selector: {current_structure.get('selector', 'N/A')}\n")
                        f.write(f"\n{extraction_code}")
                    logger.debug(f" Saved pass {pass_number} code to: {debug_code_path}")
                except Exception as e:
                    logger.debug(f" Could not save debug code: {e}")
                
                logger.info(f" Code length: {len(extraction_code)} chars")
                json_data = self._execute_extraction_code(
                    extraction_code,
                    html  # Use original HTML for extraction
                )
                logger.info(f"    Pass {pass_number}: Extracted {len(json_data)} items")
                
                # Calculate quality
                if json_data and fields:
                    total_fields = len(json_data) * len(fields)
                    filled_fields = sum(
                        1 for item in json_data
                        for v in item.values()
                        if v is not None and v != ''
                    )
                    quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
                else:
                    quality = 0.0
                
                logger.info(f"    Pass {pass_number} quality: {quality:.1f}%")
                
                # Keep best result
                if quality > best_quality or (len(json_data) > len(best_result) and quality >= best_quality * 0.9):
                    best_result = json_data
                    best_quality = quality
                    logger.info(f"    New best result: {len(best_result)} items, {best_quality:.1f}% quality")
                
                # Success threshold: >= 70% quality (lowered from 50% to catch more issues)
                if quality >= 70.0:
                    logger.info(f" Pass {pass_number} SUCCESS - Quality threshold met ({quality:.1f}% >= 70%)")
                    json_data = json_data  # Use current (best) result
                    break
                elif pass_number < 3:
                    logger.warning(f" Pass {pass_number} quality too low ({quality:.1f}% < 70%) - triggering next pass")
                else:
                    logger.warning(f" All {pass_number} passes complete - returning best attempt ({best_quality:.1f}% quality)")
                    json_data = best_result  # Use best result from all passes
            
            logger.info(f"    Final result: {len(json_data)} items from {pass_number} pass(es)")
        else:
            # Direct LLM or JSON extraction succeeded, set best_quality for later checks
            best_quality = 100.0 if json_data else 0.0
        
        # Initialize best_quality if not set (should not happen, but safety check)
        if 'best_quality' not in locals():
            best_quality = 0.0
        
        # PHASE 2.5: Semantic Pattern Fallback (if HTML code generation failed or low quality)
        # This is the NEW universal approach - resilient to layout changes!
        if (len(json_data) == 0 or best_quality < 50.0) and structure_analysis:
            logger.warning(f" HTML code generation {'failed' if len(json_data) == 0 else 'has low quality'} ({best_quality:.1f}%)")
            logger.info(" Trying Phase 2.5: Semantic Pattern Extraction...")
            
            try:
                # Import semantic extractor
                from .semantic_extractor import SemanticExtractor
                
                # Generate semantic pattern with LLM
                logger.info("   Generating semantic pattern...")
                pattern_result = self.ai_generator.generate_semantic_pattern(
                    cleaned_html=cleaned_html if 'cleaned_html' in locals() else html,
                    fields=fields,
                    url=url,
                    extraction_context=context_str if 'context_str' in locals() else None,
                    structure_analysis=structure_analysis,
                    field_hints=field_hints if 'field_hints' in locals() else None
                )
                
                logger.info(f"    Semantic pattern generated: {pattern_result['explanation'][:100]}...")
                
                # Get containers from structure analysis
                soup = BeautifulSoup(html, 'html.parser')
                containers = None
                
                if structure_analysis.get('selector'):
                    # Try to find containers with the detected selector
                    try:
                        selector = structure_analysis['selector']
                        containers = soup.select(selector)
                        logger.info(f"   Found {len(containers)} containers with selector: {selector}")
                    except Exception as e:
                        logger.warning(f"    Failed to find containers: {e}")
                
                # Extract using semantic patterns
                extractor = SemanticExtractor()
                semantic_items = extractor.extract(
                    html=html,
                    semantic_pattern=pattern_result['pattern'],
                    containers=containers
                )
                
                if semantic_items:
                    # Calculate quality
                    total_fields = len(semantic_items) * len(fields)
                    filled_fields = sum(
                        1 for item in semantic_items
                        for v in item.values()
                        if v is not None and v != ''
                    )
                    semantic_quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
                    
                    logger.info(f"    Semantic extraction: {len(semantic_items)} items, {semantic_quality:.1f}% quality")
                    
                    # Use semantic results if better than code generation
                    if semantic_quality > best_quality:
                        logger.info(f"    Semantic patterns outperformed code generation ({semantic_quality:.1f}% > {best_quality:.1f}%)")
                        json_data = semantic_items
                        extraction_source = 'semantic_patterns'
                        extraction_metadata['semantic_quality'] = semantic_quality
                        extraction_metadata['pattern'] = pattern_result['pattern']
                    else:
                        logger.info(f"   ℹ  Keeping code generation results ({best_quality:.1f}% >= {semantic_quality:.1f}%)")
                else:
                    logger.warning("    Semantic extraction returned 0 items")
                    
            except Exception as e:
                logger.error(f" Semantic pattern extraction failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # PHASE 3: LLM Fallback (if JSON, HTML, AND semantic extraction all failed)
        if len(json_data) == 0 and self.api_key and hasattr(self, 'context_manager') and self.context_manager:
            logger.warning(" JSON, HTML, and semantic extraction all returned 0 items")
            logger.info(" Trying Phase 3: LLM Direct Extraction Fallback...")
            
            try:
                json_data = await self._llm_fallback_extraction(
                    html=html,
                    json_sources=json_sources_dict if 'json_sources_dict' in locals() else {},
                    url=url,
                    fields=fields
                )
                
                if json_data:
                    logger.info(f" LLM fallback extracted {len(json_data)} items")
                    extraction_source = 'llm_fallback'
                    extraction_metadata['fallback_used'] = True
                else:
                    logger.warning(" LLM fallback also returned 0 items")
            except Exception as e:
                logger.error(f" LLM fallback failed: {e}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        logger.info(f" Extraction complete: {len(json_data)} items in {execution_time:.2f}s")
        
        # Apply schema normalization if schema manager is configured
        normalized_data = json_data
        schema_quality = None
        
        if self.schema_manager:
            logger.info(" Normalizing data to schema...")
            normalized_data = self.schema_manager.normalize_batch(json_data)
            schema_quality = self.schema_manager.get_quality_report()
            logger.info(f"   Schema Quality: {schema_quality['status']} ({schema_quality['success_rate']}% success)")
        
        return {
            'data': normalized_data,
            'metadata': {
                'url': url,
                'fields': fields,
                'items_extracted': len(normalized_data),
                'execution_time': execution_time,
                'extraction_source': extraction_source,
                'code_cached': code_cached if extraction_source == 'html' else None,
                'schema_quality': schema_quality,
                'pagination_detected': pagination_strategy['type'] if pagination_strategy else None,
                'total_pages_scraped': len(pagination_strategy.get('generated_urls', [])) if pagination_strategy and pagination_strategy.get('generated_urls') else 1,
                'auto_pagination': True if pagination_strategy and 'generated_urls' in pagination_strategy else False,
                'timestamp': time.time()
            },
            'source': extraction_source
        }
    
    async def _scrape_all_pages(
        self,
        page_urls: List[str],
        fields: List[str],
        wait_for_selector: Optional[str] = None,
        batch_size: int = 1,  # Must be 1 for browser-based scraping (shared browser instance)
        pre_warmed_mappings: Optional[Dict[str, Any]] = None  # NEW: Pre-warmed field mappings (for optimization)
    ) -> List[Dict[str, Any]]:
        """
        Scrape all paginated pages sequentially (for auto-pagination).
        Optimized to extract data quickly without full AI code generation per page.
        
        NOTE: batch_size must be 1 when using browser-based fetching because
        the browser instance is shared and cannot navigate to multiple URLs simultaneously.
        
        Args:
            page_urls: List of page URLs to scrape
            fields: Fields to extract
            wait_for_selector: Optional selector to wait for
            batch_size: Number of pages to scrape concurrently (must be 1 for browser mode)
            
        Returns:
            List of raw data objects from all pages
        """
        import asyncio
        
        all_data = []
        first_page_html = None  # NEW: Store HTML from first page for Direct LLM supplementation
        first_page_json_data = None  # NEW: Store JSON data from first page for Direct LLM supplementation
        
        # Process pages sequentially (browser limitation)
        for i in range(0, len(page_urls), batch_size):
            batch = page_urls[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(page_urls) + batch_size - 1) // batch_size
            
            logger.info(f" Processing page {batch_num}/{total_batches}...")
            
            # Fetch pages (must be sequential for browser mode)
            # FIXED: Pass fields to pagination extraction
            # NEW: Return HTML and JSON data from first page for Direct LLM supplementation
            batch_tasks = [
                self._fetch_and_extract_json(url, fields, wait_for_selector, return_html=(i == 0), return_json_data=(i == 0), pre_warmed_mappings=pre_warmed_mappings)
                for url in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collect data from successful results
            for url, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f" Failed to scrape {url}: {result}")
                    continue
                
                # Handle (items, html, json_data) tuple, (items, html) tuple, or items list
                if isinstance(result, tuple):
                    if len(result) == 3:
                        items, html, json_data = result
                        if i == 0 and not first_page_html:
                            first_page_html = html
                        if i == 0 and not first_page_json_data:
                            first_page_json_data = json_data
                        if items and isinstance(items, list):
                            all_data.extend(items)
                            logger.info(f" Page {batch_num}/{total_batches}: extracted {len(items)} items (total so far: {len(all_data)})")
                    elif len(result) == 2:
                        items, html = result
                        if i == 0 and not first_page_html:
                            first_page_html = html
                        if items and isinstance(items, list):
                            all_data.extend(items)
                            logger.info(f" Page {batch_num}/{total_batches}: extracted {len(items)} items (total so far: {len(all_data)})")
                elif result and isinstance(result, list):
                    all_data.extend(result)
                    logger.info(f" Page {batch_num}/{total_batches}: extracted {len(result)} items (total so far: {len(all_data)})")
            
            # Small delay between pages to be respectful to the server
            if i + batch_size < len(page_urls):
                await asyncio.sleep(1.0)  # 1 second delay between pages
        
        logger.info(f" Total items collected from all pages: {len(all_data)}")
        return all_data, first_page_html, first_page_json_data  # NEW: Return HTML and JSON data from first page
    
    async def _fetch_and_extract_json(
        self,
        url: str,
        fields: List[str],  # FIXED: Accept fields parameter
        wait_for_selector: Optional[str] = None,
        return_html: bool = False,  # NEW: Optionally return HTML for Direct LLM supplementation
        return_json_data: bool = False,  # NEW: Optionally return raw JSON data for Direct LLM supplementation
        pre_warmed_mappings: Optional[Dict[str, Any]] = None  # NEW: Pre-warmed field mappings (for pagination optimization)
    ) -> List[Dict[str, Any]]:
        """
        Fetch a single page and extract JSON data quickly.
        Used by auto-pagination to avoid regenerating code for each page.
        
        Args:
            url: URL to fetch
            fields: Fields to extract (FIXED: now required)
            wait_for_selector: Optional selector to wait for
            
        Returns:
            List of data objects extracted from JSON (semantically mapped to requested fields)
        """
        try:
            # Fetch the page (reuses browser if available)
            fetch_result = await self.html_fetcher.fetch(
                url,
                wait_for_selector=wait_for_selector,
                scroll_to_bottom=False,  # No scrolling for paginated pages
                click_load_more=None     # No clicking for paginated pages
            )
            
            html = fetch_result['html']
            
            # For pagination, ONLY extract from embedded JSON (not API responses)
            # Use only embedded sources to get consistent full-page data
            json_results = self.json_detector.detect_and_extract(
                html, 
                url, 
                captured_json=[]  # Don't use captured API responses for pagination
            )
            
            if json_results['json_found'] and json_results['data']:
                # NEW: Analyze JSON structure first (like HTML structure analysis)
                # OPTIMIZATION: Use pre-warmed mappings if available (skip LLM call for pages 2-N)
                json_structure_analysis = None
                
                # Check pre-warmed cache first (for pagination optimization)
                if pre_warmed_mappings:
                    json_structure_analysis = pre_warmed_mappings
                    logger.info(" Using pre-warmed field mappings for pagination (skipped LLM analysis)")
                elif self.json_structure_analyzer:
                    try:
                        logger.info(" Analyzing JSON structure before pagination extraction...")
                        # Build enriched context with website-specific field meanings
                        enriched_context = None
                        if 'chewy' in url.lower() or 'product' in url.lower():
                            enriched_context = "E-commerce product extraction | Extract FULL product names for 'title' field, not short labels like 'Trial Size' or 'Variety Pack'."
                        
                        json_structure_analysis = self.json_structure_analyzer.analyze(
                            json_data=json_results['data'][0] if json_results['data'] else None,
                            url=url,
                            fields=fields,
                            context=enriched_context
                        )
                        
                        if json_structure_analysis.get('field_mappings'):
                            logger.info(f"    JSON structure analyzed (confidence: {json_structure_analysis.get('confidence', 0):.2f}, source: {json_structure_analysis.get('source', 'unknown')})")
                    except Exception as e:
                        logger.warning(f" JSON structure analysis failed: {e}")
                
                # NEW: Pass LLM-discovered field mappings to extraction (universal fallback)
                llm_field_mappings = None
                if json_structure_analysis and json_structure_analysis.get('field_mappings'):
                    llm_field_mappings = json_structure_analysis['field_mappings']
                    logger.info("    Using LLM-discovered field mappings for universal extraction")
                
                # FIXED: Extract with requested fields (semantic mapping + LLM mappings)
                items = self.json_detector.extract_from_json(json_results['data'], fields, llm_field_mappings=llm_field_mappings)
                
                # Use structure analysis to improve extraction if available (post-processing)
                if json_structure_analysis and json_structure_analysis.get('field_mappings') and items:
                    logger.info("    Applying JSON structure analysis to improve extraction...")
                    items = self._apply_json_structure_analysis(items, json_structure_analysis, fields)
                
                if items:
                    logger.info(f" Extracted {len(items)} items from embedded JSON")
                    if return_html or return_json_data:
                        result = items
                        if return_html:
                            result = (result, html) if not isinstance(result, tuple) else (*result, html)
                        if return_json_data:
                            # Get the raw JSON data that was extracted from
                            json_data_for_llm = json_results['data'][0] if json_results.get('data') else None
                            if isinstance(result, tuple):
                                result = (*result, json_data_for_llm)
                            else:
                                result = (result, json_data_for_llm)
                        return result
                    return items
            
            if return_html or return_json_data:
                result = []
                if return_html:
                    result = (result, html)
                if return_json_data:
                    json_data_for_llm = json_results.get('data', [None])[0] if json_results.get('data') else None
                    if isinstance(result, tuple):
                        result = (*result, json_data_for_llm)
                    else:
                        result = (result, json_data_for_llm)
                return result
            return []
            
        except Exception as e:
            logger.warning(f" Error fetching {url}: {e}")
            if return_html or return_json_data:
                result = []
                if return_html:
                    result = (result, "")
                if return_json_data:
                    if isinstance(result, tuple):
                        result = (*result, None)
                    else:
                        result = (result, None)
                return result
            return []
    
    async def scrape_multiple(
        self,
        urls: List[str],
        fields: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
            fields: Fields to extract
            **kwargs: Additional arguments passed to scrape()
            
        Returns:
            List of scrape results
        """
        logger.info(f" Scraping {len(urls)} URLs...")
        
        results = []
        for i, url in enumerate(urls, 1):
            logger.info(f" Processing {i}/{len(urls)}: {url}")
            
            try:
                result = await self.scrape(url, fields, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f" Failed to scrape {url}: {str(e)}")
                results.append({
                    'data': [],
                    'metadata': {
                        'url': url,
                        'error': str(e)
                    },
                    'source': 'error'
                })
        
        logger.info(f" Batch complete: {len(results)} results")
        return results
    
    def _execute_extraction_code(
        self,
        code: str,
        html: str,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute generated extraction code with timeout protection
        
        Args:
            code: Python extraction code
            html: HTML to extract from
            timeout: Maximum execution time in seconds
            
        Returns:
            List of extracted items
        """
        import signal
        from functools import wraps
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {timeout} seconds")
        
        try:
            logger.debug(" Parsing HTML with BeautifulSoup...")
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            logger.debug(f"    Parsed {len(str(soup))} bytes of HTML")
            
            # Create execution namespace
            namespace = {
                'soup': soup,
                'BeautifulSoup': BeautifulSoup,
            }
            
            logger.debug(" Executing generated code...")
            # Set timeout alarm (Unix-like systems only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            except (AttributeError, ValueError):
                # Windows or signal not available - no timeout
                logger.warning(" Timeout not available on this platform")
            
            # Execute the code
            exec(code, namespace)
            logger.debug("    Code execution completed")
            
            # Cancel alarm
            try:
                signal.alarm(0)
            except (AttributeError, ValueError):
                pass
            
            # Call the extract_data function
            if 'extract_data' in namespace:
                logger.debug(" Calling extract_data function...")
                data = namespace['extract_data'](soup)
                logger.debug(f"    extract_data returned {type(data)}")
                
                if not isinstance(data, list):
                    logger.warning(" extract_data didn't return a list, wrapping...")
                    data = [data] if data else []
                
                logger.debug(f"    Returning {len(data)} items")
                return data
            else:
                logger.error(" Generated code doesn't have extract_data function")
                logger.error(f"Available functions: {[k for k in namespace.keys() if callable(namespace[k])]}")
                return []
        
        except TimeoutError as e:
            logger.error(f"⏱ {str(e)}")
            logger.error("   Generated code likely has an infinite loop")
            logger.error(f"   Code preview:\n{code[:1000]}")
            return []
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Special handling for CSS selector syntax errors
            if 'SelectorSyntaxError' in error_type or 'Malformed' in error_msg:
                logger.error(f" CSS Selector Syntax Error: {error_msg}")
                logger.error("    This usually happens when field names contain spaces")
                logger.error("    The LLM should use attribute selectors instead of class selectors")
                logger.error("    Example: Use [data-est-market-value] instead of .est\\. market\\ value")
                
                # Extract the problematic selector from error message
                import re
                selector_match = re.search(r'line \d+:\s*(.+?)(?:\n|$)', error_msg)
                if selector_match:
                    bad_selector = selector_match.group(1).strip()
                    logger.error(f"    Problematic selector: {bad_selector}")
                    logger.error("    Suggestion: Use attribute selectors like [data-field-name] or find actual class names from HTML")
            
            logger.error(f" Code execution failed: {error_msg}")
            logger.error(f"   Error type: {error_type}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            logger.debug(f"   Full code:\n{code}")
            return []
    
    async def _llm_fallback_extraction(
        self,
        html: str,
        json_sources: Dict[str, Any],
        url: str,
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: LLM Direct Extraction Fallback
        
        Used ONLY when:
        - JSON source selection fails to find relevant data
        - HTML code generation fails to extract items
        - As last-resort backup for edge cases
        
        This is the ScrapeGraphAI approach: Convert to Markdown + Direct LLM extraction
        
        Cost: ~$0.10 per page (expensive, but only 10% of pages need this)
        
        Args:
            html: Page HTML
            json_sources: Available JSON sources (for context)
            url: Source URL
            fields: Fields to extract
            
        Returns:
            List of extracted items
        """
        logger.info(" Phase 3: LLM Direct Extraction (last resort)")
        
        # Import html2text for Markdown conversion
        try:
            import html2text
            from urllib.parse import urlparse
        except ImportError:
            logger.error(" html2text not installed. Cannot use LLM fallback.")
            logger.info("   Install with: pip install html2text")
            return []
        
        # Convert HTML to Markdown (easier for LLM to understand)
        try:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.body_width = 0
            
            if url:
                parsed = urlparse(url)
                h.baseurl = f"{parsed.scheme}://{parsed.netloc}"
            
            markdown = h.handle(html)
            logger.info(f"    Converted HTML to Markdown ({len(markdown)} chars)")
        except Exception as e:
            logger.warning(f"    Markdown conversion failed: {e}")
            markdown = html
        
        # Build content for LLM (Markdown + JSON sources)
        content = f"=== PAGE CONTENT ===\n{markdown[:20000]}\n\n"  # Limit to 20K chars
        
        if json_sources:
            content += "=== AVAILABLE JSON DATA ===\n"
            for name, data in list(json_sources.items())[:5]:  # Limit to 5 sources
                try:
                    json_str = json.dumps(data, indent=2, default=str)[:1000]  # Limit each source
                    content += f"\n{name}:\n{json_str}\n"
                except:
                    pass
        
        # Get user's goal from context manager
        user_goal = "extract structured data"
        if hasattr(self, 'context_manager') and self.context_manager.context:
            user_goal = self.context_manager.context.goal
        
        # Build LLM prompt for direct extraction
        prompt = f"""Extract structured data from this webpage.

URL: {url}
USER'S GOAL: {user_goal}
FIELDS TO EXTRACT: {', '.join(fields) if fields else 'auto-detect all relevant fields'}

{content}

TASK:
Extract ALL matching items as a JSON array of objects.
Each object should have the requested fields.

Respond with ONLY valid JSON (no explanation):
[
  {{"field1": "value1", "field2": "value2", ...}},
  {{"field1": "value1", "field2": "value2", ...}},
  ...
]

If you can't find the data, return an empty array: []
"""
        
        try:
            # Call LLM for direct extraction
            response = litellm.completion(
                model=self.model_name or "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a web scraping data extractor. Extract structured data as JSON arrays. Be accurate and thorough."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            result = json.loads(content) if isinstance(content, str) else content
            
            logger.debug(f"   LLM response type: {type(result)}")
            
            # Extract array from result
            extracted_items = []
            if isinstance(result, list):
                extracted_items = result
            elif isinstance(result, dict):
                # Find the largest array in the dict
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > len(extracted_items):
                        extracted_items = value
            
            if extracted_items:
                logger.info(f"    LLM fallback extracted {len(extracted_items)} items")
                logger.info(f"    Cost: ~${0.10:.2f} (this is expensive, only use for edge cases)")
                return extracted_items
            else:
                logger.warning("    LLM returned no items")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"    LLM returned invalid JSON: {e}")
            if 'content' in locals():
                logger.debug(f"   Response: {content[:500]}")
            return []
        except Exception as e:
            logger.error(f"    LLM fallback failed: {type(e).__name__}: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.code_cache.get_stats()
    
    def clear_cache(self) -> bool:
        """Clear code cache"""
        return self.code_cache.clear()
    
    def export_cache(self, export_path: str) -> bool:
        """Export cache to file"""
        return self.code_cache.export_cache(export_path)
    
    def import_cache(self, import_path: str) -> bool:
        """Import cache from file"""
        return self.code_cache.import_cache(import_path)
    
    def _apply_json_structure_analysis(
        self,
        json_data: List[Dict[str, Any]],
        structure_analysis: Dict[str, Any],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply JSON structure analysis to improve extraction
        
        Uses field mappings from structure analysis to ensure correct field extraction
        """
        if not json_data or not structure_analysis.get('field_mappings'):
            return json_data
        
        field_mappings = structure_analysis['field_mappings']
        improved_data = []
        
        for item in json_data:
            improved_item = item.copy()
            
            # Apply field mappings
            for requested_field in fields:
                if requested_field in field_mappings:
                    mapping = field_mappings[requested_field]
                    json_key = mapping.get('json_key')
                    path = mapping.get('path', json_key)
                    confidence = mapping.get('confidence', 0.0)
                    
                    # Only apply if confidence is high enough
                    if confidence >= 0.6:
                        # Try to get value from path
                        value = self._get_nested_value(item, path)
                        if value is not None:
                            # Ensure field is set (may override semantic mapping)
                            improved_item[requested_field] = value
            
            improved_data.append(improved_item)
        
        return improved_data
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """
        Get nested value from object using dot-separated path
        """
        keys = path.split('.')
        current = obj
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _extract_smart_html_sample(
        self, 
        html: str, 
        structure_analysis: Optional[Dict[str, Any]],
        fields: Optional[List[str]] = None,
        url: Optional[str] = None
    ) -> str:
        """
        Extract a dynamically-sized HTML sample using the Smart HTML Sampler.
        
        This is universal and adapts to each website's structure:
        - Small product cards: includes 5 elements
        - Medium articles: includes 3 elements
        - Large content: includes 2 elements
        - Verifies field coverage to ensure completeness
        """
        detected_pattern = structure_analysis.get('detected_pattern') if structure_analysis else None
        
        # Extract domain from URL for caching
        domain = None
        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
        
        # Extract sibling_analysis from structure_analysis
        sibling_analysis = None
        if structure_analysis and isinstance(structure_analysis, dict):
            sibling_analysis = structure_analysis.get('sibling_analysis')
        
        # Use smart sampler to determine optimal sample (now with context-block support!)
        result = self.smart_sampler.extract_optimal_sample(
            html=html,
            detected_pattern=detected_pattern,
            fields=fields,
            domain=domain,
            sibling_analysis=sibling_analysis  # NEW - enables context-block extraction
        )
        
        return result['sample_html']
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.html_fetcher.close()
        logger.info(" Universal Scraper closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Note: __exit__ can't be async, so we can't await close() here
        # Users should use async context manager or call await scraper.close() explicitly
        import warnings
        warnings.warn(
            "Synchronous context manager deprecated. Use 'async with' or call 'await scraper.close()' explicitly.",
            DeprecationWarning,
            stacklevel=2
        )


# Convenience function for simple usage
def scrape(
    url: str,
    fields: List[str],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    proxy_config: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for simple scraping
    
    Args:
        url: Target URL
        fields: Fields to extract
        api_key: AI API key
        model_name: AI model name
        proxy_config: Proxy configuration
        **kwargs: Additional arguments
        
    Returns:
        Scrape result dict
    """
    with UniversalScraper(
        api_key=api_key,
        model_name=model_name,
        proxy_config=proxy_config,
        **kwargs
    ) as scraper:
        return scraper.scrape(url, fields)

