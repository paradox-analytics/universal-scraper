"""
Hybrid Fetcher - Intelligently chooses best fetching method
JSON-Forward Architecture: API Cache → Browser → Static HTML
"""

import logging
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .html_fetcher import HTMLFetcher
from .api_cache import APICache

logger = logging.getLogger(__name__)

# Lazy import browser fetchers (only if needed)
BrowserFetcher = None
CamoufoxFetcher = None
WebUnblockerFetcher = None


def _get_browser_fetcher(use_camoufox: bool = False):
    """Lazy import BrowserFetcher or CamoufoxFetcher"""
    global BrowserFetcher, CamoufoxFetcher
    
    if use_camoufox:
        if CamoufoxFetcher is None:
            try:
                from .camoufox_fetcher import CamoufoxFetcher as CF
                CamoufoxFetcher = CF
                logger.info(" Camoufox fetcher loaded")
            except ImportError:
                logger.warning(" Camoufox not available, falling back to Playwright")
                return _get_browser_fetcher(use_camoufox=False)
        return CamoufoxFetcher
    else:
        if BrowserFetcher is None:
            try:
                from .browser_fetcher import BrowserFetcher as BF
                BrowserFetcher = BF
            except ImportError:
                logger.warning("Browser fetcher not available")
                return None
        return BrowserFetcher


class HybridFetcher:
    """
    Intelligent fetcher with automatic fallback strategy
    
    Strategy:
    1. Check API cache (fastest, if available)
    2. Try static HTML (fast, works for server-rendered sites)
    3. Check if JS needed (heuristic)
    4. Use browser if needed (slower but complete)
    5. Cache discovered APIs for next time
    """
    
    # Indicators that JavaScript is required
    JS_INDICATORS = [
        # Framework indicators
        'react', 'vue', 'angular', 'next.js', 'nuxt',
        '__NEXT_DATA__', 'ng-app', 'v-app', 'reactRoot',
        
        # Empty body indicators
        'Loading...', 'Please wait', 'Rendering',
        
        # API-driven indicators
        'window.__INITIAL_STATE__', 'window.__APOLLO_STATE__',
        'data-reactroot', 'data-vue-app'
    ]
    
    # Domains known to require JS
    JS_REQUIRED_DOMAINS = [
        'leafly.com',
        'weedmaps.com',
        # Add more as discovered
    ]
    
    def __init__(
        self,
        proxy_config: Optional[Dict[str, str]] = None,
        proxy_manager: Optional[Any] = None,  # NEW: ProxyManager for rotation
        enable_cache: bool = True,
        enable_warming: bool = True,
        cache_dir: str = "./cache",
        headless: bool = True,
        browser_timeout: int = 60000,
        force_mode: Optional[str] = None,  # 'static', 'browser', or None for auto
        use_camoufox: bool = True,  # NEW: Use Camoufox instead of Playwright (better anti-detection)
        web_unblocker_api_key: Optional[str] = None,  # NEW: Bright Data Web Unblocker API key
        web_unblocker_zone: str = "web_unlocker1"  # NEW: Web Unblocker zone name
    ):
        """
        Initialize Hybrid Fetcher
        
        Args:
            proxy_config: Static proxy configuration (deprecated, use proxy_manager)
            proxy_manager: ProxyManager instance for per-request rotation (recommended)
            enable_cache: Enable code and API caching
            enable_warming: Enable session warming
            cache_dir: Cache directory
            headless: Run browser in headless mode
            browser_timeout: Browser navigation timeout
            force_mode: Force specific mode or None for auto-detection
            use_camoufox: Use Camoufox for better anti-detection (recommended)
        """
        self.proxy_config = proxy_config
        self.proxy_manager = proxy_manager  # NEW: Store ProxyManager
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.headless = headless
        self.browser_timeout = browser_timeout
        self.force_mode = force_mode
        self.use_camoufox = use_camoufox  # NEW: Store Camoufox preference
        self.web_unblocker_api_key = web_unblocker_api_key  # NEW: Web Unblocker API key
        self.web_unblocker_zone = web_unblocker_zone  # NEW: Web Unblocker zone
        
        # Initialize browser_fetcher as None (lazy initialization) - MUST be before _init_web_unblocker
        self.browser_fetcher = None
        
        # Initialize Web Unblocker fetcher if API key provided
        self.web_unblocker_fetcher = None
        if web_unblocker_api_key:
            self._init_web_unblocker()
        
        # Also check environment variables for auto-enable
        import os
        env_api_key = os.environ.get('WEB_UNBLOCKER_API_KEY') or os.environ.get('BRIGHT_DATA_API_KEY')
        if not self.web_unblocker_fetcher and env_api_key:
            logger.info(" Auto-enabling Web Unblocker from environment variable...")
            self.web_unblocker_api_key = env_api_key
            self._init_web_unblocker()
        
        # Initialize static HTML fetcher (always available)
        self.html_fetcher = HTMLFetcher(
            proxy_config=self.proxy_config,
            proxy_manager=self.proxy_manager,  # NEW: Pass ProxyManager
            enable_warming=enable_warming
        )
        
        # Initialize API cache
        self.api_cache = APICache(cache_dir=f"{self.cache_dir}/apis") if enable_cache else None
        
        # Statistics
        self.stats = {
            'api_cache_hits': 0,
            'static_html_success': 0,
            'browser_fallback': 0,
            'apis_discovered': 0
        }
        
        browser_type = " Camoufox" if self.use_camoufox else " Playwright"
        logger.info(" Hybrid Fetcher initialized")
        logger.info(f"   Mode: {'Auto-detect' if not self.force_mode else self.force_mode}")
        logger.info(f"   Browser: {browser_type}")
        logger.info(f"   API Cache: {'Enabled' if enable_cache else 'Disabled'}")
        logger.info(f"   Web Unblocker: {'Enabled' if self.web_unblocker_fetcher else 'Disabled'}")
    
    def _init_web_unblocker(self):
        """Initialize Web Unblocker fetcher (can be called lazily when blocking detected)"""
        if self.web_unblocker_fetcher:
            return  # Already initialized
        
        if not self.web_unblocker_api_key:
            return
        
        try:
            from .web_unblocker_fetcher import WebUnblockerFetcher
            self.web_unblocker_fetcher = WebUnblockerFetcher(
                api_key=self.web_unblocker_api_key,
                zone=self.web_unblocker_zone
            )
            logger.info(f" Web Unblocker enabled (zone: {self.web_unblocker_zone})")
        except ImportError:
            logger.warning(" Web Unblocker requested but module not available")
        except Exception as e:
            logger.warning(f" Failed to initialize Web Unblocker: {e}")
    
    async def fetch(
        self,
        url: str,
        fields: Optional[list] = None,
        wait_for_selector: Optional[str] = None,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch URL with intelligent method selection
        
        Args:
            url: Target URL
            fields: Fields to extract (for API cache matching)
            wait_for_selector: Selector to wait for (browser mode)
            scroll_to_bottom: Scroll to trigger lazy loading
            click_load_more: Selector for "Load More" button
            
        Returns:
            Dict with 'html', 'url', 'fetch_method', 'apis' keys
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        
        logger.info(f" Fetching: {url}")
        logger.info(f" Detection mode: {self.force_mode or 'auto'}")
        
        # Force mode if specified
        if self.force_mode == 'browser':
            browser_result = await self._fetch_with_browser(
                url,
                wait_for_selector=wait_for_selector,
                scroll_to_bottom=scroll_to_bottom,
                click_load_more=click_load_more
            )
            
            # Check if browser fetch was blocked (Kasada, etc.)
            html_content = browser_result.get('html', '')
            logger.info(f" Checking if blocked: HTML size={len(html_content)} bytes, URL={url}")
            is_blocked = self._is_blocked(html_content, url)
            logger.info(f" Blocking detection result: {is_blocked}")
            
            if is_blocked:
                logger.warning(" Browser fetch appears blocked (anti-bot protection detected)")
                logger.warning(f"   HTML size: {len(html_content)} bytes (suspiciously small)")
                logger.warning(f"   URL: {url}")
                
                # Try to auto-initialize Web Unblocker if not already initialized
                if not self.web_unblocker_fetcher:
                    self._init_web_unblocker()
                
                # Fall back to Web Unblocker if available
                if self.web_unblocker_fetcher:
                    logger.info(" Falling back to Bright Data Web Unblocker...")
                    logger.info(f"   Attempting to bypass blocking for: {url}")
                    try:
                        unblocker_result = await self.web_unblocker_fetcher.fetch_async(
                            url,
                            wait_time=wait_for_selector and 5 or 0
                        )
                        
                        unblocker_html = unblocker_result.get('html', '')
                        logger.info(f"   Web Unblocker response: {len(unblocker_html)} bytes")
                        
                        # Check if Web Unblocker succeeded
                        if not self._is_blocked(unblocker_html, url):
                            logger.info(" Web Unblocker fetch successful!")
                            logger.info(f"   HTML size: {len(unblocker_html)} bytes")
                            unblocker_result['fetch_method'] = 'web_unblocker'
                            unblocker_result['apis'] = {}
                            unblocker_result['captured_json'] = []
                            return unblocker_result
                        else:
                            logger.warning(f" Web Unblocker also appears blocked (HTML size: {len(unblocker_html)} bytes)")
                            logger.warning(f"   Response preview: {unblocker_html[:200]}...")
                    except Exception as e:
                        logger.error(f" Web Unblocker fallback failed: {e}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                        # Continue with browser result (even if blocked)
                else:
                    logger.warning("ℹ Web Unblocker not configured - skipping fallback")
                    logger.warning("   Configure webUnblockerApiKey in input to enable fallback")
            
            return browser_result
        elif self.force_mode == 'static':
            return self._fetch_with_static(url)
        
        # STEP 1: Check API cache (fastest!)
        if self.api_cache:
            cached_apis = self.api_cache.get_apis(domain)
            if cached_apis:
                logger.info(f" Found {len(cached_apis)} cached APIs for {domain}")
                self.stats['api_cache_hits'] += 1
                # Note: You'd implement direct API calls here based on fields
                # For now, we'll still fetch the page but flag APIs as available
        
        # STEP 2: Try static HTML first (fast path)
        logger.info(" Trying static HTML fetch...")
        static_result = self._fetch_with_static(url)
        
        # STEP 3: Check if JavaScript is needed
        needs_js = self._detect_js_required(static_result['html'], domain)
        
        if not needs_js:
            logger.info(" Static HTML sufficient")
            self.stats['static_html_success'] += 1
            return static_result
        
        # STEP 4: Fall back to browser
        logger.info(" JavaScript required, using browser...")
        self.stats['browser_fallback'] += 1
        
        browser_result = await self._fetch_with_browser(
            url,
            wait_for_selector=wait_for_selector,
            scroll_to_bottom=scroll_to_bottom,
            click_load_more=click_load_more
        )
        
        # STEP 4.5: Check if browser fetch was blocked (Kasada, etc.)
        html_content = browser_result.get('html', '')
        logger.info(f" Checking if blocked: HTML size={len(html_content)} bytes, URL={url}")
        is_blocked = self._is_blocked(html_content, url)
        logger.info(f" Blocking detection result: {is_blocked}")
        
        if is_blocked:
            logger.warning(" Browser fetch appears blocked (anti-bot protection detected)")
            logger.warning(f"   HTML size: {len(html_content)} bytes (suspiciously small)")
            logger.warning(f"   URL: {url}")
            
            # Try to auto-initialize Web Unblocker if not already initialized
            if not self.web_unblocker_fetcher:
                self._init_web_unblocker()
            
            # Fall back to Web Unblocker if available
            if self.web_unblocker_fetcher:
                logger.info(" Falling back to Bright Data Web Unblocker...")
                logger.info(f"   Attempting to bypass blocking for: {url}")
                try:
                    unblocker_result = await self.web_unblocker_fetcher.fetch_async(
                        url,
                        wait_time=wait_for_selector and 5 or 0
                    )
                    
                    unblocker_html = unblocker_result.get('html', '')
                    logger.info(f"   Web Unblocker response: {len(unblocker_html)} bytes")
                    
                    # Check if Web Unblocker succeeded
                    if not self._is_blocked(unblocker_html, url):
                        logger.info(" Web Unblocker fetch successful!")
                        logger.info(f"   HTML size: {len(unblocker_html)} bytes")
                        unblocker_result['fetch_method'] = 'web_unblocker'
                        unblocker_result['apis'] = {}
                        unblocker_result['captured_json'] = []
                        return unblocker_result
                    else:
                        logger.warning(f" Web Unblocker also appears blocked (HTML size: {len(unblocker_html)} bytes)")
                        logger.warning(f"   Response preview: {unblocker_html[:200]}...")
                except Exception as e:
                    logger.error(f" Web Unblocker fallback failed: {e}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    # Continue with browser result (even if blocked)
            else:
                logger.warning("ℹ Web Unblocker not configured - skipping fallback")
                logger.warning("   Configure webUnblockerApiKey in input to enable fallback")
        
        # STEP 5: Cache discovered APIs for next time
        if browser_result.get('apis') and self.api_cache:
            self.api_cache.store_discovered_apis(
                domain,
                browser_result['apis'],
                url
            )
            self.stats['apis_discovered'] += len(browser_result['apis'])
            logger.info(f" Cached {len(browser_result['apis'])} APIs for future use")
        
        return browser_result
    
    def _is_blocked(self, html: str, url: str = "") -> bool:
        """
        Detect if page is blocked by anti-bot protection (Kasada, Cloudflare, etc.)
        
        Args:
            html: HTML content to check
            url: Optional URL for domain-specific checks
            
        Returns:
            True if blocked, False otherwise
        """
        if not html or len(html) < 500:
            logger.info(f" Blocked: HTML too small ({len(html) if html else 0} bytes)")
            return True
        
        html_lower = html.lower()
        
        # E-commerce sites should have much larger pages (product listings are typically 50KB+)
        # Very small pages (< 5KB) on e-commerce domains are suspicious
        ecommerce_domains = ['chewy.com', 'amazon.com', 'ebay.com', 'etsy.com', 'walmart.com']
        is_ecommerce = any(domain in url.lower() for domain in ecommerce_domains)
        if is_ecommerce and len(html) < 5000:
            # Check if it's actually a valid product page (has product indicators)
            product_indicators = ['product-card', 'item-title', 'add-to-cart', 'product-title', 'product-name']
            has_products = any(indicator in html_lower for indicator in product_indicators)
            if not has_products:
                logger.warning(f"  Suspiciously small HTML ({len(html)} bytes) for e-commerce site without product indicators")
                logger.info(f" Blocked: E-commerce site ({url}) with small HTML ({len(html)} bytes) and no product indicators")
                return True
        
        # Kasada indicators
        kasada_indicators = ['kasada', 'kpsdk', 'ips.js', 'window.kpsdk', 'kasada.io']
        has_kasada = any(indicator in html_lower for indicator in kasada_indicators)
        if has_kasada:
            # Check if it's just the challenge script (small HTML)
            if len(html) < 2000:
                logger.info(f" Blocked: Kasada detected (HTML size: {len(html)} bytes)")
                return True
        
        # Cloudflare indicators
        cloudflare_indicators = ['cf-browser-verification', 'checking your browser', 'cloudflare', 'ray id']
        if any(indicator in html_lower for indicator in cloudflare_indicators):
            if len(html) < 5000:
                return True
        
        # Generic blocking indicators
        blocking_indicators = ['access denied', 'blocked', 'forbidden', '403', 'captcha', 'bot detection']
        if any(indicator in html_lower for indicator in blocking_indicators):
            if len(html) < 3000:
                return True
        
        return False
    
    def _fetch_with_static(self, url: str) -> Dict[str, Any]:
        """Fetch with static HTML fetcher"""
        result = self.html_fetcher.fetch(url)
        result['fetch_method'] = 'static'
        result['apis'] = {}
        result['captured_json'] = []  # No API capture in static mode
        return result
    
    async def _fetch_with_browser(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch with browser"""
        # Lazy load browser fetcher
        if self.browser_fetcher is None:
            BF = _get_browser_fetcher(use_camoufox=self.use_camoufox)
            if BF is None:
                raise ImportError(
                    f"Browser fetching not available. Install with: pip install {'camoufox' if self.use_camoufox else 'playwright'}"
                )
            
            # Initialize browser fetcher with appropriate parameters
            if self.use_camoufox:
                # Camoufox fetcher (simpler constructor)
                self.browser_fetcher = BF(
                    headless=self.headless,
                    proxy_config=self.proxy_config,
                    proxy_manager=self.proxy_manager,  # NEW: Pass ProxyManager for rotation
                    timeout=self.browser_timeout
                )
            else:
                # Playwright browser fetcher (original) - doesn't support proxy_manager yet
                self.browser_fetcher = BF(
                    headless=self.headless,
                    proxy_config=self.proxy_config,
                    timeout=self.browser_timeout,
                    capture_api_requests=True
                )
            
            await self.browser_fetcher._launch_browser()
        
        result = await self.browser_fetcher.fetch(
            url,
            wait_for_selector=wait_for_selector,
            scroll_to_bottom=scroll_to_bottom,
            click_load_more=click_load_more
        )
        result['fetch_method'] = 'browser'
        
        # CRITICAL FIX: Map json_data to captured_json for the scraper pipeline
        # The camoufox_fetcher captures API responses as 'json_data', but the
        # scraper.py expects 'captured_json' for JSON-first extraction
        if 'json_data' in result:
            json_data = result['json_data']
            if json_data:
                logger.info(f" Captured {len(json_data)} JSON API responses from browser")
                # Extract just the data portion for captured_json
                result['captured_json'] = [item.get('data') for item in json_data if item.get('data')]
                result['apis'] = {item.get('url', f'api_{i}'): item.get('data') for i, item in enumerate(json_data) if item.get('data')}
            else:
                result['captured_json'] = []
                result['apis'] = {}
        else:
            result['captured_json'] = []
            result['apis'] = {}
        
        return result
    
    def _detect_js_required(self, html: str, domain: str) -> bool:
        """
        Detect if JavaScript rendering is required
        
        IMPROVED: Only checks for JS indicators in <script> tags and validates
        that the page actually lacks content, not just contains framework names.
        
        Args:
            html: HTML content
            domain: Domain name
            
        Returns:
            True if JS is likely required
        """
        soup = BeautifulSoup(html, 'html.parser')
        body = soup.find('body')
        
        if not body:
            logger.info(" No body tag found")
            return True
        
        text_content = body.get_text(strip=True)
        
        # CRITICAL FIX: Check for STRUCTURED CONTENT FIRST!
        # Even if total text is low, structured elements indicate a good static page
        content_tags = soup.find_all(['article', 'main', 'ul', 'ol', 'table', 'p'])
        meaningful_content = sum(len(tag.get_text(strip=True)) for tag in content_tags[:20])
        
        if meaningful_content > 2000:
            # Page has substantial structured content, likely static HTML is fine
            logger.info(f" Found {meaningful_content} chars of structured content, static HTML sufficient")
            return False
        
        # NOW check if body is suspiciously empty (only if structured content check failed)
        if len(text_content) < 500:
            logger.info(" Body has minimal content (< 500 chars), likely JS-rendered")
            return True
        
        # Check for loading indicators in visible text
        if any(indicator in text_content for indicator in ['Loading', 'Please wait', 'Rendering']):
            logger.info(" Loading indicators found")
            return True
        
        # IMPROVED: Only check for framework indicators in <script> tags, not entire HTML
        # This prevents false positives from page content mentioning "React" or "Angular"
        script_tags = soup.find_all('script')
        script_content = ' '.join([script.string or '' for script in script_tags if script.string])
        script_content_lower = script_content.lower()
        
        # Framework-specific checks (only in scripts)
        framework_indicators = [
            '__NEXT_DATA__', '__NUXT__', 'window.__INITIAL_STATE__',
            'window.__APOLLO_STATE__', 'reactRoot', 'ng-app', 'v-app'
        ]
        
        for indicator in framework_indicators:
            if indicator.lower() in script_content_lower:
                logger.info(f" Detected JS framework indicator in scripts: {indicator}")
                return True
        
        # Check data attributes that indicate JS frameworks (but only if content is sparse)
        if meaningful_content < 1000:
            data_attrs = ['data-reactroot', 'data-vue-app', 'ng-app', 'v-app']
            html_lower = html.lower()
            for attr in data_attrs:
                if attr in html_lower:
                    logger.info(f" Detected framework attribute: {attr}")
                    return True
        
        # Check known JS-required domains
        for js_domain in self.JS_REQUIRED_DOMAINS:
            if js_domain in domain:
                logger.info(f" Domain {domain} known to require JS")
                return True
        
        # If we got here, static HTML is probably sufficient
        logger.info(" Static HTML appears sufficient (no JS indicators, good content)")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetching statistics"""
        return dict(self.stats)
    
    def get_api_cache_stats(self) -> Dict[str, Any]:
        """Get API cache statistics"""
        if self.api_cache:
            return self.api_cache.get_stats()
        return {}
    
    async def close(self) -> None:
        """Clean up resources"""
        if self.html_fetcher:
            self.html_fetcher.close()
        
        if self.browser_fetcher:
            await self.browser_fetcher.close()
        
        logger.info(" Hybrid Fetcher closed")
        logger.info(f" Session stats: {self.stats}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Note: This is for sync usage only (local scripts)
        # For async usage, use async with
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self.close())
        except:
            pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function
def fetch_hybrid(
    url: str,
    proxy_config: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for hybrid fetching
    
    Args:
        url: Target URL
        proxy_config: Proxy configuration
        **kwargs: Additional arguments for fetch()
        
    Returns:
        Fetch result dict
    """
    with HybridFetcher(proxy_config=proxy_config) as fetcher:
        return fetcher.fetch(url, **kwargs)

