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
from .json_detector import JSONDetector

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
                logger.info("🦊 Camoufox fetcher loaded")
            except ImportError:
                logger.warning("⚠️ Camoufox not available, falling back to Playwright")
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

    # No hardcoded domains - detection is fully dynamic and universal

    # Domain success cache (class-level to persist across requests)
    _success_cache = {}  # domain -> {'method': 'browser', 'timestamp': 12345}

    def __init__(
        self,
        proxy_config: Optional[Dict[str, str]] = None,
        proxy_manager: Optional[Any] = None,  # NEW: ProxyManager for rotation
        enable_cache: bool = True,
        enable_warming: bool = True,
        cache_dir: str = "./cache",
        headless: bool = True,
        browser_timeout: int = 120000,  # Default 120 seconds for slow-loading pages
        force_mode: Optional[str] = None,  # 'static', 'browser', or None for auto
        use_camoufox: bool = True,  # NEW: Use Camoufox instead of Playwright (better anti-detection)
        web_unblocker_api_key: Optional[str] = None,  # NEW: Bright Data Web Unblocker API key
        web_unblocker_zone: str = "web_unlocker1",  # NEW: Web Unblocker zone name
        web_unblocker_customer_id: Optional[str] = None  # NEW: Web Unblocker customer ID
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
        self.web_unblocker_customer_id = web_unblocker_customer_id  # NEW: Web Unblocker customer ID

        # Initialize Web Unblocker fetcher if API key provided
        self.web_unblocker_fetcher = None
        if web_unblocker_api_key:
            try:
                from .web_unblocker_fetcher import WebUnblockerFetcher
                self.web_unblocker_fetcher = WebUnblockerFetcher(
                    api_key=web_unblocker_api_key,
                    zone=web_unblocker_zone
                )
                logger.info(f"🌐 Web Unblocker enabled (zone: {web_unblocker_zone})")
            except Exception as e:
                logger.warning(f"⚠️ Web Unblocker requested but module not available: {e}")

        # Initialize static HTML fetcher (always available)
        self.html_fetcher = HTMLFetcher(
            proxy_config=proxy_config,
            proxy_manager=proxy_manager,  # NEW: Pass ProxyManager
            enable_warming=enable_warming
        )

        # Initialize API cache
        self.api_cache = APICache(cache_dir=f"{cache_dir}/apis") if enable_cache else None

        # Browser fetcher (lazy loaded)
        self.browser_fetcher = None

        # Statistics
        self.stats = {
            'api_cache_hits': 0,
            'static_html_success': 0,
            'browser_fallback': 0,
            'apis_discovered': 0
        }

        browser_type = "🦊 Camoufox" if use_camoufox else "🎭 Playwright"
        logger.info("🔀 Hybrid Fetcher initialized")
        logger.info(f"   Mode: {'Auto-detect' if not force_mode else force_mode}")
        logger.info(f"   Browser: {browser_type}")
        logger.info(f"   API Cache: {'Enabled' if enable_cache else 'Disabled'}")
        logger.info(f"   Web Unblocker: {'Enabled' if self.web_unblocker_fetcher else 'Disabled'}")

        # Initialize unblocker log for current request
        self.unblocker_log = []

    def _log_unblocker(self, message: str):
        """Add a message to the unblocker log"""
        entry = {
            'timestamp': time.time(),
            'message': message
        }
        self.unblocker_log.append(entry)
        logger.info(f"🛡️ [Unblocker] {message}")

    async def fetch(
        self,
        url: str,
        mode: str = "hybrid",
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
        browser_config: Optional[Dict[str, Any]] = None  # NEW: Allow overriding browser config
    ) -> Dict[str, Any]:
        """
        Fetch content from URL using hybrid strategy

        Args:
            url: Target URL
            mode: Fetch mode ('hybrid', 'static', 'browser')
            scroll_to_bottom: Scroll to bottom for infinite scroll
            click_load_more: Selector for load more button
            wait_for_selector: Wait for specific element
            browser_config: Optional browser configuration override (from strategy)

        Returns:
            Dict with 'html', 'status', 'url', 'source', 'captured_json'
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        logger.info(f"🎯 Fetching: {url}")
        self.unblocker_log = []  # Reset log for new fetch
        self._log_unblocker(f"Starting fetch for {url}")

        # Force browser mode for specific domains known to block static requests
        # Home Depot returns 404/403 for static requests even with Web Unblocker
        effective_force_mode = self.force_mode
        if "homedepot.com" in url:
             logger.info("🎯 Home Depot detected: Will attempt Universal Fast Path (JSON-First) before Browser")
             # We no longer force 'browser' mode immediately. We try the Fast Path first.
             # If Fast Path fails, the standard fallback logic will handle it.

        logger.info(f"🔍 Detection mode: {effective_force_mode or 'auto'}")

        # Check success cache
        # CRITICAL: For Home Depot and other sensitive sites, we ALWAYS use browser mode
        # for reliability even if a static fetch was previously "successful"
        is_sensitive_site = "homedepot.com" in url or "producthunt.com" in url

        if not effective_force_mode and domain in self._success_cache and not is_sensitive_site:
            cached = self._success_cache[domain]
            # Cache valid for 1 hour
            if time.time() - cached['timestamp'] < 3600:
                effective_force_mode = cached['method']
                self._log_unblocker(f"Using cached success method: {effective_force_mode}")

        # STEP 0: UNIVERSAL FAST PATH (JSON-First)
        # Attempt to extract data from initial static response (or Web Unblocker)
        # This bypasses the slow browser render if high-quality JSON is available.

        # Only try if we're not strictly forced to browser (and haven't already tried static)
        fast_path_result = None
        if effective_force_mode != 'browser':
            self._log_unblocker("🚀 Starting Universal Fast Path (JSON-First)...")

            # 1. Fetch initial HTML (Static or Web Unblocker)
            initial_res = None
            if self.web_unblocker_fetcher:
                try:
                    self._log_unblocker("   Using Web Unblocker for Fast Path...")
                    initial_res = await self.web_unblocker_fetcher.fetch_async(url)
                    initial_res['fetch_method'] = 'web_unblocker_fast_path'
                except Exception as e:
                    self._log_unblocker(f"   ⚠️ Web Unblocker Fast Path failed: {e}")

            if not initial_res:
                # Fallback to standard static fetch
                # CRITICAL: Skip naive static fetch for domains known to hang/tarpit
                if "homedepot.com" in url:
                    self._log_unblocker("   ⚠️ Skipping naive static fetch for Home Depot (avoids timeout hang)")
                else:
                    self._log_unblocker("   Using Static Fetcher for Fast Path...")
                    initial_res = self._fetch_with_static(url)
                    initial_res['fetch_method'] = 'static_fast_path'

            # 2. Analyze for High-Quality JSON
            html_content = initial_res.get('html', '') if initial_res else ''
            if html_content and len(html_content) > 1000:
                detector = JSONDetector()
                # We specifically look for "High Quality" data (products, items, hydration state)
                detection_result = detector.detect_and_extract(html_content, url)

                if detection_result['json_found']:
                    # Check if the found data is actually useful (contains items/products)
                    # The detector now includes detailed scoring/analysis logs
                    self._log_unblocker(f"   ✅ JSON Detected! Sources: {detection_result['sources']}")

                    # Store these results to return immediately
                    fast_path_result = initial_res
                    fast_path_result['captured_json'] = detection_result['data'] # Use the extracted data blobs
                    fast_path_result['json_recommended'] = True
                    fast_path_result['extraction_mode'] = 'json' # Hint to scraper

                    # Update success cache to prefer this method
                    self._success_cache[domain] = {'method': 'static', 'timestamp': time.time()}
                    self._log_unblocker("   🚀 Fast Path Successful! Skipping browser.")

                    # Log unblocker entries
                    fast_path_result['unblocker_log'] = self.unblocker_log
                    return fast_path_result
                else:
                    self._log_unblocker("   ⚠️ No significant JSON found in Fast Path.")
            else:
                 self._log_unblocker("   ⚠️ Fast Path response empty or blocked.")

            self._log_unblocker("   ⬇️ Proceeding to Standard Strategies...")

        # STEP 1: Smart Strategy Orchestration
        # We try different strategies in order of increasing cost/complexity
        strategies = []

        if effective_force_mode == 'browser':
            # If browser mode is forced, we only try browser-based strategies
            strategies = [
                {'name': 'Browser (Standard)', 'method': 'browser', 'use_unblocker': False},
                {'name': 'Browser (Web Unblocker Proxy)', 'method': 'browser', 'use_unblocker': True}
            ]
        elif effective_force_mode == 'static':
            strategies = [{'name': 'Static HTML', 'method': 'static', 'use_unblocker': False}]
        else:
            # Auto-detect: try static first, then browser
            strategies = [
                {'name': 'Static HTML', 'method': 'static', 'use_unblocker': False},
                {'name': 'Browser (Standard)', 'method': 'browser', 'use_unblocker': False},
                {'name': 'Browser (Web Unblocker Proxy)', 'method': 'browser', 'use_unblocker': True}
            ]

        last_result = None
        for strategy in strategies:
            self._log_unblocker(f"Trying strategy: {strategy['name']}")

            try:
                if strategy['method'] == 'static':
                    # Try static fetch
                    if self.web_unblocker_fetcher and strategy['use_unblocker']:
                        # This case is handled by the proactive check above, but here for completeness
                        res = await self.web_unblocker_fetcher.fetch_async(url)
                        res['fetch_method'] = 'web_unblocker'
                    else:
                        res = self._fetch_with_static(url)
                else:
                    # Try browser fetch
                    # If strategy says use_unblocker, we ensure it's passed to the browser fetcher
                    # The browser fetcher (CamoufoxFetcher) already checks self.web_unblocker_api_key
                    # but we can force it if needed.
                    res = await self._fetch_with_browser(
                        url,
                        wait_for_selector=wait_for_selector,
                        scroll_to_bottom=scroll_to_bottom,
                        click_load_more=click_load_more,
                        allow_fallback=False,
                        browser_config=browser_config, # Pass browser_config from fetch to _fetch_with_browser
                        use_web_unblocker=strategy.get('use_unblocker', False)
                    )

                # Incorporate internal logs if available
                if 'internal_log' in res:
                    for entry in res['internal_log']:
                        # Handle both dictionary and string log entries
                        msg = entry['message'] if isinstance(entry, dict) else str(entry)
                        self._log_unblocker(f"[{strategy['name']}] {msg}")

                # Validate result
                html = res.get('html', '')
                if html and len(html) > 10000: # Increased from 5000 to 10000 for shells
                    html_lower = html.lower()
                    # Check for common block patterns
                    is_blocked = (
                        'verify you are human' in html_lower or
                        'just a moment' in html_lower or
                        'access denied' in html_lower or
                        ('blocked' in html_lower and len(html) < 20000) or
                        'perimeterx' in html_lower or
                        'px-captcha' in html_lower or
                        'enable javascript' in html_lower and len(html) < 20000
                    )

                    if not is_blocked:
                        self._log_unblocker(f"✅ Strategy {strategy['name']} succeeded!")
                        # Update success cache
                        self._success_cache[domain] = {'method': strategy['method'], 'timestamp': time.time()}
                        res['unblocker_log'] = self.unblocker_log
                        return res
                    else:
                        self._log_unblocker(f"⚠️ Strategy {strategy['name']} returned a block/challenge page.")
                else:
                    self._log_unblocker(f"⚠️ Strategy {strategy['name']} returned insufficient content ({len(html)} bytes).")

                last_result = res
            except Exception as e:
                self._log_unblocker(f"❌ Strategy {strategy['name']} failed with error: {str(e)}")
                continue

        # If all strategies failed, return the last result or a failure result
        if last_result:
            last_result['unblocker_log'] = self.unblocker_log
            return last_result

        return {
            'html': '',
            'url': url,
            'fetch_method': 'failed',
            'unblocker_log': self.unblocker_log,
            'error': 'All unblocking strategies failed'
        }

        # STEP 1: Check API cache (fastest!)
        if self.api_cache:
            cached_apis = self.api_cache.get_apis(domain)
            if cached_apis:
                logger.info(f"💾 Found {len(cached_apis)} cached APIs for {domain}")
                self.stats['api_cache_hits'] += 1
                # Note: You'd implement direct API calls here based on fields
                # For now, we'll still fetch the page but flag APIs as available

        # STEP 1.5: Force browser mode if infinite scroll or "Load More" is requested
        # (Static HTML fetch can't handle scrolling or clicking)
        if scroll_to_bottom or click_load_more:
            logger.info("🦊 Infinite scroll/click requested - using browser mode...")
            try:
                return await self._fetch_with_browser(
                    url,
                    wait_for_selector=wait_for_selector,
                    scroll_to_bottom=scroll_to_bottom,
                    click_load_more=click_load_more
                )
            except (RuntimeError, Exception) as e:
                # If browser fails, we can't do scroll/click, but try static HTML anyway
                error_msg = str(e).lower()
                if 'browser' in error_msg or 'page' in error_msg or 'playwright' in error_msg:
                    logger.warning(f"⚠️ Browser failed for scroll/click request: {e}")
                    logger.info("🔄 Falling back to static HTML (scroll/click won't work)...")
                    static_result = self._fetch_with_static(url)
                    static_result['fetch_method'] = 'static_fallback'
                    static_result['fallback_reason'] = f"Browser failed: {str(e)}. Note: Scroll/click features unavailable."
                    return static_result
                else:
                    raise

        # STEP 2: Try static HTML first (fast path)
        logger.info("⚡ Trying static HTML fetch...")
        try:
            static_result = self._fetch_with_static(url)
        except Exception as e:
            # If static fetch fails (e.g., SSL errors with proxy), fall back to browser
            logger.warning(f"⚠️ Static HTML fetch failed: {e}")
            logger.info("🦊 Falling back to browser...")
            static_result = {'html': '', 'status_code': 0}  # Empty result to trigger browser fallback

        # STEP 3: Check if JavaScript is needed
        needs_js = self._detect_js_required(static_result.get('html', ''), domain)

        if not needs_js:
            logger.info("✅ Static HTML sufficient")
            self.stats['static_html_success'] += 1
            return static_result

        # STEP 4: Fall back to browser
        logger.info("🦊 JavaScript required, using browser...")
        self.stats['browser_fallback'] += 1

        try:
            browser_result = await self._fetch_with_browser(
                url,
                wait_for_selector=wait_for_selector,
                scroll_to_bottom=scroll_to_bottom,
                click_load_more=click_load_more
            )
        except (RuntimeError, Exception) as e:
            # Catch any browser-related errors and fall back to static HTML
            error_msg = str(e).lower()
            if 'browser' in error_msg or 'page' in error_msg or 'playwright' in error_msg or 'chromium' in error_msg:
                logger.warning(f"⚠️ Browser error detected: {e}")
                logger.info("🔄 Falling back to static HTML fetch...")
                browser_result = self._fetch_with_static(url)
                browser_result['fetch_method'] = 'static_fallback'
                browser_result['fallback_reason'] = f"Browser failed: {str(e)}"
            else:
                # Re-raise if it's not a browser-related error
                raise

        # STEP 4.5: Check if browser fetch was blocked (Kasada, etc.)
        # Note: If Web Unblocker was already tried proactively above, we won't try it again here
        if self._is_blocked(browser_result.get('html', '')):
            logger.warning("⚠️ Browser fetch appears blocked (Kasada challenge detected)")
            if not self.web_unblocker_fetcher:
                logger.info("ℹ️ Web Unblocker not configured - consider enabling it for better success rates")

        # STEP 5: Cache discovered APIs for next time
        if browser_result.get('apis') and self.api_cache:
            self.api_cache.store_discovered_apis(
                domain,
                browser_result['apis'],
                url
            )
            self.stats['apis_discovered'] += len(browser_result['apis'])
            logger.info(f"💾 Cached {len(browser_result['apis'])} APIs for future use")

        return browser_result

    def _is_blocked(self, html: str) -> bool:
        """
        Detect if page is blocked by anti-bot protection (Kasada, Cloudflare, etc.)

        Args:
            html: HTML content to check

        Returns:
            True if blocked, False otherwise
        """
        if not html or len(html) < 500:
            return True

        html_lower = html.lower()

        # Kasada indicators
        kasada_indicators = ['kasada', 'kpsdk', 'ips.js', 'window.kpsdk']
        if any(indicator in html_lower for indicator in kasada_indicators):
            # Check if it's just the challenge script (small HTML)
            if len(html) < 2000:
                return True

        # Cloudflare indicators
        cloudflare_indicators = ['cf-browser-verification', 'checking your browser', 'cloudflare']
        if any(indicator in html_lower for indicator in cloudflare_indicators):
            if len(html) < 5000:
                return True

        # Generic blocking indicators
        blocking_indicators = ['access denied', 'blocked', 'forbidden', '403', 'captcha']
        if any(indicator in html_lower for indicator in blocking_indicators):
            if len(html) < 3000:
                return True

        return False

    def _fetch_with_static(self, url: str) -> Dict[str, Any]:
        """Fetch with static HTML fetcher"""
        self._log_unblocker("Attempting static HTML fetch...")
        result = self.html_fetcher.fetch(url)
        result['fetch_method'] = 'static'
        result['apis'] = {}
        result['captured_json'] = []  # No API capture in static mode

        if result.get('html') and len(result['html']) > 1000:
            self._log_unblocker(f"Static fetch successful ({len(result['html'])} bytes)")
        else:
            self._log_unblocker("Static fetch failed or returned minimal content.")

        return result

    async def _fetch_with_browser(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None,
        allow_fallback: bool = True,  # NEW: Control whether to fall back to static HTML
        browser_config: Optional[Dict[str, Any]] = None,  # NEW: Browser config from cached strategy
        use_web_unblocker: bool = False  # NEW: Force Web Unblocker
    ) -> Dict[str, Any]:
        """Fetch with browser - with optional fallback to static HTML if browser fails

        Args:
            allow_fallback: If False, don't fall back to static HTML (for force_mode='browser')
            browser_config: Optional browser configuration from cached strategy
        """
        # Lazy load browser fetcher
        if self.browser_fetcher is None:
            self._log_unblocker("Initializing browser fetcher...")
            BF = _get_browser_fetcher(use_camoufox=self.use_camoufox)
            if BF is None:
                if allow_fallback:
                    self._log_unblocker("Browser fetching not available, falling back to static HTML")
                    logger.warning("⚠️ Browser fetching not available, falling back to static HTML")
                    return self._fetch_with_static(url)
                else:
                    self._log_unblocker("Browser fetching not available and fallback disabled")
                    logger.error("❌ Browser fetching not available and fallback disabled")
                    return {
                        'html': '',
                        'url': url,
                        'status_code': 0,
                        'fetch_method': 'browser_failed',
                        'fallback_reason': 'Browser fetching not available (Playwright not installed)',
                        'error': 'Browser fetching not available'
                    }

            try:
                # Initialize browser fetcher with appropriate parameters
                if self.use_camoufox:
                    # Camoufox fetcher (simpler constructor)
                    self.browser_fetcher = BF(
                        headless=self.headless,
                        proxy_config=self.proxy_config,
                        proxy_manager=self.proxy_manager,  # NEW: Pass ProxyManager for rotation
                        timeout=self.browser_timeout,
                        web_unblocker_api_key=self.web_unblocker_api_key,
                        web_unblocker_zone=self.web_unblocker_zone,
                        web_unblocker_customer_id=self.web_unblocker_customer_id
                    )
                else:
                    # Playwright browser fetcher (original) - doesn't support proxy_manager yet
                    self.browser_fetcher = BF(
                        headless=self.headless,
                        proxy_config=self.proxy_config,
                        timeout=self.browser_timeout,
                        capture_api_requests=True,
                        web_unblocker_api_key=self.web_unblocker_api_key,
                        web_unblocker_zone=self.web_unblocker_zone,
                        web_unblocker_customer_id=self.web_unblocker_customer_id
                    )

                self._log_unblocker(f"Launching browser ({'Camoufox' if self.use_camoufox else 'Playwright'})...")
                await self.browser_fetcher._launch_browser()
                self._log_unblocker("Browser launched successfully.")
                logger.info("✅ Browser launched successfully")
            except Exception as e:
                self._log_unblocker(f"Browser launch failed: {str(e)}")
                logger.error(f"❌ Browser launch failed: {e}", exc_info=True)
                self.browser_fetcher = None  # Reset so we can try again next time
                if allow_fallback:
                    self._log_unblocker("Falling back to static HTML fetch...")
                    logger.warning("⚠️ Falling back to static HTML fetch...")
                    return self._fetch_with_static(url)
                else:
                    logger.error("❌ Browser launch failed and fallback disabled")
                    return {
                        'html': '',
                        'url': url,
                        'status_code': 0,
                        'fetch_method': 'browser_failed',
                        'fallback_reason': f'Browser launch failed: {str(e)}',
                        'error': str(e)
                    }

        try:
            result = await self.browser_fetcher.fetch(
                url,
                wait_for_selector=wait_for_selector,
                scroll_to_bottom=scroll_to_bottom,
                click_load_more=click_load_more,
                browser_config=browser_config,
                use_web_unblocker=use_web_unblocker
            )
            result['fetch_method'] = 'browser'
        except Exception as e:
            logger.error(f"❌ Browser fetch failed: {e}", exc_info=True)
            # Reset browser fetcher so it can be retried next time
            try:
                await self.browser_fetcher.close()
            except Exception:
                pass
            self.browser_fetcher = None

            if allow_fallback:
                logger.warning("⚠️ Falling back to static HTML fetch...")
                return self._fetch_with_static(url)
            else:
                logger.error("❌ Browser fetch failed and fallback disabled")
                return {
                    'html': '',
                    'url': url,
                    'status_code': 0,
                    'fetch_method': 'browser_failed',
                    'fallback_reason': f'Browser fetch failed: {str(e)}',
                    'error': str(e)
                }

        # CRITICAL FIX: Map json_data to captured_json for the scraper pipeline
        # The camoufox_fetcher captures API responses as 'json_data', but the
        # scraper.py expects 'captured_json' for JSON-first extraction
        if 'json_data' in result:
            json_data = result['json_data']
            if json_data:
                logger.info(f"📦 Captured {len(json_data)} JSON API responses from browser")
                # Extract just the data portion for captured_json
                result['captured_json'] = [item.get('data') for item in json_data if item.get('data')]
                result['apis'] = {item.get('url', f'api_{i}'): item.get('data') for i, item in enumerate(json_data) if item.get('data')}
            else:
                result['captured_json'] = []
                result['apis'] = {}
        else:
            # Preserve BrowserFetcher captured_json/APIs if present
            result.setdefault('captured_json', [])
            result.setdefault('apis', result.get('apis', {}))

        return result

    def _detect_js_required(self, html: str, domain: str) -> bool:
        """
        UNIVERSAL JavaScript detection - dynamically determines if JS rendering is required.
        No hardcoded domains or site-specific logic.

        Strategy:
        1. Check for framework indicators (React, Vue, Angular, Next.js, etc.)
        2. Analyze content density (sparse content = likely JS-rendered)
        3. Check for empty/minimal body with script tags (classic SPA pattern)
        4. Look for data attributes that indicate client-side rendering

        Args:
            html: HTML content
            domain: Domain name (for logging only, not used for detection)

        Returns:
            True if JS is likely required for proper rendering
        """
        if not html or len(html) < 100:
            logger.info("🎯 HTML too small, likely needs JS")
            return True

        soup = BeautifulSoup(html, 'html.parser')
        body = soup.find('body')

        if not body:
            logger.info("🎯 No body tag found, likely JS-rendered")
            return True

        # Get all script tags (both inline and external)
        script_tags = soup.find_all('script')
        has_scripts = len(script_tags) > 0

        # Extract script content (both inline and src attributes)
        script_content = []
        script_srcs = []
        for script in script_tags:
            if script.string:
                script_content.append(script.string)
            if script.get('src'):
                script_srcs.append(script.get('src').lower())

        script_content_combined = ' '.join(script_content).lower()

        # PRIORITY 1: Check for framework indicators in scripts (most reliable)
        framework_indicators = [
            # Next.js / React
            '__next_data__', '__next', 'next.js', 'react', 'reactdom',
            # Vue / Nuxt
            '__nuxt__', 'vue', 'nuxt', 'v-app',
            # Angular
            'ng-app', 'angular', '@angular',
            # Svelte
            '__svelte', 'svelte',
            # Generic SPA patterns
            'window.__initial_state__', 'window.__apollostate__',
            'window.__redux_state__', 'window.__preloadedstate__',
            'reactroot', 'react-root', 'root',
            # Modern frameworks
            'remix', 'astro', 'solid', 'qwik'
        ]

        for indicator in framework_indicators:
            if indicator in script_content_combined:
                logger.info(f"🎯 Detected JS framework indicator: {indicator}")
                return True

        # Check script src URLs for framework indicators
        for src in script_srcs:
            if any(fw in src for fw in ['react', 'vue', 'angular', 'next', 'nuxt', 'svelte']):
                logger.info(f"🎯 Detected framework in script src: {src[:50]}")
                return True

        # PRIORITY 2: Check for SPA patterns (empty/minimal body with many scripts)
        text_content = body.get_text(strip=True)
        str(body)

        # Count meaningful content elements
        content_tags = soup.find_all(['article', 'main', 'ul', 'ol', 'table', 'p', 'div'])
        meaningful_content = sum(len(tag.get_text(strip=True)) for tag in content_tags[:30])

        # If we have many scripts but very little content, it's likely a SPA
        if has_scripts and len(script_tags) >= 3 and meaningful_content < 1000:
            logger.info(f"🎯 SPA pattern detected: {len(script_tags)} scripts but only {meaningful_content} chars of content")
            return True

        # PRIORITY 3: Check for empty/minimal body with script tags (classic SPA)
        if has_scripts and len(text_content) < 300:
            logger.info(f"🎯 Minimal body content ({len(text_content)} chars) with scripts, likely SPA")
            return True

        # PRIORITY 4: Check for framework data attributes
        data_attrs = [
            'data-reactroot', 'data-react-root', 'data-vue-app',
            'data-ng-app', 'ng-app', 'v-app', 'x-data',
            'data-svelte', 'data-nextjs'
        ]
        html_lower = html.lower()
        for attr in data_attrs:
            if attr in html_lower:
                logger.info(f"🎯 Detected framework data attribute: {attr}")
                return True

        # PRIORITY 5: Check for loading/placeholder indicators
        loading_indicators = [
            'loading...', 'please wait', 'rendering', 'initializing',
            'building...', 'compiling...', 'hydrating'
        ]
        if any(indicator in text_content.lower() for indicator in loading_indicators):
            logger.info("🎯 Loading indicators found in content")
            return True

        # PRIORITY 6: Check for root div with id but minimal content (React pattern)
        root_divs = soup.find_all('div', id=True)
        for div in root_divs[:5]:  # Check first 5 divs with IDs
            div_id = div.get('id', '').lower()
            if div_id in ['root', 'app', 'main', '__next', '__nuxt']:
                div_content = div.get_text(strip=True)
                if len(div_content) < 500:
                    logger.info(f"🎯 Found root div '{div_id}' with minimal content ({len(div_content)} chars)")
                    return True

        # PRIORITY 7: Check for modern build tool indicators
        build_indicators = [
            '_next/static', '/static/chunks/', 'webpack', 'vite',
            'esbuild', 'rollup', 'parcel'
        ]
        for indicator in build_indicators:
            if indicator in html_lower:
                logger.info(f"🎯 Detected build tool indicator: {indicator}")
                return True

        # If we have substantial structured content, static HTML is likely sufficient
        if meaningful_content > 2000:
            logger.info(f"✅ Found {meaningful_content} chars of structured content, static HTML sufficient")
            return False

        # If we have moderate content but no framework indicators, probably static
        if meaningful_content > 500 and not has_scripts:
            logger.info(f"✅ Found {meaningful_content} chars of content without scripts, static HTML sufficient")
            return False

        # Default: If uncertain and we have scripts, assume JS might be needed
        # But be conservative - only if content is very sparse
        if has_scripts and meaningful_content < 500:
            logger.info(f"⚠️ Uncertain: {len(script_tags)} scripts but only {meaningful_content} chars content - assuming JS needed")
            return True

        # Default to static HTML if we can't determine
        logger.info("✅ No clear JS indicators, assuming static HTML sufficient")
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

        logger.info("👋 Hybrid Fetcher closed")
        logger.info(f"📊 Session stats: {self.stats}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Note: This is for sync usage only (local scripts)
        # For async usage, use async with
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self.close())
        except Exception:
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
