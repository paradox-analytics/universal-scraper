"""
Browser-based HTML Fetcher using Playwright (Async)
Handles JavaScript-rendered content and captures API requests
"""

import asyncio
import time
import logging
import re
import random
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Try to import browser - prefer Playwright (more stable)
# Use async API for compatibility with Apify
try:
    from playwright.async_api import async_playwright
    BROWSER_AVAILABLE = True
    BROWSER_TYPE = 'playwright'
except ImportError:
    try:
        from camoufox.async_api import AsyncCamoufox as Camoufox
        BROWSER_AVAILABLE = True
        BROWSER_TYPE = 'camoufox'
    except ImportError:
        BROWSER_AVAILABLE = False
        BROWSER_TYPE = None
        logger.warning("No browser available. Install with: pip install playwright")


class BrowserFetcher:
    """
    Fetches HTML with JavaScript rendering using Camoufox
    Captures network requests to discover JSON APIs
    """

    def __init__(
        self,
        headless: bool = True,
        proxy_config: Optional[Dict[str, str]] = None,
        timeout: int = 30000,  # 30 seconds
        wait_for_network_idle: bool = False,
        capture_api_requests: bool = True,
        user_data_dir: Optional[str] = None,
        web_unblocker_api_key: Optional[str] = None,
        web_unblocker_zone: str = "web_unlocker1"
    ):
        """
        Initialize Browser Fetcher

        Args:
            headless: Run browser in headless mode
            proxy_config: Dict with 'server', 'username', 'password' keys
            timeout: Navigation timeout in milliseconds
            wait_for_network_idle: Wait for network to be idle before returning
            capture_api_requests: Capture API/JSON requests automatically
            user_data_dir: Directory for browser profile (for persistence)
        """
        if not BROWSER_AVAILABLE:
            raise ImportError(
                "No browser available. Install with: pip install playwright && playwright install chromium"
            )

        self.headless = headless
        self.proxy_config = proxy_config
        self.timeout = timeout
        self.wait_for_network_idle = wait_for_network_idle
        self.capture_api_requests = capture_api_requests
        self.user_data_dir = user_data_dir
        self.browser = None
        self.page = None
        self.captured_requests = []
        self.discovered_apis = {}
        self.web_unblocker_api_key = web_unblocker_api_key
        self.web_unblocker_zone = web_unblocker_zone

        logger.info(f" Browser Fetcher initialized with {BROWSER_TYPE}")

    def __enter__(self):
        # Sync context manager not supported for async fetcher
        raise RuntimeError("Use 'async with' instead of 'with' for BrowserFetcher")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        await self._launch_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _launch_browser(self) -> None:
        """Launch browser (Playwright or Camoufox)"""

        logger.info(f" Launching {BROWSER_TYPE} browser...")

        # Browser configuration
        launch_options = {
            'headless': self.headless,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-gpu'
            ]
        }

        # Add proxy if configured and valid
        if self.proxy_config:
            try:
                proxy_server = self.proxy_config.get('server')
                # Validate proxy server is present, not None, and not empty
                if proxy_server is None or not isinstance(proxy_server, str) or len(proxy_server.strip()) == 0:
                    logger.warning("⚠️ Proxy config provided but 'server' field is missing, None, or empty. Continuing without proxy...")
                else:
                    # Ensure proxy server has protocol
                    proxy_server = proxy_server.strip()
                    if not proxy_server.startswith('http'):
                        proxy_server = f"http://{proxy_server}"

                    # Validate proxy server format (basic check)
                    if '://' not in proxy_server or len(proxy_server.split('://')) != 2:
                        logger.warning(f"⚠️ Invalid proxy server format: {proxy_server}. Continuing without proxy...")
                    else:
                        launch_options['proxy'] = {
                            'server': proxy_server
                        }

                        # Add authentication if provided
                        username = self.proxy_config.get('username')
                        password = self.proxy_config.get('password')
                        if username and password:
                            launch_options['proxy']['username'] = str(username).strip()
                            launch_options['proxy']['password'] = str(password).strip()

                        logger.info(f"✅ Using proxy: {proxy_server}")
            except Exception as e:
                logger.warning(f"⚠️ Error configuring proxy: {e}. Continuing without proxy...")
                # Don't fail browser launch if proxy config is invalid
                # Remove proxy from launch_options if it was partially set
                if 'proxy' in launch_options:
                    del launch_options['proxy']

        # Add Web Unblocker as proxy if configured
        if self.web_unblocker_api_key:
            try:
                # Detect if it's proxy credentials format or API key
                separator = None
                if ':' in self.web_unblocker_api_key and self.web_unblocker_api_key.count(':') >= 3:
                    separator = ':'
                elif ',' in self.web_unblocker_api_key and self.web_unblocker_api_key.count(',') >= 3:
                    separator = ','

                if separator:
                    # Proxy credentials format: host:port:username:password or host,port,username,password
                    parts = self.web_unblocker_api_key.split(separator)
                    if len(parts) >= 4:
                        host = parts[0].strip()
                        port = parts[1].strip()
                        username = parts[2].strip()
                        password = parts[3].strip()

                        launch_options['proxy'] = {
                            'server': f"http://{host}:{port}",
                            'username': username,
                            'password': password
                        }
                        logger.info(f"✅ Using Web Unblocker as proxy for {BROWSER_TYPE}")
                else:
                    # API key (Bearer token) format: use Bright Data API proxy endpoint
                    # Format: brd-customer-<CUSTOMER_ID>-zone-<ZONE_NAME>:<PASSWORD>@brd.superproxy.io:22225
                    # However, the Web Unblocker API (Direct API Access) is usually preferred for static.
                    # For browser, we MUST use the proxy endpoint.
                    # If the user provided a Bearer token, we might not have the proxy credentials.
                    # But often the "API Key" field in our UI is used for both.
                    logger.warning("⚠️ Web Unblocker API key provided but not in proxy format. Browser mode requires proxy format (host:port:user:pass).")
            except Exception as e:
                logger.warning(f"⚠️ Error configuring Web Unblocker proxy: {e}")

        # Launch browser based on type
        try:
            if BROWSER_TYPE == 'playwright':
                # Use Playwright (standard, well-supported)
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(**launch_options)

                # Universal anti-blocking: Randomized viewport and user agent
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
                ]
                selected_ua = random.choice(user_agents)

                context = await self.browser.new_context(
                    ignore_https_errors=True,
                    viewport={
                        'width': random.choice([1920, 1366, 1536, 1440]),
                        'height': random.choice([1080, 768, 864, 900])
                    },
                    user_agent=selected_ua
                )

                # Comprehensive Universal Anti-Detection (Amazon-grade)
                await context.add_init_script("""
                    // ===== COMPREHENSIVE ANTI-DETECTION =====
                    // Universal solution for Amazon, Ticketmaster, and ALL bot-protected sites

                    // 1. Override webdriver (most basic check)
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                        configurable: true
                    });
                    delete navigator.__proto__.webdriver;

                    // 2. Realistic plugin array (exact format Chrome uses)
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [
                            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
                            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
                            { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' }
                        ],
                        configurable: true
                    });

                    // 3. Languages (look like real Chrome)
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                        configurable: true
                    });

                    // 4. Chrome runtime (critical - many sites check this)
                    if (!window.chrome) window.chrome = {};
                    window.chrome.runtime = {
                        connect: () => {},
                        sendMessage: () => {}
                    };

                    // 5. Override permissions API
                    const originalQuery = navigator.permissions?.query;
                    if (originalQuery) {
                        navigator.permissions.query = (params) => (
                            params.name === 'notifications' ?
                                Promise.resolve({ state: Notification.permission || 'default' }) :
                                originalQuery(params)
                        );
                    }

                    // 6. Battery API (desktop = always plugged in)
                    if (navigator.getBattery) {
                        navigator.getBattery = () => Promise.resolve({
                            charging: true,
                            chargingTime: 0,
                            dischargingTime: Infinity,
                            level: 1
                        });
                    }

                    // 7. Connection API (look like real user)
                    Object.defineProperty(navigator, 'connection', {
                        get: () => ({
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        }),
                        configurable: true
                    });

                    // 8. Override console.debug (headless detection)
                    const originalDebug = console.debug;
                    console.debug = function() {
                        originalDebug?.apply(console, arguments);
                    };
                """)

                self.page = await context.new_page()

                if not self.page:
                    raise RuntimeError("Failed to create browser page - context.new_page() returned None")

                logger.info(" Browser page created successfully")

            else:  # camoufox
                # Use Camoufox if preferred
                if self.user_data_dir:
                    launch_options['user_data_dir'] = self.user_data_dir
                from camoufox.sync_api import Camoufox
                self.browser = Camoufox(**launch_options)
                self.page = self.browser

                if not self.page:
                    raise RuntimeError("Failed to create browser page - Camoufox returned None")

            # Verify page is valid before proceeding
            if not self.page:
                raise RuntimeError("Browser page is None after launch - this should not happen")

            # Add request interception if needed
            if self.capture_api_requests:
                self._setup_request_interception()

            logger.info(" Browser launched successfully")

        except Exception as e:
            logger.error(f" Failed to launch browser: {e}", exc_info=True)
            # Reset state on failure
            self.browser = None
            self.page = None
            if hasattr(self, 'playwright') and self.playwright:
                try:
                    await self.playwright.stop()
                except:
                    pass
                self.playwright = None
            raise RuntimeError(f"Browser launch failed: {str(e)}") from e

    def _setup_request_interception(self) -> None:
        """Setup network request interception to capture API calls"""

        # Check if page has event listener support
        if not hasattr(self.page, 'on'):
            logger.warning(" Request interception not supported by this browser API")
            return

        def handle_request(request):
            """Capture interesting requests"""
            url = request.url
            resource_type = request.resource_type
            method = request.method

            # Capture API/JSON requests
            is_api = (
                resource_type in ['xhr', 'fetch'] or
                'api' in url.lower() or
                'json' in url.lower() or
                '/graphql' in url.lower() or
                method in ['POST', 'PUT', 'PATCH']
            )

            if is_api:
                request_data = {
                    'url': url,
                    'method': method,
                    'resource_type': resource_type,
                    'headers': request.headers,
                    'timestamp': time.time()
                }

                # Try to capture POST data
                if method in ['POST', 'PUT', 'PATCH']:
                    try:
                        request_data['post_data'] = request.post_data
                    except:
                        pass

                self.captured_requests.append(request_data)
                logger.debug(f" Captured: {method} {url[:100]}")

        async def handle_response(response):
            """Capture response data for API requests"""
            url = response.url

            # Check if this is an API request we captured
            for req in self.captured_requests:
                if req['url'] == url and 'response' not in req:
                    try:
                        # Try to get JSON response
                        if 'json' in response.headers.get('content-type', '').lower():
                            req['response'] = await response.json()
                            req['status_code'] = response.status

                            # Store as discovered API
                            self._store_discovered_api(req)

                            logger.debug(f" Captured API response: {url[:100]}")
                    except:
                        pass

        # Attach handlers
        self.page.on('request', handle_request)
        self.page.on('response', handle_response)

    def _store_discovered_api(self, request_data: Dict[str, Any]) -> None:
        """Store discovered API for future direct calls"""
        url = request_data['url']
        method = request_data['method']

        # Create a cache key based on URL pattern
        parsed = urlparse(url)
        path = parsed.path

        # Remove IDs and dynamic parts to create a pattern
        pattern = re.sub(r'/\d+', '/{id}', path)
        pattern = re.sub(r'/[a-f0-9-]{32,}', '/{uuid}', pattern)

        cache_key = f"{method}:{parsed.netloc}{pattern}"

        if cache_key not in self.discovered_apis:
            self.discovered_apis[cache_key] = {
                'url_pattern': url,
                'method': method,
                'headers': request_data.get('headers', {}),
                'sample_response': request_data.get('response'),
                'discovered_at': request_data['timestamp']
            }

            logger.info(f" Discovered API: {cache_key}")

    async def fetch(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        additional_wait: int = 2000,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None,
        geoip: Optional[bool] = None,
        browser_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch page with JavaScript rendering

        Args:
            url: Target URL
            wait_for_selector: Optional selector to wait for
            additional_wait: Additional wait time in milliseconds
            scroll_to_bottom: Scroll to bottom to trigger lazy loading
            click_load_more: Selector for "Load More" button to click

        Returns:
            Dict with 'html', 'url', 'requests', 'apis' keys
        """
        if not self.browser:
            self._launch_browser()

        # Ensure page is initialized and still connected
        if not self.page:
            logger.warning(" Page not initialized, re-launching browser...")
            self._launch_browser()

        # Check if page is still connected (browser might have been closed)
        try:
            if hasattr(self.page, 'is_closed') and self.page.is_closed():
                logger.warning(" Page was closed, re-launching browser...")
                self.browser = None
                self.page = None
                self._launch_browser()
        except Exception as e:
            logger.warning(f" Error checking page status: {e}, re-launching browser...")
            self.browser = None
            self.page = None
            self._launch_browser()

        if not self.page:
            # Don't raise here - let the caller handle the fallback
            # This allows HybridFetcher to fall back to static HTML
            logger.error("Browser page is None - browser launch failed")
            raise RuntimeError("Browser page initialization failed. This may be due to resource constraints or missing dependencies. The system will attempt to fall back to static HTML fetching.")

        # Clear captured requests
        self.captured_requests = []

        try:
            logger.info(f" Navigating to: {url}")

            # Universal wait strategy: Always use domcontentloaded for initial load
            # This prevents timeouts on sites with continuous tracking/analytics
            await self.page.goto(url, timeout=self.timeout, wait_until='domcontentloaded')
            logger.info(" DOM content loaded")

            # Try networkidle with SHORT timeout (don't fail if it times out)
            # This catches most JS rendering without getting stuck on infinite loaders
            if self.wait_for_network_idle:
                try:
                    logger.info("⏳ Waiting for network idle (15s max)...")
                    await self.page.wait_for_load_state('networkidle', timeout=15000)
                    logger.info(" Network idle reached")
                except Exception:
                    logger.info("⏱  Network idle timeout - continuing (normal for sites with tracking)")

            # Critical: Wait for JavaScript rendering to complete
            # Universal approach: Wait for content to actually appear, not just network idle
            logger.info("⏳ Waiting for JavaScript rendering...")

            # Check for Cloudflare challenge page (IMPROVED detection)
            html_preview = await self.page.content()
            body_text_preview = await self.page.evaluate('document.body.innerText.toLowerCase()')

            # Multiple Cloudflare challenge indicators
            cloudflare_indicators = [
                'just a moment' in html_preview.lower(),
                'checking your browser' in body_text_preview,
                'cloudflare' in body_text_preview and 'verify you are human' in body_text_preview,
                'cloudflare' in body_text_preview and 'ray id:' in body_text_preview,
                'challenge-platform' in html_preview,
                'turnstile' in html_preview,
                'cf-browser-verification' in html_preview,
                'cf-challenge' in html_preview
            ]

            if any(cloudflare_indicators):
                logger.warning("⚠️ Cloudflare challenge detected!")

                # Try to wait for challenge to complete
                try:
                    logger.info("⏳ Attempting to wait for Cloudflare challenge (45s max)...")
                    # Wait for the challenge indicators to disappear
                    await self.page.wait_for_function(
                        '''
                        () => {
                            const html = document.documentElement.innerHTML.toLowerCase();
                            const bodyText = document.body.innerText.toLowerCase();
                            return !html.includes('just a moment') &&
                                   !bodyText.includes('checking your browser') &&
                                   !bodyText.includes('verify you are human') &&
                                   !bodyText.includes('ray id:') &&
                                   !html.includes('cf-browser-verification') &&
                                   !html.includes('cf-challenge');
                        }
                        ''',
                        timeout=45000
                    )
                    logger.info("✅ Cloudflare challenge passed!")
                    await asyncio.sleep(3)  # Extra wait for content to stabilize
                except Exception as e:
                    logger.error(f"❌ Cloudflare challenge failed or timed out: {e}")
                    # If we have Web Unblocker, we should have used it as a proxy,
                    # but if we're here, it means the proxy didn't bypass it or wasn't used.
            else:
                logger.info("✅ No Cloudflare challenge detected")

            # UNIVERSAL: Wait for React/Next.js/Vue/Angular hydration (ONLY if framework detected)
            # This ONLY waits if we detect a framework, otherwise skips immediately
            try:
                # Quick check: Does this page use a framework?
                framework_detected = await self.page.evaluate(
                    """
                    () => {
                        // Quick framework detection (no waiting)
                        return !!(
                            document.getElementById('__NEXT_DATA__') ||
                            document.getElementById('__next') ||
                            document.getElementById('root') ||
                            document.querySelector('[data-reactroot]') ||
                            document.querySelector('[data-vue-app]') ||
                            document.querySelector('[ng-app]') ||
                            window.__NEXT_DATA__ ||
                            window.React ||
                            window.Vue ||
                            window.angular
                        );
                    }
                    """
                )

                if framework_detected:
                    logger.info("⏳ Framework detected, waiting for hydration...")
                    # Wait for React/Next.js to hydrate (check for __NEXT_DATA__ execution or React root to populate)
                    await self.page.wait_for_function(
                        """
                        () => {
                            // Check if React/Next.js has hydrated
                            const nextData = document.getElementById('__NEXT_DATA__');
                            if (nextData) {
                                try {
                                    const data = JSON.parse(nextData.textContent);
                                    // If __NEXT_DATA__ exists and has content, React is likely hydrated
                                    return true;
                                } catch (e) {
                                    return false;
                                }
                            }

                            // Check if React root has substantial content (not just empty div)
                            const reactRoot = document.getElementById('__next') || document.getElementById('root') || document.querySelector('[data-reactroot]');
                            if (reactRoot) {
                                const textContent = reactRoot.innerText || reactRoot.textContent || '';
                                // If root has substantial content (> 500 chars), likely hydrated
                                return textContent.length > 500;
                            }

                            // Check for Vue/Angular apps
                            const vueApp = document.querySelector('[data-vue-app]') || document.querySelector('[ng-app]');
                            if (vueApp) {
                                const textContent = vueApp.innerText || vueApp.textContent || '';
                                return textContent.length > 500;
                            }

                            // If no framework indicators, assume content is ready
                            return true;
                        }
                        """,
                        timeout=10000  # Wait up to 10 seconds for hydration
                    )
                    logger.info("✅ Framework hydration complete")
                else:
                    logger.info("⚡ No framework detected, skipping hydration wait")
            except Exception as e:
                logger.info(f"⏱ Framework hydration timeout: {e}")

            # Strategy: Wait for DOM to stabilize (content finished loading)
            # This handles lazy-loaded content universally
            await self._wait_for_content_loaded()

            logger.info(" JavaScript rendering complete")

            # Wait for specific selector if provided
            if wait_for_selector:
                logger.info(f"⏳ Waiting for selector: {wait_for_selector}")
                await self.page.wait_for_selector(wait_for_selector, timeout=self.timeout)

            # Scroll to bottom if requested (for infinite scroll)
            if scroll_to_bottom:
                logger.info(" Scrolling to bottom...")
                await self._scroll_to_bottom()
                # CRITICAL: Wait after scrolling to ensure all content is loaded
                logger.info(" Waiting for content to stabilize after scrolling...")
                await asyncio.sleep(2)  # Give time for lazy-loaded content
                # Wait for network idle again after scrolling
                try:
                    await self.page.wait_for_load_state('networkidle', timeout=10000)
                    logger.info(" Network idle after scrolling")
                except:
                    logger.info(" Network idle timeout after scrolling (continuing)")

            # Click "Load More" button if requested
            if click_load_more:
                logger.info(f" Clicking Load More: {click_load_more}")
                await self._click_load_more(click_load_more)

            # Additional wait for dynamic content
            if additional_wait > 0:
                await asyncio.sleep(additional_wait / 1000)

            # Get final HTML AFTER all scrolling/loading is complete
            html = await self.page.content()
            final_url = self.page.url

            logger.info(f" Page loaded: {len(html)} bytes, {len(self.captured_requests)} API requests captured")

            # Extract just the JSON blobs from captured API responses
            captured_json = []
            for req in self.captured_requests:
                if 'response' in req and req.get('response') is not None:
                    captured_json.append(req['response'])

            if captured_json:
                logger.info(f" Extracted {len(captured_json)} JSON blobs from API responses")

            return {
                'html': html,
                'url': final_url,
                'status_code': 200,
                'captured_json': captured_json,  # Simple: just the JSON blobs!
                'requests': self.captured_requests,  # Keep for debugging
                'apis': dict(self.discovered_apis),  # Keep for caching
                'cookies': await self.page.context.cookies()
            }

        except Exception as e:
            logger.error(f" Browser fetch failed: {str(e)}")
            raise

    async def _wait_for_content_loaded(self, timeout: int = 8000) -> None:
        """
        Universal method to wait for JavaScript-rendered content to appear

        Works for ANY site by waiting for:
        1. Images to load (70% threshold)
        2. DOM to stabilize (no more mutations for 1 second)
        3. Reasonable timeout to prevent infinite waits

        Args:
            timeout: Maximum wait time in milliseconds (default: 8s, reduced for speed)
        """
        try:
            # Wait for images to load (universal content indicator)
            await self.page.wait_for_function(
                """
                () => {
                    const images = Array.from(document.images);
                    // Check if most images are loaded (allow some to fail)
                    const loadedImages = images.filter(img => img.complete && img.naturalHeight > 0);
                    return images.length === 0 || loadedImages.length / images.length > 0.7;
                }
                """,
                timeout=timeout
            )
            logger.info(" Images loaded")

            # Wait for DOM to stabilize (no more mutations for 1 second)
            # This catches lazy-loaded content universally
            await self.page.evaluate(
                """
                async () => {
                    return new Promise((resolve) => {
                        let timeout;
                        const observer = new MutationObserver(() => {
                            clearTimeout(timeout);
                            timeout = setTimeout(() => {
                                observer.disconnect();
                                resolve();
                            }, 800); // Reduced: No changes for 0.8s = stable (faster)
                        });

                        observer.observe(document.body, {
                            childList: true,
                            subtree: true
                        });

                        // Initial timeout - faster bailout
                        timeout = setTimeout(() => {
                            observer.disconnect();
                            resolve();
                        }, 800);
                    });
                }
                """
            )
            logger.info(" DOM stabilized")

        except Exception as e:
            # Timeout is OK - just means content loaded quickly or took too long
            logger.debug(f"⏱  Content wait timeout (continuing): {e}")
            # Fallback to simple wait
            await asyncio.sleep(3)

    async def _scroll_to_bottom(self, max_scrolls: int = 30, scroll_pause: float = 2.0) -> None:
        """
        Scroll to bottom to trigger lazy loading

        Enhanced for sites like Reddit, Product Hunt, etc. that use continuous infinite scroll.
        Keeps scrolling until no new content loads or max_scrolls reached.
        """
        total_scrolls = 0
        no_change_count = 0  # Track consecutive scrolls with no change
        max_no_change = 5  # Stop after 5 consecutive scrolls with no change (increased for slow sites)

        # Universal item detection - find repeating patterns dynamically
        item_selector = await self.page.evaluate('''
            () => {
                // Common patterns for repeating items
                const selectors = [
                    'article',
                    '[role="article"]',
                    '[data-testid*="item"]',
                    '[data-testid*="post"]',
                    '[data-testid*="card"]',
                    '[data-testid*="product"]',
                    '[class*="item"]',
                    '[class*="card"]',
                    '[class*="product"]',
                    '[class*="post"]',
                    'div[class*="flex"][class*="gap"]',  // Product Hunt style
                    'a[href*="/products/"]',  // Product Hunt product links
                    'section > div > div',  // Generic section items
                    'main > div > div > div',  // Generic main content items
                ];

                for (const selector of selectors) {
                    try {
                        const items = document.querySelectorAll(selector);
                        // If we find 3+ items with the same selector, likely a repeating pattern
                        if (items.length >= 3) {
                            // Check if items are actually repeating (similar structure)
                            const firstItem = items[0];
                            const secondItem = items[1];
                            if (firstItem && secondItem) {
                                const firstClasses = Array.from(firstItem.classList || []).join(' ');
                                const secondClasses = Array.from(secondItem.classList || []).join(' ');
                                // If items share classes/structure, it's a repeating pattern
                                if (firstClasses && firstClasses === secondClasses) {
                                    return selector;
                                }
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }

                // Fallback: return a generic selector
                return 'article, [role="article"], div[class*="item"], div[class*="card"], a[href*="/products/"]';
            }
        ''')

        logger.info(f" Detected item selector: {item_selector}")

        for i in range(max_scrolls):
            # Get current scroll height and item count
            prev_height = await self.page.evaluate('document.body.scrollHeight')

            # Count items using detected selector
            try:
                prev_item_count = await self.page.evaluate(f'''
                    () => {{
                        const items = document.querySelectorAll('{item_selector}');
                        return items.length;
                    }}
                ''')
            except:
                prev_item_count = 0

            # Scroll to bottom
            await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            total_scrolls += 1

            # Wait for content to load (longer pause for slow sites like Product Hunt)
            await asyncio.sleep(scroll_pause)

            # Check if new content loaded (handle navigation errors)
            try:
                new_height = await self.page.evaluate('document.body.scrollHeight')

                # Count items again using detected selector
                try:
                    new_item_count = await self.page.evaluate(f'''
                        () => {{
                            const items = document.querySelectorAll('{item_selector}');
                            return items.length;
                        }}
                    ''')
                except:
                    new_item_count = prev_item_count

                height_changed = new_height > prev_height
                items_changed = new_item_count > prev_item_count

                if height_changed or items_changed:
                    no_change_count = 0  # Reset no-change counter
                    logger.info(f" Scroll #{total_scrolls}: Page grew from {prev_height}px to {new_height}px, items: {prev_item_count} → {new_item_count}")
                else:
                    no_change_count += 1
                    logger.info(f" Scroll #{total_scrolls}: No new content (no-change count: {no_change_count}/{max_no_change})")

                    if no_change_count >= max_no_change:
                        logger.info(f" Scrolling complete: No new content after {max_no_change} consecutive scrolls (total scrolls: {total_scrolls})")
                        break
            except Exception as e:
                # Handle navigation/redirect errors during scroll
                if "Execution context was destroyed" in str(e) or "navigation" in str(e).lower():
                    logger.warning(f" Page navigated during scroll (attempt {i+1}), stopping scroll")
                    break
                else:
                    # Other errors - log and continue
                    logger.warning(f" Error during scroll check: {e}, continuing...")
                    no_change_count += 1
                    if no_change_count >= max_no_change:
                        break

        if total_scrolls == 0:
            logger.info("ℹ No infinite scroll detected (page height unchanged)")
        else:
            final_height = await self.page.evaluate('document.body.scrollHeight')
            final_item_count = await self.page.evaluate(f'''
                () => {{
                    const items = document.querySelectorAll('{item_selector}');
                    return items.length;
                }}
            ''')
            logger.info(f"✅ Infinite scroll complete: {total_scrolls} scrolls performed, final page height: {final_height}px, final items: {final_item_count}")

    async def _click_load_more(self, selector: str, max_clicks: int = 10) -> None:
        """Click 'Load More' button multiple times"""
        clicks_performed = 0

        for i in range(max_clicks):
            try:
                # Check if button exists and is visible
                button_visible = await self.page.is_visible(selector, timeout=1000)

                if not button_visible:
                    if clicks_performed == 0:
                        logger.info(f" Load More button not found (selector: {selector})")
                        logger.info("   This is normal if the page has all items loaded already.")
                    else:
                        logger.info(f" Load More button disappeared after {clicks_performed} clicks (all items loaded)")
                    break

                # Click button
                logger.info(f" Clicking Load More button (attempt {i + 1})...")
                await self.page.click(selector)
                clicks_performed += 1

                # Wait longer for new API calls and content to load
                # Critical: API responses need time to be captured
                logger.info("⏳ Waiting for new content to load (5s)...")
                await asyncio.sleep(5)

                logger.info(f" Load More click #{clicks_performed} complete")

            except Exception as e:
                if clicks_performed == 0:
                    logger.info(f" Load More button not clickable: {str(e)[:100]}")
                    logger.info("   This is normal if the page doesn't need pagination.")
                else:
                    logger.info(f" Pagination complete after {clicks_performed} clicks")
                break

        if clicks_performed > 0:
            logger.info(f" Total Load More clicks: {clicks_performed}")

    async def fetch_with_interactions(
        self,
        url: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fetch page with custom interactions

        Args:
            url: Target URL
            interactions: List of interaction dicts:
                {'type': 'click', 'selector': '#button'}
                {'type': 'fill', 'selector': '#input', 'value': 'text'}
                {'type': 'scroll', 'direction': 'bottom'}
                {'type': 'wait', 'timeout': 2000}
                {'type': 'wait_for_selector', 'selector': '.product'}

        Returns:
            Dict with page data
        """
        if not self.browser:
            self._launch_browser()

        self.captured_requests = []

        try:
            # Navigate
            await self.page.goto(url, timeout=self.timeout, wait_until='domcontentloaded')

            # Execute interactions
            for interaction in interactions:
                action_type = interaction['type']

                if action_type == 'click':
                    await self.page.click(interaction['selector'])
                    await asyncio.sleep(1)  # Brief pause after click

                elif action_type == 'fill':
                    await self.page.fill(interaction['selector'], interaction['value'])

                elif action_type == 'scroll':
                    if interaction.get('direction') == 'bottom':
                        await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await asyncio.sleep(1)

                elif action_type == 'wait':
                    await asyncio.sleep(interaction['timeout'] / 1000)

                elif action_type == 'wait_for_selector':
                    await self.page.wait_for_selector(interaction['selector'], timeout=self.timeout)

            # Get final HTML
            html = await self.page.content()

            return {
                'html': html,
                'url': self.page.url,
                'status_code': 200,
                'requests': self.captured_requests,
                'apis': dict(self.discovered_apis)
            }

        except Exception as e:
            logger.error(f" Interaction fetch failed: {e}")
            raise

    def get_discovered_apis(self) -> Dict[str, Any]:
        """Get all discovered API endpoints"""
        return dict(self.discovered_apis)

    def get_captured_requests(self) -> List[Dict[str, Any]]:
        """Get all captured network requests"""
        return self.captured_requests

    async def close(self) -> None:
        """Clean up browser resources"""
        if self.page:
            try:
                await self.page.close()
            except:
                pass

        if self.browser:
            try:
                await self.browser.close()
            except:
                pass

        # Close Playwright if used
        if BROWSER_TYPE == 'playwright' and hasattr(self, 'playwright'):
            try:
                await self.playwright.stop()
            except:
                pass

        logger.info(" Browser closed")


# Convenience function
async def fetch_with_browser(
    url: str,
    headless: bool = True,
    proxy_config: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for simple browser fetching

    Args:
        url: Target URL
        headless: Run in headless mode
        proxy_config: Proxy configuration
        **kwargs: Additional arguments for fetch()

    Returns:
        Fetch result dict
    """
    async with BrowserFetcher(headless=headless, proxy_config=proxy_config) as fetcher:
        return await fetcher.fetch(url, **kwargs)

