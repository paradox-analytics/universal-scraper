"""
Browser-based HTML Fetcher using Playwright (Async)
Handles JavaScript-rendered content and captures API requests
"""

import asyncio
import time
import logging
import json
import re
import random
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin

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
        user_data_dir: Optional[str] = None
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
                '--no-sandbox'
            ]
        }
        
        # Add proxy if configured
        if self.proxy_config:
            proxy_server = self.proxy_config['server']
            if not proxy_server.startswith('http'):
                proxy_server = f"http://{proxy_server}"
            
            launch_options['proxy'] = {
                'server': proxy_server
            }
            
            if self.proxy_config.get('username') and self.proxy_config.get('password'):
                launch_options['proxy']['username'] = self.proxy_config['username']
                launch_options['proxy']['password'] = self.proxy_config['password']
            
            logger.info(f" Using proxy: {proxy_server}")
        
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
                
            else:  # camoufox
                # Use Camoufox if preferred
                if self.user_data_dir:
                    launch_options['user_data_dir'] = self.user_data_dir
                from camoufox.sync_api import Camoufox
                self.browser = Camoufox(**launch_options)
                self.page = self.browser
            
            # Add request interception if needed
            if self.capture_api_requests:
                self._setup_request_interception()
            
            logger.info(" Browser launched successfully")
            
        except Exception as e:
            logger.error(f" Failed to launch browser: {e}")
            raise
    
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
        click_load_more: Optional[str] = None
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
            
            # Click "Load More" button if requested
            if click_load_more:
                logger.info(f" Clicking Load More: {click_load_more}")
                await self._click_load_more(click_load_more)
            
            # Additional wait for dynamic content
            if additional_wait > 0:
                await asyncio.sleep(additional_wait / 1000)
            
            # Get final HTML
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
    
    async def _wait_for_content_loaded(self, timeout: int = 15000) -> None:
        """
        Universal method to wait for JavaScript-rendered content to appear
        
        Works for ANY site by waiting for:
        1. Images to load
        2. DOM to stabilize (no more mutations)
        3. Reasonable timeout to prevent infinite waits
        
        Args:
            timeout: Maximum wait time in milliseconds (default: 15s)
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
                            }, 1000); // No changes for 1 second = stable
                        });
                        
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true
                        });
                        
                        // Initial timeout
                        timeout = setTimeout(() => {
                            observer.disconnect();
                            resolve();
                        }, 1000);
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
    
    async def _scroll_to_bottom(self, max_scrolls: int = 5, scroll_pause: float = 1.0) -> None:
        """Scroll to bottom to trigger lazy loading"""
        total_scrolls = 0
        
        for i in range(max_scrolls):
            # Get current scroll height
            prev_height = await self.page.evaluate('document.body.scrollHeight')
            
            # Scroll to bottom
            await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            
            # Wait for content to load
            await asyncio.sleep(scroll_pause)
            
            # Check if new content loaded
            new_height = await self.page.evaluate('document.body.scrollHeight')
            
            if new_height > prev_height:
                total_scrolls += 1
                logger.info(f" Scroll #{total_scrolls}: Page grew from {prev_height}px to {new_height}px")
            else:
                logger.info(f" Scrolling complete: No more content to load (total scrolls: {total_scrolls})")
                break  # No new content loaded
        
        if total_scrolls == 0:
            logger.info("ℹ No infinite scroll detected (page height unchanged)")
    
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

