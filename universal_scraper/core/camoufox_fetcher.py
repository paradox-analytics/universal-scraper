"""
Camoufox Browser Fetcher - Advanced anti-detection browser automation
Inspired by the Parsera project's successful Camoufox implementation

Note: Camoufox uses Playwright's sync API, so we run it in a separate thread
to avoid conflicts with asyncio.
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import json
import time

logger = logging.getLogger(__name__)

# Import ProxyManager for per-request rotation
try:
    from .proxy_manager import ProxyManager
    PROXY_MANAGER_AVAILABLE = True
except ImportError:
    PROXY_MANAGER_AVAILABLE = False
    logger.warning(" ProxyManager not available")

try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    logger.warning(" Camoufox not installed. Install with: pip install camoufox")
    CAMOUFOX_AVAILABLE = False

# Import the universal anti-detection manager
try:
    from .anti_detection import AntiDetectionManager
    ANTI_DETECTION_AVAILABLE = True
except ImportError:
    ANTI_DETECTION_AVAILABLE = False
    logger.warning(" Anti-detection manager not available")


def _smart_wait_for_content(page, wait_for_selector: Optional[str] = None):
    """
    UNIVERSAL SOLUTION 3: Smart Wait Strategy for JS-heavy sites
    
    Adaptively waits for content to fully load without hardcoded delays.
    Works for ANY website regardless of rendering technology.
    
    Strategy:
    1. Wait for network idle (no pending requests for 500ms)
    2. Wait for DOM stability (no mutations for 500ms)
    3. If selector provided, wait for that specific element
    4. Maximum wait: 10 seconds (prevent hanging)
    
    Args:
        page: Playwright/Camoufox page object
        wait_for_selector: Optional CSS selector to wait for
    """
    start_time = time.time()
    max_wait = 10  # seconds
    
    try:
        # Strategy 1: Wait for network idle (most reliable for JS-heavy sites)
        logger.debug("   Waiting for network idle...")
        page.wait_for_load_state('networkidle', timeout=5000)
    except:
        # Timeout is OK, try other strategies
        pass
    
    # Strategy 2: Wait for specific selector if provided
    if wait_for_selector:
        try:
            logger.debug(f"   Waiting for selector: {wait_for_selector}")
            page.wait_for_selector(wait_for_selector, timeout=5000)
        except:
            # Selector not found, continue anyway
            pass
    
    # Strategy 3: Wait for common content indicators (universal patterns)
    # Check for any of these common selectors that indicate content has loaded
    content_selectors = [
        'article',
        '[role="article"]',
        '[role="listitem"]',
        '.post',
        '.item',
        '.card',
        'li',
        'tr'
    ]
    
    for selector in content_selectors:
        try:
            page.wait_for_selector(selector, timeout=2000)
            logger.debug(f"   Content detected: {selector}")
            break
        except:
            continue
    
    # Strategy 4: Minimum wait (ensures JS has time to execute)
    elapsed = time.time() - start_time
    if elapsed < 2:
        remaining = 2 - elapsed
        logger.debug(f"   Minimum wait: {remaining:.1f}s")
        time.sleep(remaining)


def _camoufox_fetch_sync(
    url: str,
    headless: bool,
    proxy_config: Optional[Dict[str, str]],
    timeout: int,
    wait_for_selector: Optional[str] = None,
    wait_time: int = 2000,
    scroll_to_bottom: bool = False,
    anti_detection_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synchronous function that runs Camoufox fetch in a separate thread
    This avoids the asyncio loop conflict with Playwright's sync API
    """
    captured_requests = []
    captured_json = []
    
    # Initialize anti-detection manager if available
    if ANTI_DETECTION_AVAILABLE and anti_detection_config:
        anti_detect = AntiDetectionManager(**anti_detection_config)
        camoufox_config = anti_detect.get_camoufox_config()
    else:
        # Fallback to basic humanization
        camoufox_config = {
            'humanize': True,
            # NOTE: 'screen' removed - Camoufox generates this internally to avoid browserforge version conflicts
        }
    
    # Enable geoip matching if using proxy (matches browser fingerprint to proxy IP location)
    if proxy_config and proxy_config.get('server'):
        camoufox_config['geoip'] = True  # Match timezone/locale to proxy IP location
        logger.debug("    GeoIP matching enabled for proxy")
    
    # CRITICAL: Need to create a new event loop in this thread to avoid conflict with parent async loop
    import asyncio
    try:
        # Try to get existing loop in this thread
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's a running loop (shouldn't happen in executor thread), create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No loop in this thread (expected in executor), create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    browser = Camoufox(headless=headless, **camoufox_config)
    
    with browser as b:
        # Get fingerprint from anti-detection manager
        if ANTI_DETECTION_AVAILABLE and anti_detection_config:
            fingerprint = anti_detect.fingerprint
            selected_ua = fingerprint.user_agent
            viewport = fingerprint.viewport
        else:
            # Fallback to random selection
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
            ]
            selected_ua = random.choice(user_agents)
            viewport = {
                'width': random.choice([1920, 1366, 1536, 1440]),
                'height': random.choice([1080, 768, 864, 900])
            }
        
        # Context options
        context_options = {
            'ignore_https_errors': True,
            'viewport': viewport,
            'user_agent': selected_ua
        }
        
        # Add proxy if configured
        if proxy_config and proxy_config.get('server'):
            # Ensure server format is correct (Camoufox expects http://host:port)
            server = proxy_config['server']
            if not server.startswith('http'):
                server = f"http://{server}"
            
            context_options['proxy'] = {
                'server': server,
                'username': proxy_config.get('username', ''),
                'password': proxy_config.get('password', '')
            }
            logger.debug(f"    Proxy configured: {server} (user: {proxy_config.get('username', '')[:20]}...)")
        else:
            logger.debug(f"    No proxy configured in context_options")
        
        # Create context and page
        context = b.new_context(**context_options)
        page = context.new_page()
        
        # Inject advanced anti-detection scripts (from Parsera project)
        page.add_init_script("""
            // Advanced anti-detection for heavy blocking sites
            
            // Override webdriver detection completely
            Object.defineProperty(navigator, 'webdriver', { 
                get: () => undefined,
                configurable: true 
            });
            
            // Realistic plugins array
            Object.defineProperty(navigator, 'plugins', {
                get: () => ({
                    length: 5,
                    0: { name: 'Chrome PDF Plugin', description: 'Portable Document Format' },
                    1: { name: 'Chromium PDF Plugin', description: 'Portable Document Format' },
                    2: { name: 'Microsoft Edge PDF Plugin', description: 'Portable Document Format' },
                    3: { name: 'PDF Viewer', description: 'Portable Document Format' },
                    4: { name: 'Chrome PDF Viewer', description: 'Portable Document Format' }
                }),
                configurable: true
            });
            
            // Realistic languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
                configurable: true
            });
            
            // Chrome app and runtime
            window.chrome = {
                app: { isInstalled: false, InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' } },
                runtime: { OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install', SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update' } }
            };
            
            // Permissions API
            if (navigator.permissions) {
                const originalQuery = navigator.permissions.query;
                navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ? 
                        Promise.resolve({ state: Notification.permission }) : 
                        originalQuery(parameters)
                );
            }
            
            // WebGL Vendor and Renderer
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Intel Inc.';
                if (parameter === 37446) return 'Intel Iris OpenGL Engine';
                return getParameter.call(this, parameter);
            };
            
            // Battery API
            if (navigator.getBattery) {
                navigator.getBattery = () => Promise.resolve({
                    charging: true,
                    chargingTime: 0,
                    dischargingTime: Infinity,
                    level: 1,
                    addEventListener: () => {},
                    removeEventListener: () => {},
                    dispatchEvent: () => true
                });
            }
            
            // Connection API
            Object.defineProperty(navigator, 'connection', {
                get: () => ({
                    effectiveType: '4g',
                    rtt: 100,
                    downlink: 10,
                    saveData: false,
                    addEventListener: () => {},
                    removeEventListener: () => {},
                    dispatchEvent: () => true
                }),
                configurable: true
            });
            
            // Hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8,
                configurable: true
            });
            
            // Device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8,
                configurable: true
            });
            
            // Screen properties
            Object.defineProperty(screen, 'colorDepth', { get: () => 24 });
            Object.defineProperty(screen, 'pixelDepth', { get: () => 24 });
        """)
        
        # Setup request/response monitoring for API capture
        def handle_response(response):
            try:
                url_resp = response.url
                content_type = response.headers.get('content-type', '')
                
                # UNIVERSAL: Detect API calls by multiple patterns
                is_api = (
                    '/api/' in url_resp.lower() or
                    '/v1/' in url_resp or '/v2/' in url_resp or '/v3/' in url_resp or  # Versioned APIs
                    '/graphql' in url_resp.lower() or  # GraphQL
                    '/rest/' in url_resp.lower() or
                    '/data/' in url_resp.lower() or
                    '/ajax/' in url_resp.lower() or
                    'json' in content_type.lower() or  # Content type check
                    (response.request.method in ['POST', 'PUT', 'PATCH'] and 'application' in content_type.lower())  # POST requests with data
                )
                
                if is_api:
                    captured_requests.append({
                        'url': url_resp,
                        'method': response.request.method,
                        'status': response.status,
                        'content_type': content_type
                    })
                    
                    # Try to extract JSON from any API-like response
                    if 'json' in content_type.lower() or response.status == 200:
                        try:
                            text = response.text()
                            if text and len(text) > 2:  # Not empty
                                data = json.loads(text)
                                # Only capture if it's a dict or list (actual data)
                                if isinstance(data, (dict, list)):
                                    captured_json.append({
                                        'source': 'api',
                                        'url': url_resp,
                                        'method': response.request.method,
                                        'data': data
                                    })
                        except:
                            pass
            except:
                pass
        
        page.on('response', handle_response)
        
        # Navigate to URL
        start_time = time.time()
        page.goto(url, wait_until='domcontentloaded', timeout=timeout)
        
        # Check if we got a Kasada challenge page (common on e-commerce sites)
        html_preview = page.content()[:500].lower()
        is_kasada_challenge = 'kasada' in html_preview or 'kpsdk' in html_preview or 'ips.js' in html_preview
        
        if is_kasada_challenge:
            logger.info("    Detected Kasada anti-bot challenge - waiting for challenge to complete...")
            # Wait longer for Kasada challenge to complete (can take 5-15 seconds)
            try:
                page.wait_for_load_state('networkidle', timeout=30000)  # Wait up to 30s for network idle
                # Additional wait for JavaScript execution
                time.sleep(5)  # Give Kasada time to solve
                logger.info("    Waited for Kasada challenge")
            except:
                logger.warning("    Kasada challenge timeout - continuing anyway")
        
        # UNIVERSAL SOLUTION 3: Smart Wait Strategy for JS-heavy sites
        # Adaptively waits for content to load without hardcoded delays
        _smart_wait_for_content(page, wait_for_selector)
        
        # Check if page content looks like a challenge/block page
        current_html = page.content()
        if len(current_html) < 2000 and ('kasada' in current_html.lower() or 'kpsdk' in current_html.lower() or 'ips.js' in current_html.lower()):
            logger.warning("    Page still appears to be Kasada challenge - waiting longer...")
            # Wait even longer and check again
            try:
                page.wait_for_load_state('networkidle', timeout=20000)
                time.sleep(10)  # Extra wait for challenge completion
                current_html = page.content()  # Refresh HTML
                logger.info(f"    After wait: {len(current_html):,} bytes")
            except:
                pass
        
        # Count initial API calls
        initial_api_count = len(captured_json)
        logger.debug(f"   Initial API calls captured: {initial_api_count}")
        
        # Additional wait time for JavaScript rendering (if explicitly requested)
        if wait_time > 0:
            time.sleep(wait_time / 1000)
        
        # UNIVERSAL ENHANCEMENT: Auto-scroll for SPAs to trigger lazy-loaded API calls
        # Many modern sites (Leafly, etc.) only load data when you scroll
        # We'll ALWAYS try scrolling on JS-heavy sites to capture more APIs
        should_scroll = scroll_to_bottom or True  # Always scroll to discover APIs
        
        if should_scroll:
            logger.debug("    Scrolling to trigger lazy-loaded content...")
            
            # Smooth scroll to trigger lazy loading
            page.evaluate("""
                (async () => {
                    const distance = 200;
                    const delay = 300;
                    const maxScrolls = 10;
                    
                    let scrollCount = 0;
                    while (scrollCount < maxScrolls && 
                           document.scrollingElement.scrollTop + window.innerHeight < document.scrollingElement.scrollHeight) {
                        document.scrollingElement.scrollBy(0, distance);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        scrollCount++;
                    }
                })();
            """)
            
            # Wait for new API calls to complete after scrolling
            # This is critical for capturing product/content APIs
            logger.debug("   ⏳ Waiting for API calls after scroll...")
            time.sleep(2)  # Give APIs time to fire
            
            try:
                # Wait for network idle again (new APIs might be loading)
                page.wait_for_load_state('networkidle', timeout=3000)
            except:
                pass  # Timeout is OK
            
            new_api_count = len(captured_json)
            if new_api_count > initial_api_count:
                logger.debug(f"    Captured {new_api_count - initial_api_count} additional APIs after scroll")
            else:
                logger.debug(f"   ℹ  No additional APIs captured")
        
        # Get final HTML
        html = page.content()
        elapsed_time = time.time() - start_time
        
        # Cleanup
        page.close()
        context.close()
        
        return {
            'html': html,
            'status': 200,
            'url': url,
            'api_calls': captured_requests,
            'json_data': captured_json,
            'elapsed_time': elapsed_time
        }


class CamoufoxFetcher:
    """
    Advanced browser fetcher using Camoufox for superior anti-detection
    
    Features:
    - Real browser fingerprints (not just stealth scripts)
    - Human-like behavior simulation
    - Better proxy support
    - Less likely to be detected than Playwright
    
    Note: Runs in a separate thread to avoid asyncio conflicts
    """
    
    def __init__(
        self,
        proxy_config: Optional[Dict[str, str]] = None,
        proxy_manager: Optional['ProxyManager'] = None,  # NEW: ProxyManager for rotation
        headless: bool = True,
        timeout: int = 60000,
        enable_js: bool = True,
        anti_detection_profile: str = 'random',  # NEW: Anti-detection profile
        humanize: bool = True,  # NEW: Enable human-like behavior
        stealth_mode: bool = True  # NEW: Maximum stealth (slower but harder to detect)
    ):
        """
        Initialize Camoufox fetcher
        
        Args:
            proxy_config: Static proxy configuration dict with 'server', 'username', 'password' (deprecated)
            proxy_manager: ProxyManager instance for per-request rotation (recommended)
            headless: Run in headless mode
            timeout: Page load timeout in milliseconds
            enable_js: Enable JavaScript rendering
            anti_detection_profile: Anti-detection profile ('random', 'windows_chrome', 'macos_chrome', 'linux_firefox')
            humanize: Enable human-like behavior (delays, mouse movement, etc.)
            stealth_mode: Maximum stealth mode (slower but harder to detect)
        """
        if not CAMOUFOX_AVAILABLE:
            raise ImportError("Camoufox is required. Install with: pip install camoufox")
        
        # Support both old (static) and new (manager) proxy approaches
        self.proxy_config = proxy_config  # For backward compatibility
        self.proxy_manager = proxy_manager  # NEW: For per-request rotation
        self.headless = headless
        self.timeout = timeout
        self.enable_js = enable_js
        
        # NEW: Store anti-detection config
        self.anti_detection_config = {
            'profile': anti_detection_profile,
            'humanize': humanize,
            'stealth_mode': stealth_mode
        }
        
        logger.info(f" Camoufox Fetcher initialized")
        logger.info(f"   Headless: {headless}, Timeout: {timeout}ms")
        logger.info(f"   Anti-Detection: Profile={anti_detection_profile}, Humanize={humanize}, Stealth={stealth_mode}")
        if proxy_manager:
            logger.info(f"   Proxy: ProxyManager enabled (per-request rotation)")
        elif proxy_config:
            logger.info(f"   Proxy: Static config enabled")
    
    async def _launch_browser(self):
        """Placeholder for compatibility - actual launch happens in _camoufox_fetch_sync"""
        pass
    
    async def fetch(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        wait_time: int = 2000,
        scroll_to_bottom: bool = False,
        click_load_more: Optional[str] = None  # For compatibility with HybridFetcher
    ) -> Dict[str, Any]:
        """
        Fetch page content with Camoufox
        
        Runs Camoufox in a separate thread to avoid asyncio conflicts
        
        Args:
            url: URL to fetch
            wait_for_selector: CSS selector to wait for before considering page loaded
            wait_time: Additional wait time in milliseconds after page load
            scroll_to_bottom: Whether to scroll to bottom for lazy-loaded content
            click_load_more: Not implemented for Camoufox (compatibility parameter)
            
        Returns:
            Dict with 'html', 'status', 'api_calls', 'json_data'
        """
        logger.info(f" Fetching with Camoufox: {url}")
        
        # NEW: Get fresh proxy for THIS request (Oxylabs approach)
        proxy_config_for_request = self.proxy_config  # Default: use static config
        
        if self.proxy_manager:
            # Try to get fresh proxy from manager (per-request rotation)
            try:
                # Check if we're in Apify context
                try:
                    from apify import Actor
                    # Get new proxy URL for THIS request
                    proxy_url = await self.proxy_manager.get_apify_proxy_url(Actor)
                    if proxy_url:
                        # Parse Apify proxy URL: http://username:password@host:port
                        # Convert to proxy_config format
                        proxy_config_for_request = self._parse_proxy_url(proxy_url)
                        logger.info(f" Using rotated Apify proxy for this request")
                except ImportError:
                    # Not in Apify context, use ProxyManager's pool
                    from urllib.parse import urlparse as parse_url
                    domain = parse_url(url).netloc
                    proxy_dict = self.proxy_manager.get_proxy(domain=domain)
                    if proxy_dict:
                        proxy_config_for_request = {
                            'server': proxy_dict['server'],
                            'username': proxy_dict.get('username', ''),
                            'password': proxy_dict.get('password', '')
                        }
                        logger.info(f" Using proxy from pool: {proxy_dict['server']}")
                    else:
                        # ProxyManager pool is empty, fall back to static config
                        logger.info(f" ProxyManager pool empty, using static proxy_config")
            except Exception as e:
                logger.warning(f" Proxy rotation failed, using fallback: {e}")
        
        # Log proxy being used
        if proxy_config_for_request:
            server = proxy_config_for_request.get('server', 'none')
            username = proxy_config_for_request.get('username', '')
            logger.info(f" Using proxy: {server} (user: {username[:30]}...)")
        else:
            logger.warning(f" No proxy configured for this request!")
        
        # Run the entire Camoufox session in a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _camoufox_fetch_sync,
            url,
            self.headless,
            proxy_config_for_request,  # Use per-request proxy!
            self.timeout,
            wait_for_selector,
            wait_time,
            scroll_to_bottom,
            self.anti_detection_config  # NEW: Pass anti-detection config
        )
        
        logger.info(f" Camoufox fetch complete: {len(result['html'])} bytes")
        logger.info(f" Captured {len(result['api_calls'])} API requests")
        logger.info(f" Extracted {len(result['json_data'])} JSON blobs")
        
        return result
    
    def _parse_proxy_url(self, proxy_url: str) -> Dict[str, str]:
        """
        Parse Apify proxy URL into proxy_config format.
        
        Args:
            proxy_url: Full proxy URL (http://username:password@host:port)
            
        Returns:
            Dict with 'server', 'username', 'password'
        """
        from urllib.parse import urlparse as parse_url
        parsed = parse_url(proxy_url)
        
        return {
            'server': f"{parsed.scheme}://{parsed.hostname}:{parsed.port}",
            'username': parsed.username or '',
            'password': parsed.password or ''
        }
    
    async def close(self):
        """Close browser and cleanup"""
        # Camoufox uses context manager, so cleanup is automatic
        logger.info(" Camoufox fetcher closed")
