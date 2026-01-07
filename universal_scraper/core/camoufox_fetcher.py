"""
Camoufox Browser Fetcher - Advanced anti-detection browser automation
Inspired by the Parsera project's successful Camoufox implementation

Note: Camoufox uses Playwright's sync API, so we run it in a separate thread
to avoid conflicts with asyncio.
"""

import asyncio
import logging
import os
import random
import re
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

# Camoufox will be imported inside the sync function to avoid asyncio conflicts
CAMOUFOX_AVAILABLE = True
try:
    import camoufox
except ImportError:
    CAMOUFOX_AVAILABLE = False
    logger.warning(" Camoufox not installed. Install with: pip install camoufox")

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
    internal_log = []
    
    def log_internal(message: str):
        internal_log.append({
            'timestamp': time.time(),
            'message': message
        })
        logger.info(f"    [Camoufox] {message}")
    
    # CRITICAL: Import Camoufox inside the thread to avoid asyncio loop detection
    from camoufox.sync_api import Camoufox
    
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
    
    # Add proxy to Camoufox constructor if configured
    if proxy_config and proxy_config.get('server'):
        server = proxy_config['server']
        # Bright Data Web Unblocker (33335) often works better with https for the proxy connection itself
        if '33335' in server and not server.startswith('http'):
            server = f"https://{server}"
        elif not server.startswith('http'):
            server = f"http://{server}"
            
        camoufox_config['proxy'] = {
            'server': server,
            'username': proxy_config.get('username', ''),
            'password': proxy_config.get('password', '')
        }
        log_internal(f"Proxy configured: {server}")
    
    # CRITICAL: Explicitly set the event loop to None in this thread.
    # Playwright Sync API checks `asyncio.get_event_loop()` and errors if it returns a running loop.
    # Even in a thread executor, some environments might leak a loop or have a default one.
    # Setting it to None ensures Playwright sees a clean state.
    import asyncio
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass  # Ignore errors if we can't set it (unlikely)
    
    # Log config (masking password)
    safe_config = camoufox_config.copy()
    if 'proxy' in safe_config:
        safe_config['proxy'] = safe_config['proxy'].copy()
        safe_config['proxy']['password'] = '********'
    log_internal(f"Launching Camoufox with config: {safe_config}")
    
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
        
        # Proxy is now handled at the browser level in Camoufox constructor
        pass
        
        # Create context and page
        context = b.new_context(**context_options)
        page = context.new_page()
        
        # IP Verification (Debug)
        try:
            page.goto("https://api.ipify.org?format=json", timeout=10000)
            ip_data = page.content()
            log_internal(f"Proxy IP check result: {ip_data[:200]}")
        except Exception as e:
            logger.warning(f"    Proxy IP check failed: {e}")
        
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
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=timeout)
        except Exception as e:
            logger.warning(f"    Navigation error: {e}")
            # If we have some content, continue. If not, return error.
            if len(page.content()) < 100:
                return {
                    'html': page.content(),
                    'status_code': 500,
                    'status': 500,
                    'url': url,
                    'error': str(e),
                    'api_calls': captured_requests,
                    'json_data': captured_json,
                    'internal_log': internal_log,
                    'elapsed_time': time.time() - start_time
                }
        
        # Check if we got a Kasada or Cloudflare challenge page
        html_preview = page.content()[:2000].lower()
        is_kasada_challenge = 'kasada' in html_preview or 'kpsdk' in html_preview or 'ips.js' in html_preview
        is_cloudflare_challenge = 'verify you are human' in html_preview or 'just a moment' in html_preview or 'cloudflare-static' in html_preview
        
        if is_kasada_challenge or is_cloudflare_challenge:
            challenge_type = "Kasada" if is_kasada_challenge else "Cloudflare"
            log_internal(f"Detected {challenge_type} challenge - waiting...")
            # Wait longer for challenge to complete (can take 5-15 seconds)
            try:
                page.wait_for_load_state('networkidle', timeout=30000)  # Wait up to 30s for network idle
                # Additional wait for JavaScript execution/challenge solving
                time.sleep(8)  # Give it time to solve
                log_internal(f"Waited for {challenge_type} challenge")
            except:
                logger.warning(f"    {challenge_type} challenge timeout - continuing anyway")
        
        # UNIVERSAL SOLUTION 3: Smart Wait Strategy for JS-heavy sites
        # Adaptively waits for content to load without hardcoded delays
        _smart_wait_for_content(page, wait_for_selector)
        
        # Check if page content still looks like a challenge/block page
        current_html = page.content()
        html_lower = current_html.lower()
        if len(current_html) < 5000 and ('kasada' in html_lower or 'kpsdk' in html_lower or 'verify you are human' in html_lower or 'just a moment' in html_lower):
            logger.warning("    Page still appears to be a challenge - waiting longer...")
            # Wait even longer and check again
            try:
                page.wait_for_load_state('networkidle', timeout=20000)
                time.sleep(12)  # Extra wait for challenge completion
                current_html = page.content()  # Refresh HTML
                log_internal(f"After extended wait: {len(current_html):,} bytes")
            except:
                pass
        
        # Count initial API calls
        initial_api_count = len(captured_json)
        logger.debug(f"   Initial API calls captured: {initial_api_count}")
        
        # Additional wait time for JavaScript rendering (if explicitly requested)
        if wait_time > 0:
            time.sleep(wait_time / 1000)
        
        # Universal infinite scroll detection and scrolling
        if scroll_to_bottom:
            logger.info("    Scrolling to trigger lazy-loaded content (infinite scroll)...")
            
            # Universal item detection - find repeating patterns dynamically
            # This works for any website by detecting common repeating structures
            scroll_result = page.evaluate("""
                (async () => {
                    // Universal item detection - find repeating containers
                    function detectRepeatingItems() {
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
                            '[id*="item"]',
                            '[id*="product"]',
                            'li[class*="item"]',
                            'div[class*="item"]',
                            'div[class*="card"]',
                            'div[class*="product"]',
                            'section > div',
                            'main > div > div',
                            '[data-component*="item"]',
                            '[data-component*="card"]'
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
                        
                        // Fallback: return a generic selector that should work
                        return 'article, [role="article"], div[class*="item"], div[class*="card"]';
                    }
                    
                    const itemSelector = detectRepeatingItems();
                    const distance = 500;
                    const delay = 500;
                    const maxScrolls = 30;  // Increased for better coverage
                    const maxNoChange = 5;  // Increased tolerance for slow-loading sites
                    
                    let scrollCount = 0;
                    let noChangeCount = 0;
                    let prevHeight = document.scrollingElement.scrollHeight;
                    let prevItemCount = document.querySelectorAll(itemSelector).length;
                    
                    while (scrollCount < maxScrolls) {
                        // Scroll down
                        document.scrollingElement.scrollBy(0, distance);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        
                        // Check if new content loaded
                        const newHeight = document.scrollingElement.scrollHeight;
                        const newItemCount = document.querySelectorAll(itemSelector).length;
                        
                        if (newHeight > prevHeight || newItemCount > prevItemCount) {
                            scrollCount++;
                            noChangeCount = 0;
                            prevHeight = newHeight;
                            prevItemCount = newItemCount;
                        } else {
                            noChangeCount++;
                            if (noChangeCount >= maxNoChange) {
                                break;  // No new content after maxNoChange tries
                            }
                        }
                    }
                    
                    return {
                        scrollCount: scrollCount,
                        finalItemCount: prevItemCount,
                        finalHeight: prevHeight,
                        itemSelector: itemSelector
                    };
                })();
            """)
            
            if scroll_result:
                log_internal(f"Scrolled {scroll_result.get('scrollCount', 0)} times, found {scroll_result.get('finalItemCount', 0)} items")
            
            # Wait for new API calls to complete after scrolling
            logger.debug("   ⏳ Waiting for API calls after scroll...")
            time.sleep(3)  # Give APIs more time to fire for Reddit
            
            try:
                # Wait for network idle again (new APIs might be loading)
                page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass  # Timeout is OK
            
            new_api_count = len(captured_json)
            if new_api_count > initial_api_count:
                logger.info(f"    Captured {new_api_count - initial_api_count} additional APIs after scroll")
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
            'status_code': 200,  # Playwright/Camoufox usually only returns if successful or handles errors
            'status': 200,       # Keep for compatibility
            'url': url,
            'api_calls': captured_requests,
            'json_data': captured_json,
            'internal_log': internal_log,
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
        stealth_mode: bool = True,  # NEW: Maximum stealth (slower but harder to detect)
        web_unblocker_api_key: Optional[str] = None,
        web_unblocker_zone: str = "web_unlocker1"
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
        self.web_unblocker_api_key = web_unblocker_api_key
        self.web_unblocker_zone = web_unblocker_zone
        
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
        
        # Add Web Unblocker if provided and no proxy yet
        if not proxy_config_for_request and self.web_unblocker_api_key:
            # Detect if it's proxy credentials format or API key
            # Split only on first colon to handle passwords with colons
            if ':' in self.web_unblocker_api_key:
                parts = self.web_unblocker_api_key.split(':', 1)  # Split only on FIRST colon
                if len(parts) == 2:
                    # user:pass format
                    username = parts[0].strip()
                    password = parts[1].strip()
                    proxy_config_for_request = {
                        'server': 'brd.superproxy.io:33335',
                        'username': username,
                        'password': password
                    }
                    logger.info(f"🔐 Using Web Unblocker proxy (user: {username[:50]}...)")
                else:
                    # Shouldn't happen with split(':', 1), but fallback just in case
                    customer_id = os.getenv('WEB_UNBLOCKER_CUSTOMER_ID', 'hl_803e8195')
                    proxy_config_for_request = {
                        'server': 'brd.superproxy.io:33335',
                        'username': f'brd-customer-{customer_id}-zone-{self.web_unblocker_zone}',
                        'password': self.web_unblocker_api_key
                    }
                    logger.info(f"🔐 Using Web Unblocker proxy (fallback, customer: {customer_id})")
            elif ',' in self.web_unblocker_api_key:
                # Comma-separated format (legacy)
                parts = self.web_unblocker_api_key.split(',')
                if len(parts) >= 4:
                    host = parts[0].strip()
                    port = parts[1].strip()
                    username = parts[2].strip()
                    password = parts[3].strip()
                    proxy_config_for_request = {
                        'server': f"{host}:{port}",
                        'username': username,
                        'password': password
                    }
                    logger.info(f"🔐 Using Web Unblocker proxy (comma-separated)")
                else:
                    # Fallback for other colon counts
                    customer_id = os.getenv('WEB_UNBLOCKER_CUSTOMER_ID', 'hl_803e8195')
                    proxy_config_for_request = {
                        'server': 'brd.superproxy.io:33335',
                        'username': f'brd-customer-{customer_id}-zone-{self.web_unblocker_zone}',
                        'password': self.web_unblocker_api_key
                    }
                    logger.info(f" Using Web Unblocker API key as proxy for Camoufox (fallback, customer: {customer_id})")
            elif ',' in self.web_unblocker_api_key and self.web_unblocker_api_key.count(',') >= 3:
                parts = self.web_unblocker_api_key.split(',')
                if len(parts) >= 4:
                    host = parts[0].strip()
                    port = parts[1].strip()
                    username = parts[2].strip()
                    password = parts[3].strip()
                    proxy_config_for_request = {
                        'server': f"{host}:{port}",
                        'username': username,
                        'password': password
                    }
                    logger.info(f" Using Web Unblocker as proxy for Camoufox (csv)")
            else:
                # Use as API key (Bearer token)
                customer_id = os.getenv('WEB_UNBLOCKER_CUSTOMER_ID', 'hl_803e8195')
                proxy_config_for_request = {
                    'server': 'brd.superproxy.io:33335',
                    'username': f'brd-customer-{customer_id}-zone-{self.web_unblocker_zone}',
                    'password': self.web_unblocker_api_key
                }
                logger.info(f" Using Web Unblocker API key as proxy for Camoufox (Bearer, customer: {customer_id})")
        
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
