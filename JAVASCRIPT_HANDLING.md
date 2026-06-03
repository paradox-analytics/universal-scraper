# JavaScript & Dynamic Content Handling Guide

## 🚨 Current Limitations

The Universal Scraper currently uses **CloudScraper** for fetching HTML, which has these characteristics:

### ✅ What It CAN Do:
- Fetch static HTML content
- Bypass Cloudflare protection  
- Handle anti-bot detection with realistic headers
- Work with residential proxies
- Retry failed requests

### ❌ What It CANNOT Do:
- **Execute JavaScript** - No JS rendering
- **Handle dynamic content** - Content loaded via AJAX/fetch won't appear
- **Interact with pages** - Can't click buttons, scroll, or fill forms
- **Wait for elements** - Can't wait for dynamically loaded content
- **Handle pagination** - Can't click "Load More" or navigate pages
- **Discover infinite scroll** - Can't trigger scroll events

## 📊 Test Results: Leafly Dispensary Menu

**URL Tested:** `https://www.leafly.com/dispensary-info/mammoth-holistics/menu`

**Result:**
```
Items extracted: 0
Source: html
Execution time: 11.07s
```

**Why it failed:**
- Leafly uses React/Next.js to render product listings
- Product data is loaded dynamically via JavaScript
- The initial HTML only contains skeleton/framework
- CloudScraper only sees empty placeholder divs

## 🔧 Solutions: Adding JavaScript Support

### Solution 1: Playwright Integration (Recommended)

**Pros:**
- ✅ Full browser automation
- ✅ JavaScript rendering
- ✅ Page interactions (click, scroll, type)
- ✅ Wait for elements
- ✅ Screenshot and video recording
- ✅ Network interception (can capture API calls!)
- ✅ Multi-browser support (Chromium, Firefox, WebKit)
- ✅ Async/await support

**Cons:**
- ❌ Slower than CloudScraper (2-5x)
- ❌ Higher resource usage (memory, CPU)
- ❌ Requires browser installation
- ❌ More complex error handling

**Implementation:**

```python
# Add to requirements.txt
playwright>=1.40.0

# New file: universal_scraper/core/browser_fetcher.py
"""
Browser-based HTML fetcher using Playwright
Handles JavaScript-rendered content
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

logger = logging.getLogger(__name__)


class BrowserFetcher:
    """
    Fetches HTML with JavaScript rendering using Playwright
    """
    
    def __init__(
        self,
        headless: bool = True,
        proxy_config: Optional[Dict[str, str]] = None,
        timeout: int = 30000,  # 30 seconds
        wait_for_selector: Optional[str] = None,
        wait_for_timeout: int = 5000  # 5 seconds after page load
    ):
        """
        Initialize Browser Fetcher
        
        Args:
            headless: Run browser in headless mode
            proxy_config: Proxy configuration
            timeout: Navigation timeout in milliseconds
            wait_for_selector: Optional selector to wait for
            wait_for_timeout: Additional wait time after page load
        """
        self.headless = headless
        self.proxy_config = proxy_config
        self.timeout = timeout
        self.wait_for_selector = wait_for_selector
        self.wait_for_timeout = wait_for_timeout
        self.playwright = None
        self.browser = None
        self.context = None
    
    async def initialize(self) -> None:
        """Initialize Playwright browser"""
        self.playwright = await async_playwright().start()
        
        # Browser launch options
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
            launch_options['proxy'] = {
                'server': self.proxy_config['server'],
                'username': self.proxy_config.get('username'),
                'password': self.proxy_config.get('password')
            }
        
        # Launch browser
        self.browser = await self.playwright.chromium.launch(**launch_options)
        
        # Create context with anti-detection
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            permissions=['geolocation']
        )
        
        # Add anti-detection script
        await self.context.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            
            // Mock chrome properties
            window.chrome = {
                runtime: {}
            };
            
            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        logger.info("🌐 Browser initialized")
    
    async def fetch(
        self,
        url: str,
        wait_for_network_idle: bool = True,
        capture_requests: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch page with JavaScript rendering
        
        Args:
            url: Target URL
            wait_for_network_idle: Wait for network to be idle
            capture_requests: Capture network requests (useful for API discovery)
            
        Returns:
            Dict with 'html', 'url', 'requests' keys
        """
        if not self.context:
            await self.initialize()
        
        page = await self.context.new_page()
        captured_requests = []
        
        try:
            # Capture network requests if requested
            if capture_requests:
                async def handle_request(request):
                    captured_requests.append({
                        'url': request.url,
                        'method': request.method,
                        'resource_type': request.resource_type,
                        'headers': request.headers
                    })
                
                page.on('request', handle_request)
            
            logger.info(f"🌐 Navigating to: {url}")
            
            # Navigate with options
            wait_until = 'networkidle' if wait_for_network_idle else 'domcontentloaded'
            await page.goto(url, timeout=self.timeout, wait_until=wait_until)
            
            # Wait for specific selector if provided
            if self.wait_for_selector:
                logger.info(f"⏳ Waiting for selector: {self.wait_for_selector}")
                await page.wait_for_selector(self.wait_for_selector, timeout=self.timeout)
            
            # Additional wait for dynamic content
            if self.wait_for_timeout > 0:
                await asyncio.sleep(self.wait_for_timeout / 1000)
            
            # Get final HTML
            html = await page.content()
            final_url = page.url
            
            logger.info(f"✅ Page loaded: {len(html)} bytes")
            
            return {
                'html': html,
                'url': final_url,
                'status_code': 200,
                'requests': captured_requests if capture_requests else []
            }
            
        except Exception as e:
            logger.error(f"❌ Browser fetch failed: {str(e)}")
            raise
        finally:
            await page.close()
    
    async def fetch_with_interaction(
        self,
        url: str,
        interactions: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fetch page with interactions (scroll, click, etc.)
        
        Args:
            url: Target URL
            interactions: List of interaction dicts:
                {'type': 'click', 'selector': '#load-more'}
                {'type': 'scroll', 'direction': 'bottom'}
                {'type': 'wait', 'timeout': 2000}
                {'type': 'fill', 'selector': '#search', 'value': 'product'}
        
        Returns:
            Dict with 'html', 'url' keys
        """
        if not self.context:
            await self.initialize()
        
        page = await self.context.new_page()
        
        try:
            # Navigate
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # Execute interactions
            for interaction in interactions:
                action_type = interaction['type']
                
                if action_type == 'click':
                    logger.info(f"🖱️ Clicking: {interaction['selector']}")
                    await page.click(interaction['selector'])
                    await page.wait_for_load_state('networkidle')
                
                elif action_type == 'scroll':
                    direction = interaction.get('direction', 'bottom')
                    if direction == 'bottom':
                        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await asyncio.sleep(1)
                
                elif action_type == 'wait':
                    await asyncio.sleep(interaction['timeout'] / 1000)
                
                elif action_type == 'fill':
                    await page.fill(interaction['selector'], interaction['value'])
                
                elif action_type == 'wait_for_selector':
                    await page.wait_for_selector(interaction['selector'])
            
            # Get final HTML
            html = await page.content()
            
            return {
                'html': html,
                'url': page.url,
                'status_code': 200
            }
            
        finally:
            await page.close()
    
    async def discover_pagination(
        self,
        url: str,
        max_pages: int = 10
    ) -> list[str]:
        """
        Discover pagination links
        
        Args:
            url: Starting URL
            max_pages: Maximum pages to discover
            
        Returns:
            List of URLs
        """
        if not self.context:
            await self.initialize()
        
        page = await self.context.new_page()
        discovered_urls = [url]
        
        try:
            await page.goto(url, wait_until='networkidle')
            
            # Common pagination selectors
            pagination_selectors = [
                'a[rel="next"]',
                '.pagination a:last-child',
                '[aria-label="Next"]',
                'button:has-text("Next")',
                'a:has-text("Next")',
                '.next',
                '#next'
            ]
            
            for i in range(max_pages - 1):
                # Try to find next button
                next_button = None
                for selector in pagination_selectors:
                    try:
                        next_button = await page.query_selector(selector)
                        if next_button:
                            break
                    except:
                        continue
                
                if not next_button:
                    logger.info(f"📄 No more pages found after page {i + 1}")
                    break
                
                # Click next
                await next_button.click()
                await page.wait_for_load_state('networkidle')
                
                current_url = page.url
                if current_url not in discovered_urls:
                    discovered_urls.append(current_url)
                    logger.info(f"📄 Discovered page {len(discovered_urls)}: {current_url}")
                else:
                    break
            
            return discovered_urls
            
        finally:
            await page.close()
    
    async def close(self) -> None:
        """Clean up browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("👋 Browser closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

**Usage Example:**

```python
import asyncio
from universal_scraper.core.browser_fetcher import BrowserFetcher

async def scrape_with_browser():
    async with BrowserFetcher(headless=True) as fetcher:
        # Simple fetch
        result = await fetcher.fetch(
            "https://www.leafly.com/dispensary-info/mammoth-holistics/menu",
            wait_for_network_idle=True
        )
        print(f"Fetched {len(result['html'])} bytes")
        
        # Fetch with interaction (load more products)
        result = await fetcher.fetch_with_interaction(
            "https://example.com/products",
            interactions=[
                {'type': 'wait', 'timeout': 2000},
                {'type': 'scroll', 'direction': 'bottom'},
                {'type': 'click', 'selector': '#load-more'},
                {'type': 'wait', 'timeout': 2000}
            ]
        )
        
        # Discover pagination
        pages = await fetcher.discover_pagination(
            "https://example.com/products",
            max_pages=10
        )
        print(f"Found {len(pages)} pages")

# Run
asyncio.run(scrape_with_browser())
```

### Solution 2: API Detection & Interception

**Often better than browser automation!**

Many JavaScript-heavy sites load data from APIs. We can:
1. Use Playwright to capture network requests
2. Identify the API endpoints
3. Call them directly (much faster!)

**Example - Leafly Site:**

```python
async def discover_api_endpoints(url: str):
    """Discover API endpoints used by a page"""
    async with BrowserFetcher() as fetcher:
        result = await fetcher.fetch(url, capture_requests=True)
        
        # Filter for JSON/API requests
        api_requests = [
            req for req in result['requests']
            if req['resource_type'] in ['xhr', 'fetch'] or
            'api' in req['url'].lower() or
            'json' in req['url'].lower()
        ]
        
        print("🔍 Discovered API endpoints:")
        for req in api_requests:
            print(f"  {req['method']} {req['url']}")
        
        return api_requests

# Then call the API directly!
import requests
api_url = "https://api.leafly.com/v1/dispensaries/mammoth-holistics/menu"
response = requests.get(api_url, headers={'Authorization': 'Bearer ...'})
data = response.json()
```

### Solution 3: Hybrid Approach (Best of Both Worlds)

```python
class HybridFetcher:
    """
    Smart fetcher that chooses the best method:
    1. Try static HTML with CloudScraper (fast)
    2. If insufficient data, try browser (slower but complete)
    3. Look for API endpoints while browsing
    """
    
    def __init__(self):
        self.static_fetcher = HTMLFetcher()
        self.browser_fetcher = None
        self.discovered_apis = {}
    
    async def fetch(self, url: str):
        # Try static first
        static_result = self.static_fetcher.fetch(url)
        
        # Check if page has content
        soup = BeautifulSoup(static_result['html'], 'html.parser')
        if self._has_sufficient_content(soup):
            return static_result
        
        # Fall back to browser
        logger.info("⚠️ Static HTML insufficient, using browser...")
        if not self.browser_fetcher:
            self.browser_fetcher = BrowserFetcher()
            await self.browser_fetcher.initialize()
        
        browser_result = await self.browser_fetcher.fetch(
            url,
            capture_requests=True
        )
        
        # Cache discovered APIs
        self._cache_api_endpoints(browser_result['requests'])
        
        return browser_result
```

## 📖 Pagination Handling

### Pagination Types:

1. **Traditional Pagination (Links)**
   ```html
   <a href="/products?page=2">Next</a>
   ```
   - Easiest to handle
   - Just extract all page URLs

2. **Load More Buttons**
   ```html
   <button onclick="loadMore()">Load More</button>
   ```
   - Requires browser interaction
   - Use `fetch_with_interaction()`

3. **Infinite Scroll**
   ```javascript
   window.addEventListener('scroll', loadMore)
   ```
   - Requires browser + scroll simulation
   - Use Playwright scroll actions

4. **API Pagination**
   ```
   GET /api/products?page=1&limit=20
   ```
   - Best performance
   - Call API directly with different params

### Implementation Example:

```python
async def scrape_with_pagination(base_url: str, scraper):
    """Scrape all pages of paginated content"""
    async with BrowserFetcher() as fetcher:
        # Discover all pages
        pages = await fetcher.discover_pagination(base_url, max_pages=50)
        
        # Scrape each page
        all_results = []
        for page_url in pages:
            result = scraper.scrape(page_url, fields=['product_name', 'price'])
            all_results.extend(result['data'])
        
        return all_results
```

## 🔍 Discovering New Pages & Elements

### Method 1: Sitemap Crawling

```python
import requests
from xml.etree import ElementTree

def discover_from_sitemap(base_url: str) -> list[str]:
    """Extract URLs from sitemap.xml"""
    sitemap_urls = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/sitemap-products.xml"
    ]
    
    discovered = []
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url)
            root = ElementTree.fromstring(response.content)
            
            # Extract URLs
            for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                discovered.append(url.text)
        except:
            continue
    
    return discovered
```

### Method 2: Link Extraction

```python
def discover_links(html: str, base_url: str, pattern: str = None) -> list[str]:
    """Extract all links matching a pattern"""
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        
        # Make absolute
        if href.startswith('/'):
            href = base_url + href
        
        # Filter by pattern
        if pattern and pattern not in href:
            continue
        
        links.append(href)
    
    return list(set(links))  # Deduplicate

# Usage
result = fetcher.fetch("https://example.com/products")
product_urls = discover_links(result['html'], "https://example.com", pattern="/product/")
```

### Method 3: Intelligent Crawling

```python
class SmartCrawler:
    """Intelligent crawler that discovers and scrapes pages"""
    
    def __init__(self, scraper, max_depth: int = 3):
        self.scraper = scraper
        self.max_depth = max_depth
        self.visited = set()
        self.discovered_data = []
    
    async def crawl(self, start_url: str, url_pattern: str, fields: list[str]):
        """Crawl and scrape pages matching pattern"""
        await self._crawl_recursive(start_url, url_pattern, fields, depth=0)
        return self.discovered_data
    
    async def _crawl_recursive(self, url: str, pattern: str, fields: list[str], depth: int):
        if depth > self.max_depth or url in self.visited:
            return
        
        self.visited.add(url)
        logger.info(f"🕷️ Crawling (depth {depth}): {url}")
        
        # Fetch and scrape
        result = self.scraper.scrape(url, fields)
        self.discovered_data.extend(result['data'])
        
        # Discover new links
        links = discover_links(result['metadata']['html'], url, pattern=pattern)
        
        # Crawl new links
        for link in links:
            await self._crawl_recursive(link, pattern, fields, depth + 1)
```

## 📊 Comparison: Static vs Browser

| Feature | CloudScraper (Current) | Playwright (Proposed) |
|---------|------------------------|----------------------|
| Speed | ⚡ Very Fast (0.5-2s) | 🐌 Slower (2-10s) |
| JavaScript | ❌ No | ✅ Yes |
| Memory | 💚 Low (~10MB) | 💛 Higher (~100MB) |
| Anti-Detection | 🟢 Good | 🟢 Excellent |
| Interactions | ❌ No | ✅ Yes |
| API Discovery | ❌ No | ✅ Yes |
| Cost per page | $ Low | $$ Higher |
| Setup | 🟢 Simple | 🟡 Moderate |

## 🎯 Recommendations

1. **For Leafly and similar JS-heavy sites:**
   - Use Playwright with API discovery
   - Capture network requests to find APIs
   - Call APIs directly for best performance

2. **For static sites:**
   - Keep using CloudScraper
   - Much faster and cheaper

3. **Hybrid approach:**
   - Try static first
   - Fall back to browser if needed
   - Cache discovered APIs for future use

4. **For pagination:**
   - Detect pagination type
   - Use appropriate method
   - Prefer API pagination when possible

## 🚀 Next Steps

To add full JavaScript support to this project:

1. Install Playwright:
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Add `BrowserFetcher` class (code provided above)

3. Update `UniversalScraper` to support mode selection:
   ```python
   scraper = UniversalScraper(
       fetch_mode='browser',  # or 'static' or 'hybrid'
       api_key="..."
   )
   ```

4. Update Apify actor to support browser mode

Would you like me to implement any of these solutions?

