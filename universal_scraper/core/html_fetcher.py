"""
HTML Fetcher with CloudScraper and Proxy Support
Handles fetching web pages with anti-bot protection and residential proxies
"""

import time
import random
import logging
from typing import Optional, Dict, Any
import cloudscraper
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class HTMLFetcher:
    """Fetches HTML content with anti-blocking and proxy support"""
    
    # Realistic user agents
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
    ]
    
    def __init__(
        self,
        proxy_config: Optional[Dict[str, str]] = None,
        proxy_manager: Optional[Any] = None,  # NEW: ProxyManager for rotation
        enable_warming: bool = True,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize HTML Fetcher
        
        Args:
            proxy_config: Static proxy dict with 'server', 'username', 'password' keys (deprecated)
            proxy_manager: ProxyManager instance for per-request rotation (recommended)
            enable_warming: Enable session warming for better success rates
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.proxy_config = proxy_config
        self.proxy_manager = proxy_manager  # NEW: Store ProxyManager
        self.enable_warming = enable_warming
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        self.warmed_domains = set()
        self.request_count = 0
        self.last_request_time = 0
        
        self._create_session()
    
    def _create_session(self) -> None:
        """Create CloudScraper session with anti-blocking features"""
        # Create cloudscraper session (handles Cloudflare, etc.)
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
        
        # Select random user agent
        selected_ua = random.choice(self.USER_AGENTS)
        selected_platform = random.choice(['"macOS"', '"Windows"', '"Linux"'])
        
        # Enhanced anti-fingerprinting headers
        self.session.headers.update({
            'User-Agent': selected_ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.9', 'en-CA,en;q=0.9']),
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': random.choice(['1', '0']),
            'Connection': 'keep-alive',
            'Cache-Control': random.choice(['max-age=0', 'no-cache']),
            'Upgrade-Insecure-Requests': '1',
            'sec-ch-ua': f'"Not_A Brand";v="8", "Chromium";v="121", "Google Chrome";v="121"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': selected_platform,
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['none', 'same-origin']),
            'Sec-Fetch-User': '?1',
        })
        
        logger.info(f" Using user agent: {selected_ua[:60]}...")
        
        # Enhanced retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Configure proxies if provided (static format only)
        # Note: Apify proxy format is handled by ProxyManager, not here
        if self.proxy_config and 'server' in self.proxy_config:
            # Static proxy format: {'server': '...', 'username': '...', 'password': '...'}
            username = self.proxy_config.get('username', '')
            password = self.proxy_config.get('password', '')
            server = self.proxy_config['server'].replace('http://', '').replace('https://', '')
            
            if username and password:
                proxy_url = f"http://{username}:{password}@{server}"
            else:
                proxy_url = f"http://{server}"
            
            self.session.proxies.update({
                'http': proxy_url,
                'https': proxy_url
            })
            logger.info(f" Using static proxy: {self.proxy_config['server']}")
    
    def fetch(self, url: str, warm_session: bool = None) -> Dict[str, Any]:
        """
        Fetch HTML content from URL
        
        Args:
            url: Target URL to fetch
            warm_session: Override default warming behavior
            
        Returns:
            Dict with 'html', 'url', 'status_code', 'headers' keys
        """
        if warm_session is None:
            warm_session = self.enable_warming
        
        # Warm session if enabled
        if warm_session:
            self._warm_session_for_domain(url)
        
        # Intelligent delay
        self._intelligent_delay()
        
        # Attempt fetch with retries
        for attempt in range(self.max_retries):
            try:
                self.request_count += 1
                self.last_request_time = time.time()
                
                logger.info(f" Fetching: {url[:80]}..." if len(url) > 80 else f" Fetching: {url}")
                
                response = self.session.get(url, timeout=self.timeout)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    backoff_delay = min(retry_after, 2 ** attempt * 30)
                    logger.warning(f"⏳ Rate limited (429). Backing off for {backoff_delay}s...")
                    time.sleep(backoff_delay)
                    continue
                
                # Log result
                if response.status_code == 200:
                    logger.info(f" Success: {response.status_code} ({len(response.text)} bytes)")
                else:
                    logger.warning(f" Response: {response.status_code}")
                
                return {
                    'html': response.text,
                    'url': response.url,
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'cookies': dict(response.cookies)
                }
                
            except Exception as e:
                logger.error(f" Fetch failed (attempt {attempt + 1}/{self.max_retries}): {str(e)[:100]}")
                if attempt < self.max_retries - 1:
                    backoff_delay = 2 ** attempt * 5
                    logger.info(f"⏳ Retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    raise
        
        raise Exception(f"Failed to fetch {url} after {self.max_retries} attempts")
    
    def _warm_session_for_domain(self, target_url: str) -> bool:
        """Warm session with domain-specific entry points"""
        parsed = urlparse(target_url)
        domain = parsed.netloc.lower()
        
        if domain in self.warmed_domains:
            return True
        
        logger.info(f" Warming session for: {domain}")
        
        # Generic warm-up URL (homepage)
        warm_url = f"{parsed.scheme}://{parsed.netloc}"
        
        try:
            self._intelligent_delay(base_delay=1, variation=1)
            response = self.session.get(warm_url, timeout=self.timeout)
            
            if response.status_code == 200:
                logger.info(f" Session warmed: {response.status_code}")
                self.warmed_domains.add(domain)
                return True
            else:
                logger.warning(f" Warm-up returned: {response.status_code}")
        except Exception as e:
            logger.warning(f" Warm-up failed: {str(e)[:50]}")
        
        # Mark as warmed even if failed to avoid repeated attempts
        self.warmed_domains.add(domain)
        return False
    
    def _intelligent_delay(self, base_delay: float = 3, variation: float = 2) -> None:
        """Intelligent delay with adaptive timing"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time if self.last_request_time > 0 else float('inf')
        
        # Adaptive delay based on request frequency
        if self.request_count > 20:
            base_delay *= 1.5
        if self.request_count > 50:
            base_delay *= 2
        
        # Calculate actual delay needed
        min_delay = base_delay
        actual_delay = max(0, min_delay - time_since_last)
        
        # Add random variation
        if actual_delay > 0:
            variation_amount = random.uniform(-variation, variation)
            actual_delay = max(0.5, actual_delay + variation_amount)
            logger.debug(f"⏳ Delay: {actual_delay:.1f}s (requests: {self.request_count})")
            time.sleep(actual_delay)
    
    def close(self) -> None:
        """Close the session"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

