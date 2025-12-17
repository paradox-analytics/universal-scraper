"""
Bright Data Web Unblocker Fetcher

Uses Bright Data's Web Unblocker API to bypass anti-bot protection (Kasada, etc.)
Falls back to this when standard residential proxies fail.

API Documentation: https://brightdata.com/products/web-unblocker
"""

import logging
import json
import time
from typing import Dict, Any, Optional
import requests
import urllib3

# Disable SSL warnings for proxy (Bright Data uses self-signed certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class WebUnblockerFetcher:
    """
    Bright Data Web Unblocker API client
    
    Automatically handles:
    - Kasada challenges
    - Cloudflare protection
    - Other anti-bot systems
    
    Usage:
        fetcher = WebUnblockerFetcher(
            api_key="your-bright-data-api-key",
            zone="web_unlocker1"  # Your Web Unblocker zone
        )
        result = fetcher.fetch("https://example.com")
    """
    
    API_BASE_URL = "https://api.brightdata.com/request"
    
    def __init__(
        self,
        api_key: str,
        zone: str = "web_unlocker1",
        timeout: int = 120,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize Web Unblocker fetcher
        
        Args:
            api_key: Bright Data API key (Bearer token) OR proxy credentials in format host:port:username:password or host,port,username,password
            zone: Web Unblocker zone name (default: web_unlocker1)
            timeout: Request timeout in seconds
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum retry attempts
        """
        # Detect if it's proxy credentials format
        separator = None
        if ':' in api_key and api_key.count(':') >= 3:
            separator = ':'
        elif ',' in api_key and api_key.count(',') >= 3:
            separator = ','
        
        if separator:
            # Proxy credentials format: use Native Proxy-Based Access
            parts = api_key.split(separator)
            if len(parts) >= 4:
                self.proxy_host = parts[0]
                self.proxy_port = parts[1]
                self.proxy_username = parts[2]
                self.proxy_password = parts[3]
                self.use_proxy_method = True
                self.api_key = None  # Not using API key method
                logger.info(f" Using Native Proxy-Based Access (proxy credentials)")
                logger.info(f"   Host: {self.proxy_host}")
                logger.info(f"   Port: {self.proxy_port}")
                logger.info(f"   Username: {self.proxy_username[:30]}...")
            else:
                # Invalid format, assume it's an API key
                self.use_proxy_method = False
                self.api_key = api_key
                logger.warning(f" Invalid proxy format, assuming API key")
        else:
            # Assume it's an API key (Bearer token) - use Direct API Access
            self.use_proxy_method = False
            self.api_key = api_key
            self.proxy_host = None
            self.proxy_port = None
            self.proxy_username = None
            self.proxy_password = None
            logger.info(f" Using Direct API Access (Bearer token)")
        
        self.zone = zone
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        
        logger.info(f" Web Unblocker Fetcher initialized")
        logger.info(f"   Zone: {zone}")
        if self.use_proxy_method:
            logger.info(f"   Method: Native Proxy-Based Access")
            logger.info(f"   Proxy: {self.proxy_host}:{self.proxy_port}")
        else:
            logger.info(f"   Method: Direct API Access")
            logger.info(f"   API: {self.API_BASE_URL}")
            logger.info(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '***'}")
    
    def fetch(
        self,
        url: str,
        format: str = "raw",
        wait_for: Optional[str] = None,
        wait_time: int = 0,
        scroll_to_bottom: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch URL using Web Unblocker (either API or Proxy method)
        
        Args:
            url: Target URL to fetch
            format: Response format ('raw' for HTML, 'json' for parsed) - only for API method
            wait_for: CSS selector to wait for (not fully supported)
            wait_time: Additional wait time in seconds
            scroll_to_bottom: Whether to scroll to bottom (not supported)
            
        Returns:
            Dict with 'html', 'status', 'url', 'api_calls', 'json_data' keys
        """
        if self.use_proxy_method:
            return self._fetch_via_proxy(url, wait_time)
        else:
            return self._fetch_via_api(url, format, wait_time)
    
    def _fetch_via_proxy(
        self,
        url: str,
        wait_time: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch URL using Native Proxy-Based Access (proxy credentials)
        
        This method uses Bright Data's proxy endpoint directly with proxy authentication.
        """
        logger.info(f" Fetching with Web Unblocker (Proxy Method): {url}")
        logger.info(f"   Proxy: {self.proxy_host}:{self.proxy_port}")
        logger.info(f"   Username: {self.proxy_username[:30]}...")
        
        # Build proxy URL
        proxy_url = f"http://{self.proxy_username}:{self.proxy_password}@{self.proxy_host}:{self.proxy_port}"
        
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"   Attempt {attempt + 1}/{self.max_retries}")
                logger.info(f"   Connecting to proxy: {self.proxy_host}:{self.proxy_port}")
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Use a shorter timeout for connection, longer for read
                connect_timeout = min(10, self.timeout // 3)  # Max 10s for connection
                read_timeout = self.timeout - connect_timeout
                
                logger.info(f"   Timeout: connect={connect_timeout}s, read={read_timeout}s")
                
                response = requests.get(
                    url,
                    proxies=proxies,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),  # (connect, read) timeout tuple
                    allow_redirects=True,
                    verify=False  # Bright Data proxy uses self-signed certs for SSL inspection
                )
                
                logger.info(f"   Response received: {response.status_code}, {len(response.text)} bytes")
                
                if response.status_code == 200:
                    html = response.text
                    logger.info(f" Web Unblocker (Proxy) fetch successful: {len(html):,} bytes")
                    
                    # Check if we got blocked
                    if len(html) < 1000:
                        html_preview = html.lower()
                        if 'kasada' in html_preview or 'kpsdk' in html_preview or 'blocked' in html_preview:
                            logger.warning(f"    Still appears blocked (Kasada challenge detected)")
                            if attempt < self.max_retries - 1:
                                logger.info(f"   Retrying...")
                                time.sleep(2)  # time is imported at module level
                                continue
                    
                    return {
                        'html': html,
                        'status': 200,
                        'url': response.url,
                        'api_calls': [],
                        'json_data': [],
                        'source': 'web_unblocker_proxy',
                        'success': True
                    }
                else:
                    error_msg = f"Proxy returned status {response.status_code}: {response.text[:200]}"
                    logger.warning(f"    {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)  # time is imported at module level
                        continue
                    raise Exception(error_msg)
                    
            except requests.exceptions.ProxyError as e:
                error_msg = f"Proxy connection failed: {str(e)}"
                logger.error(f"    {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # time is imported at module level
                    continue
                raise Exception(error_msg)
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timeout: {str(e)}"
                logger.warning(f"    {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1:
                    logger.info(f"   Retrying in 2s...")
                    time.sleep(2)
                    continue
                raise Exception(error_msg)
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {str(e)}"
                logger.error(f"    {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1:
                    logger.info(f"   Retrying in 2s...")
                    time.sleep(2)
                    continue
                raise Exception(error_msg)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"    Error: {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1 and self.retry_on_failure:
                    time.sleep(2)  # time is imported at module level
                    continue
                raise
        
        raise Exception(f"Web Unblocker (Proxy) fetch failed after {self.max_retries} attempts: {last_error}")
    
    def _fetch_via_api(
        self,
        url: str,
        format: str = "raw",
        wait_time: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch URL using Direct API Access (Bearer token)
        """
        logger.info(f" Fetching with Web Unblocker (API Method): {url}")
        logger.info(f"   Zone: {self.zone}, Format: {format}")
        logger.info(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '***'}")
        
        # Prepare API request
        payload = {
            "zone": self.zone,
            "url": url,
            "format": format
        }
        
        # Add optional parameters
        if wait_time > 0:
            payload["wait"] = wait_time * 1000  # Convert to milliseconds
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.debug(f"   Payload: {payload}")
        logger.debug(f"   Headers: Authorization: Bearer {self.api_key[:10]}...")
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"   Attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    self.API_BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                # Check response status
                if response.status_code == 200:
                    # Success!
                    if format == "raw":
                        html = response.text
                    elif format == "json":
                        html = json.dumps(response.json())
                    else:
                        html = response.text
                    
                    logger.info(f" Web Unblocker fetch successful: {len(html):,} bytes")
                    
                    # Check if we got blocked (small HTML usually means block)
                    if len(html) < 1000:
                        html_preview = html.lower()
                        if 'kasada' in html_preview or 'kpsdk' in html_preview or 'blocked' in html_preview:
                            logger.warning(f"    Still appears blocked (Kasada challenge detected)")
                            if attempt < self.max_retries - 1:
                                logger.info(f"   Retrying...")
                                time.sleep(2)
                                continue
                    
                    return {
                        'html': html,
                        'status': 200,
                        'url': url,
                        'api_calls': [],  # Web Unblocker doesn't expose API calls
                        'json_data': [],  # Web Unblocker doesn't expose JSON separately
                        'source': 'web_unblocker',
                        'success': True
                    }
                
                elif response.status_code == 401:
                    # Try fallback username if available
                    if hasattr(self, '_fallback_username') and self._fallback_username and attempt == 0:
                        logger.warning(f"    Password part failed, trying username part as API key...")
                        self.api_key = self._fallback_username
                        headers["Authorization"] = f"Bearer {self.api_key}"
                        logger.info(f"   Retrying with username part: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '***'}")
                        continue
                    
                    error_msg = "Authentication failed - check API key. Web Unblocker API requires a Bright Data API key (Bearer token)."
                    logger.error(f"    {error_msg}")
                    logger.error(f"    Tip: Web Unblocker API key is different from proxy credentials.")
                    logger.error(f"    Get your API key from: https://brightdata.com/cp/account/api")
                    logger.error(f"    The API key should be a Bearer token, not proxy credentials.")
                    raise Exception(error_msg)
                
                elif response.status_code == 402:
                    error_msg = "Insufficient credits - check Bright Data account"
                    logger.error(f"    {error_msg}")
                    raise Exception(error_msg)
                
                elif response.status_code == 429:
                    error_msg = "Rate limit exceeded - too many requests"
                    logger.warning(f"    {error_msg}")
                    if attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Exponential backoff
                        logger.info(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    raise Exception(error_msg)
                
                else:
                    error_msg = f"API returned status {response.status_code}: {response.text[:200]}"
                    logger.warning(f"    {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    raise Exception(error_msg)
                    
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout}s"
                logger.warning(f"    {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"    Error: {error_msg}")
                last_error = error_msg
                if attempt < self.max_retries - 1 and self.retry_on_failure:
                    time.sleep(2)
                    continue
                raise
        
        # All retries failed
        raise Exception(f"Web Unblocker fetch failed after {self.max_retries} attempts: {last_error}")
    
    async def fetch_async(
        self,
        url: str,
        format: str = "raw",
        wait_for: Optional[str] = None,
        wait_time: int = 0,
        scroll_to_bottom: bool = False
    ) -> Dict[str, Any]:
        """
        Async version of fetch (for compatibility with HybridFetcher)
        
        Note: Web Unblocker API is synchronous, but we wrap it in async
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.fetch,
            url,
            format,
            wait_for,
            wait_time,
            scroll_to_bottom
        )
    
    def test_connection(self) -> bool:
        """
        Test Web Unblocker connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_url = "https://geo.brdtest.com/welcome.txt?product=unlocker&method=api"
            result = self.fetch(test_url)
            return result.get('success', False)
        except Exception as e:
            logger.error(f" Web Unblocker connection test failed: {e}")
            return False

