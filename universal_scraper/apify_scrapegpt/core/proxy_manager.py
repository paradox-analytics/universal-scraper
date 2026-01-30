"""
Universal Proxy Manager

Handles proxy rotation and management for any proxy provider.
Works with Apify, Bright Data, ScraperAPI, Oxylabs, etc.

Key Features:
- Automatic proxy rotation
- Provider-agnostic interface
- Session management
- Failure tracking
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class ProxyManager:
    """
    Universal proxy manager that handles rotation and provider abstraction.
    
    Supports:
    - Static proxy (single URL)
    - Rotating proxy pool (multiple URLs)
    - Dynamic proxy (provider API - Apify, Bright Data, etc.)
    """
    
    def __init__(
        self,
        proxy_config: Optional[Dict[str, Any]] = None,
        provider: str = 'static',
        rotation_strategy: str = 'per_request',
        geo_location: Optional[str] = None  # NEW: Geographic targeting
    ):
        """
        Initialize proxy manager.
        
        Args:
            proxy_config: Proxy configuration dict
            provider: Proxy provider type ('static', 'apify', 'brightdata', etc.)
            rotation_strategy: When to rotate ('per_request', 'per_domain', 'on_failure')
            geo_location: ISO2 country code for geographic targeting (e.g., 'US', 'GB', 'DE')
        """
        self.proxy_config = proxy_config or {}
        self.provider = provider
        self.rotation_strategy = rotation_strategy
        self.geo_location = geo_location  # NEW
        
        # Proxy pool
        self.proxy_pool: List[Dict[str, Any]] = []
        self.current_proxy_index = 0
        
        # Failure tracking
        self.proxy_failures: Dict[str, int] = {}
        self.max_failures = 3
        
        # Session tracking
        self.domain_to_proxy: Dict[str, str] = {}
        
        geo_str = f", geo={geo_location}" if geo_location else ""
        logger.info(f" Proxy Manager initialized (provider={provider}, strategy={rotation_strategy}{geo_str})")
    
    def add_proxy(self, server: str, username: Optional[str] = None, password: Optional[str] = None):
        """Add a proxy to the pool."""
        proxy = {
            'server': server,
            'username': username,
            'password': password,
            'url': self._build_proxy_url(server, username, password)
        }
        self.proxy_pool.append(proxy)
        logger.debug(f"   Added proxy to pool: {server}")
    
    def _build_proxy_url(self, server: str, username: Optional[str] = None, password: Optional[str] = None) -> str:
        """Build proxy URL from components."""
        if not server:
            return ''
        
        # Parse server if it already has protocol
        if '://' in server:
            protocol, host = server.split('://', 1)
        else:
            protocol = 'http'
            host = server
        
        # Build URL with auth
        if username and password:
            return f"{protocol}://{username}:{password}@{host}"
        else:
            return f"{protocol}://{host}"
    
    def get_proxy(self, domain: Optional[str] = None, force_new: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a proxy based on rotation strategy.
        
        Args:
            domain: Domain being scraped (for per_domain strategy)
            force_new: Force a new proxy regardless of strategy
            
        Returns:
            Proxy dict or None if no proxies available
        """
        if not self.proxy_pool:
            logger.warning("    No proxies in pool")
            return None
        
        # Strategy: per_domain (sticky session per domain)
        if self.rotation_strategy == 'per_domain' and domain and not force_new:
            if domain in self.domain_to_proxy:
                proxy_url = self.domain_to_proxy[domain]
                proxy = next((p for p in self.proxy_pool if p['url'] == proxy_url), None)
                if proxy and self.proxy_failures.get(proxy_url, 0) < self.max_failures:
                    logger.debug(f"    Using sticky proxy for {domain}")
                    return proxy
        
        # Strategy: per_request or force_new
        proxy = self._get_next_proxy()
        
        # Track domain if using per_domain strategy
        if self.rotation_strategy == 'per_domain' and domain:
            self.domain_to_proxy[domain] = proxy['url']
        
        return proxy
    
    def _get_next_proxy(self) -> Dict[str, Any]:
        """Get next proxy from pool (round-robin)."""
        # Filter out failed proxies
        available_proxies = [
            p for p in self.proxy_pool
            if self.proxy_failures.get(p['url'], 0) < self.max_failures
        ]
        
        if not available_proxies:
            logger.warning("    All proxies failed, resetting failure count")
            self.proxy_failures = {}
            available_proxies = self.proxy_pool
        
        # Round-robin selection
        proxy = available_proxies[self.current_proxy_index % len(available_proxies)]
        self.current_proxy_index += 1
        
        logger.debug(f"    Rotating to proxy: {proxy['server']}")
        return proxy
    
    def report_failure(self, proxy_url: str):
        """Report a proxy failure."""
        self.proxy_failures[proxy_url] = self.proxy_failures.get(proxy_url, 0) + 1
        logger.warning(f"    Proxy failure: {proxy_url} (count: {self.proxy_failures[proxy_url]})")
    
    def report_success(self, proxy_url: str):
        """Report a proxy success (resets failure count)."""
        if proxy_url in self.proxy_failures:
            del self.proxy_failures[proxy_url]
            logger.debug(f"    Proxy success: {proxy_url}")
    
    @staticmethod
    def from_apify_config(apify_proxy_config: Optional[Dict[str, Any]] = None) -> Optional['ProxyManager']:
        """
        Create ProxyManager from Apify proxy configuration.
        
        This is a factory method specific to Apify integration.
        
        Args:
            apify_proxy_config: Apify proxy configuration dict
            
        Returns:
            ProxyManager instance or None if no proxy config
        """
        if not apify_proxy_config:
            return None
        
        manager = ProxyManager(
            proxy_config=apify_proxy_config,
            provider='apify',
            rotation_strategy='per_request'  # Apify handles rotation
        )
        
        return manager
    
    async def get_apify_proxy_url(self, actor_module: Any) -> Optional[str]:
        """
        Get a fresh proxy URL from Apify.
        
        This method is called per-request to get a new proxy from Apify's pool.
        
        Args:
            actor_module: The Apify Actor module (imported dynamically)
            
        Returns:
            Proxy URL string or None
        """
        try:
            # Add geographic targeting if specified
            proxy_config = self.proxy_config.copy()
            if self.geo_location:
                proxy_config['countryCode'] = self.geo_location
                logger.debug(f"    Requesting proxy from: {self.geo_location}")
            
            # Create proxy configuration
            proxy_configuration = await actor_module.create_proxy_configuration(
                actor_proxy_input=proxy_config
            )
            
            # Get NEW proxy URL (this is the key - Apify rotates here!)
            proxy_url = await proxy_configuration.new_url()
            
            if proxy_url:
                logger.debug(f"    Got new Apify proxy URL")
                return proxy_url
            else:
                logger.warning(f"    Apify returned empty proxy URL")
                return None
                
        except Exception as e:
            logger.error(f"    Failed to get Apify proxy: {e}")
            return None


class StaticProxyManager(ProxyManager):
    """
    Static proxy manager for simple use cases.
    
    Use this when you have a single proxy or small pool.
    """
    
    def __init__(self, server: str, username: Optional[str] = None, password: Optional[str] = None):
        super().__init__(provider='static', rotation_strategy='per_domain')
        self.add_proxy(server, username, password)


class RotatingProxyManager(ProxyManager):
    """
    Rotating proxy manager for proxy pools.
    
    Use this when you have multiple proxies and want automatic rotation.
    """
    
    def __init__(self, proxies: List[Dict[str, str]]):
        super().__init__(provider='pool', rotation_strategy='per_request')
        for proxy in proxies:
            self.add_proxy(
                server=proxy.get('server', ''),
                username=proxy.get('username'),
                password=proxy.get('password')
            )

