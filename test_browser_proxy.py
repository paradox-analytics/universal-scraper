#!/usr/bin/env python3
"""
Test script to verify browser rendering with proxy configuration works correctly.
Tests the fix for "Browser launch failed: 'server'" error.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.browser_fetcher import BrowserFetcher

async def test_browser_with_proxy():
    """Test browser rendering with proxy config"""
    print("🧪 Testing browser rendering with proxy configuration...\n")
    
    # Test URL
    test_url = "https://www.producthunt.com/categories/vibe-coding"
    
    # Test Case 1: No proxy config (should work)
    print("Test 1: Browser without proxy config")
    try:
        fetcher = HybridFetcher(
            proxy_config=None,
            headless=True,
            browser_timeout=30000,
            force_mode='browser'
        )
        result = await fetcher.fetch(test_url)
        print(f"✅ Success: Fetched {len(result.get('html', ''))} bytes")
        print(f"   Final URL: {result.get('final_url', 'N/A')}")
        print(f"   Method: {result.get('method', 'N/A')}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # Test Case 2: Invalid proxy config (should gracefully fallback)
    print("Test 2: Browser with invalid proxy config (missing server)")
    try:
        invalid_proxy = {
            'server': '',  # Empty server
            'username': 'test',
            'password': 'test'
        }
        fetcher = HybridFetcher(
            proxy_config=invalid_proxy,
            headless=True,
            browser_timeout=30000,
            force_mode='browser'
        )
        result = await fetcher.fetch(test_url)
        print(f"✅ Success: Gracefully handled invalid proxy, fetched {len(result.get('html', ''))} bytes")
        print(f"   Method: {result.get('method', 'N/A')}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # Test Case 3: Valid Bright Data proxy config
    print("Test 3: Browser with Bright Data proxy config")
    try:
        # Use Bright Data proxy format
        valid_proxy = {
            'server': 'brd.superproxy.io:33335',
            'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
            'password': 'REDACTED_PROXY_PASS'
        }
        print(f"   Proxy: {valid_proxy['server']}")
        print(f"   Username: {valid_proxy['username'][:30]}...")
        fetcher = HybridFetcher(
            proxy_config=valid_proxy,
            headless=True,
            browser_timeout=90000,  # Longer timeout for proxy
            force_mode='browser'
        )
        result = await fetcher.fetch(test_url)
        html_length = len(result.get('html', ''))
        print(f"✅ Success: Fetched {html_length} bytes with Bright Data proxy")
        print(f"   Final URL: {result.get('final_url', 'N/A')}")
        print(f"   Method: {result.get('method', 'N/A')}")
        
        # Check if HTML contains Product Hunt content
        html_preview = result.get('html', '')[:500].lower()
        if 'producthunt' in html_preview or 'vibe-coding' in html_preview:
            print(f"   ✅ HTML contains Product Hunt content (rendered successfully)")
        else:
            print(f"   ⚠️  HTML preview: {result.get('html', '')[:200]}...")
        print()
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test Case 4: Proxy config with None server (should skip proxy)
    print("Test 4: Browser with proxy config but None server")
    try:
        none_proxy = {
            'server': None,
            'username': 'test',
            'password': 'test'
        }
        fetcher = HybridFetcher(
            proxy_config=none_proxy,
            headless=True,
            browser_timeout=30000,
            force_mode='browser'
        )
        result = await fetcher.fetch(test_url)
        print(f"✅ Success: Handled None server, fetched {len(result.get('html', ''))} bytes")
        print(f"   Method: {result.get('method', 'N/A')}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    print("✅ All proxy configuration tests completed!")

if __name__ == "__main__":
    asyncio.run(test_browser_with_proxy())

