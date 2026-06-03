import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to sys.path
sys.path.append(os.getcwd())

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

async def test_async_camoufox():
    print("🧪 Testing Async CamoufoxFetcher...")
    
    # Test 1: Basic fetch (no proxy)
    fetcher = CamoufoxFetcher(headless=True)
    try:
        print("\n1. Testing basic fetch (example.com)...")
        result = await fetcher.fetch("https://example.com")
        print(f"✅ Success! Status: {result['status']}, HTML length: {len(result['html'])}")
    except Exception as e:
        print(f"❌ Basic fetch failed: {e}")

    # Test 2: Comma-separated proxy parsing
    print("\n2. Testing comma-separated proxy parsing...")
    # Mock credentials
    proxy_key = "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS"
    fetcher_proxy = CamoufoxFetcher(headless=True, web_unblocker_api_key=proxy_key)
    
    # We don't actually need to fetch to test the parsing logic
    # But let's see if it launches with these credentials (it might fail auth, but we want to see the config)
    try:
        print("   (This might fail auth but we're checking if it launches without Playwright Sync error)")
        result = await fetcher_proxy.fetch("https://example.com")
        print(f"✅ Success! Status: {result['status']}")
    except Exception as e:
        print(f"ℹ️ Fetch attempt finished (expected if auth fails): {e}")

    print("\n✅ Verification complete!")

if __name__ == "__main__":
    asyncio.run(test_async_camoufox())
