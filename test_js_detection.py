#!/usr/bin/env python3
"""
Test universal JS detection
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test_js_detection():
    """Test JS detection on Product Hunt"""
    url = "https://www.producthunt.com/categories/vibe-coding?page=1"
    
    print("Testing Universal JS Detection")
    print("=" * 60)
    
    fetcher = HybridFetcher(
        headless=True,
        browser_timeout=60000,
        use_camoufox=False
    )
    
    # First, get static HTML to test detection
    print("\n1. Fetching static HTML...")
    static_result = fetcher._fetch_with_static(url)
    html = static_result.get('html', '')
    
    print(f"   Static HTML length: {len(html)} bytes")
    
    # Test JS detection
    print("\n2. Testing JS detection...")
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    needs_js = fetcher._detect_js_required(html, domain)
    
    print(f"   JS Required: {needs_js}")
    
    if needs_js:
        print("   ✅ Correctly detected JS requirement")
    else:
        print("   ❌ Failed to detect JS requirement (should be True for Product Hunt)")
    
    # Now test with browser fetch
    print("\n3. Fetching with browser (if JS detected)...")
    try:
        browser_result = await fetcher._fetch_with_browser(url)
        browser_html = browser_result.get('html', '')
        print(f"   Browser HTML length: {len(browser_html)} bytes")
        
        if len(browser_html) > len(html) * 2:
            print("   ✅ Browser fetch got significantly more content")
        else:
            print("   ⚠️ Browser fetch didn't get much more content")
    except Exception as e:
        print(f"   ⚠️ Browser fetch failed: {e}")
    
    await fetcher.close()

if __name__ == "__main__":
    asyncio.run(test_js_detection())




