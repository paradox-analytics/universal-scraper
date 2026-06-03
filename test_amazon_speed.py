#!/usr/bin/env python3
"""
Test Amazon page rendering speed locally
"""
import asyncio
import time
from universal_scraper.core.hybrid_fetcher import HybridFetcher

AMAZON_URL = "https://www.amazon.com/stores/page/CE6A6E70-D162-4324-BE03-3C4BAFACCBB4?ingress=3&visitId=90f1be43-a9bd-419e-8d80-4400cc02e48f&channel=discovbar&ref_=nav_cs_amazonbasics"

async def test_amazon():
    print("=" * 80)
    print("Testing Amazon Page Rendering Speed")
    print("=" * 80)
    print(f"URL: {AMAZON_URL}")
    print()
    
    # Test 1: Playwright (current Cloud Run setup)
    print("Test 1: Playwright (Current Setup)")
    print("-" * 80)
    start = time.time()
    
    fetcher = HybridFetcher(
        proxy_config=None,
        headless=True,
        browser_timeout=60000,
        use_camoufox=False,  # Current Cloud Run setting
        force_mode='browser'
    )
    
    try:
        result = await fetcher.fetch(AMAZON_URL)
        elapsed = time.time() - start
        
        html = result.get('html', '')
        html_size = len(html)
        fetch_method = result.get('fetch_method', 'unknown')
        
        print(f"✅ Fetch succeeded")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   HTML Size: {html_size:,} bytes")
        print(f"   Fetch Method: {fetch_method}")
        
        # Check for Amazon content
        if 'amazon' in html.lower():
            print(f"   ✅ Amazon content detected")
        else:
            print(f"   ⚠️ No Amazon content detected")
        
        # Check for product listings
        if 'product' in html.lower() or 'item' in html.lower():
            product_count = html.lower().count('product') + html.lower().count('item')
            print(f"   Products/Items mentioned: {product_count} times")
        
        # Check for JS indicators
        if '<div id="a-page">' in html or 'amazonbasics' in html.lower():
            print(f"   ✅ Amazon structure detected (JS likely rendered)")
        else:
            print(f"   ⚠️ Amazon structure not clear")
        
        # Save HTML for inspection
        with open('/tmp/amazon_playwright.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"   HTML saved to: /tmp/amazon_playwright.html")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Fetch failed: {e}")
        print(f"   Time: {elapsed:.2f}s")
    
    print()
    
    # Test 2: Camoufox (if available)
    try:
        print("Test 2: Camoufox (Alternative)")
        print("-" * 80)
        start = time.time()
        
        fetcher_camoufox = HybridFetcher(
            proxy_config=None,
            headless=True,
            browser_timeout=60000,
            use_camoufox=True,  # Try Camoufox
            force_mode='browser'
        )
        
        result_camoufox = await fetcher_camoufox.fetch(AMAZON_URL)
        elapsed = time.time() - start
        
        html = result_camoufox.get('html', '')
        html_size = len(html)
        fetch_method = result_camoufox.get('fetch_method', 'unknown')
        
        print(f"✅ Fetch succeeded")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   HTML Size: {html_size:,} bytes")
        print(f"   Fetch Method: {fetch_method}")
        
        # Check for Amazon content
        if 'amazon' in html.lower():
            print(f"   ✅ Amazon content detected")
        
        # Save HTML
        with open('/tmp/amazon_camoufox.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"   HTML saved to: /tmp/amazon_camoufox.html")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"⚠️ Camoufox not available or failed: {e}")
        print(f"   Time: {elapsed:.2f}s")
    
    print()
    print("=" * 80)
    print("Recommendation:")
    print("  - If Playwright is slow (>10s), consider enabling Camoufox")
    print("  - If HTML is small (<50KB), JS may not be rendering")
    print("  - If Amazon blocks, try with proxies")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_amazon())




