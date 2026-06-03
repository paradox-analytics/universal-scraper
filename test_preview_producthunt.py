#!/usr/bin/env python3
"""
Test Product Hunt preview endpoint with Web Unblocker
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher

async def test_web_unblocker_direct():
    """Test Web Unblocker directly"""
    print("=" * 80)
    print("🔍 TESTING WEB UNBLOCKER DIRECTLY")
    print("=" * 80)
    
    # You'll need to set these from your Firebase settings
    api_key = input("Enter Web Unblocker API Key: ").strip()
    zone = input("Enter Web Unblocker Zone (default: web_unlocker1): ").strip() or "web_unlocker1"
    
    fetcher = WebUnblockerFetcher(
        api_key=api_key,
        zone=zone
    )
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    print(f"\n📥 Fetching: {url}")
    print(f"   Zone: {zone}")
    print(f"   Wait time: 10 seconds (for React hydration)")
    
    try:
        result = await fetcher.fetch_async(
            url,
            format="raw",
            wait_time=10  # Wait 10 seconds for React to hydrate
        )
        
        html = result.get('html', '')
        print(f"\n✅ Fetched {len(html):,} bytes")
        
        # Check for React/Next.js indicators
        if '__NEXT_DATA__' in html:
            print("✅ Found __NEXT_DATA__ (Next.js detected)")
        else:
            print("⚠️  No __NEXT_DATA__ found (might be stripped HTML)")
        
        # Check for Cloudflare challenge
        html_lower = html.lower()
        if 'verify you are human' in html_lower or 'just a moment' in html_lower:
            print("❌ Cloudflare challenge detected!")
        else:
            print("✅ No Cloudflare challenge detected")
        
        # Check for actual content
        if 'producthunt' in html_lower and len(html) > 50000:
            print("✅ Appears to have substantial content")
        else:
            print("⚠️  Content appears minimal or stripped")
        
        # Save sample
        with open("producthunt_webunblocker_test.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n💾 Saved to: producthunt_webunblocker_test.html")
        print(f"   Preview (first 500 chars):\n{html[:500]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_hybrid_fetcher():
    """Test HybridFetcher with Web Unblocker"""
    print("\n" + "=" * 80)
    print("🔍 TESTING HYBRID FETCHER WITH WEB UNBLOCKER")
    print("=" * 80)
    
    api_key = input("\nEnter Web Unblocker API Key: ").strip()
    zone = input("Enter Web Unblocker Zone (default: web_unlocker1): ").strip() or "web_unlocker1"
    
    fetcher = HybridFetcher(
        proxy_config=None,
        headless=True,
        browser_timeout=60000,
        use_camoufox=False,
        force_mode='browser',  # Force browser mode like preview endpoint
        web_unblocker_api_key=api_key,
        web_unblocker_zone=zone
    )
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    print(f"\n📥 Fetching: {url}")
    
    try:
        result = await fetcher.fetch(url)
        
        html = result.get('html', '')
        fetch_method = result.get('fetch_method', 'unknown')
        
        print(f"\n✅ Fetch method: {fetch_method}")
        print(f"✅ Fetched {len(html):,} bytes")
        
        # Check for React/Next.js indicators
        if '__NEXT_DATA__' in html:
            print("✅ Found __NEXT_DATA__ (Next.js detected)")
        else:
            print("⚠️  No __NEXT_DATA__ found")
        
        # Check for Cloudflare challenge
        html_lower = html.lower()
        if 'verify you are human' in html_lower or 'just a moment' in html_lower:
            print("❌ Cloudflare challenge detected!")
        else:
            print("✅ No Cloudflare challenge detected")
        
        # Save sample
        with open("producthunt_hybrid_test.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n💾 Saved to: producthunt_hybrid_test.html")
        print(f"   Preview (first 500 chars):\n{html[:500]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Choose test:")
    print("1. Web Unblocker Direct")
    print("2. Hybrid Fetcher (with Web Unblocker)")
    choice = input("Choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_web_unblocker_direct())
    else:
        asyncio.run(test_hybrid_fetcher())




