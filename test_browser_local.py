#!/usr/bin/env python3
"""
Test browser fetcher locally
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test_browser_fetch():
    """Test browser fetching with Product Hunt URL"""
    url = "https://www.producthunt.com/categories/vibe-coding?page=1"
    
    print(f"Testing browser fetch for: {url}")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = HybridFetcher(
        headless=True,
        browser_timeout=60000,
        use_camoufox=False  # Use Playwright
    )
    
    try:
        print("\n1. Attempting to fetch with browser...")
        result = await fetcher.fetch(url)
        
        print(f"\n✅ Fetch successful!")
        print(f"   Method: {result.get('fetch_method', 'unknown')}")
        print(f"   HTML length: {len(result.get('html', ''))}")
        print(f"   Status: {result.get('status_code', 'unknown')}")
        
        # Check if we got actual content
        html = result.get('html', '')
        if len(html) > 1000:
            print(f"   ✅ Got substantial HTML content ({len(html)} bytes)")
            # Check for Product Hunt indicators
            if 'producthunt' in html.lower() or 'product hunt' in html.lower():
                print("   ✅ Product Hunt content detected")
            else:
                print("   ⚠️ Product Hunt content not clearly detected")
        else:
            print(f"   ⚠️ HTML content seems small ({len(html)} bytes)")
        
        # Check for captured JSON
        captured_json = result.get('captured_json', [])
        if captured_json:
            print(f"   ✅ Captured {len(captured_json)} JSON responses")
        else:
            print("   ℹ️ No JSON responses captured")
            
    except Exception as e:
        print(f"\n❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            await fetcher.close()
            print("\n✅ Cleanup complete")
        except Exception as e:
            print(f"\n⚠️ Cleanup error: {e}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_browser_fetch())
    sys.exit(0 if success else 1)




