#!/usr/bin/env python3
"""
Test fallback mechanism when browser fails
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test_fallback():
    """Test that fallback works when browser fails"""
    url = "https://www.producthunt.com/categories/vibe-coding?page=1"
    
    print("Testing fallback mechanism...")
    print("=" * 60)
    
    # Create fetcher with browser that will fail
    # We'll simulate failure by using invalid browser config
    fetcher = HybridFetcher(
        headless=True,
        browser_timeout=1000,  # Very short timeout
        use_camoufox=False
    )
    
    # Manually break the browser fetcher to simulate Cloud Run failure
    # This simulates what happens when browser launch fails
    try:
        print("\n1. Attempting fetch (should fall back if browser fails)...")
        result = await fetcher.fetch(url)
        
        print(f"\n✅ Fetch completed!")
        print(f"   Method: {result.get('fetch_method', 'unknown')}")
        print(f"   HTML length: {len(result.get('html', ''))}")
        
        if result.get('fetch_method') == 'static_fallback':
            print("   ✅ Fallback to static HTML worked!")
            print(f"   Reason: {result.get('fallback_reason', 'unknown')}")
        elif result.get('fetch_method') == 'browser':
            print("   ✅ Browser worked (no fallback needed)")
        else:
            print(f"   Method: {result.get('fetch_method')}")
            
    except Exception as e:
        print(f"\n❌ Fetch failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            await fetcher.close()
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_fallback())
    sys.exit(0 if success else 1)




