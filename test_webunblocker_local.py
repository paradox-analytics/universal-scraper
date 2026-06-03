#!/usr/bin/env python3
"""
Test Web Unblocker connection locally
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_webunblocker():
    """Test Web Unblocker fetch"""
    print("=" * 80)
    print("Testing Web Unblocker Connection")
    print("=" * 80)
    
    # Get API key and zone from environment or prompt
    api_key = os.getenv('WEB_UNBLOCKER_API_KEY')
    zone = os.getenv('WEB_UNBLOCKER_ZONE', 'web_unlocker1')
    
    if not api_key:
        print("⚠️  WEB_UNBLOCKER_API_KEY not set in environment")
        api_key = input("Enter Web Unblocker API Key: ").strip()
        if not api_key:
            print("❌ API key required")
            return False
    
    print(f"\nAPI Key: {api_key[:20]}...")
    print(f"Zone: {zone}")
    print()
    
    try:
        from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher
        
        print("1. Creating WebUnblockerFetcher...")
        fetcher = WebUnblockerFetcher(
            api_key=api_key,
            zone=zone
        )
        print("   ✅ Fetcher created")
        
        print("\n2. Testing with Product Hunt (Cloudflare-protected)...")
        test_url = "https://www.producthunt.com/"
        print(f"   URL: {test_url}")
        
        result = await fetcher.fetch_async(test_url)
        print("   ✅ Fetch completed")
        
        html = result.get('html', '')
        html_size = len(html)
        
        print(f"\n3. Results:")
        print(f"   HTML Size: {html_size:,} bytes")
        
        if html_size < 1000:
            print("   ❌ Insufficient content - Web Unblocker may not be working")
            return False
        
        # Check for Cloudflare challenge
        html_lower = html.lower()
        if 'verify you are human' in html_lower or 'just a moment' in html_lower:
            print("   ❌ Cloudflare challenge detected - Web Unblocker failed to bypass")
            print(f"   First 500 chars: {html[:500]}")
            return False
        
        # Check for Product Hunt content
        if 'producthunt' in html_lower or 'product hunt' in html_lower:
            print("   ✅ Product Hunt content detected - Web Unblocker working!")
        else:
            print("   ⚠️  Product Hunt content not clearly detected, but got substantial HTML")
        
        print(f"\n4. Sample HTML (first 300 chars):")
        print(f"   {html[:300]}...")
        
        print("\n" + "=" * 80)
        print("✅ Web Unblocker test PASSED!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_webunblocker())
    sys.exit(0 if success else 1)




