#!/usr/bin/env python3
"""
Simple Test: Just fetch Chewy.com HTML with Web Unblocker proxy
No extraction, just verify we can get past Kasada.
"""
import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    print("=" * 80)
    print("🧪 SIMPLE TEST: Chewy.com HTML Fetch with Web Unblocker Proxy")
    print("=" * 80)
    
    # Web Unblocker Proxy Configuration
    web_unblocker_proxy = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-hl_803e8195-zone-web_unlocker1',
        'password': 't8mhp1qev1i1'
    }
    
    print(f"\n🌐 Web Unblocker Proxy:")
    print(f"   Server: {web_unblocker_proxy['server']}")
    print(f"   Username: {web_unblocker_proxy['username']}")
    
    url = "https://www.chewy.com/b/wet-food-389"
    
    print(f"\n📋 Fetching: {url}")
    print(f"   Using Camoufox + Web Unblocker proxy")
    
    fetcher = CamoufoxFetcher(
        proxy_config=web_unblocker_proxy,
        headless=True,
        timeout=120000
    )
    
    try:
        print(f"\n⏳ Fetching (this may take 30-60 seconds)...")
        result = await fetcher.fetch(url)
        
        html = result.get('html', '')
        html_size = len(html)
        
        print(f"\n✅ Fetch completed!")
        print(f"   HTML size: {html_size:,} bytes")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        # Check if blocked
        html_lower = html.lower()
        is_kasada = 'kasada' in html_lower or 'kpsdk' in html_lower or 'ips.js' in html_lower
        is_small = html_size < 2000
        
        print(f"\n📊 Analysis:")
        print(f"   Size check: {'⚠️ Small' if is_small else '✅ Good size'}")
        print(f"   Kasada detected: {'⚠️ Yes' if is_kasada else '✅ No'}")
        
        if is_small and is_kasada:
            print(f"\n❌ Still blocked by Kasada")
            print(f"\n   HTML preview (first 500 chars):")
            print(f"   {html[:500]}")
            return False
        elif is_small:
            print(f"\n⚠️  HTML is small but no Kasada detected")
            print(f"\n   HTML preview (first 500 chars):")
            print(f"   {html[:500]}")
            return False
        else:
            print(f"\n✅ Success! Got substantial HTML content")
            
            # Check for product indicators
            has_products = any(indicator in html_lower for indicator in [
                'product', 'chewy', 'price', 'rating', 'review',
                'add to cart', 'wet food', 'pet food'
            ])
            
            if has_products:
                print(f"   ✅ HTML contains product/Chewy content")
            else:
                print(f"   ⚠️  HTML doesn't contain expected product keywords")
            
            print(f"\n   HTML preview (first 1000 chars):")
            print(f"   {html[:1000]}")
            
            # Save HTML
            output_file = 'chewy_web_unblocker_fetch.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"\n💾 Full HTML saved to: {output_file}")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await fetcher.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

