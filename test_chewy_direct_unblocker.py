#!/usr/bin/env python3
"""
Direct Test: Chewy.com with Web Unblocker Only

Tests Web Unblocker directly (bypasses residential proxy) to verify it works.
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    print("=" * 80)
    print("🧪 DIRECT TEST: Chewy.com with Web Unblocker")
    print("=" * 80)
    
    # Get API key
    api_key = os.environ.get('BRIGHT_DATA_API_KEY')
    if not api_key:
        print("\n❌ BRIGHT_DATA_API_KEY not set!")
        print("   Set it with: export BRIGHT_DATA_API_KEY='your-api-key'")
        print("   Get your API key from: https://brightdata.com/cp/account/api")
        return False
    
    zone = os.environ.get('BRIGHT_DATA_ZONE', 'web_unlocker1')
    chewy_url = "https://www.chewy.com/b/wet-food-389"
    
    print(f"\n📋 Configuration:")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Zone: {zone}")
    print(f"   URL: {chewy_url}")
    
    # Initialize Web Unblocker fetcher
    print(f"\n⏳ Initializing Web Unblocker...")
    fetcher = WebUnblockerFetcher(
        api_key=api_key,
        zone=zone,
        timeout=120
    )
    
    # Fetch Chewy.com
    print(f"\n⏳ Fetching Chewy.com (this may take 30-60 seconds)...")
    print(f"   Web Unblocker will handle Kasada challenges automatically")
    
    try:
        result = await fetcher.fetch_async(chewy_url)
        
        html = result.get('html', '')
        print(f"\n✅ Fetch completed!")
        print(f"   Status: {result.get('status')}")
        print(f"   HTML size: {len(html):,} bytes")
        print(f"   Source: {result.get('source')}")
        
        # Check if we got blocked
        html_lower = html.lower()
        is_blocked = (
            len(html) < 2000 and (
                'kasada' in html_lower or 
                'kpsdk' in html_lower or 
                'ips.js' in html_lower
            )
        )
        
        if is_blocked:
            print(f"\n❌ Still appears blocked (Kasada challenge)")
            print(f"\n   HTML preview (first 500 chars):")
            print(f"   {html[:500]}")
            print(f"\n   Possible issues:")
            print(f"   1. Web Unblocker zone name incorrect")
            print(f"   2. Web Unblocker not enabled for your account")
            print(f"   3. Insufficient credits")
            return False
        else:
            print(f"\n✅ Success! Got full HTML content")
            print(f"\n   HTML preview (first 1000 chars):")
            print(f"   {html[:1000]}")
            
            # Check for product indicators
            has_products = any(indicator in html_lower for indicator in [
                'product', 'chewy', 'price', 'rating', 'review', 
                'add to cart', 'buy now', 'wet food'
            ])
            
            if has_products:
                print(f"\n✅ HTML contains product/Chewy content - Web Unblocker worked!")
                
                # Save HTML for inspection
                output_file = 'chewy_web_unblocker_output.html'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"\n💾 Full HTML saved to: {output_file}")
                
                return True
            else:
                print(f"\n⚠️  HTML received but doesn't contain expected product content")
                print(f"   Check saved HTML file for details")
                return False
        
    except Exception as e:
        print(f"\n❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

