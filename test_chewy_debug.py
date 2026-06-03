#!/usr/bin/env python3
"""
Debug Test: Check what HTML we're actually getting from Chewy
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

async def main():
    proxy_config = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    fetcher = CamoufoxFetcher(
        proxy_config=proxy_config,
        headless=True,
        timeout=120000
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    
    print(f"Fetching: {url}")
    print(f"Proxy: {proxy_config['server']}")
    
    try:
        result = await fetcher.fetch(url)
        html = result['html']
        
        print(f"\n✅ Fetched {len(html):,} bytes")
        print(f"\nFirst 1000 chars of HTML:")
        print("=" * 80)
        print(html[:1000])
        print("=" * 80)
        
        # Check if it's an error page
        if 'error' in html.lower() or 'blocked' in html.lower() or 'access denied' in html.lower():
            print("\n⚠️  Looks like an error/block page")
        elif len(html) < 5000:
            print("\n⚠️  HTML is very small - might be blocked")
        else:
            print("\n✅ HTML size looks reasonable")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())

