#!/usr/bin/env python3
"""
Test Product Hunt with simple fields (like the working case)
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper

async def test_simple_fields():
    # API key
    api_key = "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"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    # Simple fields (like the working case)
    fields = ['title', 'description', 'author', 'url']
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    print(f"Testing Product Hunt with SIMPLE fields (working case)...")
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print(f"Using Bright Data proxy: {proxy_config['server']}")
    print("-" * 80)
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=proxy_config,
        fetch_mode='browser',
        enable_cache=False,
        browser_timeout=120000,
        use_camoufox=False
    )
    
    try:
        print("\n🚀 Starting scrape...")
        result = await scraper.scrape(
            url=url,
            fields=fields,
            scroll_to_bottom=True,
            wait_for_selector=None
        )
        
        items = result.get('data', [])
        print(f"\n✅ Scrape completed!")
        print(f"Items extracted: {len(items)}")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Cache used: {result.get('cache_used', False)}")
        
        if items:
            print(f"\n📊 First {min(3, len(items))} items:")
            for i, item in enumerate(items[:3], 1):
                print(f"\n--- Item {i} ---")
                for field, value in item.items():
                    if value:
                        display_value = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                        print(f"  {field}: {display_value}")
        else:
            print("\n⚠️ No items extracted!")
        
        return len(items)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    count = asyncio.run(test_simple_fields())
    print(f"\n{'='*80}")
    print(f"RESULT: Extracted {count} items")
    if count >= 40:
        print("✅ SUCCESS - Got expected ~42 items!")
    elif count > 0:
        print(f"⚠️  PARTIAL - Got {count} items, expected ~42")
    else:
        print("❌ FAILED - No items extracted")
    sys.exit(0 if count >= 40 else 1)

