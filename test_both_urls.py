#!/usr/bin/env python3
"""
Test both Product Hunt and Metacritic URLs with proxies
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper

async def test_url(url, fields, name):
    # API key
    api_key = "REDACTED_OPENAI_KEY_1"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    print(f"\n{'='*80}")
    print(f"Testing {name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print(f"Using Bright Data proxy: {proxy_config['server']}")
    
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
            print(f"\n📊 Sample item (first item):")
            for field, value in list(items[0].items())[:5]:  # Show first 5 fields
                if value:
                    display_value = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                    print(f"  {field}: {display_value}")
        else:
            print("\n⚠️ No items extracted!")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        return len(items)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

async def main():
    tests = [
        (
            "https://www.producthunt.com/categories/vibe-coding",
            ['product title', 'description', 'maker', 'upvotes', 'product image'],
            "Product Hunt"
        ),
        (
            "https://www.metacritic.com/pictures/worst-movies-of-2025/",
            ['movie title', 'metascore', 'rank', 'description', 'image'],
            "Metacritic"
        )
    ]
    
    results = {}
    for url, fields, name in tests:
        count = await test_url(url, fields, name)
        results[name] = count
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, count in results.items():
        print(f"{name}: {count} items extracted")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




