#!/usr/bin/env python3
"""
Test Metacritic scraping locally
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper

async def test_metacritic():
    # API key
    api_key = "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"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    # Fields to extract (based on the page content)
    fields = [
        'movie title',
        'metascore',
        'rank',
        'description',
        'image',
        'director',
        'release date'
    ]
    
    url = "https://www.metacritic.com/pictures/worst-movies-of-2025/"
    
    print(f"Testing Metacritic scraping...")
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print(f"Using Bright Data proxy: {proxy_config['server']}")
    print("-" * 80)
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=proxy_config,
        fetch_mode='browser',  # Use browser mode for dynamic content
        enable_cache=False,  # Disable cache for testing
        browser_timeout=120000,  # 2 minutes
        use_camoufox=False  # Use Playwright
    )
    
    try:
        print("\n🚀 Starting scrape...")
        result = await scraper.scrape(
            url=url,
            fields=fields,
            scroll_to_bottom=True,  # Enable infinite scroll
            wait_for_selector=None
        )
        
        print(f"\n✅ Scrape completed!")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Items extracted: {len(result.get('data', []))}")
        print(f"Cache used: {result.get('cache_used', False)}")
        
        # Show first few items
        items = result.get('data', [])
        if items:
            print(f"\n📊 First {min(5, len(items))} items:")
            for i, item in enumerate(items[:5], 1):
                print(f"\n--- Item {i} ---")
                for field, value in item.items():
                    if value:
                        # Truncate long values
                        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"  {field}: {display_value}")
        else:
            print("\n⚠️ No items extracted!")
            print(f"Raw result keys: {result.keys()}")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        # Show extraction metadata
        if 'extraction_metadata' in result:
            metadata = result['extraction_metadata']
            print(f"\n📈 Extraction Metadata:")
            print(f"  Method: {metadata.get('method', 'unknown')}")
            print(f"  Pattern used: {metadata.get('pattern_used', 'none')}")
            print(f"  Cache hit: {metadata.get('cache_hit', False)}")
            print(f"  Items found: {metadata.get('items_found', 0)}")
        
        return 0 if items else 1
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(test_metacritic())
    sys.exit(exit_code)
