#!/usr/bin/env python3
"""
Test Product Hunt scraping locally
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper

async def test_producthunt():
    # API key
    api_key = "REDACTED_OPENAI_KEY_1"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    # Fields to extract (Product Hunt specific)
    fields = [
        'product title',
        'description',
        'maker',
        'upvotes',
        'product image',
        'tagline',
        'category'
    ]
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    print(f"Testing Product Hunt scraping...")
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
            print(f"\n📊 First {min(3, len(items))} items:")
            for i, item in enumerate(items[:3], 1):
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
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(test_producthunt())
    sys.exit(exit_code)

