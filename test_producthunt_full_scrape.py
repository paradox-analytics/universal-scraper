#!/usr/bin/env python3
"""
Test Product Hunt full scrape with discovered fields
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper

async def test_full_scrape():
    # API key
    api_key = "REDACTED_OPENAI_KEY_1"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    # Fields discovered by universal field discovery
    fields = [
        'product title',
        'tagline',
        'description',
        'maker',
        'upvotes',
        'product image',
        'category'
    ]
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    print(f"Testing Product Hunt full scrape...")
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print(f"Using Bright Data proxy: {proxy_config['server']}")
    print("-" * 80)
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=proxy_config,
        fetch_mode='browser',
        enable_cache=False,  # Disable cache to force fresh extraction
        browser_timeout=120000,
        use_camoufox=False
    )
    
    # Also disable Direct LLM cache
    scraper.direct_llm_extractor.enable_cache = False
    
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
            print(f"\n📊 First {min(5, len(items))} items:")
            for i, item in enumerate(items[:5], 1):
                print(f"\n--- Item {i} ---")
                for field, value in item.items():
                    if value:
                        # Truncate long values
                        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"  {field}: {display_value}")
            
            # Check quality
            total_fields = len(items) * len(fields)
            filled_fields = sum(
                1 for item in items
                for v in item.values()
                if v is not None and v != ''
            )
            quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
            print(f"\n📈 Quality: {quality:.1f}% field completeness")
            
            # Check if we got the expected number of items (should be ~42)
            if len(items) < 20:
                print(f"\n⚠️  Warning: Only extracted {len(items)} items, expected ~42")
            else:
                print(f"\n✅ Extracted {len(items)} items (good coverage)")
        else:
            print("\n⚠️ No items extracted!")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        return 0 if items and len(items) >= 20 else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(test_full_scrape())
    sys.exit(exit_code)

