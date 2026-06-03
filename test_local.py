#!/usr/bin/env python3
"""
Local testing script for the universal scraper.
Tests the scraper without using Apify credits.
"""

import asyncio
import json
import os
from universal_scraper.core.scraper import UniversalScraper

async def test_scraper():
    """Test the scraper locally with the same config as Apify."""
    
    # Configuration (same as Apify INPUT_SCHEMA)
    config = {
        "url": "https://www.leafly.com/dispensary-info/mammoth-holistics/menu",
        "fields": [],  # Auto-extract mode
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-key-here"),
        "fetch_mode": "browser",
        "enable_cache": True,
        "headless": True,
        "enable_llm_pagination": True,
        "proxy_url": None,  # No proxy for local testing
    }
    
    print("🚀 Starting local scraper test...")
    print(f"📋 URL: {config['url']}")
    print(f"📋 Fields: {'AUTO-EXTRACT ALL' if not config['fields'] else config['fields']}")
    print()
    
    # Initialize scraper
    scraper = UniversalScraper(
        openai_api_key=config["openai_api_key"],
        enable_cache=config["enable_cache"],
        enable_llm_pagination=config["enable_llm_pagination"]
    )
    
    # Run scraper
    result = await scraper.scrape(
        url=config["url"],
        fields=config["fields"],
        fetch_mode=config["fetch_mode"],
        headless=config["headless"],
        proxy=config["proxy_url"]
    )
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    if result.get('data'):
        items = result['data']
        print(f"✅ Extracted {len(items)} items")
        print(f"⏱️  Execution time: {result['metadata'].get('execution_time', 0):.2f}s")
        print(f"📄 Total pages scraped: {result['metadata'].get('total_pages_scraped', 1)}")
        print(f"🔄 Auto-pagination: {result['metadata'].get('auto_pagination', False)}")
        
        # Show first item as sample
        if items:
            print(f"\n📝 Sample item (first of {len(items)}):")
            print(json.dumps(items[0], indent=2)[:500] + "...")
        
        # Save to file
        output_file = "local_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
    else:
        print("❌ No data extracted")
        print(f"Metadata: {result.get('metadata', {})}")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_scraper())








