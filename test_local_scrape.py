#!/usr/bin/env python3
"""
Local test script for scraping Product Hunt
"""
import asyncio
import sys
import os
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper

async def test_scrape():
    """Test scraping Product Hunt"""
    
    # Get API key from environment or command line argument
    api_key = os.getenv("OPENAI_API_KEY")
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not provided")
        print("Usage: python3 test_local_scrape.py <api_key>")
        print("   or: export OPENAI_API_KEY='your-key' && python3 test_local_scrape.py")
        return
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    fields = ["title", "description", "url", "review score"]
    
    print(f"Testing scrape of: {url}")
    print(f"Fields: {', '.join(fields)}")
    print("-" * 80)
    
    # Configure Bright Data proxy
    proxy_config = {
        "provider": "brightdata",
        "server": "brd.superproxy.io:22225",
        "username": "brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2",
        "password": "REDACTED_PROXY_PASS"
    }
    
    print(f"Using proxy: {proxy_config['username']}")
    print("-" * 80)
    
    # Initialize scraper with proxy
    scraper = UniversalScraper(
        api_key=api_key,
        model_name="gpt-4o-mini",
        fetch_mode="hybrid",
        use_direct_llm=True,
        quality_mode="balanced",
        enable_cache=True,
        proxy_config=proxy_config,  # Add proxy config
        redis_cache=None  # Use local cache for testing
    )
    
    try:
        # Scrape
        result = await scraper.scrape(
            url=url,
            fields=fields,
            scroll_to_bottom=True,  # Product Hunt uses infinite scroll
            wait_for_selector=None
        )
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Status: {result.get('source', 'unknown')}")
        print(f"Items extracted: {len(result.get('data', []))}")
        
        metadata = result.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  - Execution time: {metadata.get('execution_time', 0) / 1000:.2f}s")
        print(f"  - Pattern cache hit: {metadata.get('pattern_cache_hit', False)}")
        print(f"  - Pattern type: {metadata.get('pattern_type', 'none')}")
        print(f"  - Incremental extraction: {metadata.get('incremental_extraction', False)}")
        if metadata.get('incremental_extraction'):
            print(f"  - Pattern fields: {metadata.get('pattern_fields', [])}")
            print(f"  - Incremental fields: {metadata.get('incremental_fields', [])}")
        
        # Show first few results
        data = result.get('data', [])
        if data:
            print(f"\nFirst 5 items:")
            for i, item in enumerate(data[:5], 1):
                print(f"\n  Item {i}:")
                for field in fields:
                    value = item.get(field, 'N/A')
                    if value and value != 'N/A':
                        # Truncate long values
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"    {field}: {value_str}")
        
        # Save full results to JSON
        output_file = "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n\nFull results saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_scrape())
    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

