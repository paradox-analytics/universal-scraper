#!/usr/bin/env python3
"""Test Baggu Crescent Bags collection page scraping"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper

async def test_baggu():
    """Test scraping Baggu Crescent Bags collection"""
    
    # Get OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    url = "https://baggu.com/collections/crescent-bags"
    
    # Recommended field specification - use as list for better parsing
    fields = ["title", "name", "price", "color", "variant", "description", "url", "image_url"]
    
    print("="*80)
    print("🛍️  TESTING BAGGU CRESCENT BAGS COLLECTION")
    print("="*80)
    print(f"\nURL: {url}")
    print(f"Fields: {fields}\n")
    
    # Initialize scraper - disable auto-pagination to only scrape first page
    scraper = UniversalScraper(
        api_key=api_key,
        headless=True,
        enable_cache=True,
        enable_auto_pagination=False  # Only scrape first page
    )
    
    try:
        # Scrape the page
        print("🚀 Starting scrape...\n")
        result = await scraper.scrape(url, fields)
        
        # Extract data from result dict
        results = result.get('data', [])
        metadata = result.get('metadata', {})
        
        print("="*80)
        print(f"✅ EXTRACTION COMPLETE")
        print("="*80)
        print(f"\nExtracted {len(results)} items")
        print(f"Source: {metadata.get('extraction_source', 'unknown')}")
        print(f"Execution time: {metadata.get('execution_time', 0):.2f}s\n")
        
        # Display first 3 results
        for i, item in enumerate(results[:3], 1):
            print(f"\n--- Product {i} ---")
            print(json.dumps(item, indent=2, ensure_ascii=False))
        
        if len(results) > 3:
            print(f"\n... and {len(results) - 3} more items")
        
        # Save to file
        output_file = "baggu_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"Total products extracted: {len(results)}")
        
        # Check field coverage
        fields_found = set()
        for item in results:
            if isinstance(item, dict):
                fields_found.update(item.keys())
        
        print(f"\nFields extracted:")
        for field in sorted(fields_found):
            count = sum(1 for item in results if field in item and item[field])
            print(f"  - {field}: {count}/{len(results)} items")
        
        # Check for color/variant extraction
        color_fields = ['color', 'variant', 'colors', 'variants']
        found_colors = False
        for field in color_fields:
            if field in fields_found:
                found_colors = True
                print(f"\n✅ Color/variant field found: '{field}'")
                # Show examples
                examples = [item.get(field) for item in results[:5] if item.get(field)]
                if examples:
                    print(f"   Examples: {', '.join(str(e) for e in examples[:3])}")
        
        if not found_colors:
            print("\n⚠️  No color/variant field detected - checking item contents...")
            for item in results[:2]:
                print(f"\n   Sample item keys: {list(item.keys())}")
                print(f"   Sample item: {json.dumps(item, indent=4, ensure_ascii=False)[:500]}...")
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_baggu())

