#!/usr/bin/env python3
"""
Test Improved System on Multiple Sources and Generate CSV Files
Tests the integrated ScrapeGraphAI improvements on various websites
"""
import asyncio
import os
import sys
import json
import csv
import time
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.scraper import UniversalScraper

# Test sources with contexts
TEST_SOURCES = [
    {
        'name': 'reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'context': 'Extract Reddit posts with title, author, upvotes, comments count',
        'wait_selector': 'shreddit-post',
        'fields': []
    },
    {
        'name': 'ebay',
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'context': 'Extract eBay product listings with title, price, shipping, condition',
        'wait_selector': None,
        'fields': []
    },
    {
        'name': 'metacritic_games',
        'url': 'https://www.metacritic.com/browse/game/',
        'context': 'Extract game listings with title, platform, score, release date',
        'wait_selector': None,
        'fields': []
    },
    {
        'name': 'hackernews',
        'url': 'https://news.ycombinator.com/',
        'context': 'Extract Hacker News posts with title, points, author, comments',
        'wait_selector': None,
        'fields': []
    },
    {
        'name': 'github_trending',
        'url': 'https://github.com/trending',
        'context': 'Extract trending repositories with name, description, stars, language',
        'wait_selector': None,
        'fields': []
    }
]

async def test_source(source_config, api_key):
    """Test a single source and return results"""
    name = source_config['name']
    url = source_config['url']
    context = source_config['context']
    wait_selector = source_config['wait_selector']
    fields = source_config['fields']
    
    # Create scraper with context for this specific source
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,  # Pass context here!
        enable_context_validation=True,
        log_level=30
    )
    
    # Disable pagination detector
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    
    print(f"\n{'='*80}")
    print(f"Testing: {name.upper()}")
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Scrape with appropriate settings
        result = await scraper.scrape(
            url,
            fields=fields,
            wait_for_selector=wait_selector
        )
        
        elapsed = time.time() - start_time
        
        items = result['data']
        source = result['metadata'].get('extraction_source', 'unknown')
        
        print(f"\n✅ Success!")
        print(f"   Items: {len(items)}")
        print(f"   Source: {source}")
        print(f"   Time: {elapsed:.1f}s")
        
        if items:
            print(f"   Fields: {list(items[0].keys())}")
            print(f"\n   Sample (first 2 items):")
            for i, item in enumerate(items[:2], 1):
                print(f"   {i}. {json.dumps(item, default=str)[:150]}...")
        
        return {
            'name': name,
            'success': True,
            'items': items,
            'count': len(items),
            'source': source,
            'time': elapsed,
            'error': None
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error: {str(e)}")
        print(f"   Time: {elapsed:.1f}s")
        
        return {
            'name': name,
            'success': False,
            'items': [],
            'count': 0,
            'source': 'error',
            'time': elapsed,
            'error': str(e)
        }
    
    finally:
        # Close scraper to clean up resources
        try:
            scraper.close()
        except:
            pass

def save_to_csv(result, output_dir='output'):
    """Save results to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    
    name = result['name']
    items = result['items']
    
    if not items:
        print(f"   ⚠️  No items to save for {name}")
        return None
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{name}_{timestamp}.csv"
    
    # Get all unique keys from all items
    all_keys = set()
    for item in items:
        all_keys.update(item.keys())
    
    fieldnames = sorted(list(all_keys))
    
    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    
    print(f"   💾 Saved: {filename} ({len(items)} rows)")
    return filename

async def main():
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE SOURCE TEST - With ScrapeGraphAI Integration")
    print("="*80)
    print(f"\nTesting {len(TEST_SOURCES)} sources:")
    for source in TEST_SOURCES:
        print(f"  • {source['name']}")
    print("="*80)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    # Test all sources (each with its own scraper instance)
    results = []
    for source in TEST_SOURCES:
        result = await test_source(source, api_key)
        results.append(result)
        
        # Save to CSV if successful
        if result['success'] and result['items']:
            save_to_csv(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n📈 Successful Extractions:")
        for r in successful:
            print(f"   • {r['name']}: {r['count']} items in {r['time']:.1f}s (source: {r['source']})")
    
    if failed:
        print(f"\n❌ Failed Extractions:")
        for r in failed:
            print(f"   • {r['name']}: {r['error']}")
    
    # Calculate totals
    total_items = sum(r['count'] for r in results)
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"\n📊 Statistics:")
    print(f"   Total items extracted: {total_items}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per source: {avg_time:.1f}s")
    
    print("\n" + "="*80)
    print("✅ Test complete! Check the 'output' directory for CSV files.")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

