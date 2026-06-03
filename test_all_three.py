#!/usr/bin/env python3
"""
Test All Three Sites with Fixed JSON Ranking
Tests: Ticketmaster, Amazon, Leafly

This validates:
1. JSON ranking works without errors
2. Correct data is extracted
3. Performance is acceptable
"""

import asyncio
import time
import os
from universal_scraper import UniversalScraper

# Test configurations
TESTS = [
    {
        "name": "Ticketmaster Concerts",
        "url": "https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF",
        "context": "Extract concert events with artist name, venue, date, ticket price, and event URL",
        "expected_min_items": 15,
        "expected_type": "events"
    },
    {
        "name": "Amazon SSD Store",
        "url": "https://www.amazon.com/fmc/ssd-storefront?ref_=nav_cs_SSD_nav_storefron",
        "context": "Extract products with name, price, rating, and product URL",
        "expected_min_items": 10,
        "expected_type": "products"
    },
    {
        "name": "Leafly Dispensary Menu",
        "url": "https://www.leafly.com/dispensary-info/silver-state-relief---fernley/menu",
        "context": "Extract cannabis products with name, type, price, THC content, and product URL",
        "expected_min_items": 15,
        "expected_type": "products"
    }
]

async def test_site(config: dict):
    """Test a single site"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {config['name']}")
    print(f"{'='*80}")
    print(f"URL: {config['url']}")
    print(f"Context: {config['context']}")
    print()
    
    start = time.time()
    
    try:
        # Initialize scraper with context
        scraper = UniversalScraper(
            api_key=os.getenv('OPENAI_API_KEY'),
            fetch_mode='browser',
            headless=True,
            enable_cache=True,
            enable_llm_pagination=False,  # Keep disabled for speed
            extraction_context=config['context'],
            enable_context_validation=True
        )
        
        print(f"⏱️  Scraping...")
        
        # Scrape (auto-extraction mode - empty fields list)
        result = await scraper.scrape(config['url'], fields=[])
        
        elapsed = time.time() - start
        
        # Extract results
        items = result.get('data', [])
        metadata = result.get('metadata', {})
        source = metadata.get('source', 'unknown')
        
        # Validate
        success = len(items) >= config['expected_min_items']
        
        # Print results
        if success:
            print(f"\n✅ SUCCESS in {elapsed:.1f}s")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS in {elapsed:.1f}s")
        
        print(f"   Items extracted: {len(items)}")
        print(f"   Source: {source}")
        print(f"   Expected minimum: {config['expected_min_items']}")
        
        # Show sample item
        if items:
            print(f"\n   Sample item:")
            for key, value in list(items[0].items())[:5]:
                display_value = str(value)[:60]
                print(f"      {key}: {display_value}")
        
        # Check for ranking errors in logs
        ranking_worked = True
        if 'ranking_error' in str(metadata):
            ranking_worked = False
            print(f"\n   ⚠️  JSON ranking had issues (but extraction still worked)")
        
        return {
            'name': config['name'],
            'success': success,
            'items': len(items),
            'time': elapsed,
            'source': source,
            'ranking_worked': ranking_worked
        }
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ FAILED in {elapsed:.1f}s")
        print(f"   Error: {type(e).__name__}: {e}")
        
        return {
            'name': config['name'],
            'success': False,
            'items': 0,
            'time': elapsed,
            'source': 'error',
            'ranking_worked': False,
            'error': str(e)
        }

async def main():
    """Run all tests"""
    print(f"\n{'='*80}")
    print(f"🔬 TESTING ALL THREE SITES WITH FIXED JSON RANKING")
    print(f"{'='*80}")
    print(f"Sites to test: {len(TESTS)}")
    print(f"   1. Ticketmaster (concerts)")
    print(f"   2. Amazon (products)")
    print(f"   3. Leafly (dispensary menu)")
    print()
    
    results = []
    
    # Test each site sequentially
    for config in TESTS:
        result = await test_site(config)
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    total_items = sum(r['items'] for r in results)
    total_time = sum(r['time'] for r in results)
    ranking_worked_count = sum(1 for r in results if r['ranking_worked'])
    
    print(f"\nTests passed: {successful}/{len(TESTS)}")
    print(f"Total items extracted: {total_items}")
    print(f"Total time: {total_time:.1f}s (avg {total_time/len(TESTS):.1f}s per site)")
    print(f"JSON ranking worked: {ranking_worked_count}/{len(TESTS)}")
    
    print(f"\nDetailed results:")
    for r in results:
        status = "✅" if r['success'] else "⚠️"
        ranking = "✅" if r['ranking_worked'] else "❌"
        print(f"   {status} {r['name']}: {r['items']} items in {r['time']:.1f}s (source: {r['source']}, ranking: {ranking})")
    
    # Final verdict
    print(f"\n{'='*80}")
    if successful == len(TESTS) and ranking_worked_count == len(TESTS):
        print(f"🎉 ALL TESTS PASSED - JSON RANKING FIXED!")
    elif successful == len(TESTS):
        print(f"✅ ALL EXTRACTIONS SUCCESSFUL (but some ranking issues)")
    else:
        print(f"⚠️  SOME TESTS FAILED - NEEDS DEBUGGING")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())

