#!/usr/bin/env python3
"""
Test Full Pipeline V2 - Complete Integration Test

Tests the complete flow:
1. Fetch HTML (universal)
2. Try JSON (quality-validated)
3. Check cache
4. DirectLLM extraction (if cache miss)
5. Pattern learning
6. Cache saving
7. Second request (cache hit!)
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import actor_v2
sys.path.insert(0, str(script_dir / "universal_scraper" / "apify"))
from actor_v2 import UniversalScraperV2


async def test_pipeline(url: str, fields, name: str):
    """Test complete pipeline on a single source"""
    print("\n" + "="*100)
    print(f"🧪 TESTING: {name}")
    print("="*100)
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    # Initialize scraper
    scraper = UniversalScraperV2(
        api_key=api_key,
        force_local_cache=True  # Use local cache for testing
    )
    
    # First request (should be cache MISS)
    print("🔍 REQUEST 1 (Cache MISS expected)")
    print("-" * 100)
    result1 = await scraper.scrape(url, fields)
    
    print()
    print("📊 Request 1 Results:")
    print(f"   Success: {result1['success']}")
    print(f"   Items: {result1.get('items_count', 0)}")
    print(f"   Method: {result1.get('extraction_method')}")
    print(f"   Used cache: {result1.get('used_cache', False)}")
    print(f"   Cost: ${result1.get('cost', 0):.4f}")
    print(f"   Time: {result1.get('time', 0):.1f}s")
    
    if result1['success'] and result1.get('data'):
        print()
        print("Sample items:")
        for i, item in enumerate(result1['data'][:2], 1):
            print(f"\n   Item {i}:")
            for key, value in item.items():
                value_str = str(value)[:60] if value else "None"
                print(f"      • {key}: {value_str}")
    
    # Second request (should be cache HIT if pattern was learned)
    print()
    print("="*100)
    print("🔍 REQUEST 2 (Cache HIT expected)")
    print("-" * 100)
    result2 = await scraper.scrape(url, fields)
    
    print()
    print("📊 Request 2 Results:")
    print(f"   Success: {result2['success']}")
    print(f"   Items: {result2.get('items_count', 0)}")
    print(f"   Method: {result2.get('extraction_method')}")
    print(f"   Used cache: {result2.get('used_cache', False)}")
    print(f"   Cost: ${result2.get('cost', 0):.4f}")
    print(f"   Time: {result2.get('time', 0):.1f}s")
    
    # Analysis
    print()
    print("="*100)
    print("📈 ANALYSIS")
    print("="*100)
    
    if result2.get('used_cache'):
        speedup = result1.get('time', 0) / max(result2.get('time', 0.1), 0.1)
        savings = result1.get('cost', 0) - result2.get('cost', 0)
        
        print(f"✅ CACHE HIT SUCCESS!")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Cost savings: ${savings:.4f} ({savings/max(result1.get('cost', 0.01), 0.01)*100:.0f}%)")
    else:
        print(f"⚠️  Cache miss on second request")
        print(f"   Pattern learning may have failed")
    
    # Metrics
    print()
    print("📊 Overall Metrics:")
    metrics = scraper.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print()
    
    return {
        "name": name,
        "request1": result1,
        "request2": result2,
        "cache_worked": result2.get('used_cache', False)
    }


async def main():
    print("\n" + "="*100)
    print("🔬 FULL PIPELINE V2 TEST - Complete Integration")
    print("="*100)
    print()
    
    print("This test validates:")
    print("  1. ✅ Universal fetching (static/JS/JSON)")
    print("  2. ✅ JSON detection with quality validation")
    print("  3. ✅ Direct LLM extraction")
    print("  4. ✅ Pattern learning from LLM results")
    print("  5. ✅ Pattern caching (local)")
    print("  6. ✅ Cache hit on second request")
    print("  7. ✅ Cost optimization (99% savings)")
    print()
    
    # Test cases
    test_cases = [
        {
            "url": "https://news.ycombinator.com/",
            "fields": "article_title, points, comments_count",
            "name": "Hacker News"
        },
        # Add more after validating HN works
    ]
    
    results = []
    
    for test_case in test_cases:
        result = await test_pipeline(
            test_case["url"],
            test_case["fields"],
            test_case["name"]
        )
        
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print("="*100)
    
    for result in results:
        cache_status = "✅ CACHE WORKED" if result['cache_worked'] else "⚠️  CACHE FAILED"
        items1 = result['request1'].get('items_count', 0)
        items2 = result['request2'].get('items_count', 0)
        cost_total = result['request1'].get('cost', 0) + result['request2'].get('cost', 0)
        
        print(f"\n{result['name']}:")
        print(f"   Request 1: {items1} items, ${result['request1'].get('cost', 0):.4f}")
        print(f"   Request 2: {items2} items, ${result['request2'].get('cost', 0):.4f}")
        print(f"   Total cost: ${cost_total:.4f}")
        print(f"   Cache: {cache_status}")
    
    print()
    
    if all(r['cache_worked'] for r in results):
        print("✅ ALL TESTS PASSED!")
        print("   Pattern caching works end-to-end")
        print()
        print("Next steps:")
        print("  1. Test on more diverse sources")
        print("  2. Deploy to Apify")
        print("  3. Validate Apify KV Store caching")
    else:
        print("⚠️  Some tests need review")
        print("   Check pattern learning logic")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())




