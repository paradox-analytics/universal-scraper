"""
Test Semantic Pattern Integration - Test on failing sites
"""

import asyncio
import os
import logging
from universal_scraper import UniversalScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_failing_sites():
    """Test semantic patterns on sites that previously failed (0% quality)"""
    print("\n" + "="*80)
    print("🧪 SEMANTIC PATTERN INTEGRATION TEST - Failing Sites")
    print("="*80)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=False,  # Use static fetching for speed
        headless=True,
        enable_auto_pagination=False,
        enable_context_validation=False  # Disable for simpler test
    )
    
    # Test sites that previously failed with 0% quality
    test_cases = [
        {
            'name': 'NPR (News)',
            'url': 'https://www.npr.org/sections/news/',
            'fields': ['headline', 'description'],
            'expected_min_items': 5,
            'note': 'Previously extracted URLs instead of headlines'
        },
        {
            'name': 'IMDb Top Movies',
            'url': 'https://www.imdb.com/chart/top/',
            'fields': ['title', 'year', 'rating'],
            'expected_min_items': 10,
            'note': 'Previously extracted raw JSON-LD objects'
        },
        {
            'name': 'Craigslist',
            'url': 'https://sfbay.craigslist.org/search/sss',
            'fields': ['title', 'price', 'location'],
            'expected_min_items': 10,
            'note': 'Previously had 0% quality despite extracting items'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"URL: {test_case['url']}")
        print(f"Fields: {', '.join(test_case['fields'])}")
        print(f"Note: {test_case['note']}")
        
        try:
            result = await scraper.scrape(
                url=test_case['url'],
                fields=test_case['fields']
            )
            
            items = result.get('data', [])
            source = result.get('source', 'unknown')
            metadata = result.get('metadata', {})
            
            # Calculate quality
            if items and test_case['fields']:
                total_fields = len(items) * len(test_case['fields'])
                filled_fields = sum(
                    1 for item in items
                    for v in item.values()
                    if v is not None and v != ''
                )
                quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
            else:
                quality = 0.0
            
            # Determine status
            if quality >= 80 and len(items) >= test_case['expected_min_items']:
                status = "✅ SUCCESS"
            elif quality >= 50:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAILED"
            
            print(f"\n📊 Results:")
            print(f"   Source: {source}")
            print(f"   Items: {len(items)}")
            print(f"   Quality: {quality:.1f}%")
            print(f"   Status: {status}")
            
            if items:
                print(f"\n   Sample (first 2 items):")
                for j, item in enumerate(items[:2], 1):
                    print(f"   {j}. {item}")
            
            results.append({
                'name': test_case['name'],
                'url': test_case['url'],
                'items': len(items),
                'quality': quality,
                'source': source,
                'status': status,
                'expected': test_case['expected_min_items']
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'name': test_case['name'],
                'url': test_case['url'],
                'items': 0,
                'quality': 0.0,
                'source': 'error',
                'status': "❌ ERROR",
                'expected': test_case['expected_min_items']
            })
    
    await scraper.close()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    print(f"\n{'Site':<25} {'Items':<8} {'Quality':<10} {'Source':<20} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<25} {result['items']:<8} {result['quality']:<10.1f}% {result['source']:<20} {result['status']:<10}")
    
    print("-" * 80)
    
    # Statistics
    success_count = sum(1 for r in results if '✅' in r['status'])
    partial_count = sum(1 for r in results if '⚠️' in r['status'])
    failed_count = sum(1 for r in results if '❌' in r['status'])
    
    semantic_count = sum(1 for r in results if 'semantic' in r['source'])
    
    print(f"\n✅ Success: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    print(f"⚠️  Partial: {partial_count}/{len(results)} ({partial_count/len(results)*100:.0f}%)")
    print(f"❌ Failed: {failed_count}/{len(results)} ({failed_count/len(results)*100:.0f}%)")
    print(f"\n🎨 Semantic patterns used: {semantic_count}/{len(results)} ({semantic_count/len(results)*100:.0f}%)")
    
    # Compare with before
    print("\n" + "="*80)
    print("📈 IMPROVEMENT ANALYSIS")
    print("="*80)
    
    print("\n**BEFORE (Code Generation Only)**:")
    print("   NPR: 0% quality")
    print("   IMDb: 0% quality")
    print("   Craigslist: 0% quality")
    print("   Average: 0%")
    
    avg_quality = sum(r['quality'] for r in results) / len(results)
    print(f"\n**AFTER (With Semantic Patterns)**:")
    for result in results:
        print(f"   {result['name']}: {result['quality']:.1f}% quality")
    print(f"   Average: {avg_quality:.1f}%")
    
    improvement = avg_quality - 0
    print(f"\n🎉 Improvement: +{improvement:.1f}% ({improvement/1 if improvement > 0 else 0:.0f}x better)")
    
    print("\n" + "="*80)
    
    if success_count == len(results):
        print("🎉 ALL TESTS PASSED - Semantic patterns work universally!")
    elif success_count + partial_count == len(results):
        print("⚠️  MOSTLY WORKING - Some refinement needed")
    else:
        print("❌ NEEDS WORK - More debugging required")
    
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_failing_sites())





