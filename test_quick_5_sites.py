#!/usr/bin/env python3
"""
Quick Test - 5 Diverse Websites
Tests core functionality without Camoufox (to avoid async issues)
"""

import asyncio
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


TEST_CASES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'fields': ['title', 'author', 'upvotes', 'comments'],
        'expected_min': 10,
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'author', 'comments'],
        'expected_min': 20,
    },
    {
        'name': 'Craigslist',
        'url': 'https://sfbay.craigslist.org/search/sss?query=laptop',
        'fields': ['title', 'price', 'location'],
        'expected_min': 20,
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'fields': ['name', 'description', 'stars'],
        'expected_min': 10,
    },
    {
        'name': 'eBay',
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'fields': ['title', 'price', 'condition'],
        'expected_min': 20,
    },
]


async def main():
    print("="*80)
    print("🧪 QUICK TEST - 5 DIVERSE WEBSITES")
    print("="*80)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set.")
        return
    
    scraper = None
    results = []
    
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=False,  # Disable for testing
            headless=True,
            enable_auto_pagination=False,
        )
        
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'#'*80}")
            print(f"# TEST {i}/{len(TEST_CASES)}: {test_case['name']}")
            print(f"{'#'*80}")
            
            try:
                result = await scraper.scrape(
                    url=test_case['url'],
                    fields=test_case['fields']
                )
                
                data = result['data']
                
                # Quick analysis
                non_null_count = 0
                all_null_count = 0
                for item in data:
                    populated = sum(1 for v in item.values() if v is not None and v != '')
                    if populated == 0:
                        all_null_count += 1
                    elif populated == len(test_case['fields']):
                        non_null_count += 1
                
                print(f"\n✅ RESULTS:")
                print(f"   • Items: {len(data)}")
                print(f"   • Complete: {non_null_count}")
                print(f"   • All null: {all_null_count}")
                print(f"   • Time: {result.get('total_time', 0):.1f}s")
                
                if data:
                    print(f"\n📋 Sample (first item):")
                    for k, v in list(data[0].items())[:5]:
                        v_str = str(v)[:80] if v else "None"
                        print(f"      • {k}: {v_str}")
                
                success = len(data) >= test_case['expected_min'] * 0.5 and all_null_count == 0
                results.append({
                    'name': test_case['name'],
                    'success': success,
                    'items': len(data),
                    'complete': non_null_count,
                    'all_null': all_null_count
                })
                
            except Exception as e:
                print(f"\n❌ FAILED: {e}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'items': 0,
                    'complete': 0,
                    'all_null': 0,
                    'error': str(e)[:100]
                })
            
            await asyncio.sleep(1)
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for r in results if r['success'])
        print(f"\n✅ Passed: {passed}/{len(results)} ({(passed/len(results)*100):.0f}%)")
        
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"\n{status} {r['name']}")
            print(f"   Items: {r['items']}, Complete: {r['complete']}, All Null: {r['all_null']}")
            if 'error' in r:
                print(f"   Error: {r['error']}")
    
    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







