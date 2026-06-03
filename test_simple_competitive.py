"""
Simple Competitive Test - Single page only, no pagination
"""

import asyncio
import json
import time
import os
from typing import Dict, List
from datetime import datetime

from universal_scraper.core.scraper import UniversalScraper

# Test sites - single page only
TEST_SITES = [
    {
        'name': 'Books to Scrape',
        'url': 'https://books.toscrape.com/',
        'fields': ['title', 'price', 'rating'],
    },
    {
        'name': 'Quotes to Scrape',
        'url': 'https://quotes.toscrape.com/',
        'fields': ['text', 'author', 'tags'],
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'author'],
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'fields': ['repository', 'description', 'stars'],
    },
    {
        'name': 'Stack Overflow',
        'url': 'https://stackoverflow.com/questions',
        'fields': ['title', 'votes', 'answers'],
    },
    {
        'name': 'Product Hunt',
        'url': 'https://www.producthunt.com/',
        'fields': ['name', 'tagline', 'votes'],
    }
]


async def test_site(site: Dict, api_key: str) -> Dict:
    """Test a single site - first page only"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {site['name']}")
    print(f"URL: {site['url']}")
    print(f"{'='*70}")
    
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="hybrid",
        enable_cache=True,
        enable_auto_pagination=False  # DISABLE auto-pagination for testing!
    )
    
    start_time = time.time()
    
    try:
        # Scrape SINGLE PAGE ONLY - no pagination
        result = await scraper.scrape(
            url=site['url'],
            fields=site['fields']
        )
        
        duration = time.time() - start_time
        
        # Calculate completeness
        items = result['data']
        total_fields = len(items) * len(site['fields'])
        filled_fields = sum(
            1 for item in items
            for field in site['fields']
            if item.get(field) not in [None, '', []]
        )
        completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        await scraper.close()
        
        print(f"✅ Success!")
        print(f"   Items: {len(items)}")
        print(f"   Time: {duration:.1f}s")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Completeness: {completeness:.0f}%")
        
        return {
            'name': site['name'],
            'success': True,
            'items': len(items),
            'time': duration,
            'completeness': completeness,
            'source': result.get('source'),
            'sample_data': items[:2] if items else []
        }
        
    except Exception as e:
        duration = time.time() - start_time
        await scraper.close()
        
        print(f"❌ Error: {str(e)[:100]}")
        
        return {
            'name': site['name'],
            'success': False,
            'items': 0,
            'time': duration,
            'completeness': 0,
            'error': str(e)[:200]
        }


async def main():
    api_key = "REDACTED_OPENAI_KEY_1"
    
    print("\n" + "="*70)
    print("🚀 Simple Competitive Test - Universal Scraper")
    print("="*70)
    print(f"Testing {len(TEST_SITES)} sites (single page only)")
    print("="*70 + "\n")
    
    results = []
    
    for i, site in enumerate(TEST_SITES, 1):
        print(f"\n[{i}/{len(TEST_SITES)}] ", end="")
        result = await test_site(site, api_key)
        results.append(result)
        
        # Small delay between tests
        if i < len(TEST_SITES):
            await asyncio.sleep(2)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 SUMMARY")
    print("="*70 + "\n")
    
    successful = sum(1 for r in results if r['success'])
    total_items = sum(r['items'] for r in results)
    total_time = sum(r['time'] for r in results)
    avg_completeness = sum(r['completeness'] for r in results if r['success']) / max(successful, 1)
    
    print("Results by Site:\n")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['name']}")
        print(f"      Items: {r['items']}, Time: {r['time']:.1f}s, Quality: {r['completeness']:.0f}%")
    
    print(f"\nOverall Statistics:")
    print(f"  Success Rate: {successful}/{len(TEST_SITES)} ({successful/len(TEST_SITES)*100:.0f}%)")
    print(f"  Total Items: {total_items}")
    print(f"  Total Time: {total_time:.1f}s (avg {total_time/len(TEST_SITES):.1f}s per site)")
    print(f"  Avg Completeness: {avg_completeness:.0f}%")
    
    # Save results
    output = {
        'test_date': datetime.now().isoformat(),
        'total_sites': len(TEST_SITES),
        'successful': successful,
        'total_items': total_items,
        'total_time': total_time,
        'avg_completeness': avg_completeness,
        'results': results
    }
    
    with open('simple_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to simple_test_results.json")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())

