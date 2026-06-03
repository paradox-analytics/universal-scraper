#!/usr/bin/env python3
"""
Test 10 diverse websites to demonstrate universal scraping architecture
"""

import asyncio
import os
import sys
from pathlib import Path
import time
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


# 10 diverse websites to test
TEST_SITES = [
    {
        "name": "Product Hunt",
        "url": "https://www.producthunt.com/",
        "fields": ["title", "description", "upvotes"],
        "type": "Tech Products"
    },
    {
        "name": "Stack Overflow",
        "url": "https://stackoverflow.com/questions",
        "fields": ["title", "votes", "answers"],
        "type": "Q&A Forum"
    },
    {
        "name": "Medium",
        "url": "https://medium.com/",
        "fields": ["title", "author", "date"],
        "type": "Blog Platform"
    },
    {
        "name": "Etsy",
        "url": "https://www.etsy.com/search?q=vintage+jewelry",
        "fields": ["title", "price", "seller"],
        "type": "E-commerce"
    },
    {
        "name": "Indeed",
        "url": "https://www.indeed.com/jobs?q=software+engineer&l=New+York",
        "fields": ["title", "company", "location"],
        "type": "Job Board"
    },
    {
        "name": "Wikipedia - List Page",
        "url": "https://en.wikipedia.org/wiki/List_of_programming_languages",
        "fields": ["name", "year", "paradigm"],
        "type": "Reference"
    },
    {
        "name": "Craigslist",
        "url": "https://newyork.craigslist.org/search/apa",
        "fields": ["title", "price", "location"],
        "type": "Classifieds"
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/",
        "fields": ["title", "author", "date"],
        "type": "News"
    },
    {
        "name": "Zillow",
        "url": "https://www.zillow.com/new-york-ny/",
        "fields": ["address", "price", "beds"],
        "type": "Real Estate"
    },
    {
        "name": "Twitter/X",
        "url": "https://twitter.com/elonmusk",
        "fields": ["tweet", "likes", "date"],
        "type": "Social Media"
    }
]


async def test_site(site: Dict, scraper: UniversalScraper) -> Dict:
    """Test a single website"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {site['name']} ({site['type']})")
    print(f"{'='*80}")
    print(f"🔗 URL: {site['url']}")
    print(f"📋 Fields: {', '.join(site['fields'])}")
    
    start_time = time.time()
    
    try:
        result = await scraper.scrape(site['url'], site['fields'])
        elapsed = time.time() - start_time
        
        data = result.get('data', [])
        source = result.get('source', 'unknown')
        
        # Analyze results
        success = len(data) > 0
        
        # Check for tracking data contamination
        if data:
            sample_keys = set()
            for item in data[:3]:
                sample_keys.update(item.keys())
            
            tracking_kw = ['session', 'tracking', 'correlation', 'guid', 'token']
            has_tracking = any(any(kw in str(k).lower() for kw in tracking_kw) for k in sample_keys)
            
            data_kw = ['title', 'price', 'name', 'text', 'description']
            has_data = any(any(kw in str(k).lower() for kw in data_kw) for k in sample_keys)
        else:
            has_tracking = False
            has_data = False
        
        # Determine status
        if not success:
            status = "❌ FAIL"
            quality = "No data"
        elif has_tracking and not has_data:
            status = "⚠️  PARTIAL"
            quality = "Tracking data"
        elif has_data:
            status = "✅ SUCCESS"
            quality = "Real data"
        else:
            status = "⚠️  PARTIAL"
            quality = "Unknown quality"
        
        print(f"\n{status}")
        print(f"   • Items: {len(data)}")
        print(f"   • Source: {source}")
        print(f"   • Quality: {quality}")
        print(f"   • Time: {elapsed:.1f}s")
        
        if data and len(data) > 0:
            print(f"\n   📋 Sample item:")
            sample = data[0]
            for key, value in list(sample.items())[:4]:  # First 4 fields
                val_str = str(value)[:50]
                print(f"      • {key}: {val_str}")
        
        return {
            'name': site['name'],
            'type': site['type'],
            'success': success,
            'items': len(data),
            'source': source,
            'quality': quality,
            'time': elapsed,
            'status': status
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR: {str(e)[:100]}")
        print(f"   • Time: {elapsed:.1f}s")
        
        return {
            'name': site['name'],
            'type': site['type'],
            'success': False,
            'items': 0,
            'source': 'error',
            'quality': 'error',
            'time': elapsed,
            'status': '❌ ERROR',
            'error': str(e)[:100]
        }


async def main():
    print("="*80)
    print("🚀 UNIVERSAL SCRAPER - 10 WEBSITE TEST")
    print("="*80)
    print(f"Testing {len(TEST_SITES)} diverse websites to demonstrate universal architecture")
    print()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create scraper (reuse for all tests)
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        headless=True,
        enable_llm_pagination=False,
        enable_auto_pagination=False
    )
    
    results = []
    
    try:
        for site in TEST_SITES:
            result = await test_site(site, scraper)
            results.append(result)
            
            # Small delay between sites
            await asyncio.sleep(2)
    
    finally:
        scraper.close()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful: {len(successes)}/{len(results)} ({len(successes)/len(results)*100:.0f}%)")
    print(f"❌ Failed: {len(failures)}/{len(results)}")
    
    # Group by website type
    print(f"\n📋 Results by Type:")
    types = {}
    for r in results:
        site_type = r['type']
        if site_type not in types:
            types[site_type] = []
        types[site_type].append(r)
    
    for site_type, type_results in sorted(types.items()):
        success_count = sum(1 for r in type_results if r['success'])
        print(f"\n   {site_type}:")
        for r in type_results:
            print(f"      {r['status']} {r['name']}: {r['items']} items ({r['time']:.1f}s)")
    
    # Detailed breakdown
    print(f"\n📊 Extraction Sources:")
    sources = {}
    for r in results:
        if r['success']:
            source = r['source']
            sources[source] = sources.get(source, 0) + 1
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {source}: {count} sites")
    
    print(f"\n⏱️  Performance:")
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"   • Average time: {avg_time:.1f}s")
    print(f"   • Total time: {sum(r['time'] for r in results):.1f}s")
    
    print(f"\n🎯 Success Rate by Quality:")
    quality_counts = {}
    for r in results:
        quality = r['quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    for quality, count in sorted(quality_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {quality}: {count} sites")


if __name__ == '__main__':
    asyncio.run(main())







