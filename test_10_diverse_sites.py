#!/usr/bin/env python3
"""
Test Universal Scraper on 10 Diverse Websites

Tests all recent improvements:
- Smart HTML Sampler (dynamic sizing)
- Universal Field Mapper (semantic field detection)
- Enhanced Attribute Extraction (custom elements)
- Temporal Field Detection (date/time fields)
- Smart Wait Strategy (JS-heavy sites)
- DOM Pattern Detection
- JSON Quality Validation
"""

import asyncio
import os
import sys
import time
from typing import Dict, List, Any
from universal_scraper import UniversalScraper

# 10 diverse websites across different categories
TEST_SITES = [
    {
        'name': 'Amazon (E-commerce)',
        'url': 'https://www.amazon.com/s?k=laptop',
        'fields': ['title', 'price', 'rating', 'reviews'],
        'expected_min_items': 10
    },
    {
        'name': 'Indeed (Job Listings)',
        'url': 'https://www.indeed.com/jobs?q=python+developer&l=San+Francisco',
        'fields': ['title', 'company', 'location', 'salary', 'posted'],
        'expected_min_items': 10
    },
    {
        'name': 'Yelp (Restaurant Reviews)',
        'url': 'https://www.yelp.com/search?find_desc=restaurants&find_loc=New+York',
        'fields': ['name', 'rating', 'reviews', 'price', 'category'],
        'expected_min_items': 10
    },
    {
        'name': 'CNN (News)',
        'url': 'https://www.cnn.com/',
        'fields': ['title', 'description', 'category', 'timestamp'],
        'expected_min_items': 10
    },
    {
        'name': 'Zillow (Real Estate)',
        'url': 'https://www.zillow.com/san-francisco-ca/',
        'fields': ['address', 'price', 'beds', 'baths', 'sqft'],
        'expected_min_items': 10
    },
    {
        'name': 'Stack Overflow (Questions)',
        'url': 'https://stackoverflow.com/questions?tab=newest',
        'fields': ['title', 'votes', 'answers', 'views', 'tags', 'author'],
        'expected_min_items': 15
    },
    {
        'name': 'Etsy (Handmade Products)',
        'url': 'https://www.etsy.com/search?q=vintage+jewelry',
        'fields': ['title', 'price', 'shop', 'rating', 'favorites'],
        'expected_min_items': 20
    },
    {
        'name': 'Medium (Articles)',
        'url': 'https://medium.com/tag/artificial-intelligence',
        'fields': ['title', 'author', 'date', 'reading_time', 'claps'],
        'expected_min_items': 10
    },
    {
        'name': 'Airbnb (Listings)',
        'url': 'https://www.airbnb.com/s/San-Francisco--CA/homes',
        'fields': ['title', 'price', 'rating', 'location', 'type'],
        'expected_min_items': 15
    },
    {
        'name': 'BBC News',
        'url': 'https://www.bbc.com/news',
        'fields': ['title', 'description', 'category', 'timestamp'],
        'expected_min_items': 15
    }
]

def calculate_quality(items: List[Dict[str, Any]]) -> float:
    """Calculate data quality as % of non-null fields"""
    if not items:
        return 0.0
    
    total_fields = 0
    filled_fields = 0
    
    for item in items:
        for value in item.values():
            total_fields += 1
            if value is not None and value != '' and value != 'N/A':
                filled_fields += 1
    
    return (filled_fields / total_fields * 100) if total_fields > 0 else 0.0

def format_result(result: Dict[str, Any], expected_min: int) -> str:
    """Format test result with color coding"""
    items = result.get('data', result.get('items', []))
    count = len(items)
    quality = calculate_quality(items)
    
    # Status indicators
    if count == 0:
        status = '❌ FAILED'
        color = '\033[91m'  # Red
    elif count >= expected_min and quality >= 70:
        status = '✅ SUCCESS'
        color = '\033[92m'  # Green
    elif count >= expected_min and quality >= 40:
        status = '⚠️  PARTIAL'
        color = '\033[93m'  # Yellow
    else:
        status = '⚠️  LOW'
        color = '\033[93m'  # Yellow
    
    reset = '\033[0m'
    
    output = f"{color}{status}{reset}"
    output += f" | Items: {count}/{expected_min}"
    output += f" | Quality: {quality:.0f}%"
    
    if items and count > 0:
        # Show first item
        first = items[0]
        null_fields = [k for k, v in first.items() if v is None or v == '' or v == 'N/A']
        if null_fields:
            output += f"\n      Null fields: {', '.join(null_fields)}"
        
        # Show sample data
        output += f"\n      Sample: {str(first)[:100]}..."
    
    return output

async def test_site(scraper: UniversalScraper, site: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single site"""
    print(f"\n{'='*80}")
    print(f"Testing: {site['name']}")
    print(f"URL: {site['url']}")
    print(f"Fields: {site['fields']}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = await scraper.scrape(
            url=site['url'],
            fields=site['fields']
        )
        
        elapsed = time.time() - start_time
        
        items = result.get('data', result.get('items', []))
        quality = calculate_quality(items)
        
        print(f"\n⏱️  Time: {elapsed:.1f}s")
        print(f"📊 Results: {format_result(result, site['expected_min_items'])}")
        
        return {
            'name': site['name'],
            'url': site['url'],
            'success': len(items) > 0,
            'items': len(items),
            'expected': site['expected_min_items'],
            'quality': quality,
            'time': elapsed,
            'error': None
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR: {str(e)}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        
        return {
            'name': site['name'],
            'url': site['url'],
            'success': False,
            'items': 0,
            'expected': site['expected_min_items'],
            'quality': 0.0,
            'time': elapsed,
            'error': str(e)
        }

async def main():
    """Run all tests"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                 Universal Scraper - 10 Website Test Suite                 ║
║                                                                            ║
║  Testing all recent improvements:                                         ║
║  ✅ Smart HTML Sampler (dynamic sizing)                                   ║
║  ✅ Universal Field Mapper (semantic field detection)                     ║
║  ✅ Enhanced Attribute Extraction (custom elements)                       ║
║  ✅ Temporal Field Detection (date/time fields)                           ║
║  ✅ Smart Wait Strategy (JS-heavy sites)                                  ║
║  ✅ DOM Pattern Detection (universal patterns)                            ║
║  ✅ JSON Quality Validation                                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize scraper
    print("\n🚀 Initializing Universal Scraper...")
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    results = []
    
    try:
        # Test each site
        for i, site in enumerate(TEST_SITES, 1):
            print(f"\n\n{'#'*80}")
            print(f"# Test {i}/{len(TEST_SITES)}")
            print(f"{'#'*80}")
            
            result = await test_site(scraper, site)
            results.append(result)
            
            # Brief pause between tests
            if i < len(TEST_SITES):
                print("\n⏸️  Waiting 2s before next test...")
                await asyncio.sleep(2)
        
    finally:
        print("\n\n🔒 Closing scraper...")
        await scraper.close()
    
    # Summary
    print("\n\n")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                              FINAL SUMMARY                                 ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    high_quality = sum(1 for r in results if r['quality'] >= 70)
    total_items = sum(r['items'] for r in results)
    avg_quality = sum(r['quality'] for r in results) / total if total > 0 else 0
    avg_time = sum(r['time'] for r in results) / total if total > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"   • Sites Tested: {total}")
    print(f"   • Successful: {successful}/{total} ({successful/total*100:.0f}%)")
    print(f"   • High Quality (≥70%): {high_quality}/{total} ({high_quality/total*100:.0f}%)")
    print(f"   • Total Items Extracted: {total_items}")
    print(f"   • Average Quality: {avg_quality:.0f}%")
    print(f"   • Average Time: {avg_time:.1f}s")
    
    print(f"\n\n{'='*80}")
    print("Detailed Results:")
    print(f"{'='*80}\n")
    
    for r in results:
        status = '✅' if r['success'] and r['quality'] >= 70 else '⚠️' if r['success'] else '❌'
        print(f"{status} {r['name']}")
        print(f"   Items: {r['items']}/{r['expected']} | Quality: {r['quality']:.0f}% | Time: {r['time']:.1f}s")
        if r['error']:
            print(f"   Error: {r['error']}")
        print()
    
    # Success criteria
    print(f"\n{'='*80}")
    print("Success Criteria:")
    print(f"{'='*80}")
    
    criteria = [
        ('At least 8/10 sites successful', successful >= 8),
        ('At least 6/10 high quality (≥70%)', high_quality >= 6),
        ('Average quality ≥60%', avg_quality >= 60),
        ('Average time <30s per site', avg_time < 30),
    ]
    
    for criterion, passed in criteria:
        status = '✅' if passed else '❌'
        print(f"{status} {criterion}")
    
    all_passed = all(passed for _, passed in criteria)
    
    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 ALL SUCCESS CRITERIA MET! 🎉")
        print("The Universal Scraper is production-ready!")
    else:
        print("⚠️  Some criteria not met. Review failures above.")
    print(f"{'='*80}\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)






