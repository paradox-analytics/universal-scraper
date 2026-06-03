#!/usr/bin/env python3
"""
Test the new architecture on 5 diverse websites:
1. Smart HTML Sampler (dynamic sizing)
2. Field Mapper (semantic understanding)
3. Camoufox (anti-detection)
"""

import asyncio
import os
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


# Test sites with different structures
TEST_SITES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/programming/',
        'fields': ['title', 'author', 'upvotes', 'comments'],
        'expected_pattern': 'Social media posts (medium size)',
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'author', 'comments'],
        'expected_pattern': 'Minimal listings (small size)',
    },
    {
        'name': 'Product Hunt',
        'url': 'https://www.producthunt.com/',
        'fields': ['product', 'description', 'upvotes', 'comments'],
        'expected_pattern': 'Product cards (medium size)',
    },
    {
        'name': 'TechCrunch',
        'url': 'https://techcrunch.com/',
        'fields': ['title', 'author', 'date', 'category'],
        'expected_pattern': 'Article previews (large size)',
    },
    {
        'name': 'Craigslist',
        'url': 'https://sfbay.craigslist.org/search/sss',
        'fields': ['title', 'price', 'location', 'date'],
        'expected_pattern': 'Classified listings (small size)',
    },
]


async def test_site(site_config: dict, api_key: str) -> dict:
    """Test a single site and return metrics"""
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {site_config['name']}")
    print(f"{'='*80}")
    print(f"URL: {site_config['url']}")
    print(f"Fields: {', '.join(site_config['fields'])}")
    print(f"Expected: {site_config['expected_pattern']}")
    print()
    
    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            use_camoufox=True,
            headless=True,
            enable_auto_pagination=False,
            enable_cache=True
        )
        
        # Scrape
        result = await scraper.scrape(
            url=site_config['url'],
            fields=site_config['fields']
        )
        
        items = result.get('data', result.get('items', []))  # Handle both 'data' and 'items' keys
        
        # Calculate quality
        if items:
            complete_items = 0
            for item in items:
                non_null = sum(1 for v in item.values() if v is not None and str(v).strip())
                if non_null == len(site_config['fields']):
                    complete_items += 1
            
            quality = (complete_items / len(items)) * 100
        else:
            quality = 0
        
        # Show results
        print(f"📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}% ({complete_items if items else 0}/{len(items)} complete)")
        
        if items:
            print(f"\n📋 Sample (first item):")
            first = items[0]
            for field, value in first.items():
                status = "✅" if value is not None and str(value).strip() else "❌"
                value_str = str(value)[:80] if value else "None"
                print(f"   {status} {field}: {value_str}")
        
        return {
            'name': site_config['name'],
            'url': site_config['url'],
            'success': len(items) > 0,
            'item_count': len(items),
            'quality': quality,
            'fields_requested': len(site_config['fields']),
            'sample': items[0] if items else None
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': site_config['name'],
            'url': site_config['url'],
            'success': False,
            'error': str(e),
            'item_count': 0,
            'quality': 0
        }
    
    finally:
        if scraper:
            await scraper.close()


async def main():
    print("="*80)
    print("🚀 TESTING NEW ARCHITECTURE ON 5 DIVERSE WEBSITES")
    print("="*80)
    print()
    print("Architecture Components:")
    print("  1. Smart HTML Sampler - Dynamic sizing per website")
    print("  2. Field Mapper - Semantic field understanding")
    print("  3. Camoufox - Advanced anti-detection")
    print("  4. DOM Pattern Detector - Fast structure analysis")
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        return
    
    results = []
    
    # Test each site
    for site_config in TEST_SITES:
        result = await test_site(site_config, api_key)
        results.append(result)
        
        # Small delay between sites
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print()
    
    successful = sum(1 for r in results if r['success'])
    avg_quality = sum(r['quality'] for r in results if r['success']) / successful if successful else 0
    total_items = sum(r['item_count'] for r in results)
    
    print(f"Sites tested: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    print(f"Average quality: {avg_quality:.1f}%")
    print(f"Total items extracted: {total_items}")
    print()
    
    print("Site-by-Site Results:")
    print()
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        quality_str = f"{result['quality']:.0f}%" if result['success'] else "Failed"
        print(f"{status} {result['name']:<20} {result['item_count']:>3} items  Quality: {quality_str:>6}")
    
    print()
    print("="*80)
    print("🎯 ARCHITECTURE VALIDATION")
    print("="*80)
    print()
    
    if successful >= 4:
        print("✅ Architecture is UNIVERSAL and PRODUCTION-READY")
        print(f"   Success rate: {successful/len(results)*100:.0f}%")
        print(f"   Average quality: {avg_quality:.1f}%")
    elif successful >= 3:
        print("⚠️  Architecture is GOOD but needs refinement")
        print(f"   Success rate: {successful/len(results)*100:.0f}%")
        print(f"   Failed sites need investigation")
    else:
        print("❌ Architecture needs significant work")
        print(f"   Success rate: {successful/len(results)*100:.0f}%")
        print(f"   Multiple sites failing")


if __name__ == '__main__':
    asyncio.run(main())

