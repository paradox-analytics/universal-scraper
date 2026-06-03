#!/usr/bin/env python3
"""
Test remaining sites after GitHub fix
"""

import asyncio
import os
import csv
from pathlib import Path
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper

# Test configurations
TESTS = [
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'fields': ['repository', 'description', 'stars', 'language'],
        'context': 'Extract trending GitHub repositories'
    },
    {
        'name': 'TechCrunch',
        'url': 'https://techcrunch.com/',
        'fields': ['title', 'author', 'date', 'url'],
        'context': 'Extract TechCrunch articles'
    },
    {
        'name': 'Medium',
        'url': 'https://medium.com/tag/technology',
        'fields': ['title', 'author', 'readTime', 'claps'],
        'context': 'Extract Medium articles'
    },
    {
        'name': 'Product Hunt',
        'url': 'https://www.producthunt.com/',
        'fields': ['name', 'tagline', 'upvotes', 'comments'],
        'context': 'Extract Product Hunt products'
    },
    {
        'name': 'Walmart',
        'url': 'https://www.walmart.com/search?q=laptop',
        'fields': ['title', 'price', 'rating', 'reviews'],
        'context': 'Extract Walmart product listings'
    },
    {
        'name': 'Etsy',
        'url': 'https://www.etsy.com/search?q=handmade',
        'fields': ['title', 'price', 'shop', 'rating'],
        'context': 'Extract Etsy product listings'
    }
]

async def test_site(test_config):
    """Test a single site"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {test_config['name']}")
    print(f"{'='*80}")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=False,  # Use Playwright for now
            headless=True,
            enable_auto_pagination=False,
            extraction_context=test_config['context']
        )
        
        print(f"🎯 URL: {test_config['url']}")
        print(f"📋 Fields: {', '.join(test_config['fields'])}")
        print()
        
        result = await scraper.scrape(test_config['url'], test_config['fields'])
        
        print(f"\n✅ RESULTS:")
        print(f"   • Items: {len(result['data'])}")
        if 'total_time' in result:
            print(f"   • Time: {result['total_time']:.1f}s")
        if 'method' in result:
            print(f"   • Source: {result.get('method', 'unknown')}")
        
        if result['data']:
            # Calculate quality
            total_fields = 0
            complete_items = 0
            for item in result['data']:
                non_null = sum(1 for v in item.values() if v not in [None, '', 'N/A'])
                if non_null == len(item):
                    complete_items += 1
                total_fields += len(item)
            
            quality = (complete_items / len(result['data']) * 100) if result['data'] else 0
            
            print(f"   • Quality: {quality:.0f}% ({complete_items}/{len(result['data'])} complete)")
            
            print(f"\n📋 Sample (first 2):")
            for i, item in enumerate(result['data'][:2], 1):
                print(f"\n   Item {i}:")
                for k, v in item.items():
                    val_str = str(v)[:100] if v else 'None'
                    print(f"      • {k}: {val_str}")
            
            # Save to CSV
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            csv_file = output_dir / f"{test_config['name'].lower().replace(' ', '_')}.csv"
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=test_config['fields'])
                writer.writeheader()
                writer.writerows(result['data'])
            
            print(f"\n   💾 Saved to: {csv_file}")
            
            return {
                'name': test_config['name'],
                'success': True,
                'items': len(result['data']),
                'quality': quality,
                'time': result.get('total_time', 0)
            }
        else:
            print("   ❌ No items extracted")
            return {
                'name': test_config['name'],
                'success': False,
                'items': 0,
                'quality': 0,
                'time': result.get('total_time', 0)
            }
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return {
            'name': test_config['name'],
            'success': False,
            'error': str(e)[:100]
        }
    
    finally:
        if scraper:
            await scraper.close()

async def main():
    print("="*80)
    print("🚀 TESTING ALL REMAINING SITES")
    print("="*80)
    
    results = []
    
    for test in TESTS:
        result = await test_site(test)
        if result:
            results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n🎯 Success Details:")
        for r in successful:
            print(f"   • {r['name']}: {r['items']} items, {r['quality']:.0f}% quality, {r['time']:.1f}s")
    
    if failed:
        print(f"\n❌ Failed Sites:")
        for r in failed:
            error_msg = r.get('error', 'Unknown error')
            print(f"   • {r['name']}: {error_msg}")
    
    # Calculate final score
    if results:
        score = (len(successful) / len(results)) * 100
        print(f"\n🏆 OVERALL SCORE: {score:.0f}% ({len(successful)}/{len(results)} sites working)")

if __name__ == '__main__':
    asyncio.run(main())

