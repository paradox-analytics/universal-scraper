#!/usr/bin/env python3
"""
Quick test: 3 diverse sites to verify content-based DOM detection
"""

import asyncio
import os
from universal_scraper import UniversalScraper

SITES = [
    {
        'name': 'Stack Overflow',
        'url': 'https://stackoverflow.com/questions?tab=newest',
        'fields': ['title', 'votes'],
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'fields': ['repository', 'description', 'stars'],
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'comments'],
    }
]

async def test_site(scraper, site):
    """Test a single site"""
    try:
        print(f"\n{'='*75}")
        print(f"🔍 Testing: {site['name']}")
        print(f"{'='*75}")
        
        result = await scraper.scrape(
            url=site['url'],
            fields=site['fields']
        )
        
        items = result.get('data', [])
        
        if not items:
            print(f"❌ 0 items extracted")
            return {'name': site['name'], 'items': 0, 'quality': 0, 'status': '❌'}
        
        # Calculate quality
        total_fields = len(items) * len(site['fields'])
        filled_fields = sum(
            1 for item in items 
            for v in item.values() 
            if v is not None and v != ''
        )
        quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        
        # Show first 2 items
        print(f"\n   Sample Items:")
        for i, item in enumerate(items[:2], 1):
            print(f"   {i}. {item}")
        
        # Determine status
        if len(items) >= 10 and quality >= 70:
            status = '✅'
            print(f"\n✅ SUCCESS!")
        elif len(items) >= 5 and quality >= 50:
            status = '⚠️'
            print(f"\n⚠️  PARTIAL - Need to improve quality")
        else:
            status = '❌'
            print(f"\n❌ FAILED")
        
        return {
            'name': site['name'],
            'items': len(items),
            'quality': quality,
            'status': status
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'name': site['name'], 'items': 0, 'quality': 0, 'status': '❌'}

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║           Quick Test: Content-Based DOM Detection (3 Sites)               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    results = []
    try:
        for site in SITES:
            result = await test_site(scraper, site)
            results.append(result)
        
        # Summary
        print(f"\n{'='*75}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*75}")
        print(f"{'Site':<25} {'Items':<10} {'Quality':<12} {'Status':<10}")
        print(f"{'-'*75}")
        
        for r in results:
            print(f"{r['name']:<25} {r['items']:<10} {r['quality']:.0f}%{'':<8} {r['status']:<10}")
        
        success = sum(1 for r in results if r['status'] == '✅')
        print(f"{'-'*75}")
        print(f"✅ Success: {success}/{len(SITES)}")
        print(f"{'='*75}")
            
    finally:
        await scraper.close()

if __name__ == '__main__':
    asyncio.run(main())






