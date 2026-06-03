#!/usr/bin/env python3
"""
Test Enhanced Content-Based DOM Detector on 10 Diverse Sites

Verify that content-based scoring works universally
"""

import asyncio
import os
from universal_scraper import UniversalScraper

SITES = [
    {
        'name': 'Stack Overflow',
        'url': 'https://stackoverflow.com/questions?tab=newest',
        'fields': ['title', 'votes', 'answers', 'views'],
        'expected_items': 15
    },
    {
        'name': 'Zillow',
        'url': 'https://www.zillow.com/homes/San-Francisco,-CA_rb/',
        'fields': ['address', 'price', 'beds', 'baths'],
        'expected_items': 15
    },
    {
        'name': 'Amazon',
        'url': 'https://www.amazon.com/s?k=laptop',
        'fields': ['title', 'price', 'rating', 'reviews'],
        'expected_items': 20
    },
    {
        'name': 'Indeed',
        'url': 'https://www.indeed.com/jobs?q=python+developer',
        'fields': ['title', 'company', 'location', 'salary'],
        'expected_items': 15
    },
    {
        'name': 'Medium',
        'url': 'https://medium.com/tag/technology',
        'fields': ['title', 'author', 'date', 'claps'],
        'expected_items': 10
    },
    {
        'name': 'CNN',
        'url': 'https://www.cnn.com/world',
        'fields': ['title', 'description', 'date', 'category'],
        'expected_items': 15
    },
    {
        'name': 'Etsy',
        'url': 'https://www.etsy.com/search?q=handmade+jewelry',
        'fields': ['title', 'price', 'seller', 'rating'],
        'expected_items': 20
    },
    {
        'name': 'Yelp',
        'url': 'https://www.yelp.com/search?find_desc=restaurants&find_loc=San+Francisco',
        'fields': ['name', 'rating', 'reviews', 'category'],
        'expected_items': 10
    },
    {
        'name': 'Airbnb',
        'url': 'https://www.airbnb.com/s/San-Francisco--CA/homes',
        'fields': ['title', 'price', 'rating', 'location'],
        'expected_items': 15
    },
    {
        'name': 'BBC News',
        'url': 'https://www.bbc.com/news',
        'fields': ['title', 'description', 'date', 'category'],
        'expected_items': 15
    }
]

async def test_site(scraper, site):
    """Test a single site"""
    try:
        print(f"\n🔍 Testing: {site['name']}")
        result = await scraper.scrape(
            url=site['url'],
            fields=site['fields']
        )
        
        items = result.get('data', [])
        
        if not items:
            print(f"   ❌ 0 items extracted")
            return {'name': site['name'], 'items': 0, 'quality': 0, 'status': '❌'}
        
        # Calculate quality
        total_fields = len(items) * len(site['fields'])
        filled_fields = sum(
            1 for item in items 
            for v in item.values() 
            if v is not None and v != ''
        )
        quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        # Determine status
        if len(items) >= site['expected_items'] * 0.7 and quality >= 70:
            status = '✅'
        elif len(items) > 0 and quality >= 50:
            status = '⚠️'
        else:
            status = '❌'
        
        print(f"   {status} Items: {len(items)}, Quality: {quality:.0f}%")
        
        return {
            'name': site['name'],
            'items': len(items),
            'quality': quality,
            'status': status
        }
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return {'name': site['name'], 'items': 0, 'quality': 0, 'status': '❌'}

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║     Testing Content-Based DOM Detector on 10 Diverse Sites                ║
╚═══════════════════════════════════════════════════════════════════════════╝

✅ Content-based scoring (text density, semantic HTML, structure)
✅ Universal frequency penalty (>100 instances heavily penalized)
✅ No keyword ontology (works on ANY website)
✅ Reverted to GPT-4o-mini (cost savings)
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
            await asyncio.sleep(2)  # Rate limiting
        
        # Summary
        print(f"\n{'='*75}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*75}")
        print(f"{'Site':<20} {'Items':<10} {'Quality':<12} {'Status':<10}")
        print(f"{'-'*75}")
        
        success_count = 0
        partial_count = 0
        fail_count = 0
        
        for r in results:
            print(f"{r['name']:<20} {r['items']:<10} {r['quality']:.0f}%{'':<8} {r['status']:<10}")
            if r['status'] == '✅':
                success_count += 1
            elif r['status'] == '⚠️':
                partial_count += 1
            else:
                fail_count += 1
        
        print(f"{'-'*75}")
        print(f"✅ Success: {success_count}/10")
        print(f"⚠️  Partial: {partial_count}/10")
        print(f"❌ Failed:  {fail_count}/10")
        print(f"{'='*75}")
        
        if success_count >= 7:
            print(f"\n🎉 EXCELLENT! Content-based DOM detection is working universally!")
        elif success_count + partial_count >= 7:
            print(f"\n👍 GOOD! Most sites working, need to improve field extraction")
        else:
            print(f"\n⚠️  Need more work on DOM detection or field extraction")
            
    finally:
        await scraper.close()

if __name__ == '__main__':
    asyncio.run(main())






