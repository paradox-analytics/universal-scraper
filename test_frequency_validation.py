"""Test frequency-based validation on 10 diverse sites"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_site(scraper, url, fields, name):
    """Test a single site"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {name}")
    print(f"{'='*80}")
    
    try:
        result = await scraper.scrape(url=url, fields=fields)
        items = result.get('data', [])
        
        if not items:
            print(f"   ❌ 0 items extracted")
            return {'name': name, 'items': 0, 'quality': 0, 'status': '❌'}
        
        # Calculate quality
        total_fields = len(fields)
        filled_count = sum(
            1 for item in items
            for field in fields
            if item.get(field) not in (None, '', [])
        )
        quality = (filled_count / (len(items) * total_fields) * 100) if items else 0
        
        # Sample items
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        print(f"   Sample Items:")
        for i, item in enumerate(items[:3], 1):
            print(f"   {i}. {item}")
        
        # Status
        if quality >= 80:
            status = "✅ SUCCESS!"
            symbol = '✅'
        elif quality >= 50:
            status = "⚠️  PARTIAL - Need to improve quality"
            symbol = '⚠️'
        else:
            status = "❌ FAILED"
            symbol = '❌'
        
        print(status)
        
        return {
            'name': name,
            'items': len(items),
            'quality': quality,
            'status': symbol
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'name': name, 'items': 0, 'quality': 0, 'status': '❌'}


async def main():
    print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Testing Frequency-Based Validation on 10 Diverse Sites               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print("✅ JSON frequency check: < 5 items = reject")
    print("✅ Sibling detection + context blocks")
    print("✅ Frequency-based field matching")
    print()
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=False,
        headless=True,
        enable_auto_pagination=False
    )
    
    sites = [
        ("https://stackoverflow.com/questions?tab=newest", ['title', 'votes'], "Stack Overflow"),
        ("https://github.com/trending", ['repository', 'description', 'stars'], "GitHub Trending"),
        ("https://news.ycombinator.com/", ['title', 'points', 'comments'], "Hacker News"),
        ("https://www.reddit.com/r/programming/", ['title', 'score', 'comments'], "Reddit"),
        ("https://www.producthunt.com/", ['name', 'tagline', 'votes'], "Product Hunt"),
        ("https://www.zillow.com/homes/for_sale/", ['address', 'price', 'beds'], "Zillow"),
        ("https://www.amazon.com/s?k=laptop", ['title', 'price', 'rating'], "Amazon"),
        ("https://www.bbc.com/news", ['title', 'summary'], "BBC News"),
        ("https://techcrunch.com/", ['title', 'author'], "TechCrunch"),
        ("https://medium.com/", ['title', 'author', 'claps'], "Medium"),
    ]
    
    results = []
    
    for url, fields, name in sites:
        result = await test_site(scraper, url, fields, name)
        results.append(result)
    
    await scraper.close()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"{'Site':<25} {'Items':<10} {'Quality':<12} {'Status':<10}")
    print("-"*80)
    
    success_count = 0
    partial_count = 0
    failed_count = 0
    
    for r in results:
        print(f"{r['name']:<25} {r['items']:<10} {r['quality']:.0f}%{'':<9} {r['status']:<10}")
        if r['status'] == '✅':
            success_count += 1
        elif r['status'] == '⚠️':
            partial_count += 1
        else:
            failed_count += 1
    
    print("-"*80)
    print(f"✅ Success: {success_count}/10")
    print(f"⚠️  Partial: {partial_count}/10")
    print(f"❌ Failed:  {failed_count}/10")
    print("="*80)
    
    if success_count >= 7:
        print("🎉 EXCELLENT - Frequency validation working!")
    elif success_count >= 5:
        print("✅ GOOD - Most sites working with frequency validation")
    elif partial_count + success_count >= 6:
        print("⚠️  PARTIAL - Frequency validation helps, needs refinement")
    else:
        print("❌ NEEDS WORK - More debugging required")

if __name__ == "__main__":
    asyncio.run(main())





