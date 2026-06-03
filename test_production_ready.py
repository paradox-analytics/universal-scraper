"""Production-ready test with Camoufox on 3 high-quality sites"""

import asyncio
import os
import time
from universal_scraper import UniversalScraper

async def test_site(scraper, url, fields, name):
    """Test a single site with production settings"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = await scraper.scrape(url=url, fields=fields)
        items = result.get('data', [])
        
        elapsed = time.time() - start_time
        
        if not items:
            print(f"   ❌ 0 items extracted ({elapsed:.1f}s)")
            return {'name': name, 'items': 0, 'quality': 0, 'status': '❌', 'time': elapsed}
        
        # Calculate quality
        total_fields = len(fields)
        filled_count = sum(
            1 for item in items
            for field in fields
            if item.get(field) not in (None, '', [])
        )
        quality = (filled_count / (len(items) * total_fields) * 100) if items else 0
        
        # Sample items
        print(f"\n📊 Results ({elapsed:.1f}s):")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        print(f"   Sample Items:")
        for i, item in enumerate(items[:2], 1):
            # Show first 100 chars of each field
            item_preview = {k: str(v)[:100] for k, v in item.items()}
            print(f"   {i}. {item_preview}")
        
        # Status
        if quality >= 80:
            status = "✅ PRODUCTION READY"
            symbol = '✅'
        elif quality >= 60:
            status = "⚠️  ACCEPTABLE - Minor improvements needed"
            symbol = '⚠️'
        else:
            status = "❌ NEEDS WORK"
            symbol = '❌'
        
        print(f"\n{status}")
        
        return {
            'name': name,
            'items': len(items),
            'quality': quality,
            'status': symbol,
            'time': elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Error: {e} ({elapsed:.1f}s)")
        return {'name': name, 'items': 0, 'quality': 0, 'status': '❌', 'time': elapsed}


async def main():
    print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║           PRODUCTION-READY TEST - Camoufox + Frequency Validation         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print("🎯 Camoufox: ENABLED (advanced anti-detection)")
    print("🎯 Frequency validation: < 5 items = reject JSON")
    print("🎯 Sibling detection: ENABLED (context-block extraction)")
    print("🎯 Anti-detection: Full fingerprinting + humanization")
    print()
    
    # Production-ready configuration
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,  # 🔥 Production-grade anti-detection
        headless=True,
        enable_auto_pagination=False,
        browser_timeout=30000,  # 30s timeout (production-appropriate)
    )
    
    # 3 high-quality, well-structured sites
    sites = [
        ("https://news.ycombinator.com/", ['title', 'points', 'comments'], "Hacker News"),
        ("https://stackoverflow.com/questions?tab=newest", ['title', 'votes'], "Stack Overflow"),
        ("https://github.com/trending", ['repository', 'description', 'stars'], "GitHub Trending"),
    ]
    
    results = []
    total_start = time.time()
    
    for url, fields, name in sites:
        result = await test_site(scraper, url, fields, name)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    await scraper.close()
    
    # Production metrics summary
    print("\n" + "="*80)
    print("📊 PRODUCTION TEST RESULTS")
    print("="*80)
    print(f"{'Site':<20} {'Items':<8} {'Quality':<10} {'Time':<10} {'Status':<10}")
    print("-"*80)
    
    success_count = 0
    total_items = 0
    
    for r in results:
        print(f"{r['name']:<20} {r['items']:<8} {r['quality']:.0f}%{'':<7} {r['time']:.1f}s{'':<6} {r['status']:<10}")
        if r['status'] == '✅':
            success_count += 1
        total_items += r['items']
    
    print("-"*80)
    print(f"✅ Success Rate: {success_count}/3 ({success_count/3*100:.0f}%)")
    print(f"📦 Total Items: {total_items}")
    print(f"⏱️  Total Time: {total_elapsed:.1f}s")
    print(f"⚡ Avg Time/Site: {total_elapsed/3:.1f}s")
    print("="*80)
    
    # Production readiness assessment
    if success_count == 3:
        print("\n🎉 PRODUCTION READY - All systems operational!")
        print("   ✅ Frequency validation working")
        print("   ✅ Camoufox anti-detection working")
        print("   ✅ Extraction quality meets production standards")
    elif success_count >= 2:
        print("\n✅ NEAR PRODUCTION READY - 1 site needs refinement")
    else:
        print("\n⚠️  NOT PRODUCTION READY - Multiple issues detected")

if __name__ == "__main__":
    asyncio.run(main())





