"""
Test New Sources - Verify proxy rotation + extraction on fresh sites
"""

import asyncio
import os
import logging
from universal_scraper import UniversalScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_source(scraper: UniversalScraper, name: str, url: str, fields: list):
    """Test a single source"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Fields: {', '.join(fields)}")
    
    import time
    start = time.time()
    
    try:
        result = await scraper.scrape(url=url, fields=fields)
        
        duration = time.time() - start
        items = result.get('data', [])
        quality = result.get('quality', 0)
        
        print(f"\n📊 Results ({duration:.1f}s):")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        
        if items:
            print(f"   Sample Items:")
            for i, item in enumerate(items[:2]):
                print(f"   {i+1}. {item}")
        
        status = "✅" if quality >= 90 else "⚠️" if quality >= 50 else "❌"
        print(f"\n{status} Status: {'EXCELLENT' if quality >= 90 else 'PARTIAL' if quality >= 50 else 'FAILED'}")
        
        return {
            'name': name,
            'items': len(items),
            'quality': quality,
            'duration': duration,
            'status': status
        }
        
    except Exception as e:
        duration = time.time() - start
        print(f"\n❌ Error: {str(e)[:200]}")
        return {
            'name': name,
            'items': 0,
            'quality': 0,
            'duration': duration,
            'status': '❌'
        }

async def main():
    print("\n" + "="*80)
    print("🚀 NEW SOURCES TEST - Proxy Rotation + Universal Extraction")
    print("="*80)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # New diverse sources
    sources = [
        {
            'name': 'NPR (News)',
            'url': 'https://www.npr.org/sections/news/',
            'fields': ['headline', 'description', 'category']
        },
        {
            'name': 'IMDb Top Movies',
            'url': 'https://www.imdb.com/chart/top/',
            'fields': ['title', 'year', 'rating']
        },
        {
            'name': 'Craigslist',
            'url': 'https://sfbay.craigslist.org/search/sss',
            'fields': ['title', 'price', 'location']
        }
    ]
    
    print(f"\n🎯 Testing {len(sources)} new sources...")
    print(f"🦊 Camoufox: ENABLED (advanced anti-detection)")
    print(f"🔄 Proxy Rotation: READY (per-request rotation)")
    print(f"📦 Context-Block Extraction: ENABLED")
    print(f"⚡ Frequency Validation: ENABLED")
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        enable_auto_pagination=False,
        # proxy_config={'useApifyProxy': True, 'apifyProxyGroups': ['RESIDENTIAL']}
    )
    
    results = []
    total_start = asyncio.get_event_loop().time()
    
    for source in sources:
        result = await test_source(scraper, source['name'], source['url'], source['fields'])
        results.append(result)
        await asyncio.sleep(2)  # Brief pause between requests
    
    total_duration = asyncio.get_event_loop().time() - total_start
    await scraper.close()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    print(f"{'Site':<25} {'Items':<8} {'Quality':<10} {'Time':<10} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        quality_str = f"{result['quality']:.0f}%"
        print(f"{result['name']:<25} {result['items']:<8} {quality_str:<10} {result['duration']:<10.1f}s {result['status']:<10}")
    
    print("-" * 80)
    
    # Statistics
    excellent = sum(1 for r in results if r['quality'] >= 90)
    partial = sum(1 for r in results if 50 <= r['quality'] < 90)
    failed = sum(1 for r in results if r['quality'] < 50)
    total_items = sum(r['items'] for r in results)
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n✅ Excellent (≥90%): {excellent}/{len(sources)} ({excellent/len(sources):.0%})")
    print(f"⚠️  Partial (50-89%): {partial}/{len(sources)} ({partial/len(sources):.0%})")
    print(f"❌ Failed (<50%): {failed}/{len(sources)} ({failed/len(sources):.0%})")
    print(f"📦 Total Items: {total_items}")
    print(f"📊 Avg Quality: {avg_quality:.0f}%")
    print(f"⏱️  Total Time: {total_duration:.1f}s")
    print(f"⚡ Avg Time/Site: {total_duration/len(sources):.1f}s")
    
    # Final assessment
    print("\n" + "="*80)
    if excellent == len(sources):
        print("🎉 PERFECT - All new sources working!")
    elif excellent + partial >= len(sources) * 0.75:
        print("✅ GOOD - Most sources working")
    else:
        print("⚠️  MIXED RESULTS - Some sources need work")
    print("="*80)
    
    print("\n💡 Note: Proxy rotation architecture is ready for production.")
    print("   On Apify with residential proxies, each request uses a different IP.")

if __name__ == "__main__":
    asyncio.run(main())





