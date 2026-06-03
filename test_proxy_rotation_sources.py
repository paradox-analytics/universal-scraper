"""
Test Proxy Rotation + Universal Scraping with Diverse Sources

Tests 4 diverse sources to verify:
1. Proxy rotation is working (if configured)
2. Universal extraction works across domains
3. All features integrated correctly
"""

import asyncio
import os
import logging
from universal_scraper import UniversalScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_source(scraper: UniversalScraper, name: str, url: str, fields: list):
    """Test a single source and return results"""
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
        print(f"\n{status} Status: {'EXCELLENT' if quality >= 90 else 'NEEDS WORK' if quality >= 50 else 'FAILED'}")
        
        return {
            'name': name,
            'url': url,
            'items': len(items),
            'quality': quality,
            'duration': duration,
            'status': status,
            'sample': items[:2] if items else []
        }
        
    except Exception as e:
        duration = time.time() - start
        print(f"\n❌ Error: {str(e)[:200]}")
        return {
            'name': name,
            'url': url,
            'items': 0,
            'quality': 0,
            'duration': duration,
            'status': '❌',
            'error': str(e)[:200]
        }

async def main():
    print("\n" + "="*80)
    print("🚀 PROXY ROTATION + UNIVERSAL SCRAPING TEST")
    print("="*80)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Test sources (diverse domains)
    sources = [
        {
            'name': 'Product Hunt',
            'url': 'https://www.producthunt.com/',
            'fields': ['title', 'description', 'votes']
        },
        {
            'name': 'BBC News',
            'url': 'https://www.bbc.com/news',
            'fields': ['headline', 'description']
        },
        {
            'name': 'Reddit (r/Python)',
            'url': 'https://www.reddit.com/r/Python/',
            'fields': ['title', 'author', 'upvotes']
        },
        {
            'name': 'Medium',
            'url': 'https://medium.com/',
            'fields': ['title', 'author', 'read_time']
        }
    ]
    
    # Initialize scraper with optional proxy rotation
    # In production (Apify), this would enable per-request rotation
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,  # Advanced anti-detection
        enable_auto_pagination=False,
        # proxy_config={'useApifyProxy': True, 'apifyProxyGroups': ['RESIDENTIAL']}  # Uncomment for Apify
    )
    
    print(f"\n🎯 Testing {len(sources)} sources...")
    print(f"🦊 Camoufox: ENABLED")
    print(f"🔄 Proxy Rotation: {'ENABLED (per-request)' if scraper.html_fetcher else 'NOT CONFIGURED (local testing)'}")
    print(f"📦 Context-Block Extraction: ENABLED")
    print(f"⚡ Frequency Validation: ENABLED")
    
    results = []
    total_start = asyncio.get_event_loop().time()
    
    for source in sources:
        result = await test_source(
            scraper,
            source['name'],
            source['url'],
            source['fields']
        )
        results.append(result)
        
        # Brief pause between requests (good practice)
        await asyncio.sleep(2)
    
    total_duration = asyncio.get_event_loop().time() - total_start
    
    # Close scraper
    await scraper.close()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"{'Site':<20} {'Items':<8} {'Quality':<10} {'Time':<10} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        quality_str = f"{result['quality']:.0f}%"
        print(f"{result['name']:<20} {result['items']:<8} {quality_str:<10} {result['duration']:<10.1f}s {result['status']:<10}")
    
    print("-" * 80)
    
    # Statistics
    successful = sum(1 for r in results if r['quality'] >= 90)
    partial = sum(1 for r in results if 50 <= r['quality'] < 90)
    failed = sum(1 for r in results if r['quality'] < 50)
    total_items = sum(r['items'] for r in results)
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n✅ Excellent: {successful}/{len(sources)} ({successful/len(sources):.0%})")
    print(f"⚠️  Partial: {partial}/{len(sources)} ({partial/len(sources):.0%})")
    print(f"❌ Failed: {failed}/{len(sources)} ({failed/len(sources):.0%})")
    print(f"📦 Total Items: {total_items}")
    print(f"📊 Avg Quality: {avg_quality:.0f}%")
    print(f"⏱️  Total Time: {total_duration:.1f}s")
    print(f"⚡ Avg Time/Site: {total_duration/len(sources):.1f}s")
    
    # Final assessment
    print("\n" + "="*80)
    if successful == len(sources):
        print("🎉 PERFECT - All sources working!")
    elif successful + partial >= len(sources) * 0.75:
        print("✅ GOOD - Most sources working, minor issues")
    else:
        print("⚠️  NEEDS WORK - Multiple sources need attention")
    print("="*80)
    
    # Proxy rotation note
    if not scraper.html_fetcher:
        print("\n💡 Note: Proxy rotation is configured but not active in local testing.")
        print("   On Apify with residential proxies, each request would use a different IP.")
        print("   This prevents IP-based rate limiting and blocking.")

if __name__ == "__main__":
    asyncio.run(main())

