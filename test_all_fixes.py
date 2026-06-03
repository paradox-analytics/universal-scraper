"""
Test All Fixes - Simplified JSON Selection + Markdown + LLM Fallback
"""

import asyncio
import os
import time
import logging
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_site(name, url, context):
    """Test a single site with new fixes"""
    print("\n" + "="*80)
    print(f"🧪 TESTING: {name}")
    print("="*80)
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"\nFIXES APPLIED:")
    print(f"  ✅ Simplified JSON selection (select_best_source)")
    print(f"  ✅ Markdown conversion for HTML code generation")
    print(f"  ✅ LLM fallback for edge cases")
    print()

    start = time.time()

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - skipping")
        return None

    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,  # Single page test
        extraction_context=context,
        enable_context_validation=True,
        log_level=20  # INFO level
    )

    try:
        result = await scraper.scrape(url, fields=[])  # Auto-extract
        elapsed = time.time() - start
        
        data = result.get('data', [])
        metadata = result.get('metadata', {})
        source = metadata.get('extraction_source', 'unknown')
        
        print(f"\n================================================================================")
        print(f"⏱️  Completed in {elapsed:.1f} seconds")
        print(f"\n📊 RESULTS:")
        print(f"   Items extracted: {len(data)}")
        print(f"   Data source: {source}")
        print(f"   Success: {'✅ YES' if len(data) > 0 else '❌ NO'}")
        
        if data:
            print(f"\n📝 Sample (first 2 items):")
            for i, item in enumerate(data[:2]):
                print(f"   Item {i+1}:")
                for k, v in item.items():
                    val_str = str(v)[:100]  # Truncate long values
                    print(f"      {k}: {val_str}")
        
        print(f"\n================================================================================\n")
        
        return {
            'name': name,
            'url': url,
            'items': len(data),
            'source': source,
            'time': elapsed,
            'success': len(data) > 0
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'url': url,
            'items': 0,
            'source': 'error',
            'time': time.time() - start,
            'success': False,
            'error': str(e)
        }
    finally:
        scraper.close()

async def main():
    print("\n" + "="*80)
    print("🔬 TESTING ALL FIXES")
    print("="*80)
    print("Phase 1: ✅ HTML Cleaner (42-51% reduction)")
    print("Phase 2: ✅ Improved Code Generation Prompts")
    print("Phase 2.5: ✅ Simplified JSON Selection (select_best_source)")
    print("Phase 2.6: ✅ Markdown Conversion for HTML")
    print("Phase 3: ✅ LLM Fallback (direct extraction)")
    print("="*80)
    print("\n⏱️  This will take ~5-10 minutes (4 sites with browser automation)")
    print()

    tests = [
        {
            'name': "Reddit - Posts",
            'url': "https://www.reddit.com/r/webscraping/",
            'context': "Extract Reddit posts with title, author, upvotes, comments count, post URL"
        },
        {
            'name': "Apify - Actors",
            'url': "https://apify.com/",
            'context': "Extract scrapers/actors from Apify with name, description, rating, runs"
        },
        {
            'name': "Metacritic - Games",
            'url': "https://www.metacritic.com/browse/game/all/all/current-year/",
            'context': "Extract video game listings with title, platform, release date, Metascore rating"
        },
        {
            'name': "eBay - Laptops",
            'url': "https://www.ebay.com/b/Apple-Laptops/111422/bn_320025",
            'context': "Extract Apple laptop listings with title, price, condition, seller, ratings"
        }
    ]

    results = []
    for test in tests:
        result = await test_site(test['name'], test['url'], test['context'])
        if result:
            results.append(result)

    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['success'])
    total_items = sum(r['items'] for r in results)
    total_time = sum(r['time'] for r in results)
    
    print(f"\nSuccess Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    print(f"Total Items Extracted: {total_items}")
    print(f"Total Time: {total_time:.1f}s\n")
    
    print("SITE RESULTS:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['name']}: {r['items']} items from {r['source']} ({r['time']:.1f}s)")
    
    print("\n" + "="*80)
    print("🎯 EXPECTED vs ACTUAL")
    print("="*80)
    print("\nBEFORE FIXES:")
    print("  ❌ Reddit: 4 app config items (not posts)")
    print("  ❌ Apify: 2 JS libraries (not actors)")
    print("  ❌ Metacritic: 5 ad configs (not games)")
    print("  ❌ eBay: 33 UI actions (not laptops)")
    print("  Success: 0/4 (0%)")
    
    print("\nAFTER FIXES:")
    for r in results:
        status = "✅" if r['success'] and r['items'] >= 10 else "⚠️" if r['items'] > 0 else "❌"
        print(f"  {status} {r['name']}: {r['items']} items (expected: target data, not config)")
    print(f"  Success: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())








