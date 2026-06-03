import asyncio
import time
import json
import os
from universal_scraper.core.scraper import UniversalScraper

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

async def test_single_site(url: str, context: str, site_name: str):
    """Test a single site with no pagination - first page only"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {site_name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"Pagination: DISABLED (first page only)")
    print()
    
    start = time.time()
    
    # Initialize scraper with pagination completely disabled
    scraper = UniversalScraper(
        api_key=OPENAI_API_KEY,
        fetch_mode="browser",
        model_name="gpt-4o-mini",
        enable_cache=True,
        enable_llm_pagination=False,  # Disable LLM pagination
        extraction_context=context,
        enable_context_validation=True
    )
    
    # Disable fast pagination detector by setting it to None
    scraper.fast_pagination_detector = None
    
    print(f"⏱️  Scraping (first page only)...")
    
    # Scrape with empty fields for auto-extraction
    result = await scraper.scrape(url, fields=[])
    
    elapsed = time.time() - start
    
    # Extract results
    items = result.get('data', [])
    metadata = result.get('metadata', {})
    validation = metadata.get('validation_status', {})
    
    print(f"\n{'='*80}")
    print(f"⏱️  Completed in {elapsed:.1f} seconds")
    print(f"{'='*80}")
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   Items extracted: {len(items)}")
    print(f"   Data source: {metadata.get('source', 'N/A')}")
    print(f"   JSON ranking: {'✅ SUCCESS' if metadata.get('json_ranking_success') else '❌ FAILED'}")
    
    if validation:
        print(f"\n🔍 VALIDATION:")
        print(f"   Is target data: {'✅ YES' if validation.get('is_target_data') else '❌ NO'}")
        print(f"   Confidence: {validation.get('confidence', 0):.2f}")
        print(f"   Reasoning: {validation.get('reasoning', 'N/A')}")
    
    if items:
        print(f"\n{'='*80}")
        print(f"📦 EXTRACTED DATA (showing first 3 items):")
        print(f"{'='*80}")
        for i, item in enumerate(items[:3], 1):
            print(f"\n--- Item {i} ---")
            print(json.dumps(item, indent=2, ensure_ascii=False))
        
        if len(items) > 3:
            print(f"\n... and {len(items) - 3} more items")
    else:
        print(f"\n⚠️  No items extracted!")
        if metadata.get('error'):
            print(f"   Error: {metadata.get('error')}")
    
    print(f"\n{'='*80}\n")
    
    return {
        'site': site_name,
        'success': len(items) > 0,
        'items_count': len(items),
        'time': elapsed,
        'source': metadata.get('source'),
        'validation_passed': validation.get('is_target_data', False)
    }

async def main():
    print("\n" + "="*80)
    print("🔬 SINGLE PAGE TEST - NO PAGINATION")
    print("="*80)
    print("This test will scrape ONLY the first page of each site")
    print("to quickly validate the context-driven extraction.\n")
    
    # Test configurations
    tests = [
        {
            'site': 'Apify Homepage',
            'url': 'https://apify.com/',
            'context': 'Extract featured Actors/scrapers with their name, description, author, run count, and rating'
        },
        {
            'site': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes, comments count, post URL, and timestamp'
        }
    ]
    
    # Run tests
    results = []
    for test in tests:
        result = await test_single_site(test['url'], test['context'], test['site'])
        results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"\nTotal sites tested: {len(results)}")
    print(f"Successful extractions: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Total items extracted: {sum(r['items_count'] for r in results)}")
    print(f"Total time: {sum(r['time'] for r in results):.1f}s")
    
    print("\nResults by site:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        validation = "✅" if r['validation_passed'] else "⚠️"
        print(f"   {status} {r['site']}: {r['items_count']} items in {r['time']:.1f}s | Source: {r['source']} | Validation: {validation}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())








