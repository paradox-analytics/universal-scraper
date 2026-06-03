import asyncio
import time
import json
import os
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.context_manager import ExtractionContext

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

async def run_test(config):
    print(f"\n{'='*80}\n🧪 TEST: {config['name']}\n{'='*80}")
    print(f"URL: {config['url']}")
    print(f"Context: {config['context']}")
    
    start = time.time()
    scraper = UniversalScraper(
        api_key=OPENAI_API_KEY,
        fetch_mode="browser",
        model_name="gpt-4o-mini",
        enable_cache=True,
        enable_llm_pagination=False,  # Keep disabled for speed
        extraction_context=config['context'],
        enable_context_validation=True
    )
    
    print(f"⏱️  Scraping...")
    
    # Scrape (auto-extraction mode - empty fields list)
    result = await scraper.scrape(config['url'], fields=[])
    
    elapsed = time.time() - start
    
    # Extract results
    items = result.get('data', [])
    metadata = result.get('metadata', {})
    
    print(f"\n{'='*80}")
    if metadata.get('validation_status', {}).get('is_target_data', False) or len(items) > 0:
        print(f"✅ COMPLETE in {elapsed:.1f}s")
        print(f"   Items extracted: {len(items)}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
        if metadata.get('validation_status'):
            print(f"   Validation: {'PASS' if metadata['validation_status']['is_target_data'] else 'FAIL'} (confidence: {metadata['validation_status']['confidence']:.2f})")
            print(f"   Reasoning: {metadata['validation_status']['reasoning']}")
        if items:
            print(f"\n   📦 Sample items ({min(3, len(items))} of {len(items)}):")
            for i, item in enumerate(items[:3]):
                print(f"\n   Item {i+1}:")
                item_str = json.dumps(item, indent=6)
                # Truncate very long items
                if len(item_str) > 500:
                    item_str = item_str[:500] + "..."
                print(f"      {item_str}")
        return True, len(items), metadata.get('json_ranking_success', False)
    else:
        print(f"❌ FAILED in {elapsed:.1f}s")
        print(f"   Error: {metadata.get('error', 'Unknown error')}")
        return False, 0, metadata.get('json_ranking_success', False)

async def main():
    test_configs = [
        {
            'name': 'Apify Homepage',
            'url': 'https://apify.com/',
            'context': 'Extract featured Actors/scrapers with their name, description, author, run count, and rating'
        },
        {
            'name': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes, comments count, post URL, and timestamp'
        }
    ]
    
    print("\n{'='*80}\n🔬 TESTING APIFY & REDDIT\n{'='*80}")
    print(f"Sites to test: {len(test_configs)}")
    for i, config in enumerate(test_configs):
        print(f"   {i+1}. {config['name']}")
    
    results = []
    total_items_extracted = 0
    total_time_seconds = 0
    json_ranking_success_count = 0
    
    for config in test_configs:
        test_start = time.time()
        success, items_count, json_ranking_worked = await run_test(config)
        test_elapsed = time.time() - test_start
        
        results.append({
            'name': config['name'],
            'success': success,
            'items': items_count,
            'json_ranking_worked': json_ranking_worked,
            'time': test_elapsed
        })
        total_items_extracted += items_count
        total_time_seconds += test_elapsed
        if json_ranking_worked:
            json_ranking_success_count += 1
    
    print(f"\n{'='*80}\n📊 SUMMARY\n{'='*80}")
    
    passed_tests = sum(1 for r in results if r['success'])
    print(f"\nTests passed: {passed_tests}/{len(test_configs)}")
    print(f"Total items extracted: {total_items_extracted}")
    print(f"Total time: {total_time_seconds:.1f}s (avg {total_time_seconds/len(test_configs):.1f}s per site)")
    print(f"JSON ranking worked: {json_ranking_success_count}/{len(test_configs)}")
    
    print("\nDetailed results:")
    for r in results:
        status = "✅" if r['success'] else "⚠️"
        ranking_status = "✅" if r['json_ranking_worked'] else "❌"
        print(f"   {status} {r['name']}: {r['items']} items in {r['time']:.1f}s (ranking: {ranking_status})")
    
    if passed_tests < len(test_configs):
        print(f"\n{'='*80}\n⚠️  SOME TESTS FAILED - NEEDS DEBUGGING\n{'='*80}")
    else:
        print(f"\n{'='*80}\n🎉 ALL TESTS PASSED!\n{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())

