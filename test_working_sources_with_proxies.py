"""Test working sources (Reddit, Hacker News) with and without Apify proxies"""
import asyncio
import os
import sys
import csv
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


# Working sources only
WORKING_SOURCES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'context': 'Extract Reddit posts with title, author, upvotes, comments',
        'fields': ['title', 'author', 'upvotes', 'comments_count'],
        'wait_for': 'shreddit-post'
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'context': 'Extract HN posts with title, points, author, comments',
        'fields': ['title', 'points', 'author', 'comments'],
        'wait_for': None
    }
]


async def test_source(source, use_proxy=False):
    """Test a single source with or without proxy"""
    proxy_label = "WITH PROXY" if use_proxy else "NO PROXY"
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {source['name']} ({proxy_label})")
    print(f"{'='*80}\n")
    
    # Build proxy config if needed
    proxy_config = None
    if use_proxy:
        apify_token = os.environ.get('APIFY_TOKEN')
        if not apify_token:
            print("❌ APIFY_TOKEN not set!")
            return None
        
        proxy_config = {
            'server': f'http://proxy.apify.com:8000',
            'username': 'auto',
            'password': apify_token
        }
        print(f"🔐 Using Apify residential proxy\n")
    
    # Create scraper
    scraper = None
    start_time = time.time()
    
    try:
        scraper = UniversalScraper(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4o-mini",
            extraction_context=source['context'],
            proxy_config=proxy_config,
            fetch_mode="browser",
            headless=True,
            browser_timeout=120000 if use_proxy else 60000,  # Longer timeout for proxies
            enable_llm_pagination=False
        )
        
        # Disable pagination
        if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
            scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
        if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
            scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
        
        # Scrape
        result = await scraper.scrape(
            source['url'],
            fields=source['fields'],
            wait_for_selector=source.get('wait_for')
        )
        
        elapsed = time.time() - start_time
        
        # Extract data
        if isinstance(result, dict) and 'data' in result:
            items = result['data']
            metadata = result.get('metadata', {})
        elif isinstance(result, list):
            items = result
            metadata = {}
        else:
            items = []
            metadata = {}
        
        # Print results
        print(f"\n✅ RESULTS:")
        print(f"   • Items extracted: {len(items)}")
        print(f"   • Extraction source: {metadata.get('extraction_source', 'unknown')}")
        print(f"   • Total time: {elapsed:.1f}s")
        print(f"   • Code cached: {metadata.get('code_cached', 'N/A')}")
        
        if items:
            # Show sample
            print(f"\n📋 Sample (first 2 items):")
            for i, item in enumerate(items[:2], 1):
                print(f"\n   Item {i}:")
                for key, value in list(item.items())[:4]:  # First 4 fields
                    if not key.startswith('_'):
                        value_str = str(value) if value else 'N/A'
                        value_str = value_str[:60]
                        if value and len(str(value)) > 60:
                            value_str += "..."
                        print(f"     • {key}: {value_str}")
            
            # Quality metrics
            complete = sum(1 for item in items if all(item.get(f) for f in source['fields']))
            completeness = (complete / len(items)) * 100 if items else 0
            
            print(f"\n📈 Quality:")
            print(f"   • Complete items: {complete}/{len(items)} ({completeness:.0f}%)")
            print(f"   • Proxy used: {'Yes' if use_proxy else 'No'}")
        
        return {
            'success': True,
            'items': items,
            'count': len(items),
            'completeness': (sum(1 for item in items if all(item.get(f) for f in source['fields'])) / len(items)) * 100 if items else 0,
            'time': elapsed,
            'proxy_used': use_proxy,
            'extraction_source': metadata.get('extraction_source'),
            'code_cached': metadata.get('code_cached', False)
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'time': elapsed,
            'proxy_used': use_proxy
        }
    
    finally:
        if scraper:
            scraper.close()


def save_csv(items, filename):
    """Save items to CSV"""
    if not items:
        print(f"⚠️  No data to save for {filename}")
        return
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / filename
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if items:
            # Get all unique keys
            keys = set()
            for item in items:
                keys.update(k for k in item.keys() if not k.startswith('_'))
            
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            
            for item in items:
                clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
                writer.writerow(clean_item)
    
    print(f"   💾 Saved to: {csv_path}")


async def main():
    """Test all working sources with and without proxies"""
    
    print(f"\n{'#'*80}")
    print("🚀 TESTING WORKING SOURCES WITH APIFY RESIDENTIAL PROXIES")
    print(f"{'#'*80}")
    print("\nSources to test:")
    for source in WORKING_SOURCES:
        print(f"  • {source['name']}")
    print()
    
    results = {}
    
    for source in WORKING_SOURCES:
        source_name = source['name']
        results[source_name] = {}
        
        # Test WITHOUT proxy
        result_no_proxy = await test_source(source, use_proxy=False)
        results[source_name]['no_proxy'] = result_no_proxy
        
        if result_no_proxy and result_no_proxy.get('success') and result_no_proxy.get('items'):
            filename = f"{source_name.lower().replace(' ', '_')}_no_proxy.csv"
            save_csv(result_no_proxy['items'], filename)
        
        print("\n" + "-"*80 + "\n")
        
        # Test WITH proxy
        result_with_proxy = await test_source(source, use_proxy=True)
        results[source_name]['with_proxy'] = result_with_proxy
        
        if result_with_proxy and result_with_proxy.get('success') and result_with_proxy.get('items'):
            filename = f"{source_name.lower().replace(' ', '_')}_with_proxy.csv"
            save_csv(result_with_proxy['items'], filename)
        
        print("\n")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON: PROXY vs NO PROXY")
    print(f"{'='*80}\n")
    
    for source_name, source_results in results.items():
        print(f"{'='*80}")
        print(f"📌 {source_name}")
        print(f"{'='*80}")
        
        no_proxy = source_results.get('no_proxy', {})
        with_proxy = source_results.get('with_proxy', {})
        
        if no_proxy.get('success'):
            print(f"\n✅ WITHOUT PROXY:")
            print(f"   • Items: {no_proxy.get('count', 0)}")
            print(f"   • Quality: {no_proxy.get('completeness', 0):.0f}%")
            print(f"   • Time: {no_proxy.get('time', 0):.1f}s")
            print(f"   • Cached: {no_proxy.get('code_cached', False)}")
        else:
            print(f"\n❌ WITHOUT PROXY: Failed - {no_proxy.get('error', 'Unknown')}")
        
        if with_proxy.get('success'):
            print(f"\n✅ WITH PROXY:")
            print(f"   • Items: {with_proxy.get('count', 0)}")
            print(f"   • Quality: {with_proxy.get('completeness', 0):.0f}%")
            print(f"   • Time: {with_proxy.get('time', 0):.1f}s")
            print(f"   • Cached: {with_proxy.get('code_cached', False)}")
        else:
            print(f"\n❌ WITH PROXY: Failed - {with_proxy.get('error', 'Unknown')}")
        
        # Comparison
        if no_proxy.get('success') and with_proxy.get('success'):
            item_diff = with_proxy.get('count', 0) - no_proxy.get('count', 0)
            time_diff = with_proxy.get('time', 0) - no_proxy.get('time', 0)
            quality_diff = with_proxy.get('completeness', 0) - no_proxy.get('completeness', 0)
            
            print(f"\n📊 COMPARISON:")
            print(f"   • Items difference: {item_diff:+d}")
            print(f"   • Quality difference: {quality_diff:+.0f}%")
            print(f"   • Time difference: {time_diff:+.1f}s")
        
        print()
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())







