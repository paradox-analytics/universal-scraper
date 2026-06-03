"""
Test all sources with Apify Residential Proxies
Rerun all previous tests to compare with and without proxies
"""

import asyncio
import sys
import os
from datetime import datetime
import csv
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


# Test sources (same as before)
TEST_SOURCES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'context': 'Extract Reddit posts with title, author, upvotes, comments count',
        'fields': ['title', 'author', 'upvotes', 'comments_count'],
        'wait_for': 'shreddit-post'
    },
    {
        'name': 'eBay',
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'context': 'Extract eBay product listings with title, price, shipping, condition',
        'fields': ['title', 'price', 'shipping', 'condition'],
        'wait_for': '.s-item'
    },
    {
        'name': 'Metacritic Games',
        'url': 'https://www.metacritic.com/browse/game/',
        'context': 'Extract video games with title, score, platform, release date',
        'fields': ['title', 'score', 'platform', 'release_date'],
        'wait_for': '.c-finderProductCard'
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'context': 'Extract top stories with title, author, points, comments',
        'fields': ['title', 'author', 'points', 'comments'],
        'wait_for': '.athing'
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'context': 'Extract trending repositories with name, description, language, stars',
        'fields': ['repository_name', 'description', 'programming_language', 'stars_count'],
        'wait_for': 'article.Box-row'
    }
]


def get_apify_proxy_config():
    """
    Get Apify proxy configuration from environment
    
    Apify proxies format: http://auto:<APIFY_PROXY_PASSWORD>@proxy.apify.com:8000
    Or: http://groups-RESIDENTIAL:<APIFY_PROXY_PASSWORD>@proxy.apify.com:8000
    
    For Apify users, the password is constructed from your API token.
    """
    apify_token = os.environ.get('APIFY_TOKEN')
    
    if not apify_token:
        print("⚠️  APIFY_TOKEN not found in environment")
        print("   Set it with: export APIFY_TOKEN='apify_api_...'")
        return None
    
    # Apify proxy configuration
    # Using RESIDENTIAL proxy group for better anti-bot protection
    proxy_password = apify_token
    
    return {
        'server': 'http://proxy.apify.com:8000',
        'username': 'groups-RESIDENTIAL,session-default',  # Use residential proxies with sticky session
        'password': proxy_password
    }


async def test_source(source, use_proxy=False):
    """Test a single source"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {source['name']}")
    print(f"{'='*80}")
    print(f"📋 URL: {source['url']}")
    print(f"📋 Context: {source['context']}")
    print(f"📋 Fields: {', '.join(source['fields'])}")
    if use_proxy:
        print(f"🌐 Proxy: Apify Residential Proxies (ENABLED)")
    else:
        print(f"🌐 Proxy: None (Direct connection)")
    print()
    
    # Get proxy config if requested
    proxy_config = get_apify_proxy_config() if use_proxy else None
    
    if use_proxy and not proxy_config:
        print("❌ Cannot test with proxies - APIFY_TOKEN not configured")
        return None
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY environment variable found")
        return None
    
    scraper = None
    try:
        # Initialize scraper with proxy config
        # Note: When using proxies, increase timeout to allow for proxy warmup
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            extraction_context=source['context'],
            fetch_mode="browser",  # Use browser mode for JavaScript-rendered content
            headless=True,
            browser_timeout=120000 if use_proxy else 60000,  # 120s for proxies, 60s without
            enable_llm_pagination=False,  # Disable LLM pagination
            proxy_config=proxy_config  # Pass proxy config here (works for ALL fetchers)
        )
        
        # IMPORTANT: Disable ALL pagination detectors to ensure single-page only
        if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
            scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
        if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
            scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
        
        print("⚠️  Pagination detection DISABLED (single page only)")
        print("⏱️  Scraping single page...\n")
        
        # Scrape with wait selector
        result = await scraper.scrape(
            source['url'],
            fields=source['fields'],
            wait_for_selector=source.get('wait_for')
        )
        
        # Print results
        print(f"\n{'='*80}")
        print(f"✅ RESULTS for {source['name']}")
        print(f"{'='*80}\n")
        
        if result:
            # Extract actual data (scraper returns dict with 'data' key)
            if isinstance(result, dict) and 'data' in result:
                result_list = result['data']
            elif isinstance(result, list):
                result_list = result
            else:
                result_list = []
            
            if len(result_list) > 0:
                print(f"📊 Items extracted: {len(result_list)}")
                print(f"\n📋 First 3 items:")
                for i, item in enumerate(result_list[:3], 1):
                    print(f"\n   Item {i}:")
                    for key, value in item.items():
                        if not key.startswith('_'):  # Skip metadata
                            value_str = str(value) if value else 'N/A'
                            value_str = value_str[:100]
                            if value and len(str(value)) > 100:
                                value_str += "..."
                            print(f"     • {key}: {value_str}")
                
                # Check field completeness
                total_fields = len(source['fields'])
                complete_items = sum(1 for item in result_list if all(item.get(f) for f in source['fields']))
                completeness = (complete_items / len(result_list)) * 100 if result_list else 0
                
                print(f"\n📈 Quality Metrics:")
                print(f"   • Items with all fields: {complete_items}/{len(result_list)} ({completeness:.1f}%)")
                print(f"   • Extraction method: {result_list[0].get('_extraction_method', 'Unknown')}" if result_list else "")
                
                return result_list
            else:
                print(f"⚠️  No items extracted (empty result)")
                return []
        else:
            print(f"⚠️  No items extracted")
            return []
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        if scraper:
            scraper.close()  # NOT async


def save_to_csv(data, source_name, use_proxy=False):
    """Save results to CSV file"""
    if not data or len(data) == 0:
        print(f"⚠️  No data to save for {source_name}")
        return
    
    # Create output directory
    output_dir = Path("output_with_proxies" if use_proxy else "output_no_proxies")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = source_name.lower().replace(' ', '_')
    filename = output_dir / f"{safe_name}_{timestamp}.csv"
    
    # Get all unique keys from all items
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    # Remove metadata fields
    fieldnames = sorted([k for k in all_keys if not k.startswith('_')])
    
    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            # Filter out metadata fields
            row = {k: v for k, v in item.items() if k in fieldnames}
            writer.writerow(row)
    
    print(f"\n💾 Saved to: {filename}")
    print(f"   📊 {len(data)} items written")


async def main():
    """Main test runner"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                 🧪 UNIVERSAL SCRAPER - PROXY COMPARISON TEST 🧪                ║
║                                                                               ║
║  Testing all sources WITH and WITHOUT Apify residential proxies              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check for required environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        print("   export OPENAI_API_KEY='sk-proj-...'")
        return
    
    if not os.environ.get('APIFY_TOKEN'):
        print("\n⚠️  WARNING: APIFY_TOKEN not set")
        print("   Proxy tests will be skipped")
        print("   To enable proxies: export APIFY_TOKEN='apify_api_...'")
        test_with_proxies = False
    else:
        print(f"\n✅ APIFY_TOKEN found: {os.environ.get('APIFY_TOKEN')[:20]}...")
        test_with_proxies = True
    
    print(f"\n📝 Testing {len(TEST_SOURCES)} sources")
    print(f"🔄 Each source will be tested {'with AND without' if test_with_proxies else 'without'} proxies\n")
    
    results_summary = []
    
    for i, source in enumerate(TEST_SOURCES, 1):
        print(f"\n{'#'*80}")
        print(f"# Source {i}/{len(TEST_SOURCES)}: {source['name']}")
        print(f"{'#'*80}")
        
        # Test WITHOUT proxies first
        print(f"\n🔵 TEST 1: WITHOUT PROXIES (Direct connection)")
        result_no_proxy = await test_source(source, use_proxy=False)
        if result_no_proxy:
            save_to_csv(result_no_proxy, source['name'], use_proxy=False)
        
        items_no_proxy = len(result_no_proxy) if result_no_proxy else 0
        
        # Test WITH proxies if available
        items_with_proxy = 0
        if test_with_proxies:
            print(f"\n🟢 TEST 2: WITH APIFY RESIDENTIAL PROXIES")
            result_with_proxy = await test_source(source, use_proxy=True)
            if result_with_proxy:
                save_to_csv(result_with_proxy, source['name'], use_proxy=True)
            
            items_with_proxy = len(result_with_proxy) if result_with_proxy else 0
        
        # Summary for this source
        results_summary.append({
            'source': source['name'],
            'without_proxy': items_no_proxy,
            'with_proxy': items_with_proxy if test_with_proxies else 'N/A'
        })
        
        # Brief pause between sources
        if i < len(TEST_SOURCES):
            print(f"\n⏳ Pausing 5s before next source...\n")
            await asyncio.sleep(5)
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"📊 FINAL SUMMARY - ALL SOURCES")
    print(f"{'='*80}\n")
    
    print(f"{'Source':<25} {'Without Proxy':<15} {'With Proxy':<15} {'Improvement':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    for result in results_summary:
        source = result['source']
        no_proxy = result['without_proxy']
        with_proxy = result['with_proxy']
        
        if with_proxy != 'N/A' and no_proxy > 0:
            improvement = f"+{with_proxy - no_proxy}" if with_proxy > no_proxy else str(with_proxy - no_proxy)
            if with_proxy > no_proxy:
                improvement += " ✅"
            elif with_proxy < no_proxy:
                improvement += " ⚠️"
        else:
            improvement = "-"
        
        print(f"{source:<25} {str(no_proxy) + ' items':<15} {str(with_proxy) + ' items' if with_proxy != 'N/A' else 'N/A':<15} {improvement:<15}")
    
    print(f"\n{'='*80}\n")
    print("✅ Test complete! Check the output directories:")
    print(f"   📁 output_no_proxies/     - Results without proxies")
    if test_with_proxies:
        print(f"   📁 output_with_proxies/   - Results with Apify residential proxies")
    print()


if __name__ == "__main__":
    asyncio.run(main())

