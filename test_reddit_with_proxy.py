"""
Quick test of Reddit with and without Apify proxies
Single page only - testing proxy effectiveness
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def test_reddit(use_proxy=False):
    """Test Reddit scraping"""
    url = 'https://www.reddit.com/r/webscraping/'
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing Reddit - {'WITH' if use_proxy else 'WITHOUT'} Apify Proxies")
    print(f"{'='*80}\n")
    
    # Get proxy config if needed
    proxy_config = None
    if use_proxy:
        apify_token = os.environ.get('APIFY_TOKEN')
        if not apify_token:
            print("❌ APIFY_TOKEN not set")
            return None
        
        proxy_config = {
            'server': 'http://proxy.apify.com:8000',
            'username': 'groups-RESIDENTIAL,session-default',
            'password': apify_token
        }
        print(f"🌐 Using Apify Residential Proxies")
    else:
        print(f"🌐 Direct connection (no proxy)")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    scraper = None
    try:
        # Initialize scraper
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            extraction_context="Extract Reddit posts with title, author, upvotes, comments count",
            fetch_mode="browser",
            headless=True,
            enable_llm_pagination=False,
            proxy_config=proxy_config
        )
        
        # DISABLE pagination completely
        if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
            scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
        if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
            scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
        
        print("⚠️  Pagination DISABLED (single page only)")
        print("⏱️  Scraping...\n")
        
        # Scrape
        result = await scraper.scrape(
            url,
            fields=['title', 'author', 'upvotes', 'comments_count'],
            wait_for_selector='shreddit-post'
        )
        
        # Print results
        print(f"\n{'='*80}")
        print(f"✅ RESULTS")
        print(f"{'='*80}\n")
        
        if result and len(result) > 0:
            print(f"📊 Items extracted: {len(result)}")
            print(f"\n📋 First 3 items:")
            # Convert to list if needed
            result_list = list(result) if not isinstance(result, list) else result
            for i, item in enumerate(result_list[:3], 1):
                print(f"\n   Item {i}:")
                for key, value in item.items():
                    if not key.startswith('_'):
                        value_str = str(value)[:80]
                        print(f"     • {key}: {value_str}")
            return result_list
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


async def main():
    """Main test runner"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                 🧪 REDDIT PROXY TEST - Single Page Only 🧪                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Test without proxy first
    print("\n🔵 TEST 1: WITHOUT PROXY")
    result_no_proxy = await test_reddit(use_proxy=False)
    items_no_proxy = len(result_no_proxy) if result_no_proxy else 0
    
    print("\n\n⏳ Pausing 5 seconds...\n")
    await asyncio.sleep(5)
    
    # Test with proxy
    print("\n🟢 TEST 2: WITH APIFY RESIDENTIAL PROXY")
    result_with_proxy = await test_reddit(use_proxy=True)
    items_with_proxy = len(result_with_proxy) if result_with_proxy else 0
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    print(f"  Without Proxy: {items_no_proxy} items")
    print(f"  With Proxy:    {items_with_proxy} items")
    
    if items_with_proxy > items_no_proxy:
        print(f"\n  ✅ Proxy helped! (+{items_with_proxy - items_no_proxy} items)")
    elif items_with_proxy < items_no_proxy:
        print(f"\n  ⚠️  Proxy resulted in fewer items (-{items_no_proxy - items_with_proxy} items)")
    else:
        print(f"\n  ➡️  Same number of items")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())

