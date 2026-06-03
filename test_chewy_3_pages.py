#!/usr/bin/env python3
"""
Quick Test: Chewy.com - First 3 Pages Only
Extracts products from first 3 pages with timeout protection.
"""
import asyncio
import json
import sys
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def scrape_limited_pages(scraper, base_url, fields, max_pages=3):
    """Scrape limited number of pages manually"""
    all_data = []
    
    # Scrape first page
    print(f"\n📄 Scraping page 1/{max_pages}...")
    result1 = await scraper.scrape(base_url, fields)
    data1 = result1.get('data', [])
    all_data.extend(data1)
    print(f"   ✅ Page 1: {len(data1)} items (total: {len(all_data)})")
    
    # Scrape additional pages if needed
    for page_num in range(2, max_pages + 1):
        page_url = f"{base_url}?pageNumber={page_num}"
        print(f"\n📄 Scraping page {page_num}/{max_pages}...")
        try:
            result = await scraper.scrape(page_url, fields)
            data = result.get('data', [])
            all_data.extend(data)
            print(f"   ✅ Page {page_num}: {len(data)} items (total: {len(all_data)})")
        except Exception as e:
            print(f"   ⚠️  Page {page_num} failed: {e}")
            break
    
    # Return combined result
    return {
        'data': all_data,
        'total_items': len(all_data),
        'pages_scraped': min(max_pages, len([d for d in [result1] if d.get('data')])),
        'success': len(all_data) > 0
    }


async def main():
    print("=" * 80)
    print("🧪 QUICK TEST: Chewy.com - First 3 Pages Only")
    print("=" * 80)
    
    # Web Unblocker Proxy Configuration
    web_unblocker_proxy = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    print(f"\n🌐 Web Unblocker Proxy:")
    print(f"   Server: {web_unblocker_proxy['server']}")
    print(f"   Username: {web_unblocker_proxy['username']}")
    
    # Get OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set - extraction will be limited")
        api_key = 'sk-dummy-key'
    
    # Initialize scraper with auto-pagination DISABLED
    print(f"\n🚀 Initializing scraper (auto-pagination DISABLED)...")
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=web_unblocker_proxy,
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=120000,
        use_direct_llm=True,
        enable_cache=False,
        enable_auto_pagination=False,  # DISABLE auto-pagination
        log_level=logging.INFO
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["name", "price", "rating", "reviewCount", "image"]
    max_pages = 3
    
    print(f"\n📋 Scraping Configuration:")
    print(f"   Base URL: {url}")
    print(f"   Fields: {', '.join(fields)}")
    print(f"   Max Pages: {max_pages}")
    print(f"   Estimated Time: ~{max_pages * 45} seconds")
    print(f"\n⏳ Starting scrape...")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Scrape limited pages manually
        result = await scrape_limited_pages(scraper, url, fields, max_pages=max_pages)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        print(f"\n✅ Scrape completed in {elapsed:.1f} seconds!")
        print(f"   Pages scraped: {result.get('pages_scraped', 0)}/{max_pages}")
        print(f"   Total items: {result.get('total_items', 0)}")
        print(f"   Success: {result.get('success', False)}")
        
        # Show extracted data
        data = result.get('data', [])
        if data:
            print(f"\n🎯 Extracted Products (showing first 15):")
            for i, item in enumerate(data[:15], 1):
                name = item.get('name', 'Unknown')[:60]
                price = item.get('price', 'N/A')
                rating = item.get('rating', 'N/A')
                reviews = item.get('reviewCount', 'N/A')
                print(f"\n   {i}. {name}")
                print(f"      Price: {price} | Rating: {rating} | Reviews: {reviews}")
            
            # Save results
            output_file = 'chewy_3_pages_results.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Full results saved to: {output_file}")
            
            if len(data) >= 10:
                print(f"\n✅ TEST PASSED: Successfully extracted {len(data)} products from {result.get('pages_scraped', 0)} pages!")
                print(f"   ⏱️  Time taken: {elapsed:.1f} seconds")
                print(f"   📊 Average: {len(data) / result.get('pages_scraped', 1):.1f} items per page")
                return True
            else:
                print(f"\n⚠️  TEST WARNING: Only extracted {len(data)} products")
                return False
        else:
            print(f"\n❌ TEST FAILED: No products extracted")
            return False
            
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"\n❌ Scrape failed after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)

