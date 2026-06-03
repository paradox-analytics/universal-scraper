#!/usr/bin/env python3
"""
Test Chewy.com with Web Unblocker Proxy Endpoint

Uses Web Unblocker proxy endpoint (not API) - configured proxy credentials.
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


async def main():
    print("=" * 80)
    print("🧪 TEST: Chewy.com with Web Unblocker Proxy")
    print("=" * 80)
    
    # Web Unblocker Proxy Configuration
    web_unblocker_proxy = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    print(f"\n🌐 Web Unblocker Proxy Configuration:")
    print(f"   Server: {web_unblocker_proxy['server']}")
    print(f"   Username: {web_unblocker_proxy['username']}")
    print(f"   Zone: web_unlocker1")
    
    # Get OpenAI API key (optional for testing)
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set - LLM features will be limited")
        api_key = 'sk-dummy-key'
    
    # Initialize scraper with Web Unblocker proxy
    print(f"\n🚀 Initializing UniversalScraper with Web Unblocker proxy...")
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=web_unblocker_proxy,  # Use Web Unblocker proxy directly
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        use_direct_llm=True,
        enable_cache=False,
        log_level=logging.INFO
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["name", "price", "rating", "reviewCount", "image"]
    
    print(f"\n📋 Scraping Configuration:")
    print(f"   URL: {url}")
    print(f"   Fields: {', '.join(fields)}")
    print(f"   Proxy: Web Unblocker (brd.superproxy.io:33335)")
    print(f"\n⏳ Starting scrape...")
    print(f"   Web Unblocker proxy should bypass Kasada automatically")
    
    try:
        result = await scraper.scrape(url, fields)
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        print(f"\n✅ Scrape completed!")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Fetch Method: {result.get('fetch_method', 'unknown')}")
        print(f"   Items extracted: {len(result.get('data', []))}")
        print(f"   Success: {result.get('success', False)}")
        
        # Check HTML size
        html_size = len(result.get('html', ''))
        print(f"   HTML size: {html_size:,} bytes")
        
        # Show extracted data
        data = result.get('data', [])
        if data:
            print(f"\n🎯 Extracted Products (showing first 5):")
            for i, item in enumerate(data[:5], 1):
                print(f"\n   {i}. {item.get('name', 'Unknown Product')}")
                print(f"      Price: {item.get('price', 'N/A')}")
                print(f"      Rating: {item.get('rating', 'N/A')}")
                print(f"      Reviews: {item.get('reviewCount', 'N/A')}")
            
            # Save results
            output_file = 'chewy_web_unblocker_proxy_results.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Full results saved to: {output_file}")
            
            if len(data) >= 5:
                print(f"\n✅ TEST PASSED: Successfully extracted {len(data)} products!")
                print(f"   🌐 Web Unblocker proxy bypassed Kasada successfully!")
                return True
            else:
                print(f"\n⚠️  TEST WARNING: Only extracted {len(data)} products")
                return False
        else:
            print(f"\n❌ TEST FAILED: No products extracted")
            
            # Check if HTML is small (still blocked)
            if html_size < 2000:
                print(f"\n   ⚠️  HTML is very small ({html_size} bytes) - might still be blocked")
                html = result.get('html', '')
                if html:
                    print(f"\n   HTML preview:")
                    print(f"   {html[:500]}")
            else:
                print(f"\n   ✅ HTML size looks good ({html_size:,} bytes)")
                print(f"   Check logs above for extraction issues")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Scrape failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

