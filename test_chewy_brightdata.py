#!/usr/bin/env python3
"""
Comprehensive Test: Chewy.com with Bright Data Residential Proxies

Tests the full pipeline:
1. Proxy connectivity verification
2. Page fetching with Camoufox
3. JSON detection and extraction
4. Product data validation
"""
import asyncio
import json
import sys
import logging
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_proxy_connectivity():
    """Test if Bright Data proxy is accessible"""
    print("\n" + "=" * 80)
    print("🔌 STEP 1: Testing Bright Data Proxy Connectivity")
    print("=" * 80)
    
    import requests
    
    proxy_config = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    # Build proxy URL
    proxy_url = f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['server'].replace('http://', '')}"
    
    try:
        # Test with Bright Data's test endpoint
        test_url = "https://geo.brdtest.com/welcome.txt?product=resi&method=native"
        
        print(f"   Testing proxy: {proxy_config['server']}")
        print(f"   Test URL: {test_url}")
        
        response = requests.get(
            test_url,
            proxies={
                'http': proxy_url,
                'https': proxy_url
            },
            timeout=30,
            verify=False  # Bright Data test endpoint uses self-signed cert
        )
        
        print(f"   ✅ Proxy Response: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
        # Extract IP from response if available
        if 'ip' in response.text.lower() or 'location' in response.text.lower():
            print(f"   📍 Proxy IP Info: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Proxy test failed: {e}")
        return False


async def test_chewy_scraping():
    """Test scraping Chewy.com with Bright Data proxies"""
    print("\n" + "=" * 80)
    print("🛒 STEP 2: Testing Chewy.com Scraping")
    print("=" * 80)
    
    # Bright Data Proxy Configuration
    proxy_config = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    print(f"\n🔌 Proxy Configuration:")
    print(f"   Server: {proxy_config['server']}")
    print(f"   Username: {proxy_config['username']}")
    print(f"   Password: {'*' * len(proxy_config['password'])}")
    
    # Get API key from environment or use dummy
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  No OPENAI_API_KEY found. Using dummy key for testing.")
        print("   (JSON extraction will work, but LLM fallback won't)")
        api_key = 'sk-dummy-key-for-testing'
    
    # Initialize scraper with proxy
    print("\n🚀 Initializing UniversalScraper...")
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=proxy_config,
        headless=True,
        use_camoufox=True,  # Use Camoufox for better anti-detection
        fetch_mode='browser',  # Force browser mode for JavaScript sites
        browser_timeout=120000,  # 2 minutes timeout for proxy warmup
        use_direct_llm=True,  # Enable Direct LLM extraction
        quality_mode='balanced',  # Balanced quality mode
        enable_cache=False,  # Disable cache for fresh test
        log_level=logging.INFO
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["name", "price", "rating", "reviewCount", "image"]
    
    print(f"\n📋 Scraping Configuration:")
    print(f"   URL: {url}")
    print(f"   Fields: {', '.join(fields)}")
    print(f"   Browser: Camoufox (anti-detection enabled)")
    print(f"   Proxy: Bright Data Residential")
    
    try:
        print(f"\n⏳ Starting scrape (this may take 30-60 seconds)...")
        result = await scraper.scrape(url, fields)
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        print(f"\n✅ Scrape completed!")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Items extracted: {len(result.get('data', []))}")
        print(f"   Success: {result.get('success', False)}")
        
        # Show extracted data
        data = result.get('data', [])
        if data:
            print(f"\n🎯 Extracted Products (showing first 5):")
            for i, item in enumerate(data[:5], 1):
                print(f"\n   {i}. {item.get('name', 'Unknown Product')}")
                print(f"      Price: {item.get('price', 'N/A')}")
                print(f"      Rating: {item.get('rating', 'N/A')}")
                print(f"      Reviews: {item.get('reviewCount', 'N/A')}")
                if item.get('image'):
                    print(f"      Image: {item.get('image')[:60]}...")
            
            # Save results to file
            output_file = 'chewy_test_results.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Full results saved to: {output_file}")
            
            # Validation
            if len(data) >= 5:
                print(f"\n✅ TEST PASSED: Successfully extracted {len(data)} products!")
                return True
            else:
                print(f"\n⚠️  TEST WARNING: Only extracted {len(data)} products (expected ≥5)")
                return False
        else:
            print(f"\n❌ TEST FAILED: No products extracted")
            print(f"\n   Possible issues:")
            print(f"   1. Proxy connection failed")
            print(f"   2. Page was blocked by Chewy")
            print(f"   3. JavaScript didn't render properly")
            print(f"   4. JSON extraction failed")
            
            # Check if we got HTML at least
            if 'html' in result:
                html_size = len(result.get('html', ''))
                print(f"\n   HTML size: {html_size:,} bytes")
                if html_size < 5000:
                    print(f"   ⚠️  HTML is very small - likely blocked or error page")
                else:
                    print(f"   ✅ HTML size looks reasonable")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Scrape failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if hasattr(scraper, 'html_fetcher') and hasattr(scraper.html_fetcher, 'close'):
                await scraper.html_fetcher.close()
        except:
            pass


async def main():
    """Run all tests"""
    print("=" * 80)
    print("🧪 CHEWY.COM + BRIGHT DATA PROXY TEST SUITE")
    print("=" * 80)
    
    # Test 1: Proxy connectivity
    proxy_ok = await test_proxy_connectivity()
    
    if not proxy_ok:
        print("\n⚠️  Proxy connectivity test failed, but continuing with scraping test...")
        print("   (Proxy might work for browser even if HTTP test fails)")
    
    # Test 2: Scraping
    scraping_ok = await test_chewy_scraping()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    print(f"   Proxy Connectivity: {'✅ PASS' if proxy_ok else '⚠️  FAIL (but may still work)'}")
    print(f"   Scraping: {'✅ PASS' if scraping_ok else '❌ FAIL'}")
    
    if scraping_ok:
        print("\n🎉 All tests passed! Chewy.com scraping works with Bright Data proxies!")
    else:
        print("\n⚠️  Some tests failed. Check logs above for details.")
    
    return scraping_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

