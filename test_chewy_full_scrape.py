#!/usr/bin/env python3
"""
Full Scraping Test: Chewy.com with Web Unblocker
Extracts products with timeout protection to prevent hanging.
"""
import asyncio
import json
import sys
import logging
import os
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

# Configure logging with timeout warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global timeout flag
timeout_occurred = False

def timeout_handler(signum, frame):
    """Handle timeout"""
    global timeout_occurred
    timeout_occurred = True
    logger.error("⏱️ TIMEOUT: Scraping took too long, aborting...")
    raise TimeoutError("Scraping timeout exceeded")


async def scrape_with_timeout(scraper, url, fields, timeout_seconds=180):
    """Run scrape with timeout protection"""
    global timeout_occurred
    
    # Set up signal handler for timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        # Use asyncio timeout as backup
        result = await asyncio.wait_for(
            scraper.scrape(url, fields),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Asyncio timeout after {timeout_seconds}s")
        raise
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm


async def main():
    print("=" * 80)
    print("🧪 FULL SCRAPING TEST: Chewy.com with Web Unblocker")
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
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        api_key = 'sk-dummy-key'
    
    # Initialize scraper
    print(f"\n🚀 Initializing scraper...")
    scraper = UniversalScraper(
        api_key=api_key,
        proxy_config=web_unblocker_proxy,
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=120000,  # 2 minutes for browser
        use_direct_llm=True,
        enable_cache=False,
        log_level=logging.INFO
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["name", "price", "rating", "reviewCount", "image"]
    
    print(f"\n📋 Scraping Configuration:")
    print(f"   URL: {url}")
    print(f"   Fields: {', '.join(fields)}")
    print(f"   Timeout: 180 seconds (3 minutes)")
    print(f"\n⏳ Starting scrape (with timeout protection)...")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await scrape_with_timeout(scraper, url, fields, timeout_seconds=180)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        print(f"\n✅ Scrape completed in {elapsed:.1f} seconds!")
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
            print(f"\n🎯 Extracted Products (showing first 10):")
            for i, item in enumerate(data[:10], 1):
                name = item.get('name', 'Unknown')[:60]
                price = item.get('price', 'N/A')
                rating = item.get('rating', 'N/A')
                reviews = item.get('reviewCount', 'N/A')
                print(f"\n   {i}. {name}")
                print(f"      Price: {price} | Rating: {rating} | Reviews: {reviews}")
            
            # Save results
            output_file = 'chewy_full_scrape_results.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Full results saved to: {output_file}")
            
            # Save sample HTML
            if result.get('html'):
                html_file = 'chewy_full_scrape_html.html'
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(result['html'][:100000])  # First 100KB
                print(f"💾 HTML sample saved to: {html_file}")
            
            if len(data) >= 5:
                print(f"\n✅ TEST PASSED: Successfully extracted {len(data)} products!")
                print(f"   ⏱️  Time taken: {elapsed:.1f} seconds")
                return True
            else:
                print(f"\n⚠️  TEST WARNING: Only extracted {len(data)} products (expected ≥5)")
                return False
        else:
            print(f"\n❌ TEST FAILED: No products extracted")
            
            # Check if HTML is good
            if html_size > 100000:
                print(f"   ✅ HTML size is good ({html_size:,} bytes)")
                print(f"   ⚠️  Extraction failed - check logs above")
                if not api_key or api_key == 'sk-dummy-key':
                    print(f"   💡 Set OPENAI_API_KEY for LLM-based extraction")
            else:
                print(f"   ⚠️  HTML size is small ({html_size} bytes) - might be blocked")
            
            return False
            
    except asyncio.TimeoutError:
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"\n❌ TEST FAILED: Timeout after {elapsed:.1f} seconds")
        print(f"   Scraping took too long - check network/proxy connection")
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

