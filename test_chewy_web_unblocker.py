#!/usr/bin/env python3
"""
Local test for Chewy.com with Web Unblocker
Tests the core UniversalScraper with Web Unblocker fallback
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Test Chewy with Web Unblocker"""
    
    # Web Unblocker credentials (proxy format - supports both : and , separators)
    web_unblocker_credential = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"
    
    # Pass the credential as-is - WebUnblockerFetcher will handle parsing
    web_unblocker_api_key = web_unblocker_credential
    logger.info(f"✅ Using Web Unblocker credential: {web_unblocker_api_key[:30]}...")
    
    # Get OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Test URL and fields
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["title", "rating", "review count", "price"]
    
    logger.info("="*80)
    logger.info("🧪 LOCAL TEST: Chewy.com with Web Unblocker")
    logger.info("="*80)
    logger.info(f"   URL: {url}")
    logger.info(f"   Fields: {', '.join(fields)}")
    logger.info(f"   Web Unblocker Zone: web_unlocker1")
    logger.info("="*80)
    logger.info("")
    
    # Initialize scraper with Web Unblocker
    scraper = UniversalScraper(
        api_key=openai_api_key,
        proxy_config=None,  # No proxy (Web Unblocker handles it)
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',  # Use browser mode to trigger Web Unblocker fallback
        browser_timeout=120000,
        use_direct_llm=True,
        enable_cache=False,  # Disable local cache for testing
        web_unblocker_api_key=web_unblocker_api_key,  # Web Unblocker API key
        web_unblocker_zone="web_unlocker1",
        log_level=logging.INFO
    )
    
    # Limit to first 1 page for testing (prevent hanging)
    scraper._max_pages_limit = 1
    logger.info(f"📄 Limiting pagination to {scraper._max_pages_limit} page for testing")
    
    # Reduce timeout for Web Unblocker to prevent hanging
    if hasattr(scraper, 'html_fetcher') and scraper.html_fetcher and hasattr(scraper.html_fetcher, 'web_unblocker_fetcher') and scraper.html_fetcher.web_unblocker_fetcher:
        scraper.html_fetcher.web_unblocker_fetcher.timeout = 60  # 60s timeout instead of 120s
    
    try:
        # Scrape the URL
        logger.info(f"🚀 Starting scrape...")
        result = await scraper.scrape(url, fields)
        
        # Display results
        logger.info("")
        logger.info("="*80)
        logger.info("📊 RESULTS")
        logger.info("="*80)
        
        items = result.get('data', [])
        logger.info(f"   Total items extracted: {len(items)}")
        
        if items:
            # Check field coverage
            logger.info("")
            logger.info("   Field Coverage Analysis:")
            all_fields_found = set()
            for item in items:
                all_fields_found.update(item.keys())
            
            requested_fields = set(fields)
            missing_fields = requested_fields - all_fields_found
            found_fields = requested_fields & all_fields_found
            
            logger.info(f"   ✅ Found fields: {', '.join(sorted(found_fields))}")
            if missing_fields:
                logger.warning(f"   ❌ Missing fields: {', '.join(sorted(missing_fields))}")
            else:
                logger.info(f"   ✅ All requested fields found!")
            
            logger.info("")
            logger.info("   First 5 items:")
            for i, item in enumerate(items[:5], 1):
                logger.info(f"   {i}. {json.dumps(item, indent=6)}")
            
            # Check for product URLs
            logger.info("")
            logger.info("   URL Analysis:")
            unique_urls = set()
            for item in items:
                url = item.get('_url') or item.get('url') or item.get('href')
                if url:
                    unique_urls.add(url)
            logger.info(f"   Unique URLs found: {len(unique_urls)}")
            if len(unique_urls) == 1:
                logger.warning(f"   ⚠️ All items have the same URL: {list(unique_urls)[0]}")
            else:
                logger.info(f"   ✅ Multiple unique URLs found (good!)")
                logger.info(f"   Sample URLs: {list(unique_urls)[:3]}")
        
        # Save results to file
        output_file = "chewy_web_unblocker_local_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'url': url,
                'fields': fields,
                'items': items,
                'total_items': len(items),
                'result': result
            }, f, indent=2)
        
        logger.info("")
        logger.info(f"💾 Results saved to: {output_file}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Scrape failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
