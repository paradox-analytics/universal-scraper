#!/usr/bin/env python3
"""
Test universal scraper on multiple different websites
Tests: Metacritic, Reddit, Leafly
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


async def test_url(url: str, fields: list, site_name: str):
    """Test a single URL"""
    logger.info("="*80)
    logger.info(f"🧪 TESTING: {site_name}")
    logger.info("="*80)
    logger.info(f"   URL: {url}")
    logger.info(f"   Fields: {', '.join(fields)}")
    logger.info("="*80)
    
    # Get OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return None
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=openai_api_key,
        proxy_config=None,
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=60000,
        use_direct_llm=True,
        enable_cache=True,  # Enable cache to test pre-warming
        web_unblocker_api_key=None,  # Not using Web Unblocker for these tests
        log_level=logging.INFO
    )
    
    # Limit to 1 page for testing
    scraper._max_pages_limit = 1
    
    try:
        logger.info(f"🚀 Starting scrape...")
        result = await scraper.scrape(url, fields)
        
        items = result.get('data', [])
        logger.info("")
        logger.info("="*80)
        logger.info(f"📊 RESULTS: {site_name}")
        logger.info("="*80)
        logger.info(f"   Total items extracted: {len(items)}")
        
        if items:
            # Field coverage
            all_fields_found = set()
            for item in items:
                all_fields_found.update(item.keys())
            
            requested_fields = set(fields)
            missing_fields = requested_fields - all_fields_found
            found_fields = requested_fields & all_fields_found
            
            logger.info("")
            logger.info("   Field Coverage:")
            logger.info(f"   ✅ Found: {', '.join(sorted(found_fields))}")
            if missing_fields:
                logger.warning(f"   ❌ Missing: {', '.join(sorted(missing_fields))}")
            else:
                logger.info(f"   ✅ All fields present!")
            
            # Show first 3 items
            logger.info("")
            logger.info("   First 3 items:")
            for i, item in enumerate(items[:3], 1):
                logger.info(f"   {i}. {json.dumps(item, indent=4, default=str)}")
            
            # Extraction source
            extraction_source = result.get('metadata', {}).get('extraction_source', 'unknown')
            logger.info("")
            logger.info(f"   Extraction Source: {extraction_source}")
            
            # Execution time
            exec_time = result.get('metadata', {}).get('execution_time', 0)
            logger.info(f"   Execution Time: {exec_time:.2f}s")
        else:
            logger.error("   ❌ No items extracted!")
        
        logger.info("="*80)
        logger.info("")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Scrape failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        await scraper.close()


async def main():
    """Test multiple sites"""
    
    test_cases = [
        {
            "name": "Metacritic",
            "url": "https://www.metacritic.com/pictures/november-2025-movie-preview-wicked-hamnet-running-man/",
            "fields": ["title", "metascore", "release_date", "director", "description"]
        },
        {
            "name": "Reddit",
            "url": "https://www.reddit.com/r/github/",
            "fields": ["title", "author", "score", "comments", "url", "subreddit"]
        },
        {
            "name": "Leafly",
            "url": "https://www.leafly.com/dispensary-info/mammoth-holistics/menu",
            "fields": ["product_name", "price", "category", "description", "thc_percentage"]
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        result = await test_url(
            test_case["url"],
            test_case["fields"],
            test_case["name"]
        )
        results[test_case["name"]] = result
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    
    for name, result in results.items():
        if result:
            items_count = len(result.get('data', []))
            exec_time = result.get('metadata', {}).get('execution_time', 0)
            source = result.get('metadata', {}).get('extraction_source', 'unknown')
            logger.info(f"   {name}: {items_count} items, {exec_time:.1f}s, source={source}")
        else:
            logger.warning(f"   {name}: FAILED")
    
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())







