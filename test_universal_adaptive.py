import asyncio
import logging
import os
import sys
import json
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.scraping_strategy_detector import ScrapingStrategyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('universal_adaptive_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Test URLs covering different challenges
TEST_SITES = [
    {
        'name': 'Home Depot (E-commerce, Anti-bot)',
        'url': 'https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745',
        'expected_fields': ['price', 'name', 'sku']
    },
    {
        'name': 'Amazon (E-commerce, Anti-bot)',
        'url': 'https://www.amazon.com/dp/B08N5KWB9H', # PS5 or similar popular item
        'expected_fields': ['price', 'title', 'rating']
    },
    {
        'name': 'Reddit (SPA, Dynamic)',
        'url': 'https://www.reddit.com/r/programming/comments/18w456/example_post/', # Generic placeholder, will need real URL or main page
        # Better to use a stable subreddit page
        'url': 'https://www.reddit.com/r/technology/',
        'expected_fields': ['title', 'author', 'upvotes']
    }
]

async def test_site(scraper: UniversalScraper, site: Dict):
    """Test a single site with adaptive scraping"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing: {site['name']}")
    logger.info(f"🔗 URL: {site['url']}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. First Run - Should adapt and learn
        logger.info(f"🏃 Run 1: Initial Adaptive Scrape...")
        start_time = asyncio.get_event_loop().time()
        
        result = await scraper.scrape(
            url=site['url'],
            fields=site.get('expected_fields')
        )
        
        duration = asyncio.get_event_loop().time() - start_time
        
        # Analyze Result
        success = result.get('status_code') == 200 and len(result.get('data', [])) > 0
        logger.info(f"📊 Run 1 Result: {'✅ Success' if success else '❌ Failed'}")
        logger.info(f"   Time: {duration:.2f}s")
        logger.info(f"   Items: {len(result.get('data', []))}")
        logger.info(f"   Strategy Used: {result.get('strategy', 'unknown')}") # Scraper needs to return this
        
        if not success:
            logger.error(f"   Error: {result.get('error')}")
            return
            
        # 2. Verify Caching
        domain = site['url'].split('/')[2]
        detector = scraper.strategy_detector
        cached_strategy = detector.get_strategy(site['url'])
        
        if cached_strategy:
            logger.info(f"💾 Strategy Cached for {domain}:")
            logger.info(f"   Method: {cached_strategy.get('recommended_strategy', {}).get('extraction_method')}")
            logger.info(f"   Proxy: {cached_strategy.get('recommended_strategy', {}).get('proxy_type')}")
        else:
            logger.warning(f"⚠️ No strategy cached for {domain}")
            
        # 3. Second Run - Should use cache and be faster/more reliable
        logger.info(f"\n🏃 Run 2: Cached Strategy Scrape...")
        start_time = asyncio.get_event_loop().time()
        
        result_2 = await scraper.scrape(
            url=site['url'],
            fields=site.get('expected_fields')
        )
        
        duration_2 = asyncio.get_event_loop().time() - start_time
        
        success_2 = result_2.get('status_code') == 200 and len(result_2.get('data', [])) > 0
        logger.info(f"📊 Run 2 Result: {'✅ Success' if success_2 else '❌ Failed'}")
        logger.info(f"   Time: {duration_2:.2f}s")
        
        # Compare
        logger.info(f"\n📈 Improvement:")
        logger.info(f"   Time Delta: {duration - duration_2:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Test failed for {site['name']}: {e}", exc_info=True)

async def main():
    # Initialize Scraper
    # Ensure we have API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found in environment")
        return

    scraper = UniversalScraper(
        headless=True,
        browser_timeout=60000,
        # Use credentials from env (loaded by scraper)
    )
    
    try:
        # Run tests sequentially
        for site in TEST_SITES:
            await test_site(scraper, site)
            await asyncio.sleep(5) # Polite delay between sites
            
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
