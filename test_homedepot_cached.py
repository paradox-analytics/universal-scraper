import asyncio
import logging
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_homedepot():
    """Test Home Depot with cached strategy"""
    
    # Credentials
    openai_key = "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"
    unblocker_pass = "t8mhp1qev1i1"
    
    # Force set env var to ensure all libs see it
    import os
    os.environ['OPENAI_API_KEY'] = openai_key
    
    # Initialize Scraper
    logger.info("🚀 Initializing Universal Scraper...")
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True,
        headless=True,
        browser_timeout=120000,
        web_unblocker_api_key=unblocker_pass,
        web_unblocker_zone="web_unlocker1"
    )
    
    url = "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745"
    
    logger.info(f"\\n{'='*60}")
    logger.info(f"🧪 Testing: Home Depot")
    logger.info(f"🔗 URL: {url}")
    logger.info(f"{'='*60}")
    
    try:
        # Check for cached strategy
        if scraper.strategy_detector:
            cached = scraper.strategy_detector.get_strategy(url)
            if cached:
                logger.info(f"📚 Cached Strategy Found:")
                logger.info(f"   Extraction Method: {cached['extraction_method']}")
                logger.info(f"   Proxy Type: {cached['proxy_type']}")
                logger.info(f"   Confidence: {cached['confidence']}")
                logger.info(f"   Details: {cached.get('extraction_details')}")
        
        # Execute Scrape
        result = await scraper.scrape(
            url=url,
            fields=[]
        )
        
        # Analyze Result
        status = result.get('status_code', 0)
        success = status == 200
        data_count = len(result.get('data', []))
        
        logger.info(f"\\n📊 Result:")
        logger.info(f"   Status: {status}")
        logger.info(f"   Success: {success}")
        logger.info(f"   Items Extracted: {data_count}")
        logger.info(f"   Data: {result.get('data', [])[:2]}")  # First 2 items
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    
    await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_homedepot())
