import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_smart_unblocker():
    # Initialize HybridFetcher with Camoufox enabled
    fetcher = HybridFetcher(
        use_camoufox=True,
        headless=True,
        web_unblocker_api_key=os.getenv("WEB_UNBLOCKER_API_KEY"),
        web_unblocker_zone=os.getenv("WEB_UNBLOCKER_ZONE", "web_unlocker1")
    )
    
    url = "https://www.homedepot.com/p/Milwaukee-M18-FUEL-18V-Lithium-Ion-Brushless-Cordless-HACKZALL-Reciprocating-Saw-Tool-Only-2719-20/302190765"
    
    logger.info(f"Testing Smart Unblocker for: {url}")
    
    # This should trigger browser mode for Home Depot
    result = await fetcher.fetch(url)
    
    logger.info(f"Fetch method: {result.get('fetch_method')}")
    logger.info(f"HTML length: {len(result.get('html', ''))}")
    
    unblocker_log = result.get('unblocker_log', [])
    logger.info("Unblocker Log:")
    for entry in unblocker_log:
        logger.info(f"  [{entry['timestamp']}] {entry['message']}")
    
    if len(result.get('html', '')) > 1000:
        logger.info("✅ Success: HTML content retrieved")
    else:
        logger.error("❌ Failure: HTML content empty or too small")
        
    if unblocker_log:
        logger.info("✅ Success: Unblocker log populated")
    else:
        logger.error("❌ Failure: Unblocker log empty")

if __name__ == "__main__":
    asyncio.run(test_smart_unblocker())
