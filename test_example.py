import asyncio
import logging
from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_example():
    creds = "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS"
    url = "https://example.com"
    
    logger.info(f"Testing Web Unblocker with example.com...")
    fetcher = WebUnblockerFetcher(api_key=creds)
    
    try:
        result = await fetcher.fetch_async(url)
        logger.info(f"Result: status={result.get('status')}, length={len(result.get('html', ''))}")
        if result.get('status') == 200:
            logger.info("✅ SUCCESS: Web Unblocker works for example.com")
        else:
            logger.error(f"❌ FAILURE: Status {result.get('status')}")
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_example())
