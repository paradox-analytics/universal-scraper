import asyncio
import logging
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_camoufox_unblocker():
    # Web Unblocker as Proxy
    unblocker_creds = "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS"
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    logger.info("--- Testing Camoufox with Web Unblocker as Proxy ---")
    fetcher = CamoufoxFetcher(web_unblocker_api_key=unblocker_creds, headless=True)
    
    try:
        result = await fetcher.fetch(url)
        html = result.get('html', '')
        status = result.get('status')
        
        logger.info(f"Camoufox Result: status={status}, length={len(html)}")
        
        if status == 200 and len(html) > 5000:
            logger.info("✅ SUCCESS: Camoufox + Web Unblocker got a 200 status!")
            soup = BeautifulSoup(html, 'html.parser')
            logger.info(f"Title: {soup.title.text.strip() if soup.title else 'No Title'}")
        else:
            logger.warning(f"⚠️ Camoufox returned status {status}")
            if len(html) > 0:
                logger.info(f"Snippet: {html[:500]}")
    except Exception as e:
        logger.error(f"❌ Camoufox ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_camoufox_unblocker())
