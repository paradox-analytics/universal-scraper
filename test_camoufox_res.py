import asyncio
import logging
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_camoufox_res():
    # Residential Proxy
    res_proxy_config = {
        'server': 'brd.superproxy.io:22225',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    logger.info("--- Testing Camoufox with Residential Proxy ---")
    fetcher = CamoufoxFetcher(proxy_config=res_proxy_config, headless=True)
    
    try:
        result = await fetcher.fetch(url)
        html = result.get('html', '')
        status = result.get('status')
        
        logger.info(f"Camoufox Result: status={status}, length={len(html)}")
        
        if status == 200 and len(html) > 5000:
            logger.info("✅ SUCCESS: Camoufox got a 200 status!")
            soup = BeautifulSoup(html, 'html.parser')
            logger.info(f"Title: {soup.title.text.strip() if soup.title else 'No Title'}")
        else:
            logger.warning(f"⚠️ Camoufox returned status {status}")
            if len(html) > 5000:
                logger.info("But it still got substantial content.")
    except Exception as e:
        logger.error(f"❌ Camoufox ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_camoufox_res())
