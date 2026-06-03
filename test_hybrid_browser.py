import asyncio
import logging
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hybrid_browser_unblocker():
    # Web Unblocker as Proxy
    unblocker_creds = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    logger.info("--- Testing HybridFetcher with force_mode='browser' and Web Unblocker ---")
    fetcher = HybridFetcher(
        web_unblocker_api_key=unblocker_creds,
        use_camoufox=True,
        force_mode='browser',
        headless=True
    )
    
    try:
        result = await fetcher.fetch(url)
        html = result.get('html', '')
        status = result.get('status_code')
        method = result.get('fetch_method')
        
        logger.info(f"HybridFetcher Result: status={status}, length={len(html)}, method={method}")
        
        if status == 200 and len(html) > 5000:
            logger.info("✅ SUCCESS: HybridFetcher got a 200 status in browser mode!")
            soup = BeautifulSoup(html, 'html.parser')
            logger.info(f"Title: {soup.title.text.strip() if soup.title else 'No Title'}")
        else:
            logger.warning(f"⚠️ HybridFetcher returned status {status}")
            if len(html) > 0:
                logger.info(f"Snippet: {html[:500]}")
    except Exception as e:
        logger.error(f"❌ HybridFetcher ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_browser_unblocker())
