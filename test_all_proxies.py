import asyncio
import logging
from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher
from universal_scraper.core.html_fetcher import HTMLFetcher
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_all_credentials():
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    # 1. Test Residential Proxy
    res_proxy_config = {
        'server': 'brd.superproxy.io:22225',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    logger.info("--- Testing Residential Proxy (HTMLFetcher) ---")
    html_fetcher = HTMLFetcher(proxy_config=res_proxy_config)
    try:
        result = html_fetcher.fetch(url)
        html = result.get('html', '')
        status = result.get('status_code')
        logger.info(f"Residential Proxy Result: status={status}, length={len(html)}")
        if status == 200 and len(html) > 5000:
            logger.info("✅ Residential Proxy SUCCESS")
            soup = BeautifulSoup(html, 'html.parser')
            logger.info(f"Title: {soup.title.text.strip() if soup.title else 'No Title'}")
        else:
            logger.warning(f"⚠️ Residential Proxy failed or returned status {status}")
    except Exception as e:
        logger.error(f"❌ Residential Proxy ERROR: {e}")

    # 2. Test Web Unblocker
    unblocker_creds = "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS"
    logger.info("--- Testing Web Unblocker (WebUnblockerFetcher) ---")
    unblocker_fetcher = WebUnblockerFetcher(api_key=unblocker_creds)
    try:
        result = await unblocker_fetcher.fetch_async(url)
        html = result.get('html', '')
        status = result.get('status')
        logger.info(f"Web Unblocker Result: status={status}, length={len(html)}")
        if len(html) > 5000:
            logger.info("✅ Web Unblocker SUCCESS (received content)")
            soup = BeautifulSoup(html, 'html.parser')
            logger.info(f"Title: {soup.title.text.strip() if soup.title else 'No Title'}")
            if "Refrigerator" in html:
                logger.info("✅ Verified: Content contains product info.")
        else:
            logger.warning(f"⚠️ Web Unblocker failed or returned status {status}")
    except Exception as e:
        logger.error(f"❌ Web Unblocker ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_credentials())
