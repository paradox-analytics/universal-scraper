import asyncio
import logging
from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_credentials():
    # Credentials provided by the user
    creds = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    logger.info(f"Testing Web Unblocker with credentials: {creds[:20]}...")
    
    fetcher = WebUnblockerFetcher(api_key=creds)
    
    try:
        logger.info(f"Fetching Home Depot URL: {url}")
        result = await fetcher.fetch_async(url)
        
        html = result.get('html', '')
        status = result.get('status')
        
        logger.info(f"Fetch result: status={status}, length={len(html)}")
        
        if len(html) > 5000:
            logger.info("✅ SUCCESS: Received substantial content.")
            
            # Parse with BeautifulSoup to check for product info
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            if title:
                logger.info(f"Page Title: {title.text.strip()}")
            
            # Check for product name in the content
            if "Refrigerator" in html or "GNE27JYMFS" in html:
                logger.info("✅ Verified: Content contains product-specific keywords.")
            else:
                logger.warning("⚠️ Warning: Keywords not found in content.")
                # Print a snippet of the body
                body = soup.find('body')
                if body:
                    logger.info(f"Body snippet: {body.text[:500].strip()}")
        else:
            logger.error(f"❌ FAILURE: Received insufficient content. Status: {status}")
            
    except Exception as e:
        logger.error(f"❌ ERROR: Test failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_credentials())
