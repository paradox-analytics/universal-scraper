import asyncio
import logging
from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def save_homedepot_html():
    creds = "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS"
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    fetcher = WebUnblockerFetcher(api_key=creds)
    try:
        # We'll try to catch the exception and check if we got ANY content
        result = await fetcher.fetch_async(url)
        html = result.get('html', '')
        if html:
            with open("homedepot_debug.html", "w") as f:
                f.write(html)
            logger.info(f"Saved {len(html)} bytes to homedepot_debug.html")
        else:
            logger.warning("No HTML content received.")
    except Exception as e:
        logger.error(f"Error during fetch: {e}")
        # Check if the exception message contains HTML (sometimes it does if it's a status error)
        # But better to just look at the logs.

if __name__ == "__main__":
    asyncio.run(save_homedepot_html())
