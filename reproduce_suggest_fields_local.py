import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())

from universal_scraper.core.hybrid_fetcher import HybridFetcher

# Web Unblocker Credentials (from previous context)
WEB_UNBLOCKER_API_KEY = "brd-customer-hl_803e8195-zone-web_unlocker1:9u6410007858"
WEB_UNBLOCKER_ZONE = "web_unlocker1"

async def reproduce_suggest_fields():
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    logger.info(f"🧪 Testing suggest-fields logic for: {url}")
    
    # Mimic the logic in api/main.py suggest_fields_endpoint
    force_mode = None
    if "homedepot.com" in url:
        force_mode = "browser"
        logger.info("🎯 Home Depot detected: Forcing browser mode")

    # Initialize HybridFetcher exactly as in the API
    fetcher = HybridFetcher(
        proxy_config=None,  # We'll use Web Unblocker directly via api_key
        headless=True,
        browser_timeout=90000,
        force_mode=force_mode,
        use_camoufox=True,
        web_unblocker_api_key=WEB_UNBLOCKER_API_KEY,
        web_unblocker_zone=WEB_UNBLOCKER_ZONE
    )
    
    try:
        logger.info("🚀 Starting fetch...")
        result = await fetcher.fetch(url)
        
        html = result.get('html', '')
        status_code = result.get('status_code')
        fetch_method = result.get('fetch_method')
        
        logger.info(f"✅ Fetch complete!")
        logger.info(f"   Status Code: {status_code}")
        logger.info(f"   Fetch Method: {fetch_method}")
        logger.info(f"   HTML Length: {len(html)} bytes")
        
        if status_code == 200 and len(html) > 1000:
            logger.info("🎉 SUCCESS: Backend logic is working correctly!")
        else:
            logger.error("❌ FAILURE: Backend logic failed to retrieve content.")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce_suggest_fields())
