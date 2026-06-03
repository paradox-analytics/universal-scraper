import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.field_discovery import FieldDiscovery

# Add current dir to path
sys.path.append(os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_full_field_discovery():
    url = "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745"
    
    # Check for credentials
    api_key = os.getenv("WEB_UNBLOCKER_API_KEY")
    customer_id = os.getenv("WEB_UNBLOCKER_CUSTOMER_ID")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ WEB_UNBLOCKER_API_KEY not found in environment")
        return

    logger.info(f"🚀 Starting full test for: {url}")
    logger.info(f"🔐 Using Web Unblocker Customer ID: {customer_id or 'REDACTED_CUSTOMER_ID (default)'}")
    
    # 1. Initialize Fetcher (browser mode for Home Depot)
    fetcher = HybridFetcher(
        web_unblocker_api_key=api_key,
        web_unblocker_customer_id=customer_id,
        use_camoufox=True,  # Match production
        headless=True
    )
    
    try:
        # 2. Fetch HTML (this uses the 90s timeout logic I just fixed)
        logger.info("📡 Fetching content...")
        fetch_result = await fetcher.fetch(url)
        
        html = fetch_result.get('html', '')
        status = fetch_result.get('status', 'unknown')
        method = fetch_result.get('fetch_method', 'unknown')
        
        logger.info(f"✅ Fetch complete. Method: {method}, Status: {status}, HTML Size: {len(html)} bytes")
        
        if not html or len(html) < 200:
            logger.error("❌ Failed to fetch usable HTML")
            # Log internal logs to see why
            for entry in fetch_result.get('unblocker_log', []):
                logger.info(f"🛡️ [Unblocker Log] {entry.get('message')}")
            return

        # 3. Field Discovery
        logger.info("🧠 Running field discovery...")
        discovery = FieldDiscovery(api_key=openai_key)
        
        # Test with LLM if key available, else fallback
        use_llm = True if openai_key else False
        logger.info(f"   Using LLM: {use_llm}")
        
        discovery_result = await discovery.discover_fields(html, url, use_llm=use_llm)
        
        logger.info(f"✨ Discovery Result: {discovery_result}")
        
    except Exception as e:
        logger.error(f"💥 Test failed: {e}", exc_info=True)
    finally:
        if fetcher.browser_fetcher:
            await fetcher.browser_fetcher.close()

if __name__ == "__main__":
    asyncio.run(test_full_field_discovery())
