import asyncio
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.scraping_strategy_detector import ScrapingStrategyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_multi_site_strategy():
    """
    Test smart strategy detection across multiple challenging sites.
    """
    # Credentials provided by user
    openai_key = "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"
    
    # Residential Proxy (corrected password from .env)
    res_host = "brd.superproxy.io"
    res_port = "33335"
    res_user = "brd-customer-hl_803e8195-zone-residential_proxy2"
    res_pass = "rs2mvj79xi2t"  # CORRECTED: was using unblocker password by mistake
    res_proxy_url = f"http://{res_user}:{res_pass}@{res_host}:{res_port}"
    
    # Web Unblocker
    unblocker_user = "brd-customer-hl_803e8195-zone-web_unlocker1"
    unblocker_pass = "t8mhp1qev1i1"
    # Note: Web Unblocker usually uses the same host/port but different auth
    
    # Test URLs
    test_urls = [
        {
            "name": "Product Hunt (Next.js/React)",
            "url": "https://www.producthunt.com/categories/vibe-coding",
            "expected_method": "json" # Expecting inline JSON or API
        },
        {
            "name": "Home Depot (Cached Strategy)",
            "url": "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745",
            "expected_method": "json_ld" # Should use cached strategy
        },
        {
            "name": "Metacritic (New Domain)",
            "url": "https://www.metacritic.com/pictures/best-movies-of-2025/",
            "expected_method": "html" # Likely HTML or JSON-LD
        },
        {
            "name": "Amazon (High Difficulty)",
            "url": "https://www.amazon.com/Energizer-Batteries-Double-Long-Lasting-Alkaline/dp/B09RTVD1GF",
            "expected_method": "html" # Likely HTML or JSON-LD
        }
    ]

    # Initialize Scraper
    logger.info("🚀 Initializing Universal Scraper with Strategy Detection...")
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True,
        headless=True,
        browser_timeout=120000, # 120s timeout (matches cached Home Depot strategy)
        web_unblocker_api_key=unblocker_pass, # ENABLED for Home Depot cached strategy
        web_unblocker_zone="web_unlocker1",
        proxy_config={
            "server": f"http://{res_host}:{res_port}",
            "username": res_user,
            "password": res_pass
        }
    )

    results = []

    for test in test_urls:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing: {test['name']}")
        logger.info(f"🔗 URL: {test['url']}")
        logger.info(f"{'='*60}")

        try:
            # Check for cached strategy first
            if scraper.strategy_detector:
                cached = scraper.strategy_detector.get_strategy(test['url'])
                if cached:
                    logger.info(f"📚 Found cached strategy: {cached['extraction_method']} via {cached['proxy_type']}")
                else:
                    logger.info("🆕 No cached strategy found. Will detect automatically.")

            # Execute Scrape
            result = await scraper.scrape(
                url=test['url'],
                fields=[], # Auto-extract
                scroll_to_bottom=True # Enable scrolling for dynamic content
            )

            # Analyze Result
            status = result.get('status_code', 0)
            success = status == 200
            data_count = len(result.get('data', []))
            
            # Determine method used
            method_used = "html"
            if result.get('metadata', {}).get('strategy', {}).get('method'):
                method_used = result['metadata']['strategy']['method']
            elif len(result.get('json_data', [])) > 0:
                method_used = "json"
            
            logger.info(f"📊 Result: {'✅ Success' if success else '❌ Failed'}")
            logger.info(f"   Status: {status}")
            logger.info(f"   Items Extracted: {data_count}")
            logger.info(f"   Method Used: {method_used}")

            results.append({
                "name": test['name'],
                "url": test['url'],
                "success": success,
                "status": status,
                "items": data_count,
                "method": method_used,
                "cached": bool(cached) if scraper.strategy_detector else False
            })

        except Exception as e:
            logger.error(f"❌ Error testing {test['name']}: {e}")
            results.append({
                "name": test['name'],
                "url": test['url'],
                "success": False,
                "error": str(e)
            })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📈 Test Summary")
    logger.info(f"{'='*60}")
    
    for res in results:
        icon = "✅" if res.get('success') else "❌"
        logger.info(f"{icon} {res['name']}: Status {res.get('status', 'ERR')} | Items: {res.get('items', 0)} | Method: {res.get('method', 'N/A')} | Cached: {res.get('cached', False)}")

    await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_multi_site_strategy())
