import asyncio
import logging
import os
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_samples():
    """Test Universal Scraper on user-provided samples"""
    
    # Credentials
    openai_key = "REDACTED_OPENAI_KEY_1"
    unblocker_pass = "REDACTED_PROXY_PASS"
    
    # Force set env var
    os.environ['OPENAI_API_KEY'] = openai_key
    
    # Initialize Scraper
    logger.info("🚀 Initializing Universal Scraper...")
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True,
        headless=True,
        browser_timeout=120000,
        web_unblocker_api_key=unblocker_pass,
        web_unblocker_zone="web_unlocker1"
    )
    
    urls = [
        "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745",
        "https://www.homedepot.com/p/Frigidaire-28-Cu-Ft-Standard-Depth-French-Door-Refrigerator-in-Smudge-Proof-Stainless-Steel-ENERGY-STAR-FRFS2823AF/336155094?MERCH=REC-_-personalizedDeals-_-n/a-_-1-_-n/a-_-n/a-_-n/a-_-n/a-_-n/a",
        "https://www.producthunt.com/categories/vibe-coding"
    ]
    
    results = {}
    
    for i, url in enumerate(urls):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing URL {i+1}/{len(urls)}")
        logger.info(f"🔗 {url}")
        logger.info(f"{'='*60}")
        
        try:
            # Execute Scrape
            result = await scraper.scrape(
                url=url,
                fields=[] # Auto-detect
            )
            
            # Analyze Result
            status = result.get('status_code', 0)
            data = result.get('data', [])
            success = status == 200 and len(data) > 0
            
            results[url] = {
                'success': success,
                'status': status,
                'items': len(data),
                'sample': data[:1] if data else []
            }
            
            logger.info(f"✅ Result for {url}:")
            logger.info(f"   Success: {success}")
            logger.info(f"   Items: {len(data)}")
            logger.info(f"   Sample: {data[:1]}")
            
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}", exc_info=True)
            results[url] = {'success': False, 'error': str(e)}
            
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    for url, res in results.items():
        status_icon = "✅" if res.get('success') else "❌"
        logger.info(f"{status_icon} {url}")
        logger.info(f"   Items: {res.get('items', 0)}")
        if res.get('sample'):
            logger.info(f"   Data: {res['sample']}")
            
    await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_samples())
