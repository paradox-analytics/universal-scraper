"""
Adaptive Anti-Blocking Discovery Test for Home Depot
Systematically tests different configurations to find the optimal strategy
"""
import asyncio
import logging
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.adaptive_antiblocking_agent import AdaptiveAntiBlockingAgent
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_adaptive_discovery():
    """
    Run adaptive anti-blocking discovery for Home Depot
    Tests multiple configurations to find what works
    """
    
    # Credentials
    openai_key = "REDACTED_OPENAI_KEY_1"
    
    # Residential Proxy
    res_host = "brd.superproxy.io"
    res_port = "33335"
    res_user = "brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2"
    res_pass = "REDACTED_PROXY_PASS"
    
    # Web Unblocker
    unblocker_user = "brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1"
    unblocker_pass = "REDACTED_PROXY_PASS"
    
    # Home Depot URL
    url = "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745"
    
    logger.info("\n" + "="*80)
    logger.info("🚀 Adaptive Anti-Blocking Discovery Test")
    logger.info("="*80)
    logger.info(f"Target: {url}")
    logger.info("")
    
    # Test configurations to try
    test_configs = [
        {
            "name": "Web Unblocker + Browser",
            "proxy_type": "web_unblocker",
            "web_unblocker_api_key": unblocker_pass,
            "web_unblocker_zone": "web_unlocker1",
            "use_camoufox": True,
            "browser_timeout": 120000
        },
        {
            "name": "Residential Proxy + Browser",
            "proxy_type": "residential",
            "proxy_config": {
                "server": f"http://{res_host}:{res_port}",
                "username": res_user,
                "password": res_pass
            },
            "use_camoufox": True,
            "browser_timeout": 120000
        },
        {
            "name": "Web Unblocker + Static",
            "proxy_type": "web_unblocker",
            "web_unblocker_api_key": unblocker_pass,
            "web_unblocker_zone": "web_unlocker1",
            "use_camoufox": False,
            "browser_timeout": 60000
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 Test {i}/{len(test_configs)}: {config['name']}")
        logger.info(f"{'='*80}")
        
        try:
            # Initialize scraper with this configuration
            scraper_args = {
                "api_key": openai_key,
                "headless": True
            }
            
            # Add config-specific args
            if config.get("web_unblocker_api_key"):
                scraper_args["web_unblocker_api_key"] = config["web_unblocker_api_key"]
                scraper_args["web_unblocker_zone"] = config.get("web_unblocker_zone", "web_unlocker1")
            
            if config.get("proxy_config"):
                scraper_args["proxy_config"] = config["proxy_config"]
            
            if "use_camoufox" in config:
                scraper_args["use_camoufox"] = config["use_camoufox"]
            
            if "browser_timeout" in config:
                scraper_args["browser_timeout"] = config["browser_timeout"]
            
            logger.info(f"📋 Configuration:")
            logger.info(f"   Proxy Type: {config['proxy_type']}")
            logger.info(f"   Browser: {'Camoufox' if config.get('use_camoufox') else 'Static'}")
            logger.info(f"   Timeout: {config.get('browser_timeout', 60000)}ms")
            
            # Create scraper
            scraper = UniversalScraper(**scraper_args)
            
            # Attempt scrape
            logger.info(f"\n📡 Fetching {url}...")
            result = await scraper.scrape(
                url=url,
                fields=[]  # Auto-extract
            )
            
            # Analyze result
            status = result.get('status_code', 0)
            data_count = len(result.get('data', []))
            html_size = len(result.get('html', ''))
            
            success = status == 200 and data_count > 0
            
            logger.info(f"\n📊 Result:")
            logger.info(f"   Status Code: {status}")
            logger.info(f"   Items Extracted: {data_count}")
            logger.info(f"   HTML Size: {html_size} bytes")
            logger.info(f"   Success: {'✅ YES' if success else '❌ NO'}")
            
            if data_count > 0:
                logger.info(f"\n📦 Sample Data:")
                for item in result['data'][:2]:
                    logger.info(f"   {item}")
            
            results.append({
                "config": config['name'],
                "proxy_type": config['proxy_type'],
                "success": success,
                "status": status,
                "items": data_count,
                "html_size": html_size
            })
            
            await scraper.close()
            
            # If successful, we found a working config!
            if success:
                logger.info(f"\n🎉 SUCCESS! Found working configuration: {config['name']}")
                logger.info(f"   This configuration will be cached for future use.")
                break
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            results.append({
                "config": config['name'],
                "proxy_type": config['proxy_type'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📈 Discovery Test Summary")
    logger.info(f"{'='*80}")
    
    for result in results:
        icon = "✅" if result.get('success') else "❌"
        logger.info(f"{icon} {result['config']}")
        logger.info(f"   Proxy: {result['proxy_type']}")
        if result.get('success'):
            logger.info(f"   Items: {result.get('items', 0)}")
            logger.info(f"   HTML: {result.get('html_size', 0)} bytes")
        elif result.get('error'):
            logger.info(f"   Error: {result['error'][:100]}")
    
    # Check if any succeeded
    successful = [r for r in results if r.get('success')]
    if successful:
        logger.info(f"\n✅ Discovery Complete! Found {len(successful)} working configuration(s).")
        logger.info(f"   Best: {successful[0]['config']}")
    else:
        logger.error(f"\n❌ Discovery Failed. No configurations worked.")
        logger.error(f"   Recommendation: Check proxy credentials and try with longer timeout.")

if __name__ == "__main__":
    asyncio.run(test_adaptive_discovery())
