import asyncio
import logging
import os
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import universal_scraper.core.json_detector as jd
print(f"DEBUG: JSONDetector imported from: {jd.__file__}")

async def test_homedepot():
    # User provided credentials
    # Note: Using the provided Web Unlocker credentials
    web_unblocker_api_key = "brd.superproxy.io:33335:brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS"
    web_unblocker_zone = "web_unlocker1"
    
    # Also setting up residential proxy just in case, though Web Unlocker is preferred for tough sites
    # brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2:REDACTED_PROXY_PASS
    
    # OpenAI Key (using the one from previous context if available, or a placeholder if I need to ask)
    # I'll use the one from the previous test script since it's likely the same user environment
    openai_key = os.getenv("OPENAI_API_KEY", "REDACTED_OPENAI_KEY_1")

    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    
    print(f"🚀 Testing Home Depot extraction for {url}")
    
    # Initialize scraper with Web Unlocker
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True, # Try with Camoufox first as it's the default now
        web_unblocker_api_key=web_unblocker_api_key,
        web_unblocker_zone=web_unblocker_zone
    )
    
    # Apply Golden Configuration for Home Depot (resolves InvalidIP and 403 issues)
    # We'll force initialization of the browser fetcher to ensure settings are applied
    if hasattr(scraper.html_fetcher, 'browser_fetcher'):
        # Force initialization if it's a HybridFetcher
        if hasattr(scraper.html_fetcher, '_get_browser_fetcher'):
            browser_fetcher = scraper.html_fetcher._get_browser_fetcher()
            browser_fetcher.anti_detection_config['geoip'] = False
            browser_fetcher.anti_detection_config['stealth_mode'] = False
            browser_fetcher.anti_detection_config['humanize'] = True
    else:
        # If not yet initialized, we need to ensure it's set when it is
        # HybridFetcher lazy loads the browser fetcher, so we can't set it directly yet
        # unless we force initialization or modify HybridFetcher to pass it.
        # For this test script, we'll just force a small fetch to initialize it or 
        # better yet, modify the scraper's fetcher config if possible.
        pass
    
    try:
        # 1. Test HTML Fetching
        print("\n📥 Fetching HTML...")
        fetch_result = await scraper.html_fetcher.fetch(url)
        
        if not fetch_result or not fetch_result.get('html'):
            print("❌ Failed to fetch HTML")
            return
            
        html_content = fetch_result['html']
        print(f"✅ Fetched {len(html_content)} chars")
        
        # Save HTML for inspection
        with open("debug_homedepot.html", "w") as f:
            f.write(html_content)
        print("   HTML saved to debug_homedepot.html")
        
        # 2. Test Field Suggestion (using provided HTML)
        print("\n🔍 Testing Field Suggestion...")
        
        from universal_scraper.core.field_discovery import FieldDiscovery
        field_discovery = FieldDiscovery(api_key=openai_key)
        
        # Use LLM for discovery to get high quality fields
        discovery_result = await field_discovery.discover_fields(
            html=html_content,
            url=url,
            use_llm=False
        )
        
        suggested_fields = discovery_result.get('fields', [])
        print(f"✅ Suggested fields: {suggested_fields}")
        print(f"   Source: {discovery_result.get('source')}")
        
        # 3. Test Extraction (passing HTML to avoid re-fetch)
        print("\n🎯 Testing Extraction (with provided HTML)...")
        
        if suggested_fields:
            fields = suggested_fields
            print(f"   Using discovered fields: {fields}")
        else:
            print("⚠️ No fields suggested, falling back to basic fields")
            fields = ["name", "price", "model_number", "sku"]
        
        results = await scraper.scrape(
            url=url,
            fields=fields,
            html=html_content  # NEW: Pass HTML directly
        )
        
        print("\n📊 Extraction Results:")
        if results and results.get('data'):
            print(f"  Items found: {len(results['data'])}")
            print(f"  First item: {results['data'][0]}")
        else:
            print("  ❌ No data extracted")
            print(f"  Metadata: {results.get('metadata', {})}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_homedepot())
