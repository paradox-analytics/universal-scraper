import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to sys.path
sys.path.append(os.getcwd())

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

async def test_homedepot_async():
    print("🧪 Testing Async CamoufoxFetcher on Home Depot...")
    
    # Actual credentials provided by user
    proxy_key = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    
    fetcher = CamoufoxFetcher(
        headless=True, 
        web_unblocker_api_key=proxy_key,
        timeout=60000
    )
    
    try:
        print(f"\nTesting fetch for: {url}")
        print("Using credentials: brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,********")
        
        result = await fetcher.fetch(url)
        
        print(f"\n✅ Success!")
        print(f"Status: {result['status']}")
        print(f"URL: {result['url']}")
        print(f"HTML length: {len(result['html'])}")
        print(f"Captured JSON blobs: {len(result['json_data'])}")
        
        if "Access Denied" in result['html'] or "Access Denied" in result['html']:
            print("⚠️ Warning: Page contains 'Access Denied' text. Might still be blocked.")
        elif len(result['html']) < 5000:
            print("⚠️ Warning: HTML is very short. Might be a block page.")
        else:
            print("🎉 Content looks valid!")
            
    except Exception as e:
        print(f"\n❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_homedepot_async())
