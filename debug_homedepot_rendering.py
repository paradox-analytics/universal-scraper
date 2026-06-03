import asyncio
import sys
from pathlib import Path
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

async def debug_rendering():
    print("=" * 80)
    print("🕵️  DEBUGGING HOME DEPOT RENDERING")
    print("=" * 80)
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    
    # Use CamoufoxFetcher for consistent configuration
    fetcher = CamoufoxFetcher(
        proxy_config={
            'server': 'http://brd.superproxy.io:33335',
            'username': 'brd-customer-hl_803e8195-zone-web_unlocker1',
            'password': 't8mhp1qev1i1'
        },
        headless=True,
        timeout=300000,
        humanize=True,
        stealth_mode=False, # Golden Configuration
        # Disable geoip to bypass initial IP verification which might fail with Web Unblocker
        # geoip=False is not a direct param of CamoufoxFetcher, but we can pass it via anti_detection_config
    )
    # Actually, let's modify CamoufoxFetcher to accept geoip or just hack it here
    fetcher.anti_detection_config['geoip'] = False
    
    print(f"🚀 Starting fetch with CamoufoxFetcher...")
    try:
        result = await fetcher.fetch(
            url=url,
            wait_for_selector="h1.sui-h2-bold",
            wait_time=5000
        )
        
        print(f"   ✅ Fetch Complete (Status: {result.get('status_code')})")
        print(f"   📄 HTML Size: {len(result.get('html', ''))} bytes")
        
        # Save results
        with open("debug_final.html", "w") as f:
            f.write(result.get('html', ''))
        print("   📄 Final HTML saved")
        
        if len(result.get('html', '')) < 1000:
            print("\n⚠️  WARNING: HTML size is suspiciously small. Likely still blocked.")
            if "Access Denied" in result.get('html', ''):
                print("   ❌ Confirmed: Access Denied page detected.")
        else:
            print("\n🎉 SUCCESS: HTML size looks good!")
        
    except Exception as e:
        print(f"\n❌ Fetch Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_rendering())
