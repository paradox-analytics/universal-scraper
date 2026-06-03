import asyncio
import time
import logging
from camoufox.async_api import AsyncCamoufox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_camoufox_proxy():
    print("🚀 Starting Camoufox Proxy Test...")
    
    url = "https://api.ipify.org?format=json"
    proxy = {
        'server': 'http://brd.superproxy.io:22225',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    print(f"Target URL: {url}")
    print(f"Proxy Server: {proxy['server']}")
    
    try:
        # Camoufox takes proxy in the constructor
        # We use ignore_https_errors=True because Bright Data uses self-signed certs for interception
        async with AsyncCamoufox(headless=True, proxy=proxy) as browser:
            print("Browser launched successfully with proxy")
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()
            
            print(f"Navigating to {url}...")
            start_time = time.time()
            # Set a long timeout
            response = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            elapsed = time.time() - start_time
            
            print(f"Navigation finished in {elapsed:.1f}s")
            print(f"Status: {response.status}")
            
            content = await page.content()
            print(f"HTML Size: {len(content)} bytes")
            print(f"Content: {content}")
            
            if response.status == 200 and "ip" in content:
                print("✅ Camoufox proxy test PASSED")
                return True
            else:
                print(f"❌ Camoufox proxy test FAILED (status: {response.status})")
                return False
                
    except Exception as e:
        print(f"❌ Camoufox proxy test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_camoufox_proxy())
