import asyncio
import time
import logging
from camoufox.async_api import AsyncCamoufox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_camoufox_basic():
    print("🚀 Starting Basic Camoufox Test (No Proxy)...")
    
    url = "https://example.com"
    print(f"Target URL: {url}")
    
    try:
        async with AsyncCamoufox(headless=True) as browser:
            print("Browser launched successfully")
            context = await browser.new_context()
            page = await context.new_page()
            
            print(f"Navigating to {url}...")
            start_time = time.time()
            response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            elapsed = time.time() - start_time
            
            print(f"Navigation finished in {elapsed:.1f}s")
            print(f"Status: {response.status}")
            
            content = await page.content()
            print(f"HTML Size: {len(content)} bytes")
            
            if response.status == 200 and len(content) > 500:
                print("✅ Basic Camoufox test PASSED")
                return True
            else:
                print(f"❌ Basic Camoufox test FAILED (status: {response.status}, size: {len(content)})")
                return False
                
    except Exception as e:
        print(f"❌ Basic Camoufox test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_camoufox_basic())
