import asyncio
from camoufox.async_api import AsyncCamoufox
import logging

async def test_camoufox():
    logging.basicConfig(level=logging.INFO)
    
    proxy = {
        "server": "brd.superproxy.io:22225",
        "username": "brd-customer-hl_803e8195-zone-web_unlocker1",
        "password": "t8mhp1qev1i1"
    }
    
    url = "https://www.homedepot.com/p/Ryobi-18V-ONE-Lithium-Ion-Cordless-1-2-in-Drill-Driver-Kit-with-1-1-5-Ah-Battery-and-Charger-P215K/309677412"
    
    async with AsyncCamoufox(
        proxy=proxy,
        headless=True,
        humanize=True,
        geoip=False,
        os="windows"
    ) as browser:
        page = await browser.new_page(ignore_https_errors=True)
        print(f"🚀 Navigating to {url}...")
        try:
            response = await page.goto(url, wait_until="networkidle", timeout=60000)
            print(f"✅ Status: {response.status}")
            content = await page.content()
            print(f"✅ Content length: {len(content)}")
            print(f"✅ Content preview: {content[:500]}")
            
            with open("direct_camoufox.html", "w") as f:
                f.write(content)
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_camoufox())
