"""
Debug eBay HTML to understand the structure
"""
import asyncio
import os
from bs4 import BeautifulSoup
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher

async def debug_ebay():
    print("\n🔍 Debugging eBay HTML Structure\n")
    
    fetcher = CamoufoxFetcher(headless=True)
    
    try:
        result = await fetcher.fetch('https://www.ebay.com/sch/i.html?_nkw=laptop')
        html = result['html']
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check what elements exist
        print("🔍 Checking key selectors:")
        
        selectors_to_check = [
            'li.s-item',  # Common eBay product item
            'li.s-card',  # Alternative
            'div.s-item__wrapper',
            'div.s-item__info',
            '.srp-results .s-item',  # Full path
        ]
        
        for selector in selectors_to_check:
            elements = soup.select(selector)
            print(f"   {selector}: {len(elements)} elements")
            
            if elements:
                # Show first element structure
                first = elements[0]
                print(f"      Classes: {first.get('class', [])}")
                
                # Try to find title
                title = first.select_one('div.s-item__title, h3.s-item__title, span.s-item__title, [role="heading"]')
                if title:
                    print(f"      Title: {title.get_text(strip=True)[:50]}...")
                
                # Try to find price
                price = first.select_one('span.s-item__price, div.s-item__price')
                if price:
                    print(f"      Price: {price.get_text(strip=True)}")
                
                print()
        
        # Check if page was blocked
        if 'To better protect your account' in html or 'captcha' in html.lower():
            print("⚠️ Page might be blocked or showing CAPTCHA")
        
        # Show sample of HTML
        print(f"\n📄 HTML Sample (first product):")
        items = soup.select('li.s-item')
        if items:
            print(str(items[0])[:1500])
        else:
            print("No li.s-item found, trying s-card...")
            cards = soup.select('li.s-card')
            if cards:
                print(str(cards[0])[:1500])
            else:
                print("❌ No items found with common selectors")
        
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(debug_ebay())





