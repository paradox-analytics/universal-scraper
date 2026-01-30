"""Quick debug for eBay"""
import asyncio
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from bs4 import BeautifulSoup
from universal_scraper.core.browser_fetcher import BrowserFetcher


async def debug_ebay():
    print("🔍 Debugging eBay...\n")
    
    # Fetch page directly with browser
    fetcher = BrowserFetcher(headless=True)
    await fetcher._launch_browser()  # Initialize browser
    
    try:
        result = await fetcher.fetch("https://www.ebay.com/sch/i.html?_nkw=laptop")
        
        html = result['html'] if isinstance(result, dict) else result
        
        print(f"✅ Fetched: {len(html):,} bytes\n")
        
        # Parse
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for product items
        print("🔍 Looking for product containers:\n")
        
        # Try common eBay selectors
        selectors_to_try = [
            ('li with s-item class', soup.find_all('li', class_=lambda x: x and 's-item' in str(x))),
            ('div with s-item class', soup.find_all('div', class_=lambda x: x and 's-item' in str(x))),
            ('article tags', soup.find_all('article')),
            ('li tags', soup.find_all('li')),
        ]
        
        for name, elements in selectors_to_try:
            if elements:
                print(f"   ✅ {name}: Found {len(elements)}")
                if len(elements) > 0:
                    # Show first element structure
                    first_html = str(elements[0])[:800]
                    print(f"      First element preview:\n      {first_html}...\n")
        
        # Save sample
        output_dir = Path(__file__).parent / "debug_output"
        output_dir.mkdir(exist_ok=True)
        
        sample_path = output_dir / "ebay_full.html"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"💾 Saved full HTML to: {sample_path}\n")
        
        await fetcher.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_ebay())

