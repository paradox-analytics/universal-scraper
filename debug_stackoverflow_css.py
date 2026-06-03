#!/usr/bin/env python3
"""
Debug Stack Overflow CSS Selector Issue

Test if BeautifulSoup can handle escaped CSS selectors
"""

import asyncio
import os
from bs4 import BeautifulSoup
from universal_scraper import UniversalScraper

async def main():
    print("Testing BeautifulSoup CSS selector escaping...")
    
    # Test HTML with Tailwind classes containing `:`
    test_html = """
    <html>
        <li class="h:bg-black-150 item">Item 1</li>
        <li class="h:bg-black-150 item">Item 2</li>
        <li class="h:bg-black-150 item">Item 3</li>
    </html>
    """
    
    soup = BeautifulSoup(test_html, 'lxml')
    
    # Test 1: Without escaping (should fail)
    try:
        items = soup.select('li.h:bg-black-150')
        print(f"\n❌ Unescaped selector 'li.h:bg-black-150': {len(items)} items")
    except Exception as e:
        print(f"\n❌ Unescaped selector 'li.h:bg-black-150': ERROR - {e}")
    
    # Test 2: With escaping (should work)
    try:
        items = soup.select('li.h\\:bg-black-150')
        print(f"✅ Escaped selector 'li.h\\\\:bg-black-150': {len(items)} items")
    except Exception as e:
        print(f"❌ Escaped selector 'li.h\\\\:bg-black-150': ERROR - {e}")
    
    # Test 3: Using class_ attribute (alternative)
    try:
        items = soup.find_all('li', class_='h:bg-black-150')
        print(f"✅ Using class_ attribute: {len(items)} items")
    except Exception as e:
        print(f"❌ Using class_ attribute: ERROR - {e}")
    
    # Test 4: Real Stack Overflow scrape to see the actual HTML
    print("\n\n" + "="*80)
    print("Fetching real Stack Overflow HTML...")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
        from universal_scraper.core.anti_detection import AntiDetectionManager
        
        anti_detect = AntiDetectionManager(
            profile='random',
            humanize=True,
            stealth_mode=True
        )
        
        fetcher = CamoufoxFetcher(anti_detection=anti_detect)
        result = await fetcher.fetch('https://stackoverflow.com/questions?tab=newest')
        html = result['html']
        
        print(f"\nFetched {len(html)} bytes")
        
        # Parse and look for the pattern
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all li elements
        all_li = soup.find_all('li')
        print(f"Total <li> elements: {len(all_li)}")
        
        # Find li with classes containing `:
        li_with_colon = [li for li in all_li if li.get('class') and any(':' in c for c in li.get('class'))]
        print(f"<li> with `:` in classes: {len(li_with_colon)}")
        
        if li_with_colon:
            print(f"\nSample <li> with colon:")
            sample = li_with_colon[0]
            print(f"Classes: {sample.get('class')}")
            print(f"HTML: {str(sample)[:200]}...")
            
            # Try to select it
            classes_str = '.'.join(sample.get('class', []))
            print(f"\nUnescaped selector: li.{classes_str}")
            
            # Try escaped
            escaped_classes = '.'.join([c.replace(':', '\\:') for c in sample.get('class', [])])
            print(f"Escaped selector: li.{escaped_classes}")
            
            try:
                selected = soup.select(f'li.{escaped_classes}')
                print(f"✅ Escaped selector found {len(selected)} items")
            except Exception as e:
                print(f"❌ Escaped selector failed: {e}")
        
        await fetcher.close()
        
    finally:
        await scraper.close()

if __name__ == '__main__':
    asyncio.run(main())






