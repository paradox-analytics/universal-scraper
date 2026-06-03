"""Debug Stack Overflow sibling extraction"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def debug():
    print("\n🔍 Debugging Stack Overflow - Why votes=None?\n")
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=False,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes']
        )
        
        items = result.get('data', [])
        
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        
        # Check vote quality
        votes_filled = sum(1 for item in items if item.get('votes') not in (None, '', [], 'None'))
        print(f"   Votes extracted: {votes_filled}/{len(items)}")
        
        print(f"\n   First 5 items:")
        for i, item in enumerate(items[:5], 1):
            print(f"      {i}. title={item.get('title')[:50]}...")
            print(f"         votes={item.get('votes')}")
        
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(debug())





