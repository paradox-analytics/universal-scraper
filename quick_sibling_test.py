"""Quick test of sibling + frequency detection"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test():
    print("\n🎯 Testing Sibling + Frequency Detection on Stack Overflow")
    print("="*70)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=False,  # Use browser for API detection
        headless=True,
        enable_auto_pagination=False
        # Let it try JSON first - should reject and fall back to HTML!
    )
    
    try:
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes']
        )
        items = result.get('data', [])
        
        # Calculate vote quality
        votes_filled = sum(1 for item in items if item.get('votes') not in (None, '', []))
        votes_quality = (votes_filled / len(items) * 100) if items else 0
        
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Votes extracted: {votes_filled}/{len(items)} ({votes_quality:.0f}%)")
        print(f"\n   Sample:")
        for i, item in enumerate(items[:5], 1):
            print(f"      {i}. votes={item.get('votes', 'None')}")
        
        if votes_quality >= 80:
            print(f"\n✅ SUCCESS! Sibling + frequency detection WORKING!")
        elif votes_quality > 0:
            print(f"\n⚠️  PARTIAL - {votes_quality:.0f}% working")
        else:
            print(f"\n❌ FAILED - Still not extracting votes")
        
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test())

