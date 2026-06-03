"""
Test eBay extraction locally to diagnose issues
"""
import asyncio
import os
import logging
from universal_scraper import UniversalScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ebay():
    print("\n" + "="*80)
    print("🔍 Testing eBay Laptop Search")
    print("="*80 + "\n")
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,  # Advanced anti-detection
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        result = await scraper.scrape(
            url='https://www.ebay.com/sch/i.html?_nkw=laptop',
            fields=['title', 'price', 'condition']
        )
        
        items = result.get('data', [])
        quality = result.get('quality', 0)
        
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        
        if items:
            print(f"\n   First 3 items:")
            for i, item in enumerate(items[:3], 1):
                print(f"   {i}. {item}")
                
            # Check field coverage
            if items:
                first = items[0]
                filled = sum(1 for v in first.values() if v not in (None, '', []))
                print(f"\n   Field Coverage: {filled}/{len(first)} fields ({filled/len(first)*100:.0f}%)")
        else:
            print("\n   ❌ No items extracted!")
            print("\n   Possible issues:")
            print("   1. eBay detected bot and blocked request")
            print("   2. HTML structure too complex/obfuscated")
            print("   3. Need residential proxies")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_ebay())





