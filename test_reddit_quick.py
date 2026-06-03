"""
Quick Reddit Test - Validate Simplified JSON Selection
"""

import asyncio
import os
import time
import logging
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

async def main():
    print("\n" + "="*80)
    print("🧪 QUICK TEST - Reddit (Simplified JSON Selection)")
    print("="*80)
    print("Expected: Extract Reddit posts (not app config)")
    print("="*80 + "\n")

    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes"

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY")
        return

    start = time.time()
    
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True
    )

    try:
        result = await scraper.scrape(url, fields=[])
        elapsed = time.time() - start
        
        data = result.get('data', [])
        metadata = result.get('metadata', {})
        
        print(f"\n{'='*80}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📊 Items: {len(data)}")
        print(f"📍 Source: {metadata.get('extraction_source', 'unknown')}")
        
        if metadata.get('json_source'):
            print(f"🎯 JSON Source: {metadata['json_source']}")
        
        if len(data) > 0:
            print(f"\n✅ SUCCESS! Extracted {len(data)} items")
            print(f"\n📝 Sample (first item):")
            item = data[0]
            for k, v in list(item.items())[:5]:  # First 5 fields
                print(f"   {k}: {str(v)[:80]}")
        else:
            print("\n❌ FAILED - 0 items extracted")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()

if __name__ == "__main__":
    asyncio.run(main())








