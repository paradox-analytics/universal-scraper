"""
Quick test for Reddit only - Phase 1 + 2 validation
"""
import asyncio
import time
import logging
from universal_scraper.core.scraper import UniversalScraper

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    print("\n" + "="*80)
    print("🧪 TESTING REDDIT ONLY - Phase 1 + 2")
    print("="*80)
    print("Phase 1: Better HTML cleaning (42% reduction vs 99.9%)")
    print("Phase 2: Improved code generation prompts (few-shot + context)")
    print("="*80 + "\n")
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count, post URL, and timestamp"
    
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"\n⏱️  Scraping...\n")
    
    start = time.time()
    
    scraper = UniversalScraper(
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
    )
    
    result = await scraper.scrape(url, fields=[])
    
    elapsed = time.time() - start
    
    print("\n" + "="*80)
    print(f"⏱️  Completed in {elapsed:.1f} seconds")
    print("="*80 + "\n")
    
    # Check results
    if result:
        data = result.get('data', [])
        source = result.get('metadata', {}).get('source', 'unknown')
        
        print("📊 EXTRACTION SUMMARY:")
        print(f"   Items extracted: {len(data)}")
        print(f"   Data source: {source}")
        
        if len(data) > 0:
            print(f"\n✅ SUCCESS! Extracted {len(data)} Reddit posts")
            print(f"\n📝 Sample (first 3 posts):")
            for i, item in enumerate(data[:3], 1):
                print(f"\n   Post {i}:")
                for key, value in item.items():
                    value_str = str(value)[:60]
                    print(f"      {key}: {value_str}")
        else:
            print(f"\n⚠️  No items extracted!")
    else:
        print("❌ No result returned")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

