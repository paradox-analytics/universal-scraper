#!/usr/bin/env python3
"""
REDDIT DEBUG TEST - Single page with full logging
"""
import asyncio
import os
import sys
import logging
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable DEBUG logging for ALL modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

from universal_scraper.core.scraper import UniversalScraper


async def main():
    print("\n" + "="*80)
    print("🔍 REDDIT DEBUG TEST - Full Logging Enabled")
    print("="*80)
    print("You'll see every step in real-time")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: {url}")
    print(f"📋 Context: {context}")
    print(f"\n{'='*80}")
    print("⏱️  Starting scrape... (watch for each step below)")
    print(f"{'='*80}\n")
    
    start = time.time()
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,  # Disable for speed
        extraction_context=context,
        enable_context_validation=True,
        log_level=logging.DEBUG  # Full debug output
    )
    
    # Scrape
    result = await scraper.scrape(url, fields=[])
    
    elapsed = time.time() - start
    
    # Show results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📦 Items extracted: {len(result['data'])}")
    print(f"📍 Source: {result['metadata'].get('extraction_source', 'unknown')}")
    
    if len(result['data']) > 0:
        print(f"\n📝 First item keys: {list(result['data'][0].keys())}")
        print(f"\n🔍 First item preview:")
        for k, v in list(result['data'][0].items())[:5]:
            print(f"   {k}: {str(v)[:80]}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())








