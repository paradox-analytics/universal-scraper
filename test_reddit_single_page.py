#!/usr/bin/env python3
"""
REDDIT SINGLE PAGE TEST - No pagination at all
"""
import asyncio
import os
import sys
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.scraper import UniversalScraper


async def main():
    print("\n" + "="*80)
    print("🔍 REDDIT SINGLE PAGE TEST - Pagination Completely Disabled")
    print("="*80)
    print("This will scrape ONLY the first page, no pagination at all")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: {url}")
    print(f"📋 Context: {context}\n")
    
    start = time.time()
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
    )
    
    # MONKEY PATCH: Disable pagination detectors
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
        scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
    
    print("⚠️  Pagination detection DISABLED (monkey-patched)")
    
    print("⏱️  Scraping single page...\n")
    
    # Scrape with wait for Reddit's custom post elements to load
    result = await scraper.scrape(
        url, 
        fields=[],
        wait_for_selector="shreddit-post"  # Wait for Reddit posts to load
    )
    
    elapsed = time.time() - start
    
    # Show results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📦 Items extracted: {len(result['data'])}")
    print(f"📍 Source: {result['metadata'].get('extraction_source', 'unknown')}")
    
    if len(result['data']) > 0:
        first_item = result['data'][0]
        print(f"\n📝 First item keys: {list(first_item.keys())[:10]}")
        
        # Check if we got Reddit posts or config
        keys_str = str(list(first_item.keys())).lower()
        if 'title' in keys_str or 'author' in keys_str or 'subreddit' in keys_str:
            print("\n✅ SUCCESS: Extracted Reddit posts (correct data)!")
        elif 'account_manager' in keys_str or 'apple_sso' in keys_str:
            print("\n❌ FAIL: Extracted app config (wrong data)")
        else:
            print("\n⚠️  UNCLEAR: Check first item below")
        
        print(f"\n🔍 First item preview:")
        for k, v in list(first_item.items())[:5]:
            val_str = str(v)[:80]
            print(f"   {k}: {val_str}")
    else:
        print("\n❌ FAIL: No items extracted")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

