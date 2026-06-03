#!/usr/bin/env python3
"""
Test Reddit with explicit wait for content to load
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
    print("🔍 REDDIT TEST - With Wait for Dynamic Content")
    print("="*80 + "\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: {url}")
    print(f"📋 Context: {context}\n")
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
    )
    
    # Disable pagination
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    
    print("⏱️  Fetching with wait for 'shreddit-post' selector...\n")
    
    # Scrape with wait for the actual Reddit post element
    result = await scraper.scrape(
        url, 
        fields=[],
        wait_for_selector="shreddit-post"  # Wait for Reddit's custom post element
    )
    
    # Show results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"📦 Items extracted: {len(result['data'])}")
    print(f"📍 Source: {result['metadata'].get('extraction_source', 'unknown')}")
    
    if len(result['data']) > 0:
        first_item = result['data'][0]
        print(f"\n📝 First item keys: {list(first_item.keys())}")
        print(f"\n🔍 First item preview:")
        import json
        print(json.dumps(first_item, indent=2, default=str)[:500])
        
        # Check if we got Reddit posts
        keys_str = str(list(first_item.keys())).lower()
        if 'title' in keys_str or 'author' in keys_str:
            print("\n✅ SUCCESS: Got Reddit posts!")
        else:
            print("\n⚠️  Got data but might not be posts")
    else:
        print("\n❌ FAIL: No items extracted")
    
    print("\n" + "="*80 + "\n")
    
    scraper.close()

if __name__ == "__main__":
    asyncio.run(main())







