#!/usr/bin/env python3
"""
QUICK VALIDATION TEST - Single site, single page, show results immediately
Should take < 30 seconds
"""
import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.scraper import UniversalScraper


async def main():
    print("\n" + "="*80)
    print("⚡ QUICK VALIDATION - Single Page Test")
    print("="*80)
    print("Purpose: Verify JSON ranking fix works (extract correct data, not config)")
    print("Expected time: ~20-30 seconds")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY environment variable set")
        return
    
    # Test Reddit only (single page)
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: Reddit r/webscraping")
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"\n⏱️  Scraping (this will take ~20 seconds)...\n")
    
    # Initialize scraper with context validation
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,  # Disable pagination for speed
        extraction_context=context,
        enable_context_validation=True,
    )
    
    # Scrape single page
    result = await scraper.scrape(url, fields=[])
    
    # Show results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    items = result['data']
    metadata = result['metadata']
    
    print(f"✅ Extracted: {len(items)} items")
    print(f"📍 Source: {metadata.get('extraction_source', 'unknown')}")
    print(f"🎯 JSON ranking used: {metadata.get('json_ranking_success', False)}")
    
    if len(items) > 0:
        print(f"\n📝 First 3 items (showing all fields):")
        for i, item in enumerate(items[:3], 1):
            print(f"\n   --- Item {i} ---")
            for key, value in item.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:100] + "..."
                print(f"   {key}: {str_value}")
        
        # Quick validation check
        print("\n" + "="*80)
        print("🔍 VALIDATION CHECK")
        print("="*80)
        
        first_item = items[0]
        keys = list(first_item.keys())
        
        # Check if we got Reddit posts (good) or config data (bad)
        good_keys = ['title', 'author', 'upvotes', 'comments', 'subreddit', 'permalink']
        bad_keys = ['USE_DEBUG', 'ACCOUNT_MANAGER_ORIGIN', 'APPLE_SSO_CLIENT_ID', 'MANIFEST_FILE']
        
        has_good = any(k in str(keys).lower() for k in good_keys)
        has_bad = any(k in str(keys) for k in bad_keys)
        
        if has_bad:
            print("❌ FAIL: Extracted APP CONFIG (not Reddit posts)")
            print(f"   Found config keys: {keys[:5]}")
            print("\n   🐛 The JSON ranking is NOT working - it selected the wrong JSON source")
        elif has_good:
            print("✅ SUCCESS: Extracted REDDIT POSTS (correct data)")
            print(f"   Found post keys: {keys[:5]}")
            print("\n   🎉 The JSON ranking IS working!")
        else:
            print("⚠️  UNCLEAR: Extracted data doesn't match expected patterns")
            print(f"   Found keys: {keys[:5]}")
    else:
        print("\n❌ FAIL: No items extracted")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())








