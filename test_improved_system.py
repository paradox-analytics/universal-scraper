#!/usr/bin/env python3
"""
Test Improved System with ScrapeGraphAI Integration
- HTML Structure Analysis before code generation
- Multi-iteration code refinement with error feedback
- Smart content sampling
- Pattern detection
"""
import asyncio
import os
import sys
import json
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.scraper import UniversalScraper

async def main():
    print("\n" + "="*80)
    print("🚀 IMPROVED SYSTEM TEST - With ScrapeGraphAI Integration")
    print("="*80)
    print("\nNew Features:")
    print("  ✅ HTML Structure Analysis (from ScrapeGraphAI)")
    print("  ✅ Multi-iteration Code Refinement (from ScrapeGraphAI)")
    print("  ✅ Smart Content Sampling (our innovation)")
    print("  ✅ Attribute Detection (our innovation)")
    print("="*80 + "\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: {url}")
    print(f"📋 Context: {context}\n")
    
    start = time.time()
    
    # Initialize scraper with all new features enabled
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
        log_level=20  # INFO level
    )
    
    # Disable pagination for this test
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    
    print("⏱️  Scraping with improved system...\n")
    
    # Scrape with wait for posts to load
    result = await scraper.scrape(
        url,
        fields=[],
        wait_for_selector="shreddit-post"
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
        print(f"\n📝 First item keys: {list(first_item.keys())}")
        
        # Check data quality
        has_title = bool(first_item.get('title'))
        has_author = bool(first_item.get('author'))
        has_upvotes = first_item.get('upvotes') is not None
        has_comments = first_item.get('comments_count') is not None or first_item.get('comments') is not None
        
        print(f"\n✅ Data Quality:")
        print(f"   Title: {'✅ ' + str(first_item.get('title'))[:50] if has_title else '❌ Missing'}")
        print(f"   Author: {'✅ ' + str(first_item.get('author')) if has_author else '❌ Missing'}")
        print(f"   Upvotes: {'✅ ' + str(first_item.get('upvotes')) if has_upvotes else '❌ Missing'}")
        print(f"   Comments: {'✅ ' + str(first_item.get('comments_count') or first_item.get('comments')) if has_comments else '❌ Missing'}")
        
        print(f"\n🔍 First 3 items:")
        for i, item in enumerate(result['data'][:3], 1):
            title = item.get('title', 'No title')[:60]
            author = item.get('author', 'No author')
            upvotes = item.get('upvotes', 0)
            comments = item.get('comments_count') or item.get('comments', 0)
            print(f"\n{i}. {title}...")
            print(f"   By: {author} | Upvotes: {upvotes} | Comments: {comments}")
        
        if all([has_title, has_author, has_upvotes, has_comments]):
            print(f"\n🎉 SUCCESS: All fields extracted correctly!")
            print(f"✅ The improved system works!")
        else:
            print(f"\n⚠️  PARTIAL: Some fields missing")
    else:
        print("\n❌ FAIL: No items extracted")
    
    print("\n" + "="*80 + "\n")
    
    scraper.close()

if __name__ == "__main__":
    asyncio.run(main())







