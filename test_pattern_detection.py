#!/usr/bin/env python3
"""
Test Pattern Detection System - NO HARDCODING
Uses LLM to detect if site uses attribute-based or nested element extraction
"""
import asyncio
import os
import sys
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.pattern_detector import PatternDetector
from universal_scraper.core.attribute_extractor import AttributeExtractor

async def test_pattern_detection(url: str, context: str):
    """
    Test the scalable pattern detection approach:
    1. Detect pattern using LLM (no hardcoding!)
    2. Route to appropriate extractor
    3. Extract data
    """
    print("\n" + "="*80)
    print("🔍 PATTERN DETECTION TEST - LLM-Based (Scalable)")
    print("="*80)
    print(f"\nURL: {url}")
    print(f"Context: {context}\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    # Step 1: Fetch HTML
    print("📥 Step 1: Fetching HTML...")
    fetcher = HybridFetcher(headless=True)
    result = await fetcher.fetch(url, wait_for_selector="shreddit-post")
    html = result['html']
    print(f"✅ Fetched {len(html)} bytes")
    
    # Step 2: Clean HTML
    print("\n🧹 Step 2: Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(html)
    cleaned = clean_result['html']
    print(f"✅ Cleaned to {len(cleaned)} bytes")
    
    # Step 3: LLM-based Pattern Detection (NO HARDCODING!)
    print("\n🤖 Step 3: LLM Pattern Detection (analyzing structure)...")
    detector = PatternDetector(api_key=api_key)
    pattern = detector.detect_pattern(cleaned, url)
    
    print(f"\n📊 Detection Result:")
    print(f"   Strategy: {pattern['strategy']}")
    print(f"   Confidence: {pattern['confidence']:.2f}")
    print(f"   Reasoning: {pattern['reasoning']}")
    if pattern.get('element_name'):
        print(f"   Element: {pattern['element_name']}")
        print(f"   Key Attributes: {pattern.get('key_attributes', [])[:10]}")
    
    # Step 4: Route to appropriate extractor
    print(f"\n⚡ Step 4: Extracting with {pattern['strategy']} strategy...")
    
    fields = ['title', 'author', 'upvotes', 'comments_count']
    
    if pattern['strategy'] == 'attributes':
        # Use attribute extraction (fast, reliable, no AI code generation!)
        print("   → Using AttributeExtractor (no AI code generation needed)")
        extractor = AttributeExtractor()
        items = extractor.extract(html, fields, pattern)
    else:
        # Use traditional AI code generation
        print("   → Would use AI code generation for nested elements")
        items = []  # Not implemented in this test
    
    # Step 5: Show results
    print(f"\n📊 Results:")
    print(f"✅ Extracted {len(items)} items")
    
    if items:
        print(f"\n📝 First 3 items:")
        for i, item in enumerate(items[:3], 1):
            print(f"\n{i}. {json.dumps(item, indent=2)}")
        
        # Validate data quality
        first_item = items[0]
        has_title = bool(first_item.get('title'))
        has_author = bool(first_item.get('author'))
        has_upvotes = first_item.get('upvotes') is not None
        has_comments = first_item.get('comments_count') is not None
        
        print(f"\n✅ Data Quality Check:")
        print(f"   Title: {'✅' if has_title else '❌'}")
        print(f"   Author: {'✅' if has_author else '❌'}")
        print(f"   Upvotes: {'✅' if has_upvotes else '❌'}")
        print(f"   Comments: {'✅' if has_comments else '❌'}")
        
        if all([has_title, has_author, has_upvotes, has_comments]):
            print(f"\n🎉 SUCCESS: All fields extracted correctly!")
        else:
            print(f"\n⚠️  PARTIAL: Some fields missing")
    else:
        print("\n❌ FAIL: No items extracted")
    
    print("\n" + "="*80 + "\n")
    
    await fetcher.close()

async def main():
    # Test on Reddit (attribute-based site)
    await test_pattern_detection(
        url="https://www.reddit.com/r/webscraping/",
        context="Extract Reddit posts with title, author, upvotes, comments count"
    )

if __name__ == "__main__":
    asyncio.run(main())







