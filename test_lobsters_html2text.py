#!/usr/bin/env python3
"""
Test if HTML-to-text is causing the Lobsters issue
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor


async def test_html2text_impact():
    """Test if html2text is the problem"""
    
    print("\n" + "="*80)
    print("🧪 Testing HTML-to-Text Impact on Lobsters")
    print("="*80)
    print()
    
    url = "https://lobste.rs/"
    
    # Fetch
    fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
    fetch_result = await fetcher.fetch(url)
    
    # Clean
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(fetch_result['html'])
    cleaned_html = clean_result['html']
    
    print(f"✅ Fetched & cleaned: {len(cleaned_html):,} bytes\n")
    
    # Test 1: WITH html2text (current)
    print("="*80)
    print("Test 1: WITH HTML-to-Text Conversion (current)")
    print("="*80)
    
    extractor_with = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=True  # ENABLED
    )
    
    items_with = await extractor_with.extract(
        cleaned_html,
        ['title', 'points', 'comments'],
        context="Extract all story listings with title, points/score, and comments"
    )
    
    points_with = sum(1 for item in items_with if item.get('points') not in [None, '', 'N/A'])
    comments_with = sum(1 for item in items_with if item.get('comments') not in [None, '', 'N/A'])
    
    print(f"✅ Extracted: {len(items_with)} items")
    print(f"   Points coverage: {points_with}/{len(items_with)} ({points_with/len(items_with)*100:.0f}%)")
    print(f"   Comments coverage: {comments_with}/{len(items_with)} ({comments_with/len(items_with)*100:.0f}%)")
    print()
    
    print("Samples:")
    for i, item in enumerate(items_with[:3], 1):
        print(f"  {i}. {item.get('title', '')[:50]}")
        print(f"     points={item.get('points')}, comments={item.get('comments')}")
    print()
    
    # Test 2: WITHOUT html2text
    print("="*80)
    print("Test 2: WITHOUT HTML-to-Text Conversion (raw HTML)")
    print("="*80)
    
    extractor_without = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=False  # DISABLED
    )
    
    items_without = await extractor_without.extract(
        cleaned_html,
        ['title', 'points', 'comments'],
        context="Extract all story listings with title, points/score, and comments"
    )
    
    points_without = sum(1 for item in items_without if item.get('points') not in [None, '', 'N/A'])
    comments_without = sum(1 for item in items_without if item.get('comments') not in [None, '', 'N/A'])
    
    print(f"✅ Extracted: {len(items_without)} items")
    print(f"   Points coverage: {points_without}/{len(items_without)} ({points_without/len(items_without)*100:.0f}%)")
    print(f"   Comments coverage: {comments_without}/{len(items_without)} ({comments_without/len(items_without)*100:.0f}%)")
    print()
    
    print("Samples:")
    for i, item in enumerate(items_without[:3], 1):
        print(f"  {i}. {item.get('title', '')[:50]}")
        print(f"     points={item.get('points')}, comments={item.get('comments')}")
    print()
    
    # Comparison
    print("="*80)
    print("📊 COMPARISON")
    print("="*80)
    print()
    
    print(f"{'Method':<30} {'Items':<10} {'Points Coverage':<20} {'Comments Coverage':<20}")
    print("-" * 80)
    print(f"{'WITH html2text':<30} {len(items_with):<10} {points_with}/{len(items_with)} ({points_with/len(items_with)*100:.0f}%){'':<7} {comments_with}/{len(items_with)} ({comments_with/len(items_with)*100:.0f}%)")
    print(f"{'WITHOUT html2text':<30} {len(items_without):<10} {points_without}/{len(items_without)} ({points_without/len(items_without)*100:.0f}%){'':<7} {comments_without}/{len(items_without)} ({comments_without/len(items_without)*100:.0f}%)")
    print()
    
    # Verdict
    print("="*80)
    print("🎯 VERDICT")
    print("="*80)
    print()
    
    improvement_points = points_without - points_with
    improvement_comments = comments_without - comments_with
    
    if improvement_points > 5 or improvement_comments > 5:
        print("🔴 HTML-to-text IS causing the problem!")
        print(f"   • WITHOUT html2text: {points_without} items have points")
        print(f"   • WITH html2text: {points_with} items have points")
        print(f"   • Improvement: +{improvement_points} items")
        print()
        print("💡 SOLUTION: Disable html2text for Lobsters")
        print("   OR improve html2text conversion to preserve structure")
    elif improvement_points < -5 or improvement_comments < -5:
        print("🔵 HTML-to-text is HELPING!")
        print(f"   • WITH html2text: {points_with} items have points (better)")
        print(f"   • WITHOUT html2text: {points_without} items have points")
        print()
        print("💡 The problem is elsewhere (not html2text)")
    else:
        print("⚪ HTML-to-text makes little difference")
        print(f"   • WITH: {points_with} items, WITHOUT: {points_without} items")
        print()
        print("💡 The problem is likely:")
        print("   • HTML structure is unusual (score in <a class='upvoter'>)")
        print("   • LLM can't find it regardless of format")
        print("   • Need better prompt or field mapping")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_html2text_impact())



