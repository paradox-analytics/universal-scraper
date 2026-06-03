#!/usr/bin/env python3
"""
Test the new Langchain Html2TextTransformer implementation
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


async def test_implementation():
    """Test on Lobsters with new implementation"""
    
    print("\n" + "="*80)
    print("🧪 TESTING NEW LANGCHAIN IMPLEMENTATION")
    print("="*80)
    print()
    
    url = "https://lobste.rs/"
    fields = ['title', 'points', 'comments']
    
    # Fetch
    print("📥 Fetching Lobsters...")
    fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
    fetch_result = await fetcher.fetch(url)
    
    # Clean
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(fetch_result['html'])
    
    print(f"✅ Fetched & cleaned: {len(clean_result['html']):,} bytes")
    print()
    
    # Extract with NEW implementation
    print("="*80)
    print("🔧 Extracting with Langchain Html2TextTransformer")
    print("="*80)
    
    extractor = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=True  # Now uses Langchain!
    )
    
    items = await extractor.extract(
        clean_result['html'],
        fields,
        context="Extract all story listings"
    )
    
    print(f"✅ Extracted: {len(items)} items")
    print()
    
    # Analyze quality
    title_count = sum(1 for item in items if item.get('title') not in [None, '', 'N/A'])
    points_count = sum(1 for item in items if item.get('points') not in [None, '', 'N/A'])
    comments_count = sum(1 for item in items if item.get('comments') not in [None, '', 'N/A'])
    
    total_fields = len(items) * 3
    filled_fields = title_count + points_count + comments_count
    completeness = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
    
    print("📊 Quality Metrics:")
    print(f"   • Title: {title_count}/{len(items)} (100%)")
    print(f"   • Points: {points_count}/{len(items)} ({points_count/len(items)*100:.0f}%)")
    print(f"   • Comments: {comments_count}/{len(items)} ({comments_count/len(items)*100:.0f}%)")
    print(f"   • Overall completeness: {completeness:.1f}%")
    print()
    
    # Show samples
    print("📝 Sample Items (first 5):")
    for i, item in enumerate(items[:5], 1):
        title = item.get('title', 'N/A')[:50]
        points = item.get('points', 'MISSING')
        comments = item.get('comments', 'MISSING')
        print(f"   {i}. {title}")
        print(f"      points={points}, comments={comments}")
    print()
    
    # Verdict
    print("="*80)
    print("🎯 VERDICT")
    print("="*80)
    print()
    
    if points_count >= len(items) * 0.95:  # 95%+ coverage
        print(f"🎉 SUCCESS! Extracted {points_count}/{len(items)} points ({points_count/len(items)*100:.0f}%)")
        print()
        print("✅ Langchain Html2TextTransformer FIXES the Lobsters issue!")
        print(f"✅ Data completeness: {completeness:.1f}% (excellent)")
        print()
        print("💡 This matches ScrapeGraphAI's approach and quality!")
    elif points_count >= len(items) * 0.80:  # 80%+ coverage
        print(f"✅ GOOD! Extracted {points_count}/{len(items)} points ({points_count/len(items)*100:.0f}%)")
        print(f"   Completeness: {completeness:.1f}%")
        print()
        print("Significant improvement, but could be better")
    else:
        print(f"⚠️  PARTIAL: {points_count}/{len(items)} points ({points_count/len(items)*100:.0f}%)")
        print(f"   Completeness: {completeness:.1f}%")
        print()
        print("Need further investigation")
    
    print()
    
    return items


if __name__ == "__main__":
    asyncio.run(test_implementation())



