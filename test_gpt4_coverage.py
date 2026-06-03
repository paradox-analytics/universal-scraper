#!/usr/bin/env python3
"""
Test if GPT-4 can achieve 100% coverage (all 30 items)
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


async def test_gpt4():
    """Test GPT-4 vs GPT-4o-mini on same HTML"""
    
    print("\n" + "="*100)
    print("🧪 GPT-4 vs GPT-4o-mini Coverage Test")
    print("="*100)
    print()
    
    url = "https://news.ycombinator.com/"
    fields = ["title", "points", "comments"]
    
    # Fetch HTML once
    print("📥 Fetching HTML...")
    fetcher = HybridFetcher(
        proxy_config=None,
        enable_cache=False,
        headless=True,
        use_camoufox=False
    )
    
    fetch_result = await fetcher.fetch(url)
    raw_html = fetch_result['html']
    
    # Clean HTML
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(raw_html)
    cleaned_html = clean_result['html']
    
    print(f"✅ HTML prepared: {len(cleaned_html):,} bytes")
    print()
    
    # Test with GPT-4o-mini
    print("="*100)
    print("🤖 Testing GPT-4o-mini")
    print("="*100)
    
    extractor_mini = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        quality_mode="balanced"
    )
    
    items_mini = await extractor_mini.extract(
        cleaned_html,
        fields,
        context="Extract all article listings with title, points, and comments count"
    )
    
    print(f"✅ GPT-4o-mini extracted: {len(items_mini)} items")
    print()
    
    # Show sample
    if items_mini:
        print("Sample (first 3):")
        for i, item in enumerate(items_mini[:3], 1):
            print(f"  {i}. {item.get('title', 'N/A')[:60]}")
            print(f"     Points: {item.get('points')}, Comments: {item.get('comments')}")
        print()
    
    # Test with GPT-4
    print("="*100)
    print("🚀 Testing GPT-4 (Full Model)")
    print("="*100)
    print("⚠️  This will cost ~$0.10 for this test")
    print()
    
    extractor_gpt4 = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o",  # GPT-4 Omni (latest, cheaper than gpt-4)
        quality_mode="balanced"
    )
    
    items_gpt4 = await extractor_gpt4.extract(
        cleaned_html,
        fields,
        context="Extract all article listings with title, points, and comments count"
    )
    
    print(f"✅ GPT-4 extracted: {len(items_gpt4)} items")
    print()
    
    # Show sample
    if items_gpt4:
        print("Sample (first 3):")
        for i, item in enumerate(items_gpt4[:3], 1):
            print(f"  {i}. {item.get('title', 'N/A')[:60]}")
            print(f"     Points: {item.get('points')}, Comments: {item.get('comments')}")
        print()
    
    # Comparison
    print("="*100)
    print("📊 COMPARISON")
    print("="*100)
    print()
    
    print(f"{'Model':<20} {'Items':<10} {'Coverage':<12} {'Cost (1K pages)'}")
    print("-" * 70)
    print(f"{'GPT-4o-mini':<20} {len(items_mini):<10} {len(items_mini)/30*100:>5.1f}%     ~$0.50")
    print(f"{'GPT-4o':<20} {len(items_gpt4):<10} {len(items_gpt4)/30*100:>5.1f}%     ~$5.00")
    print(f"{'Target (HN)':<20} {'30':<10} {'100.0%':<12} {'N/A'}")
    print()
    
    # Verdict
    print("="*100)
    print("🎯 VERDICT")
    print("="*100)
    print()
    
    if len(items_gpt4) >= 30:
        print("✅ SUCCESS! GPT-4 achieves 100% coverage")
        print(f"   Upgrade path confirmed: {len(items_mini)}/30 → {len(items_gpt4)}/30")
        print()
        print("💡 Recommendation:")
        print("   • Use gpt-4o-mini by default (cheap, 77% coverage)")
        print("   • Offer gpt-4o upgrade for 100% coverage needs")
        print("   • Cost difference: $0.50 → $5.00 (10x more, but still cheaper than ScrapeGraphAI)")
    elif len(items_gpt4) > len(items_mini):
        print(f"✅ IMPROVEMENT! GPT-4 gets {len(items_gpt4)}/30 (vs {len(items_mini)}/30)")
        print(f"   Coverage improved by {len(items_gpt4) - len(items_mini)} items")
    else:
        print(f"⚠️  NO IMPROVEMENT. GPT-4 also gets {len(items_gpt4)}/30")
        print("   Root cause is NOT model capability")
        print("   Need to investigate ScrapeGraphAI's preprocessing")
    
    print()
    
    return {
        'mini_count': len(items_mini),
        'gpt4_count': len(items_gpt4),
        'mini_items': items_mini,
        'gpt4_items': items_gpt4
    }


if __name__ == "__main__":
    result = asyncio.run(test_gpt4())



