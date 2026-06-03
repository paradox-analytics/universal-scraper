#!/usr/bin/env python3
"""
Quick test of quality modes using Hacker News (faster than Amazon)
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def test_quality_modes_quick():
    """Quick test on Hacker News"""
    print("\n" + "="*100)
    print("🔬 QUALITY MODE QUICK TEST - Hacker News")
    print("="*100)
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Use Hacker News (much faster than Amazon)
    url = "https://news.ycombinator.com/"
    fields = ["title", "points", "comments"]
    
    print(f"📥 Fetching: {url}")
    print("⏱️  This should be quick (5-10 seconds)...")
    print()
    
    # Fetch HTML
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=False, enable_cache=False)
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes")
    
    # Clean HTML
    cleaner = SmartHTMLCleaner()
    cleaned_result = cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned to {len(cleaned_html):,} bytes")
    print()
    
    # Test each quality mode
    modes = ['conservative', 'balanced', 'aggressive']
    results = {}
    
    for mode in modes:
        print("-" * 100)
        print(f"🎯 Mode: {mode.upper()}")
        print()
        
        extractor = DirectLLMExtractor(api_key=api_key, quality_mode=mode)
        items = await extractor.extract(cleaned_html, fields)
        
        print(f"   Items: {len(items)}")
        
        if items:
            # Calculate completeness
            for field in fields:
                filled = sum(1 for item in items if item.get(field) and str(item.get(field)).strip())
                fill_rate = (filled / len(items)) * 100
                print(f"   {field}: {fill_rate:.1f}% filled")
            
            # Show sample
            print()
            print("   Sample item:")
            item = items[0]
            for key, value in item.items():
                value_type = type(value).__name__
                print(f"     • {key} ({value_type}): {value}")
            
            # Store results
            total_filled = sum(
                1 for item in items
                for field in fields
                if item.get(field) and str(item.get(field)).strip()
            )
            avg_fill = (total_filled / (len(items) * len(fields))) * 100
            results[mode] = {
                'count': len(items),
                'avg_fill': avg_fill
            }
        else:
            results[mode] = {'count': 0, 'avg_fill': 0}
        
        print()
    
    # Summary
    print("\n" + "="*100)
    print("📊 COMPARISON SUMMARY")
    print("="*100)
    print()
    print("| Mode         | Items | Avg Fill Rate | Quality Score |")
    print("|--------------|-------|---------------|---------------|")
    
    for mode in modes:
        r = results[mode]
        quality_score = (r['avg_fill'] / 100) * r['count']
        print(f"| {mode:<12} | {r['count']:>5} | {r['avg_fill']:>13.1f}% | {quality_score:>13.1f} |")
    
    print()
    print("Key Insights:")
    print()
    print("✅ CONSERVATIVE: Like ScrapeGraphAI - fewer items, highest quality")
    print("⚖️  BALANCED: Default - good compromise")
    print("🚀 AGGRESSIVE: Maximum extraction - most items")
    print()
    
    # Show difference vs ScrapeGraphAI
    print("Comparison with ScrapeGraphAI test:")
    print(f"  • ScrapeGraphAI (HN): 30 items, 100% completeness")
    print(f"  • Our conservative: {results['conservative']['count']} items, {results['conservative']['avg_fill']:.1f}% completeness")
    print(f"  • Our balanced: {results['balanced']['count']} items, {results['balanced']['avg_fill']:.1f}% completeness")
    print()


if __name__ == "__main__":
    asyncio.run(test_quality_modes_quick())



