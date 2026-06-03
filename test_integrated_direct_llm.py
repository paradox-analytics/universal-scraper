#!/usr/bin/env python3
"""
Test the integrated DirectLLMExtractor in the main UniversalScraper
Compare with ScrapeGraphAI's output
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper


async def test_integrated_scraper():
    """Test UniversalScraper with DirectLLM enabled"""
    print("\n" + "="*100)
    print("🚀 TESTING INTEGRATED DIRECT LLM EXTRACTION")
    print("="*100)
    print()
    print("Testing UniversalScraper with DirectLLM enabled (like ScrapeGraphAI)")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Test on Hacker News (faster than Amazon)
    url = "https://news.ycombinator.com/"
    fields = ["title", "points", "comments"]
    
    print(f"📋 URL: {url}")
    print(f"📋 Fields: {fields}")
    print()
    
    # Test with different quality modes
    modes = ['conservative', 'balanced', 'aggressive']
    results = {}
    
    for mode in modes:
        print("\n" + "="*100)
        print(f"🎯 MODE: {mode.upper()}")
        print("="*100)
        print()
        
        # Create scraper with DirectLLM enabled
        scraper = UniversalScraper(
            api_key=api_key,
            use_direct_llm=True,
            quality_mode=mode,
            fetch_mode="hybrid",  # Use hybrid (supports all parameters)
            use_camoufox=False,  # Don't need Camoufox for HN
            enable_cache=False,
            enable_llm_pagination=False,  # Disable pagination for this test
            enable_auto_pagination=False
        )
        
        # Scrape
        result = await scraper.scrape(
            url=url,
            fields=fields,
            force_html=True,  # Skip JSON detection for clean test
            scroll_to_bottom=False,  # No need for scrolling on HN
            click_load_more=None,
            wait_for_selector=None
        )
        
        await scraper.close()
        
        # Analyze results
        items = result.get('data', [])
        metadata = result.get('metadata', {})
        source = result.get('source', 'unknown')
        
        print()
        print(f"📊 Results:")
        print(f"   Source: {metadata.get('extraction_source', source)}")
        print(f"   Items: {len(items)}")
        print(f"   Execution time: {metadata.get('execution_time', 0):.2f}s")
        
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
            avg_fill = (total_filled / (len(items) * len(fields))) * 100 if items else 0
            results[mode] = {
                'count': len(items),
                'avg_fill': avg_fill,
                'source': metadata.get('extraction_source', source)
            }
        else:
            print("   ❌ No items extracted")
            results[mode] = {'count': 0, 'avg_fill': 0, 'source': 'failed'}
    
    # Comparison
    print("\n" + "="*100)
    print("📊 COMPARISON WITH SCRAPEGRAPHAI")
    print("="*100)
    print()
    print("| Mode         | Items | Completeness | Source      | vs ScrapeGraphAI |")
    print("|--------------|-------|--------------|-------------|------------------|")
    
    scrapegraphai_count = 30  # From our earlier test
    
    for mode in modes:
        r = results[mode]
        vs_scrapegraphai = f"{r['count']}/{scrapegraphai_count}"
        match = "✅" if r['count'] >= 25 else "⚠️" if r['count'] >= 15 else "❌"
        print(f"| {mode:<12} | {r['count']:>5} | {r['avg_fill']:>12.1f}% | {r['source']:<11} | {match} {vs_scrapegraphai:>14} |")
    
    print()
    print("Goal: Match or exceed ScrapeGraphAI's 30 items with 100% completeness")
    print()
    
    # Best mode
    best_mode = max(results.keys(), key=lambda k: results[k]['count'])
    best = results[best_mode]
    
    if best['count'] >= 25:
        print(f"✅ SUCCESS: {best_mode.upper()} mode extracted {best['count']} items (close to ScrapeGraphAI's 30)")
    elif best['count'] >= 15:
        print(f"⚠️  PARTIAL: {best_mode.upper()} mode extracted {best['count']} items (need to improve)")
    else:
        print(f"❌ FAILED: Best mode only extracted {best['count']} items (much lower than ScrapeGraphAI's 30)")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_integrated_scraper())

