#!/usr/bin/env python3
"""
Quick multi-source test - 3 diverse sites
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor


# 3 diverse test sources (fast, reliable sites)
TEST_SOURCES = [
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "fields": ["title", "points", "comments"],
        "context": "Extract all article listings",
        "expected_items": 30,
        "type": "news_aggregator"
    },
    {
        "name": "Lobsters",
        "url": "https://lobste.rs/",
        "fields": ["title", "points", "comments"],
        "context": "Extract all story listings",
        "expected_items": 25,
        "type": "news_aggregator"
    },
    {
        "name": "GitHub Trending",
        "url": "https://github.com/trending",
        "fields": ["repository", "description", "stars"],
        "context": "Extract all trending repositories",
        "expected_items": 25,
        "type": "repository_list"
    },
]


async def test_source_quick(source_config):
    """Quick test of a single source"""
    
    print(f"\n{'='*80}")
    print(f"🧪 {source_config['name']} ({source_config['type']})")
    print(f"{'='*80}")
    
    result = {
        "name": source_config['name'],
        "type": source_config['type'],
        "items": 0,
        "completeness": 0.0,
        "perfect_items": 0,
        "success": False,
        "error": None
    }
    
    try:
        # Fetch
        fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
        fetch_result = await fetcher.fetch(source_config['url'])
        
        # Clean
        cleaner = SmartHTMLCleaner()
        clean_result = cleaner.clean(fetch_result['html'])
        
        print(f"📥 Fetched & cleaned: {len(clean_result['html']):,} bytes")
        
        # Extract
        extractor = DirectLLMExtractor(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4o-mini",
            max_tokens_per_chunk=4000,
            quality_mode="balanced",
            use_html2text=True
        )
        
        items = await extractor.extract(
            clean_result['html'],
            source_config['fields'],
            context=source_config['context']
        )
        
        # Analyze
        if items:
            total_fields = len(items) * len(source_config['fields'])
            filled_fields = sum(
                1 for item in items 
                for field in source_config['fields']
                if item.get(field) not in [None, '', 'N/A']
            )
            perfect_items = sum(
                1 for item in items
                if all(item.get(f) not in [None, '', 'N/A'] for f in source_config['fields'])
            )
            
            completeness = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
            
            result['items'] = len(items)
            result['completeness'] = completeness
            result['perfect_items'] = perfect_items
            result['success'] = True
            
            print(f"✅ Extracted: {len(items)} items")
            print(f"   Perfect: {perfect_items} ({perfect_items/len(items)*100:.0f}%)")
            print(f"   Complete: {completeness:.1f}%")
            
            # Show 2 sample items
            print(f"\n   Samples:")
            for i, item in enumerate(items[:2], 1):
                sample = ", ".join([f"{k}={str(v)[:30]}" for k, v in item.items() if v not in [None, '', 'N/A']])
                print(f"   {i}. {sample[:70]}")
        else:
            result['error'] = "No items extracted"
            print(f"❌ No items extracted")
        
    except Exception as e:
        result['error'] = str(e)[:100]
        print(f"❌ Error: {str(e)[:100]}")
    
    return result


async def run_quick_tests():
    """Run quick tests on 3 sources"""
    
    print("\n" + "="*80)
    print("🚀 QUICK MULTI-SOURCE TEST (3 Sites)")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    results = []
    
    for i, source in enumerate(TEST_SOURCES, 1):
        print(f"\n[{i}/{len(TEST_SOURCES)}]", end=" ")
        result = await test_source_quick(source)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    total_items = sum(r['items'] for r in results if r['success'])
    avg_completeness = sum(r['completeness'] for r in results if r['success']) / successful if successful > 0 else 0
    
    print(f"\n{'Source':<20} {'Items':<8} {'Perfect':<10} {'Complete':<12} {'Status':<8}")
    print("-" * 80)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        items = str(r['items']) if r['success'] else "N/A"
        perfect = str(r['perfect_items']) if r['success'] else "N/A"
        complete = f"{r['completeness']:.1f}%" if r['success'] else "N/A"
        
        print(f"{r['name']:<20} {items:<8} {perfect:<10} {complete:<12} {status:<8}")
    
    print()
    print(f"✅ Success rate: {successful}/{len(TEST_SOURCES)} ({successful/len(TEST_SOURCES)*100:.0f}%)")
    print(f"📊 Total items: {total_items}")
    print(f"📊 Avg completeness: {avg_completeness:.1f}%")
    
    # Verdict
    print(f"\n{'='*80}")
    print("🎯 VERDICT")
    print(f"{'='*80}\n")
    
    if successful == len(TEST_SOURCES) and avg_completeness >= 90:
        print("🎉 EXCELLENT! All sources work with high quality.")
        print("   ✅ Ready for production deployment")
    elif successful == len(TEST_SOURCES) and avg_completeness >= 80:
        print("✅ GOOD! All sources work with acceptable quality.")
        print("   ✅ Ready for production with monitoring")
    elif successful >= len(TEST_SOURCES) * 0.7:
        print("⚠️  PARTIAL SUCCESS. Most sources work.")
        print("   💡 Investigate failed sources before full deployment")
    else:
        print("❌ NEEDS WORK. Too many failures.")
        print("   🔧 Debug issues before deployment")
    
    print()


if __name__ == "__main__":
    asyncio.run(run_quick_tests())



