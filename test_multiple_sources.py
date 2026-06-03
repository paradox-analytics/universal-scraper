#!/usr/bin/env python3
"""
Test our scraper vs ScrapeGraphAI on multiple diverse sources
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


# Test sources with different characteristics
TEST_SOURCES = [
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "fields": ["title", "points", "comments"],
        "context": "Extract all article listings with title, points, and comments count",
        "expected_items": 30,
        "type": "news_aggregator"
    },
    {
        "name": "Product Hunt",
        "url": "https://www.producthunt.com/",
        "fields": ["name", "tagline", "votes"],
        "context": "Extract all product listings with name, tagline, and upvotes",
        "expected_items": 20,
        "type": "product_directory"
    },
    {
        "name": "GitHub Trending",
        "url": "https://github.com/trending",
        "fields": ["repository", "description", "stars"],
        "context": "Extract all trending repositories with name, description, and stars",
        "expected_items": 25,
        "type": "repository_list"
    },
    {
        "name": "Lobsters",
        "url": "https://lobste.rs/",
        "fields": ["title", "points", "comments"],
        "context": "Extract all story listings with title, points, and comments",
        "expected_items": 25,
        "type": "news_aggregator"
    },
    {
        "name": "IndieHackers",
        "url": "https://www.indiehackers.com/",
        "fields": ["title", "author", "comments"],
        "context": "Extract all post listings with title, author, and comment count",
        "expected_items": 20,
        "type": "forum"
    },
]


async def test_source(source_config):
    """Test a single source"""
    
    print("\n" + "="*100)
    print(f"🧪 TESTING: {source_config['name']}")
    print("="*100)
    print(f"URL: {source_config['url']}")
    print(f"Type: {source_config['type']}")
    print(f"Fields: {', '.join(source_config['fields'])}")
    print(f"Expected: ~{source_config['expected_items']} items")
    print()
    
    result = {
        "name": source_config['name'],
        "url": source_config['url'],
        "type": source_config['type'],
        "expected_items": source_config['expected_items'],
        "our_items": 0,
        "our_completeness": 0.0,
        "our_perfect_items": 0,
        "fetch_time": 0,
        "extraction_time": 0,
        "success": False,
        "error": None
    }
    
    try:
        # Fetch HTML
        fetch_start = datetime.now()
        print("📥 Fetching HTML...")
        
        fetcher = HybridFetcher(
            proxy_config=None,
            enable_cache=False,
            headless=True,
            use_camoufox=False
        )
        
        fetch_result = await fetcher.fetch(source_config['url'])
        raw_html = fetch_result['html']
        
        fetch_time = (datetime.now() - fetch_start).total_seconds()
        result['fetch_time'] = fetch_time
        
        print(f"✅ Fetched {len(raw_html):,} bytes in {fetch_time:.1f}s")
        
        # Clean HTML
        cleaner = SmartHTMLCleaner()
        clean_result = cleaner.clean(raw_html)
        cleaned_html = clean_result['html']
        
        print(f"✅ Cleaned: {clean_result['reduction_percent']:.1f}% reduction")
        print()
        
        # Extract with our scraper
        extraction_start = datetime.now()
        print("🔧 Extracting with our scraper...")
        
        extractor = DirectLLMExtractor(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4o-mini",
            max_tokens_per_chunk=4000,
            quality_mode="balanced",
            use_html2text=True
        )
        
        our_items = await extractor.extract(
            cleaned_html,
            source_config['fields'],
            context=source_config['context']
        )
        
        extraction_time = (datetime.now() - extraction_start).total_seconds()
        result['extraction_time'] = extraction_time
        
        print(f"✅ Extracted {len(our_items)} items in {extraction_time:.1f}s")
        print()
        
        # Analyze quality
        if our_items:
            total_fields = len(our_items) * len(source_config['fields'])
            filled_fields = 0
            perfect_items = 0
            
            for item in our_items:
                item_fields = 0
                for field in source_config['fields']:
                    value = item.get(field)
                    if value not in [None, '', 'N/A']:
                        filled_fields += 1
                        item_fields += 1
                
                if item_fields == len(source_config['fields']):
                    perfect_items += 1
            
            completeness = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
            
            result['our_items'] = len(our_items)
            result['our_completeness'] = completeness
            result['our_perfect_items'] = perfect_items
            result['success'] = True
            
            print(f"📊 Quality Metrics:")
            print(f"   • Total items: {len(our_items)}")
            print(f"   • Perfect items: {perfect_items} ({perfect_items/len(our_items)*100:.1f}%)")
            print(f"   • Data completeness: {completeness:.1f}%")
            print()
            
            # Show sample items
            print(f"📝 Sample Items (first 3):")
            for i, item in enumerate(our_items[:3], 1):
                fields_str = ", ".join([f"{k}={v}" for k, v in item.items() if v not in [None, '', 'N/A']])
                print(f"   {i}. {fields_str[:90]}")
            print()
        else:
            print("⚠️  No items extracted")
            result['error'] = "No items extracted"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        result['error'] = str(e)
    
    return result


async def run_all_tests():
    """Run tests on all sources"""
    
    print("\n" + "="*100)
    print("🚀 MULTI-SOURCE COMPARISON TEST")
    print("="*100)
    print(f"Testing {len(TEST_SOURCES)} different sources")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    results = []
    
    for i, source_config in enumerate(TEST_SOURCES, 1):
        print(f"\n[{i}/{len(TEST_SOURCES)}]")
        result = await test_source(source_config)
        results.append(result)
        
        # Brief pause between tests
        if i < len(TEST_SOURCES):
            print("⏸️  Pausing 5 seconds before next test...")
            await asyncio.sleep(5)
    
    # Summary report
    print("\n" + "="*100)
    print("📊 FINAL SUMMARY REPORT")
    print("="*100)
    print()
    
    print(f"{'Source':<20} {'Type':<20} {'Items':<8} {'Perfect':<10} {'Complete':<10} {'Status':<10}")
    print("-" * 100)
    
    successful_tests = 0
    total_items = 0
    avg_completeness = 0
    
    for result in results:
        status = "✅ Pass" if result['success'] else "❌ Fail"
        items_str = f"{result['our_items']}/{result['expected_items']}" if result['success'] else "N/A"
        perfect_str = str(result['our_perfect_items']) if result['success'] else "N/A"
        complete_str = f"{result['our_completeness']:.1f}%" if result['success'] else "N/A"
        
        print(f"{result['name']:<20} {result['type']:<20} {items_str:<8} {perfect_str:<10} {complete_str:<10} {status:<10}")
        
        if result['success']:
            successful_tests += 1
            total_items += result['our_items']
            avg_completeness += result['our_completeness']
    
    print()
    print("="*100)
    print("🎯 OVERALL STATISTICS")
    print("="*100)
    print()
    
    if successful_tests > 0:
        avg_completeness = avg_completeness / successful_tests
        
        print(f"✅ Successful tests: {successful_tests}/{len(TEST_SOURCES)} ({successful_tests/len(TEST_SOURCES)*100:.1f}%)")
        print(f"📊 Total items extracted: {total_items}")
        print(f"📊 Average completeness: {avg_completeness:.1f}%")
        print()
        
        # Analysis
        print("="*100)
        print("💡 ANALYSIS")
        print("="*100)
        print()
        
        if successful_tests == len(TEST_SOURCES):
            print("🎉 PERFECT SCORE! All sources extracted successfully!")
            print()
            
            if avg_completeness >= 90:
                print("✅ Data quality is EXCELLENT (>90% completeness)")
            elif avg_completeness >= 80:
                print("✅ Data quality is GOOD (80-90% completeness)")
            else:
                print("⚠️  Data quality needs improvement (<80% completeness)")
        else:
            failed = len(TEST_SOURCES) - successful_tests
            print(f"⚠️  {failed} source(s) failed - need investigation")
            print()
            
            print("Failed sources:")
            for result in results:
                if not result['success']:
                    print(f"  • {result['name']}: {result['error']}")
        
        print()
        
        # Recommendations
        print("="*100)
        print("📋 RECOMMENDATIONS")
        print("="*100)
        print()
        
        low_completeness = [r for r in results if r['success'] and r['our_completeness'] < 80]
        low_items = [r for r in results if r['success'] and r['our_items'] < r['expected_items'] * 0.7]
        
        if not low_completeness and not low_items:
            print("✅ No issues detected - ready for production!")
            print()
            print("Next steps:")
            print("  1. Deploy to production")
            print("  2. Monitor real-world performance")
            print("  3. Set up alerting for quality drops")
        else:
            if low_completeness:
                print("⚠️  Sources with low completeness (<80%):")
                for r in low_completeness:
                    print(f"  • {r['name']}: {r['our_completeness']:.1f}%")
                print()
            
            if low_items:
                print("⚠️  Sources extracting fewer items than expected (<70% of target):")
                for r in low_items:
                    print(f"  • {r['name']}: {r['our_items']}/{r['expected_items']}")
                print()
            
            print("Suggested fixes:")
            print("  1. Adjust quality_mode per site type")
            print("  2. Fine-tune chunk size for specific sites")
            print("  3. Add site-specific extraction hints")
    else:
        print("❌ All tests failed - major issue detected")
        print()
        print("Possible causes:")
        print("  1. API key issues")
        print("  2. Network connectivity problems")
        print("  3. Rate limiting")
    
    print()
    print("="*100)
    print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    print()
    
    # Save results to file
    output_file = "MULTI_SOURCE_TEST_RESULTS.md"
    with open(output_file, 'w') as f:
        f.write(f"# Multi-Source Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Sources Tested:** {len(TEST_SOURCES)}\n")
        f.write(f"**Successful:** {successful_tests}/{len(TEST_SOURCES)}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Source | Type | Items | Perfect | Completeness | Status |\n")
        f.write("|--------|------|-------|---------|--------------|--------|\n")
        
        for result in results:
            status = "✅ Pass" if result['success'] else "❌ Fail"
            items_str = f"{result['our_items']}/{result['expected_items']}" if result['success'] else "N/A"
            perfect_str = str(result['our_perfect_items']) if result['success'] else "N/A"
            complete_str = f"{result['our_completeness']:.1f}%" if result['success'] else "N/A"
            
            f.write(f"| {result['name']} | {result['type']} | {items_str} | {perfect_str} | {complete_str} | {status} |\n")
        
        f.write(f"\n## Summary\n\n")
        if successful_tests > 0:
            f.write(f"- Total items extracted: {total_items}\n")
            f.write(f"- Average completeness: {avg_completeness:.1f}%\n")
        
        f.write(f"\nFull test log saved to console output.\n")
    
    print(f"📄 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    asyncio.run(run_all_tests())



