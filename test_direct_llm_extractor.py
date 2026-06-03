#!/usr/bin/env python3
"""
Test DirectLLMExtractor on our failing sources
Validates that direct LLM extraction produces quality data
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def test_source(url: str, fields: List[str], name: str, expected_min: int):
    """Test direct LLM extraction on a single source"""
    print("\n" + "="*100)
    print(f"🧪 TESTING: {name}")
    print("="*100)
    print(f"URL: {url}")
    print(f"Fields: {fields}")
    print(f"Expected: {expected_min}+ items")
    print()
    
    # Fetch HTML
    print("📥 Fetching HTML...")
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes via {result.get('fetch_method')}")
    print()
    
    # Clean HTML
    print("🧹 Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    cleaned_result = cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned: {len(html):,} → {len(cleaned_html):,} bytes ({cleaned_result['reduction_percent']:.1f}% reduction)")
    print()
    
    # Direct LLM extraction
    print("🤖 Direct LLM extraction...")
    api_key = os.environ.get('OPENAI_API_KEY')
    extractor = DirectLLMExtractor(api_key=api_key)
    
    # Estimate cost
    cost = extractor.estimate_cost(len(cleaned_html), len(fields))
    print(f"   Estimated cost: ${cost:.4f}")
    print()
    
    items = await extractor.extract(cleaned_html, fields)
    
    print()
    print("="*100)
    print(f"📊 RESULTS - {name}")
    print("="*100)
    print(f"Items extracted: {len(items)}")
    print()
    
    if items:
        print("Sample items:")
        for i, item in enumerate(items[:3], 1):
            print(f"\nItem {i}:")
            for key, value in item.items():
                value_str = str(value)[:80] if value else "None"
                print(f"  • {key}: {value_str}")
        
        if len(items) > 3:
            print(f"\n... and {len(items) - 3} more items")
        
        # Quality check
        print()
        print("📈 Quality Analysis:")
        
        # Check empty rates
        for field in fields:
            empty_count = sum(1 for item in items if not item.get(field))
            empty_rate = (empty_count / len(items)) * 100
            status = "✅" if empty_rate < 20 else ("⚠️" if empty_rate < 50 else "❌")
            print(f"  {status} {field}: {empty_rate:.1f}% empty ({empty_count}/{len(items)})")
        
        # Check for analytics garbage
        has_garbage = False
        for item in items[:5]:
            for key, value in item.items():
                if value and isinstance(value, str):
                    if any(kw in value.lower() for kw in ['_optimistic_', 'operationid', 'correlation', 'si=', 'c=1,']):
                        has_garbage = True
                        print(f"  ⚠️  Analytics pattern detected: {value[:50]}")
                        break
        
        if not has_garbage:
            print(f"  ✅ No analytics garbage detected")
        
        # Success determination
        print()
        if len(items) >= expected_min:
            print(f"✅ SUCCESS: {len(items)} items (≥ {expected_min} expected)")
        else:
            print(f"⚠️  PARTIAL: {len(items)} items (< {expected_min} expected)")
    else:
        print("❌ FAILED: No items extracted")
    
    print()
    return len(items)


async def main():
    print("\n" + "="*100)
    print("🔬 DIRECT LLM EXTRACTOR TEST - Validating on Failing Sources")
    print("="*100)
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Test on our previously failing sources
    test_cases = [
        {
            "url": "https://www.amazon.com/s?k=laptop",
            "fields": ["product_title", "price", "rating"],
            "name": "Amazon (Previously FAILED - wrong data)",
            "expected_min": 10
        },
        {
            "url": "https://news.ycombinator.com/",
            "fields": ["article_title", "points", "comments_count"],
            "name": "Hacker News (Previously FAILED - 97% empty)",
            "expected_min": 20
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        items_count = await test_source(
            test_case["url"],
            test_case["fields"],
            test_case["name"],
            test_case["expected_min"]
        )
        
        results.append({
            "name": test_case["name"],
            "items": items_count,
            "expected": test_case["expected_min"],
            "success": items_count >= test_case["expected_min"]
        })
        
        # Brief pause between tests
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print("="*100)
    
    successful = sum(1 for r in results if r['success'])
    
    for result in results:
        status = "✅" if result['success'] else "⚠️"
        print(f"{status} {result['name']}: {result['items']}/{result['expected']} items")
    
    print()
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    print()
    
    if successful == len(results):
        print("✅ ALL TESTS PASSED! Direct LLM extraction works!")
        print()
        print("Next step: Implement pattern learning to cache these results")
    else:
        print("⚠️  Some tests need review")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())

