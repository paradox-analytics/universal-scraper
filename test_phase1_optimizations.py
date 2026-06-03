#!/usr/bin/env python3
"""
Test Phase 1 Optimizations (Early Exit)
Tests that early exit is working correctly for JSON and Direct LLM extraction
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

# Test URLs - different scenarios
TEST_URLS = {
    "json_early_exit": {
        "url": "https://www.chewy.com/b/wet-food-389",
        "fields": ["title", "price", "rating", "review count"],
        "expected_source": "json",
        "description": "Chewy.com - Should trigger JSON early exit (high quality JSON)"
    },
    "direct_llm_early_exit": {
        "url": "https://news.ycombinator.com/",
        "fields": ["title", "points", "comments"],
        "expected_source": "direct_llm",
        "description": "Hacker News - Should trigger Direct LLM early exit (if quality ≥ 60%)"
    },
    "html_fallback": {
        "url": "https://example.com",
        "fields": ["title", "description"],
        "expected_source": "html",
        "description": "Simple HTML site - Should fall back to HTML extraction"
    }
}

async def test_early_exit():
    """Test that early exit optimizations are working"""
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("="*80)
    print("🧪 PHASE 1 OPTIMIZATION TEST: Early Exit")
    print("="*80)
    print()
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_direct_llm=True,
        enable_cache=True
    )
    
    results = {}
    
    for test_name, test_config in TEST_URLS.items():
        print(f"\n{'='*80}")
        print(f"📋 Test: {test_name}")
        print(f"   URL: {test_config['url']}")
        print(f"   Fields: {', '.join(test_config['fields'])}")
        print(f"   Expected: {test_config['expected_source']} extraction")
        print(f"   Description: {test_config['description']}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            result = await scraper.scrape(
                url=test_config['url'],
                fields=test_config['fields']
            )
            
            execution_time = time.time() - start_time
            
            # Check results
            items = result.get('data', [])
            metadata = result.get('metadata', {})
            source = result.get('source', 'unknown')
            early_exit = metadata.get('early_exit', False)
            
            print(f"\n✅ Extraction Complete:")
            print(f"   Items extracted: {len(items)}")
            print(f"   Extraction source: {source}")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Early exit: {'✅ YES' if early_exit else '❌ NO'}")
            
            if early_exit:
                print(f"   ⚡ PHASE 1 OPTIMIZATION WORKING: Early exit triggered!")
                print(f"   Time saved: ~{execution_time * 0.3:.1f}-{execution_time * 0.5:.1f}s (estimated)")
            
            # Show sample items
            if items:
                print(f"\n   Sample items (first 3):")
                for i, item in enumerate(items[:3], 1):
                    print(f"   {i}. {json.dumps(item, indent=6, default=str)[:200]}...")
            
            # Validate early exit
            if source == test_config['expected_source']:
                if early_exit:
                    print(f"\n   ✅ PASS: Early exit triggered for {source} extraction")
                else:
                    print(f"\n   ⚠️  WARNING: Expected {test_config['expected_source']} but no early exit")
                    print(f"      (This is OK if quality threshold not met)")
            else:
                print(f"\n   ⚠️  NOTE: Used {source} instead of {test_config['expected_source']}")
            
            results[test_name] = {
                'success': True,
                'items': len(items),
                'source': source,
                'early_exit': early_exit,
                'execution_time': execution_time,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = {
                'success': False,
                'error': str(e)
            }
        
        print(f"\n{'─'*80}\n")
    
    # Summary
    print("="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print()
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.get('success'))
    early_exits = sum(1 for r in results.values() if r.get('early_exit'))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}/{total_tests}")
    print(f"Early exits triggered: {early_exits}/{total_tests}")
    print()
    
    for test_name, result in results.items():
        if result.get('success'):
            status = "✅" if result.get('early_exit') else "⚠️"
            print(f"{status} {test_name}:")
            print(f"   Items: {result['items']}")
            print(f"   Source: {result['source']}")
            print(f"   Time: {result['execution_time']:.2f}s")
            print(f"   Early exit: {'YES' if result['early_exit'] else 'NO'}")
        else:
            print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")
        print()
    
    print("="*80)
    print("💡 EXPECTED BEHAVIOR:")
    print("="*80)
    print("1. JSON extraction should trigger early exit if:")
    print("   - Quality is high (validation confidence > 0.5)")
    print("   - All or most fields are present")
    print()
    print("2. Direct LLM extraction should trigger early exit if:")
    print("   - Quality ≥ 60%")
    print("   - Items extracted > 0")
    print()
    print("3. HTML extraction should NOT trigger early exit:")
    print("   - It's the final fallback")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_early_exit())







