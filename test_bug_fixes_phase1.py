#!/usr/bin/env python3
"""
Test Bug Fixes Phase 1 & 2
Tests the implemented fixes for:
- Null value extraction (Priority 1)
- Enhanced anti-detection (Priority 2)
"""

import asyncio
import os
from pathlib import Path
import logging
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


# Test cases for previously failing sites
TEST_CASES = [
    {
        'name': 'Craigslist',
        'url': 'https://sfbay.craigslist.org/search/sss?query=laptop',
        'fields': ['title', 'price', 'location'],
        'expected_items': 50,  # Should get many items
        'context': 'Extract classified ads with title, price, and location',
        'issue': 'NULL VALUES - All fields were None despite finding items'
    },
    {
        'name': 'TechCrunch',
        'url': 'https://techcrunch.com/',
        'fields': ['title', 'author', 'date', 'category'],
        'expected_items': 10,
        'context': 'Extract news articles with title, author, date, and category',
        'issue': 'NULL VALUES - All fields were None'
    },
    {
        'name': 'Medium',
        'url': 'https://medium.com/tag/technology',
        'fields': ['title', 'author', 'reading_time'],
        'expected_items': 5,
        'context': 'Extract articles with title, author, and reading time',
        'issue': 'SINGLE ITEM - Only extracted 1 item when multiple visible'
    },
    {
        'name': 'Etsy',
        'url': 'https://www.etsy.com/search?q=laptop',
        'fields': ['title', 'price', 'seller'],
        'expected_items': 10,
        'context': 'Extract product listings with title, price, and seller',
        'issue': 'ANTI-BOT - 403 Forbidden'
    },
]


async def test_site(scraper: UniversalScraper, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single site
    
    Returns:
        Dict with test results
    """
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Issue: {test_case['issue']}")
    print(f"URL: {test_case['url']}")
    print(f"Fields: {', '.join(test_case['fields'])}")
    print()
    
    try:
        result = await scraper.scrape(
            url=test_case['url'],
            fields=test_case['fields']
        )
        
        data = result['data']
        
        # Analyze results
        null_value_analysis = analyze_null_values(data, test_case['fields'])
        
        print(f"\n✅ RESULTS:")
        print(f"   • Items extracted: {len(data)}")
        print(f"   • Expected: ~{test_case['expected_items']}")
        print(f"   • Extraction source: {result['extraction_source']}")
        print(f"   • Total time: {result['total_time']:.1f}s")
        print()
        
        print(f"📊 NULL VALUE ANALYSIS:")
        print(f"   • Items with ALL null values: {null_value_analysis['all_null_count']}")
        print(f"   • Items with SOME null values: {null_value_analysis['some_null_count']}")
        print(f"   • Items with NO null values: {null_value_analysis['no_null_count']}")
        print(f"   • Average fields populated: {null_value_analysis['avg_populated_fields']:.1f}/{len(test_case['fields'])}")
        
        # Determine success
        success = False
        if null_value_analysis['all_null_count'] == 0 and len(data) >= test_case['expected_items'] * 0.5:
            success = True
            print(f"\n✅ TEST PASSED!")
        elif null_value_analysis['all_null_count'] > 0:
            print(f"\n❌ TEST FAILED: Still has items with all null values")
        elif len(data) < test_case['expected_items'] * 0.5:
            print(f"\n⚠️ TEST PARTIAL: Items extracted but below expected count")
        else:
            print(f"\n✅ TEST PASSED (with warnings)")
            success = True
        
        # Show samples
        if data:
            print(f"\n📋 Sample items (first 2):")
            for i, item in enumerate(data[:2], 1):
                print(f"\n   Item {i}:")
                for k, v in item.items():
                    status = "✅" if v is not None and v != '' else "❌"
                    print(f"      {status} {k}: {v}")
        
        return {
            'name': test_case['name'],
            'success': success,
            'items_count': len(data),
            'null_analysis': null_value_analysis,
            'error': None
        }
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'name': test_case['name'],
            'success': False,
            'items_count': 0,
            'null_analysis': None,
            'error': str(e)
        }


def analyze_null_values(data: List[Dict[str, Any]], expected_fields: List[str]) -> Dict[str, Any]:
    """
    Analyze null values in extracted data
    
    Returns:
        Dict with null value statistics
    """
    if not data:
        return {
            'all_null_count': 0,
            'some_null_count': 0,
            'no_null_count': 0,
            'avg_populated_fields': 0
        }
    
    all_null_count = 0
    some_null_count = 0
    no_null_count = 0
    total_populated = 0
    
    for item in data:
        populated_count = sum(1 for v in item.values() if v is not None and v != '')
        total_populated += populated_count
        
        if populated_count == 0:
            all_null_count += 1
        elif populated_count < len(expected_fields):
            some_null_count += 1
        else:
            no_null_count += 1
    
    return {
        'all_null_count': all_null_count,
        'some_null_count': some_null_count,
        'no_null_count': no_null_count,
        'avg_populated_fields': total_populated / len(data) if data else 0
    }


async def main():
    """Run all tests"""
    print("="*80)
    print("🧪 BUG FIXES PHASE 1 & 2 TEST SUITE")
    print("="*80)
    print()
    print("Testing fixes for:")
    print("  1. Null value extraction (Priority 1)")
    print("  2. Enhanced anti-detection (Priority 2)")
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        return
    
    scraper = None
    results = []
    
    try:
        # Initialize scraper with enhanced anti-detection
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=True,
            headless=True,
            enable_auto_pagination=False,  # Single page for testing
        )
        
        # Test each site
        for test_case in TEST_CASES:
            result = await test_site(scraper, test_case)
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for r in results if r['success'])
        failed = len(results) - passed
        
        print(f"\n✅ Passed: {passed}/{len(results)}")
        print(f"❌ Failed: {failed}/{len(results)}")
        print(f"📈 Success rate: {(passed/len(results)*100):.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"\n{status} {r['name']}")
            print(f"   Items: {r['items_count']}")
            if r['null_analysis']:
                print(f"   All null: {r['null_analysis']['all_null_count']}")
                print(f"   Avg fields: {r['null_analysis']['avg_populated_fields']:.1f}")
            if r['error']:
                print(f"   Error: {r['error']}")
        
        # Print fixes status
        print(f"\n{'='*80}")
        print("🔧 FIX STATUS")
        print(f"{'='*80}")
        
        # Check if null value fix worked
        null_value_fixed = all(
            r['null_analysis'] and r['null_analysis']['all_null_count'] == 0
            for r in results[:2]  # First 2 are null value tests
            if r['null_analysis']
        )
        
        print(f"\n{'✅' if null_value_fixed else '❌'} Null Value Fix (Priority 1)")
        if null_value_fixed:
            print("   All items have at least some non-null fields!")
        else:
            print("   Still finding items with all null values")
        
        # Check if anti-detection helped
        anti_detect_results = [r for r in results if 'Etsy' in r['name']]
        anti_detect_fixed = any(r['success'] for r in anti_detect_results)
        
        print(f"\n{'✅' if anti_detect_fixed else '⚠️'} Anti-Detection Fix (Priority 2)")
        if anti_detect_fixed:
            print("   Successfully bypassed anti-bot detection!")
        else:
            print("   Still being blocked (may need residential proxies)")
        
    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







