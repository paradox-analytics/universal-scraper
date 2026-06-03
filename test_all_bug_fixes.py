#!/usr/bin/env python3
"""
Comprehensive Test for All Bug Fixes (Phases 1-4)
Tests all fixes on all previously failing sites
"""

import asyncio
import os
from pathlib import Path
import logging
import json
import csv
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


# Comprehensive test cases (all 10 from original test + problematic ones)
TEST_CASES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'fields': ['title', 'author', 'upvotes', 'comments'],
        'expected_min': 10,
        'context': 'Extract Reddit posts with title, author, upvotes, and comments',
        'difficulty': 'medium'
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'author', 'comments'],
        'expected_min': 20,
        'context': 'Extract HN posts with title, points, author, comments',
        'difficulty': 'easy'
    },
    {
        'name': 'Craigslist',
        'url': 'https://sfbay.craigslist.org/search/sss?query=laptop',
        'fields': ['title', 'price', 'location'],
        'expected_min': 20,
        'context': 'Extract classified ads with title, price, and location',
        'difficulty': 'easy',
        'previous_issue': 'NULL VALUES - All fields were None'
    },
    {
        'name': 'TechCrunch',
        'url': 'https://techcrunch.com/',
        'fields': ['title', 'author', 'date'],
        'expected_min': 5,
        'context': 'Extract news articles with title, author, date',
        'difficulty': 'medium',
        'previous_issue': 'NULL VALUES - All fields were None'
    },
    {
        'name': 'Medium',
        'url': 'https://medium.com/tag/technology',
        'fields': ['title', 'author', 'reading_time'],
        'expected_min': 5,
        'context': 'Extract articles with title, author, and reading time',
        'difficulty': 'medium',
        'previous_issue': 'SINGLE ITEM - Only 1 item extracted'
    },
    {
        'name': 'Product Hunt',
        'url': 'https://www.producthunt.com/',
        'fields': ['name', 'tagline', 'upvotes'],
        'expected_min': 5,
        'context': 'Extract product listings with name, tagline, and upvotes',
        'difficulty': 'hard',
        'previous_issue': '0 ITEMS - Unknown cause (Next.js)'
    },
    {
        'name': 'Etsy',
        'url': 'https://www.etsy.com/search?q=laptop',
        'fields': ['title', 'price', 'seller'],
        'expected_min': 10,
        'context': 'Extract product listings with title, price, and seller',
        'difficulty': 'hard',
        'previous_issue': 'ANTI-BOT - 403 Forbidden'
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'fields': ['name', 'description', 'stars'],
        'expected_min': 10,
        'context': 'Extract trending repositories with name, description, and stars',
        'difficulty': 'easy'
    },
    {
        'name': 'eBay',
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'fields': ['title', 'price', 'condition'],
        'expected_min': 20,
        'context': 'Extract product listings with title, price, and condition',
        'difficulty': 'hard'
    },
    {
        'name': 'Walmart',
        'url': 'https://www.walmart.com/search?q=laptop',
        'fields': ['title', 'price', 'rating'],
        'expected_min': 10,
        'context': 'Extract product listings with title, price, and rating',
        'difficulty': 'hard'
    },
]


def analyze_results(data: List[Dict[str, Any]], expected_fields: List[str]) -> Dict[str, Any]:
    """Comprehensive result analysis"""
    if not data:
        return {
            'all_null_count': 0,
            'some_null_count': 0,
            'no_null_count': 0,
            'avg_populated_fields': 0,
            'completeness': 0,
            'quality_score': 0
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
    
    avg_populated = total_populated / len(data) if data else 0
    completeness = no_null_count / len(data) if data else 0
    quality_score = (completeness * 0.7) + (min(len(data) / 20, 1.0) * 0.3)  # Weighted score
    
    return {
        'all_null_count': all_null_count,
        'some_null_count': some_null_count,
        'no_null_count': no_null_count,
        'avg_populated_fields': avg_populated,
        'completeness': completeness,
        'quality_score': quality_score
    }


async def test_site(scraper: UniversalScraper, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single site"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Difficulty: {test_case['difficulty'].upper()}")
    if 'previous_issue' in test_case:
        print(f"Previous Issue: {test_case['previous_issue']}")
    print(f"URL: {test_case['url']}")
    print(f"Fields: {', '.join(test_case['fields'])}")
    print()
    
    try:
        result = await scraper.scrape(
            url=test_case['url'],
            fields=test_case['fields']
        )
        
        data = result['data']
        analysis = analyze_results(data, test_case['fields'])
        
        print(f"\n✅ RESULTS:")
        print(f"   • Items extracted: {len(data)}")
        print(f"   • Expected minimum: {test_case['expected_min']}")
        print(f"   • Extraction source: {result.get('extraction_source', 'unknown')}")
        print(f"   • Total time: {result.get('total_time', 0):.1f}s")
        print(f"   • Quality score: {analysis['quality_score']:.2f}/1.0")
        
        print(f"\n📊 DATA QUALITY:")
        print(f"   • Items with ALL null values: {analysis['all_null_count']}")
        print(f"   • Items with SOME null values: {analysis['some_null_count']}")
        print(f"   • Items with NO null values: {analysis['no_null_count']}")
        print(f"   • Avg fields populated: {analysis['avg_populated_fields']:.1f}/{len(test_case['fields'])}")
        print(f"   • Completeness: {analysis['completeness']*100:.1f}%")
        
        # Determine success
        success = (
            analysis['all_null_count'] == 0 and 
            len(data) >= test_case['expected_min'] * 0.5 and
            analysis['quality_score'] > 0.4
        )
        
        if success:
            print(f"\n✅ TEST PASSED!")
        else:
            print(f"\n❌ TEST FAILED")
            if analysis['all_null_count'] > 0:
                print(f"   Reason: {analysis['all_null_count']} items with all null values")
            if len(data) < test_case['expected_min'] * 0.5:
                print(f"   Reason: Only {len(data)} items (expected {test_case['expected_min']}+)")
            if analysis['quality_score'] <= 0.4:
                print(f"   Reason: Low quality score ({analysis['quality_score']:.2f})")
        
        # Show samples
        if data:
            print(f"\n📋 Sample items (first 2):")
            for i, item in enumerate(data[:2], 1):
                print(f"\n   Item {i}:")
                for k, v in item.items():
                    status = "✅" if v is not None and v != '' else "❌"
                    v_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
                    print(f"      {status} {k}: {v_str}")
        
        # Save to CSV
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        csv_file = output_dir / f"{test_case['name'].lower().replace(' ', '_')}_results.csv"
        
        if data:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=test_case['fields'])
                writer.writeheader()
                writer.writerows(data)
            print(f"\n💾 Saved to: {csv_file}")
        
        return {
            'name': test_case['name'],
            'success': success,
            'items_count': len(data),
            'analysis': analysis,
            'difficulty': test_case['difficulty'],
            'previous_issue': test_case.get('previous_issue', None),
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
            'analysis': None,
            'difficulty': test_case['difficulty'],
            'previous_issue': test_case.get('previous_issue', None),
            'error': str(e)
        }


async def main():
    """Run all tests"""
    print("="*80)
    print("🧪 COMPREHENSIVE BUG FIXES TEST SUITE (ALL PHASES)")
    print("="*80)
    print()
    print("Testing ALL bug fixes:")
    print("  ✅ Phase 1: Null value extraction")
    print("  ✅ Phase 2: Anti-detection (Camoufox + AntiDetectionManager)")
    print("  ✅ Phase 3: Single-item detection")
    print("  ✅ Phase 4: Product Hunt / Next.js support")
    print()
    print(f"Testing {len(TEST_CASES)} websites...")
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        return
    
    scraper = None
    results = []
    
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=False,  # Disable Camoufox for testing (async loop conflict)
            headless=True,
            enable_auto_pagination=False,
        )
        
        # Test each site
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'#'*80}")
            print(f"# TEST {i}/{len(TEST_CASES)}")
            print(f"{'#'*80}")
            
            result = await test_site(scraper, test_case)
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for r in results if r['success'])
        failed = len(results) - passed
        
        print(f"\n✅ Overall Results:")
        print(f"   • Passed: {passed}/{len(results)} ({(passed/len(results)*100):.1f}%)")
        print(f"   • Failed: {failed}/{len(results)}")
        
        # Group by difficulty
        by_difficulty = {'easy': [], 'medium': [], 'hard': []}
        for r in results:
            by_difficulty[r['difficulty']].append(r)
        
        print(f"\n📊 Results by Difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            tests = by_difficulty[diff]
            if tests:
                passed_diff = sum(1 for t in tests if t['success'])
                print(f"   • {diff.upper()}: {passed_diff}/{len(tests)} passed ({(passed_diff/len(tests)*100):.1f}%)")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for r in results:
            status = "✅" if r['success'] else "❌"
            issue_fixed = ""
            if r['previous_issue']:
                issue_fixed = f" (Fixed: {r['previous_issue'].split(' - ')[0]})"
            
            print(f"\n{status} {r['name']}{issue_fixed}")
            print(f"   Items: {r['items_count']}")
            print(f"   Difficulty: {r['difficulty']}")
            if r['analysis']:
                print(f"   Quality: {r['analysis']['quality_score']:.2f}")
                print(f"   All null: {r['analysis']['all_null_count']}")
            if r['error']:
                print(f"   Error: {r['error'][:100]}...")
        
        # Fix status
        print(f"\n{'='*80}")
        print("🔧 BUG FIX STATUS")
        print(f"{'='*80}")
        
        # Phase 1: Null values (Craigslist, TechCrunch)
        null_value_sites = [r for r in results if r.get('previous_issue') and 'NULL VALUES' in r['previous_issue']]
        null_value_fixed = all(r['success'] and r['analysis'] and r['analysis']['all_null_count'] == 0 for r in null_value_sites)
        print(f"\n{'✅' if null_value_fixed else '❌'} Phase 1: Null Value Fix")
        print(f"   Sites: {', '.join(r['name'] for r in null_value_sites)}")
        print(f"   Status: {'All fixed!' if null_value_fixed else 'Still has issues'}")
        
        # Phase 2: Anti-bot (Etsy)
        anti_bot_sites = [r for r in results if r.get('previous_issue') and 'ANTI-BOT' in r['previous_issue']]
        anti_bot_fixed = any(r['success'] for r in anti_bot_sites)
        print(f"\n{'✅' if anti_bot_fixed else '⚠️'} Phase 2: Anti-Detection Fix")
        print(f"   Sites: {', '.join(r['name'] for r in anti_bot_sites)}")
        print(f"   Status: {'Bypassed!' if anti_bot_fixed else 'Still blocked (may need proxies)'}")
        
        # Phase 3: Single item (Medium)
        single_item_sites = [r for r in results if r.get('previous_issue') and 'SINGLE ITEM' in r['previous_issue']]
        single_item_fixed = all(r['success'] and r['items_count'] > 1 for r in single_item_sites)
        print(f"\n{'✅' if single_item_fixed else '❌'} Phase 3: Single-Item Detection Fix")
        print(f"   Sites: {', '.join(r['name'] for r in single_item_sites)}")
        print(f"   Status: {'Fixed!' if single_item_fixed else 'Still extracting only 1 item'}")
        
        # Phase 4: Product Hunt
        product_hunt = [r for r in results if r['name'] == 'Product Hunt']
        product_hunt_fixed = any(r['success'] and r['items_count'] > 0 for r in product_hunt)
        print(f"\n{'✅' if product_hunt_fixed else '❌'} Phase 4: Product Hunt / Next.js Fix")
        print(f"   Status: {'Fixed!' if product_hunt_fixed else 'Still returning 0 items'}")
        
        # Overall assessment
        print(f"\n{'='*80}")
        print("🎯 OVERALL ASSESSMENT")
        print(f"{'='*80}")
        success_rate = (passed / len(results)) * 100
        
        if success_rate >= 90:
            grade = "A+ (EXCELLENT)"
            emoji = "🎉"
        elif success_rate >= 80:
            grade = "A (VERY GOOD)"
            emoji = "🎊"
        elif success_rate >= 70:
            grade = "B (GOOD)"
            emoji = "👍"
        elif success_rate >= 60:
            grade = "C (AVERAGE)"
            emoji = "😐"
        else:
            grade = "D (NEEDS WORK)"
            emoji = "😕"
        
        print(f"\n{emoji} Success Rate: {success_rate:.1f}% - Grade: {grade}")
        print(f"\nTarget: 90%+ success rate")
        print(f"Current: {success_rate:.1f}%")
        print(f"Gap: {max(0, 90 - success_rate):.1f}%")
        
        if success_rate >= 90:
            print(f"\n🎉 MISSION ACCOMPLISHED! All bug fixes are working!")
        else:
            print(f"\n💪 Good progress! {len([r for r in results if not r['success']])} site(s) still need attention.")
    
    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())

