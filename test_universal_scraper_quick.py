"""
Quick Universal Scraper Test Suite - Tests only fast/reliable sites
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper

# Quick test cases: Only reliable/fast sites
QUICK_TEST_CASES = [
    {
        "name": "Real Estate (Auction.com)",
        "url": "https://www.auction.com/residential/TX/Dallas_ct/active_lt/auction_date_order,resi_sort_v2_st/y_nbs/100_nsr",
        "fields": ["est. market value", "bedrooms", "bathrooms", "square footage", "property url"],
        "domain_type": "real_estate",
        "expected_min_items": 5,
        "expected_fields": ["est. market value", "bedrooms", "bathrooms"],
        "timeout": 120
    },
    {
        "name": "E-commerce (Chewy.com)",
        "url": "https://www.chewy.com/b/wet-food-389",
        "fields": ["title", "price", "rating", "review count", "product url"],
        "domain_type": "ecommerce",
        "expected_min_items": 10,
        "expected_fields": ["title", "price"],
        "timeout": 180
    },
    {
        "name": "E-commerce with Variants (Baggu.com)",
        "url": "https://baggu.com/collections/crescent-bags",
        "fields": ["title", "price", "color", "product detail url"],
        "domain_type": "ecommerce_variants",
        "expected_min_items": 5,
        "expected_fields": ["title", "price", "color"],
        "timeout": 120
    },
    {
        "name": "News/Articles (Hacker News)",
        "url": "https://news.ycombinator.com/",
        "fields": ["title", "url", "score", "author", "comments"],
        "domain_type": "news",
        "expected_min_items": 20,
        "expected_fields": ["title", "url"],
        "timeout": 90
    },
    {
        "name": "Social Media (Reddit)",
        "url": "https://www.reddit.com/r/github/",
        "fields": ["title", "author", "score", "comments", "url"],
        "domain_type": "social_media",
        "expected_min_items": 10,
        "expected_fields": ["title", "score"],
        "timeout": 120
    }
]

class TestResult:
    def __init__(self, test_case: Dict[str, Any]):
        self.test_case = test_case
        self.success = False
        self.items_extracted = 0
        self.execution_time = 0.0
        self.extraction_source = None
        self.items = []
        self.issues = []
        self.field_coverage = {}
        self.quality_score = 0.0
        self.error = None
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "test_case": self.test_case,
            "success": self.success,
            "items_extracted": self.items_extracted,
            "execution_time": self.execution_time,
            "extraction_source": self.extraction_source,
            "field_coverage": self.field_coverage,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "error": str(self.error) if self.error else None,
            "timestamp": self.timestamp,
            "sample_item": self.items[0] if self.items else None
        }

def analyze_results(result: TestResult) -> List[str]:
    """Analyze test results and identify issues"""
    issues = []
    
    if result.items_extracted < result.test_case["expected_min_items"]:
        issues.append(f"⚠️ Low item count: Expected ≥{result.test_case['expected_min_items']}, got {result.items_extracted}")
    
    missing_fields = []
    for expected_field in result.test_case["expected_fields"]:
        if expected_field not in result.field_coverage:
            missing_fields.append(expected_field)
        elif result.field_coverage[expected_field] == 0:
            missing_fields.append(expected_field)
    
    if missing_fields:
        issues.append(f"⚠️ Missing critical fields: {', '.join(missing_fields)}")
    
    total_fields = len(result.test_case["fields"])
    covered_fields = sum(1 for f in result.test_case["fields"] 
                         if result.field_coverage.get(f, 0) > 0)
    coverage_pct = (covered_fields / total_fields * 100) if total_fields > 0 else 0
    
    if coverage_pct < 60:
        issues.append(f"⚠️ Low field coverage: {coverage_pct:.1f}% ({covered_fields}/{total_fields} fields)")
    
    if result.quality_score < 50:
        issues.append(f"⚠️ Low quality score: {result.quality_score:.1f}%")
    
    if result.items:
        sample = result.items[0]
        empty_fields = [k for k, v in sample.items() if not v or v == "null" or v == "None"]
        if empty_fields:
            issues.append(f"⚠️ Empty/null values in sample: {', '.join(empty_fields[:5])}")
    
    if result.execution_time > 120:
        issues.append(f"⚠️ Slow execution: {result.execution_time:.1f}s (> 2 minutes)")
    
    return issues

def calculate_field_coverage(items: List[Dict], fields: List[str]) -> Dict[str, int]:
    """Calculate how many items have each field"""
    coverage = {field: 0 for field in fields}
    
    for item in items:
        for field in fields:
            if field in item and item[field] and item[field] not in ["null", "None", ""]:
                coverage[field] += 1
    
    return coverage

async def run_test(test_case: Dict[str, Any], api_key: str, timeout: int = 180) -> TestResult:
    """Run a single test case with timeout"""
    result = TestResult(test_case)
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {test_case['name']}")
    print(f"   URL: {test_case['url']}")
    print(f"   Fields: {', '.join(test_case['fields'])}")
    print(f"   Timeout: {timeout}s")
    print(f"{'='*80}")
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=60000,
        use_direct_llm=True,
        enable_auto_pagination=False,
        log_level=30
    )
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        scrape_result = await asyncio.wait_for(
            scraper.scrape(test_case['url'], test_case['fields']),
            timeout=timeout
        )
        
        end_time = asyncio.get_event_loop().time()
        
        result.execution_time = end_time - start_time
        result.items = scrape_result.get('data', [])
        result.items_extracted = len(result.items)
        result.extraction_source = scrape_result.get('source', 'unknown')
        result.quality_score = scrape_result.get('metadata', {}).get('direct_llm_quality', 0.0)
        
        result.field_coverage = calculate_field_coverage(result.items, test_case['fields'])
        result.issues = analyze_results(result)
        
        result.success = (
            result.items_extracted >= test_case['expected_min_items'] and
            len(result.issues) == 0 and
            result.error is None
        )
        
        print(f"\n📊 Results:")
        print(f"   ✅ Items extracted: {result.items_extracted}")
        print(f"   ⏱️  Execution time: {result.execution_time:.1f}s")
        print(f"   📦 Source: {result.extraction_source}")
        print(f"   🎯 Quality: {result.quality_score:.1f}%")
        print(f"   📋 Field coverage:")
        for field, count in result.field_coverage.items():
            pct = (count / result.items_extracted * 100) if result.items_extracted > 0 else 0
            print(f"      - {field}: {count}/{result.items_extracted} ({pct:.1f}%)")
        
        if result.issues:
            print(f"\n   ⚠️  Issues found:")
            for issue in result.issues:
                print(f"      {issue}")
        else:
            print(f"\n   ✅ No issues detected!")
        
        if result.items:
            print(f"\n   📄 Sample item:")
            sample = result.items[0]
            for key, value in list(sample.items())[:5]:
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"      {key}: {val_str}")
        
    except asyncio.TimeoutError:
        result.error = f"Test timed out after {timeout}s"
        result.success = False
        result.issues.append(f"⏱️ Test timed out after {timeout}s")
        print(f"\n⏱️ Test timed out after {timeout}s")
    except Exception as e:
        result.error = e
        result.success = False
        result.issues.append(f"❌ Error: {str(e)}")
        print(f"\n❌ Error: {e}")
    finally:
        try:
            await scraper.close()
        except:
            pass
    
    return result

async def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    print("🚀 Universal Scraper Quick Test Suite")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print(f"🧪 Test cases: {len(QUICK_TEST_CASES)}")
    print("💡 This is a quick test with reliable sites only")
    
    results = []
    
    for i, test_case in enumerate(QUICK_TEST_CASES, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(QUICK_TEST_CASES)}: {test_case['name']}")
        print(f"{'#'*80}")
        
        try:
            timeout = test_case.get('timeout', 180)
            result = await run_test(test_case, api_key, timeout=timeout)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            result = TestResult(test_case)
            result.error = e
            result.success = False
            results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUITE SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    
    print(f"\n✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    print(f"⏱️  Total execution time: {sum(r.execution_time for r in results):.1f}s")
    
    # Save results
    output_file = "universal_scraper_quick_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'total_execution_time': sum(r.execution_time for r in results),
                'timestamp': datetime.now().isoformat()
            },
            'results': [r.to_dict() for r in results]
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n{'='*80}")
    print("✅ Quick test suite complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())







