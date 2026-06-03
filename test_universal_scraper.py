"""
Comprehensive Universal Scraper Test Suite
Tests multiple different website types to ensure universality
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

# Test cases: Different domain types with realistic fields
TEST_CASES = [
    {
        "name": "Real Estate (Auction.com)",
        "url": "https://www.auction.com/residential/TX/Dallas_ct/active_lt/auction_date_order,resi_sort_v2_st/y_nbs/100_nsr",
        "fields": ["est. market value", "bedrooms", "bathrooms", "square footage", "property url"],
        "domain_type": "real_estate",
        "expected_min_items": 5,
        "expected_fields": ["est. market value", "bedrooms", "bathrooms"]
    },
    {
        "name": "E-commerce (Chewy.com)",
        "url": "https://www.chewy.com/b/wet-food-389",
        "fields": ["title", "price", "rating", "review count", "product url"],
        "domain_type": "ecommerce",
        "expected_min_items": 10,
        "expected_fields": ["title", "price"]
    },
    {
        "name": "E-commerce with Variants (Baggu.com)",
        "url": "https://baggu.com/collections/crescent-bags",
        "fields": ["title", "price", "color", "product detail url"],
        "domain_type": "ecommerce_variants",
        "expected_min_items": 5,
        "expected_fields": ["title", "price", "color"]
    },
    {
        "name": "Job Listings (Indeed.com)",
        "url": "https://www.indeed.com/jobs?q=software+engineer&l=San+Francisco%2C+CA",
        "fields": ["job title", "company", "location", "salary", "job url"],
        "domain_type": "job_listings",
        "expected_min_items": 5,
        "expected_fields": ["job title", "company"],
        "timeout": 120  # Shorter timeout for potentially slow site
    },
    {
        "name": "News/Articles (Hacker News)",
        "url": "https://news.ycombinator.com/",
        "fields": ["title", "url", "score", "author", "comments"],
        "domain_type": "news",
        "expected_min_items": 20,
        "expected_fields": ["title", "url"]
    },
    {
        "name": "Product Reviews (Metacritic)",
        "url": "https://www.metacritic.com/pictures/november-2025-movie-preview-wicked-hamnet-running-man/",
        "fields": ["title", "metascore", "director", "release date", "description"],
        "domain_type": "reviews",
        "expected_min_items": 1,
        "expected_fields": ["title"]
    },
    {
        "name": "Social Media (Reddit)",
        "url": "https://www.reddit.com/r/github/",
        "fields": ["title", "author", "score", "comments", "url"],
        "domain_type": "social_media",
        "expected_min_items": 10,
        "expected_fields": ["title", "score"]
    },
    {
        "name": "Dispensary Menu (Leafly)",
        "url": "https://www.leafly.com/dispensary-info/mammoth-holistics/menu",
        "fields": ["product name", "price", "strain type", "thc", "product url"],
        "domain_type": "cannabis_menu",
        "expected_min_items": 5,
        "expected_fields": ["product name", "price"]
    },
    {
        "name": "Restaurant Menu (Yelp - example)",
        "url": "https://www.yelp.com/biz/tonys-little-star-pizza-san-francisco",
        "fields": ["dish name", "price", "description", "category"],
        "domain_type": "restaurant_menu",
        "expected_min_items": 3,
        "expected_fields": ["dish name"],
        "timeout": 120
    },
    {
        "name": "Classifieds (Craigslist - example)",
        "url": "https://sfbay.craigslist.org/search/sss?query=laptop",
        "fields": ["title", "price", "location", "posting date", "url"],
        "domain_type": "classifieds",
        "expected_min_items": 5,
        "expected_fields": ["title", "price"],
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
    
    # Check item count
    if result.items_extracted < result.test_case["expected_min_items"]:
        issues.append(f"⚠️ Low item count: Expected ≥{result.test_case['expected_min_items']}, got {result.items_extracted}")
    
    # Check field coverage
    missing_fields = []
    for expected_field in result.test_case["expected_fields"]:
        if expected_field not in result.field_coverage:
            missing_fields.append(expected_field)
        elif result.field_coverage[expected_field] == 0:
            missing_fields.append(expected_field)
    
    if missing_fields:
        issues.append(f"⚠️ Missing critical fields: {', '.join(missing_fields)}")
    
    # Check field coverage percentage
    total_fields = len(result.test_case["fields"])
    covered_fields = sum(1 for f in result.test_case["fields"] 
                         if result.field_coverage.get(f, 0) > 0)
    coverage_pct = (covered_fields / total_fields * 100) if total_fields > 0 else 0
    
    if coverage_pct < 60:
        issues.append(f"⚠️ Low field coverage: {coverage_pct:.1f}% ({covered_fields}/{total_fields} fields)")
    
    # Check quality score
    if result.quality_score < 50:
        issues.append(f"⚠️ Low quality score: {result.quality_score:.1f}%")
    
    # Check for empty/null values
    if result.items:
        sample = result.items[0]
        empty_fields = [k for k, v in sample.items() if not v or v == "null" or v == "None"]
        if empty_fields:
            issues.append(f"⚠️ Empty/null values in sample: {', '.join(empty_fields[:5])}")
    
    # Check execution time (warn if > 2 minutes)
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
        browser_timeout=60000,  # 60s browser timeout
        use_direct_llm=True,
        enable_auto_pagination=False,  # Single page only
        log_level=30  # WARNING level to reduce noise
    )
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Run with overall timeout
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
        
        # Calculate field coverage
        result.field_coverage = calculate_field_coverage(result.items, test_case['fields'])
        
        # Analyze results
        result.issues = analyze_results(result)
        
        # Determine success
        result.success = (
            result.items_extracted >= test_case['expected_min_items'] and
            len(result.issues) == 0 and
            result.error is None
        )
        
        # Print summary
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
        import traceback
        traceback.print_exc()
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
    
    print("🚀 Universal Scraper Test Suite")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print(f"🧪 Test cases: {len(TEST_CASES)}")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"{'#'*80}")
        
        try:
            # Use shorter timeout for faster feedback
            timeout = test_case.get('timeout', 180)  # Default 3 minutes
            result = await run_test(test_case, api_key, timeout=timeout)
            results.append(result)
            
            # If test took too long, warn user
            if result.execution_time > 120:
                print(f"\n⚠️ Warning: Test took {result.execution_time:.1f}s - consider skipping slow sites")
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
            print("💡 Tip: You can modify TEST_CASES to skip slow sites")
            break
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            result = TestResult(test_case)
            result.error = e
            result.success = False
            results.append(result)
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print("📊 TEST SUITE SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests
    
    print(f"\n✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {failed_tests}/{total_tests}")
    print(f"⏱️  Total execution time: {sum(r.execution_time for r in results):.1f}s")
    
    # Group issues by type
    all_issues = []
    for result in results:
        all_issues.extend([(result.test_case['name'], issue) for issue in result.issues])
    
    if all_issues:
        print(f"\n⚠️  Issues Found ({len(all_issues)} total):")
        issue_types = {}
        for test_name, issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append((test_name, issue))
        
        for issue_type, occurrences in issue_types.items():
            print(f"\n   {issue_type} ({len(occurrences)} occurrences):")
            for test_name, issue in occurrences[:5]:  # Show first 5
                print(f"      - {test_name}: {issue}")
            if len(occurrences) > 5:
                print(f"      ... and {len(occurrences) - 5} more")
    
    # Domain type analysis
    print(f"\n📈 Results by Domain Type:")
    domain_stats = {}
    for result in results:
        domain_type = result.test_case['domain_type']
        if domain_type not in domain_stats:
            domain_stats[domain_type] = {'total': 0, 'passed': 0, 'avg_items': 0, 'avg_time': 0}
        domain_stats[domain_type]['total'] += 1
        if result.success:
            domain_stats[domain_type]['passed'] += 1
        domain_stats[domain_type]['avg_items'] += result.items_extracted
        domain_stats[domain_type]['avg_time'] += result.execution_time
    
    for domain_type, stats in domain_stats.items():
        avg_items = stats['avg_items'] / stats['total'] if stats['total'] > 0 else 0
        avg_time = stats['avg_time'] / stats['total'] if stats['total'] > 0 else 0
        pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   {domain_type}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%), "
              f"avg {avg_items:.1f} items, {avg_time:.1f}s")
    
    # Save detailed results
    output_file = "universal_scraper_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'total_execution_time': sum(r.execution_time for r in results),
                'timestamp': datetime.now().isoformat()
            },
            'results': [r.to_dict() for r in results],
            'domain_stats': domain_stats
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Generate recommendations
    print(f"\n\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = []
    
    # Check for common patterns in failures
    low_coverage_tests = [r for r in results if r.quality_score < 50]
    if low_coverage_tests:
        recommendations.append({
            'issue': 'Low quality scores',
            'count': len(low_coverage_tests),
            'tests': [r.test_case['name'] for r in low_coverage_tests],
            'recommendation': 'Review HTML structure analysis and field mapping for these domains. Consider improving semantic field detection.'
        })
    
    slow_tests = [r for r in results if r.execution_time > 120]
    if slow_tests:
        recommendations.append({
            'issue': 'Slow execution times',
            'count': len(slow_tests),
            'tests': [r.test_case['name'] for r in slow_tests],
            'recommendation': 'Optimize extraction for these sites. Consider caching or parallel processing.'
        })
    
    missing_field_tests = [r for r in results if any(f in r.issues for f in r.issues if 'Missing critical fields' in f)]
    if missing_field_tests:
        recommendations.append({
            'issue': 'Missing critical fields',
            'count': len(missing_field_tests),
            'tests': [r.test_case['name'] for r in missing_field_tests],
            'recommendation': 'Improve field detection and mapping. Review semantic synonyms and LLM field context.'
        })
    
    if recommendations:
        for rec in recommendations:
            print(f"\n🔧 {rec['issue']} ({rec['count']} tests):")
            print(f"   Affected: {', '.join(rec['tests'])}")
            print(f"   Recommendation: {rec['recommendation']}")
    else:
        print("\n✅ No major issues detected! The scraper appears to be working well across all domain types.")
    
    print(f"\n{'='*80}")
    print("✅ Test suite complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())

