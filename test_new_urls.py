"""
Test new URLs locally with appropriate fields
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper

# Test cases with appropriate fields for each domain
TEST_CASES = [
    {
        "name": "Real Estate (Auction.com)",
        "url": "https://www.auction.com/residential/TX/Dallas_ct/active_lt/auction_date_order,resi_sort_v2_st/y_nbs/100_nsr",
        "fields": ["est. market value", "bedrooms", "bathrooms", "square footage", "property url"],
        "timeout": 120
    },
    {
        "name": "Movie Preview (Metacritic)",
        "url": "https://www.metacritic.com/pictures/december-2025-movie-preview-avatar-kill-bill-marty-supreme/",
        "fields": ["title", "metascore", "director", "release date", "description"],
        "timeout": 120
    },
    {
        "name": "Job Listings (Monster.com)",
        "url": "https://www.monster.com/jobs/search?q=Data+Engineer&where=Remote&page=1&so=m.s.sh",
        "fields": ["job title", "company", "location", "salary", "job url"],
        "timeout": 120
    },
    {
        "name": "E-commerce (Lowes.com)",
        "url": "https://www.lowes.com/search?searchTerm=bathroom%20vanity%20with%20sink",
        "fields": ["title", "price", "rating", "review count", "product url"],
        "timeout": 120
    }
]

async def run_test(test_case, api_key):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {test_case['name']}")
    print(f"   URL: {test_case['url']}")
    print(f"   Fields: {', '.join(test_case['fields'])}")
    print(f"{'='*80}")
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=60000,
        use_direct_llm=True,
        enable_auto_pagination=False,  # Single page only
        log_level=30  # WARNING level
    )
    
    result_data = {
        "test_case": test_case,
        "success": False,
        "items_extracted": 0,
        "execution_time": 0.0,
        "extraction_source": None,
        "field_coverage": {},
        "quality_score": 0.0,
        "issues": [],
        "error": None,
        "sample_item": None
    }
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        scrape_result = await asyncio.wait_for(
            scraper.scrape(test_case['url'], test_case['fields']),
            timeout=test_case.get('timeout', 180)
        )
        
        end_time = asyncio.get_event_loop().time()
        
        result_data['execution_time'] = end_time - start_time
        items = scrape_result.get('data', [])
        result_data['items_extracted'] = len(items)
        result_data['extraction_source'] = scrape_result.get('source', 'unknown')
        result_data['quality_score'] = scrape_result.get('metadata', {}).get('direct_llm_quality', 0.0)
        
        # Calculate field coverage
        field_coverage = {field: 0 for field in test_case['fields']}
        for item in items:
            for field in test_case['fields']:
                if field in item and item[field] and item[field] not in ["null", "None", ""]:
                    field_coverage[field] += 1
        result_data['field_coverage'] = field_coverage
        
        # Check for issues
        if result_data['items_extracted'] == 0:
            result_data['issues'].append("⚠️ No items extracted")
        
        missing_fields = [f for f in test_case['fields'] if field_coverage.get(f, 0) == 0]
        if missing_fields:
            result_data['issues'].append(f"⚠️ Missing fields: {', '.join(missing_fields)}")
        
        if result_data['quality_score'] < 50:
            result_data['issues'].append(f"⚠️ Low quality score: {result_data['quality_score']:.1f}%")
        
        result_data['success'] = len(result_data['issues']) == 0
        result_data['sample_item'] = items[0] if items else None
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   ✅ Items extracted: {result_data['items_extracted']}")
        print(f"   ⏱️  Execution time: {result_data['execution_time']:.1f}s")
        print(f"   📦 Source: {result_data['extraction_source']}")
        print(f"   🎯 Quality: {result_data['quality_score']:.1f}%")
        print(f"   📋 Field coverage:")
        for field, count in field_coverage.items():
            pct = (count / result_data['items_extracted'] * 100) if result_data['items_extracted'] > 0 else 0
            print(f"      - {field}: {count}/{result_data['items_extracted']} ({pct:.1f}%)")
        
        if result_data['issues']:
            print(f"\n   ⚠️  Issues:")
            for issue in result_data['issues']:
                print(f"      {issue}")
        else:
            print(f"\n   ✅ No issues detected!")
        
        if result_data['sample_item']:
            print(f"\n   📄 Sample item:")
            for key, value in list(result_data['sample_item'].items())[:5]:
                val_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                print(f"      {key}: {val_str}")
        
    except asyncio.TimeoutError:
        result_data['error'] = f"Test timed out after {test_case.get('timeout', 180)}s"
        result_data['issues'].append(f"⏱️ Test timed out")
        print(f"\n⏱️ Test timed out after {test_case.get('timeout', 180)}s")
    except Exception as e:
        result_data['error'] = str(e)
        result_data['issues'].append(f"❌ Error: {str(e)}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await scraper.close()
        except:
            pass
    
    return result_data

async def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    print("🚀 Testing New URLs Locally")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print(f"🧪 Test cases: {len(TEST_CASES)}")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"{'#'*80}")
        
        try:
            result = await run_test(test_case, api_key)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            result = {
                "test_case": test_case,
                "success": False,
                "error": str(e),
                "items_extracted": 0
            }
            results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get('success', False))
    
    print(f"\n✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    print(f"⏱️  Total execution time: {sum(r.get('execution_time', 0) for r in results):.1f}s")
    
    # Save results
    output_file = "new_urls_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'total_execution_time': sum(r.get('execution_time', 0) for r in results),
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n{'='*80}")
    print("✅ Test complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())







