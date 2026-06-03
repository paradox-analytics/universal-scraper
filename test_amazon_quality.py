#!/usr/bin/env python3
"""
Quick test to verify content quality validation rejects Amazon analytics garbage
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector

async def test_amazon():
    print("\n" + "="*80)
    print("🧪 Testing Amazon Content Quality Validation")
    print("="*80)
    print("\nThis test should:")
    print("  ✅ Extract JSON data (analytics garbage)")
    print("  ✅ Detect it's low quality (analytics/tracking)")
    print("  ✅ Reject JSON and fall back to HTML")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Initialize components
    pattern_gen = SemanticPatternGenerator(api_key=api_key)
    fetcher = HybridFetcher(
        proxy_config=None,
        headless=True,
        use_camoufox=True,
        enable_cache=False
    )
    json_detector = JSONDetector()
    
    # Test URL
    url = "https://www.amazon.com/s?k=laptop"
    fields_nl = "Get product title, price, and rating"
    
    print(f"📥 Fetching: {url}")
    print(f"📝 Fields: {fields_nl}")
    print()
    
    # Step 1: Fetch
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes via {result.get('fetch_method')}")
    print()
    
    # Step 2: Parse fields
    print("📝 Parsing natural language fields...")
    fields = await pattern_gen._parse_natural_language_fields(fields_nl, html[:3000])
    print(f"✅ Parsed to: {fields}")
    print()
    
    # Step 3: JSON Detection
    print("🔍 Universal JSON Detection...")
    json_data_captured = result.get('json_data', [])
    json_detection_result = json_detector.detect_and_extract(
        html=html,
        url=url,
        captured_json=json_data_captured
    )
    
    if json_detection_result['json_found']:
        json_sources = json_detection_result['sources']
        all_json = json_detection_result['data']
        
        print(f"✅ Found JSON from: {', '.join(json_sources)}")
        print(f"📦 Total JSON sources: {len(all_json)}")
        print()
        
        # Step 4: Extract
        print("🔬 Extracting items...")
        extracted_items = json_detector.extract_from_json(
            json_data=all_json,
            fields=fields
        )
        
        print(f"✅ Extracted {len(extracted_items)} items")
        print()
        
        if extracted_items and len(extracted_items) >= 3:
            # Show sample data
            print("📋 Sample extracted data:")
            sample = extracted_items[0]
            for key, value in sample.items():
                if key != '_metadata':
                    print(f"  • {key}: {str(value)[:80]}")
            print()
            
            # Step 5: CRITICAL - Quality validation
            print("🎯 Validating content quality...")
            is_sufficient = json_detector.is_json_sufficient(
                json_results=json_detection_result,
                fields=fields
            )
            
            print()
            print("="*80)
            print("📊 TEST RESULT")
            print("="*80)
            
            if is_sufficient:
                print("❌ FAILED: Quality validator accepted analytics garbage!")
                print("   This should have been rejected and fallen back to HTML.")
            else:
                print("✅ SUCCESS: Quality validator correctly rejected analytics data!")
                print("   The system will now fall back to HTML extraction.")
            print()
        else:
            print(f"ℹ️  Only {len(extracted_items)} items extracted (needs ≥3)")
    else:
        print("ℹ️  No JSON detected")
    
    print("="*80)
    print("🏁 TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_amazon())




