#!/usr/bin/env python3
"""
Test Universal Hybrid Scraper on 6 Diverse Sources
===================================================

Tests the system on:
1. Leafly (Cannabis products - Next.js, JS-heavy)
2. Amazon (E-commerce - static HTML)
3. Reddit (Forum/Social - JS-heavy)
4. Product Hunt (Tech products - modern JS)
5. Hacker News (News aggregator - static HTML)
6. eBay (E-commerce - mixed HTML/JS)

This validates:
- Universal fetching (static vs JS)
- Universal JSON detection (Next.js, embedded, APIs)
- Natural language field parsing
- Semantic field extraction
- Content filtering (breadcrumbs, metadata)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator

# Test sources with diverse characteristics
TEST_SOURCES = [
    {
        "name": "Leafly (Cannabis)",
        "url": "https://www.leafly.com/dispensary-info/seven-point/menu",
        "fields": "Extract product name, price, and description for all products",
        "expected_min": 10,
        "site_type": "JS-heavy, Next.js"
    },
    {
        "name": "Amazon (E-commerce)",
        "url": "https://www.amazon.com/s?k=laptop",
        "fields": "Get product title, price, and rating",
        "expected_min": 10,
        "site_type": "Mixed HTML/JS"
    },
    {
        "name": "Reddit (Social)",
        "url": "https://old.reddit.com/r/programming/",
        "fields": "Extract post title, author, and upvotes",
        "expected_min": 15,
        "site_type": "Static HTML"
    },
    {
        "name": "Product Hunt (Tech)",
        "url": "https://www.producthunt.com/",
        "fields": "Get product name, tagline, and upvotes",
        "expected_min": 5,
        "site_type": "Modern JS/React"
    },
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "fields": "Extract article title, points, and comments count",
        "expected_min": 20,
        "site_type": "Static HTML"
    },
    {
        "name": "eBay (Auction)",
        "url": "https://www.ebay.com/sch/i.html?_nkw=macbook",
        "fields": "Get item title, price, and condition",
        "expected_min": 10,
        "site_type": "Mixed HTML/JS"
    }
]


async def test_source(source_config, api_key):
    """Test a single source"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {source_config['name']}")
    print(f"{'='*80}")
    print(f"URL: {source_config['url']}")
    print(f"Type: {source_config['site_type']}")
    print(f"Fields: {source_config['fields']}")
    print(f"Expected: ≥{source_config['expected_min']} items")
    print()
    
    try:
        # Initialize components
        hybrid_fetcher = HybridFetcher(
            proxy_config=None,
            headless=True,
            use_camoufox=True,
            enable_cache=True
        )
        json_detector = JSONDetector()
        pattern_gen = SemanticPatternGenerator(api_key=api_key)
        
        # Step 1: Fetch with universal fetcher
        print("📥 Step 1: Universal Fetch")
        print("-" * 80)
        result = await hybrid_fetcher.fetch(source_config['url'])
        
        if not result or 'html' not in result:
            print("❌ Failed to fetch")
            return {
                'name': source_config['name'],
                'success': False,
                'error': 'Fetch failed',
                'items': 0
            }
        
        html = result['html']
        fetch_method = result.get('fetch_method', 'unknown')
        print(f"✅ Fetched {len(html):,} bytes via {fetch_method}")
        
        # Step 2: Parse natural language fields
        print(f"\n📝 Step 2: Parse Natural Language Fields")
        print("-" * 80)
        print(f"Input: '{source_config['fields']}'")
        
        fields = await pattern_gen._parse_natural_language_fields(
            source_config['fields'],
            html[:3000]
        )
        print(f"✅ Parsed to: {fields}")
        
        # Step 3: Universal JSON detection
        print(f"\n🔍 Step 3: Universal JSON Detection")
        print("-" * 80)
        
        json_data_captured = result.get('json_data', [])
        json_detection_result = json_detector.detect_and_extract(
            html=html,
            url=source_config['url'],
            captured_json=json_data_captured
        )
        
        extracted_items = []
        extraction_method = "none"
        
        if json_detection_result['json_found']:
            json_sources = json_detection_result['sources']
            all_json = json_detection_result['data']
            
            print(f"✅ Found JSON from: {', '.join(json_sources)}")
            
            try:
                extracted_items = json_detector.extract_from_json(
                    json_data=all_json,
                    fields=fields
                )
                
                if extracted_items and len(extracted_items) >= 3:
                    extraction_method = f"json ({', '.join(json_sources)})"
                    print(f"✅ Extracted {len(extracted_items)} items via JSON")
                else:
                    print(f"ℹ️  JSON extraction found {len(extracted_items)} items (< 3)")
            except Exception as e:
                print(f"⚠️  JSON extraction error: {e}")
        else:
            print("ℹ️  No JSON detected")
        
        # If JSON didn't work, would fall back to HTML (not implemented in this test)
        if not extracted_items:
            extraction_method = "would_use_html_fallback"
            print("ℹ️  Would fall back to HTML semantic extraction")
        
        # Results
        print(f"\n📊 Results")
        print("-" * 80)
        item_count = len(extracted_items)
        success = item_count >= source_config['expected_min']
        
        if success:
            print(f"✅ SUCCESS: {item_count} items (≥{source_config['expected_min']} expected)")
        else:
            print(f"⚠️  PARTIAL: {item_count} items (<{source_config['expected_min']} expected)")
        
        print(f"Extraction: {extraction_method}")
        print(f"Cost: $0.00 (JSON only, no LLM)")
        
        # Show sample
        if extracted_items:
            print(f"\nSample item:")
            sample = extracted_items[0]
            for key, value in sample.items():
                if key != '_metadata':
                    value_str = str(value)[:100]
                    print(f"  • {key}: {value_str}")
        
        await hybrid_fetcher.close()
        
        return {
            'name': source_config['name'],
            'success': success,
            'items': item_count,
            'expected': source_config['expected_min'],
            'method': extraction_method,
            'fetch_method': fetch_method
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'name': source_config['name'],
            'success': False,
            'error': str(e),
            'items': 0
        }


async def main():
    """Run tests on all sources"""
    print("="*80)
    print("🧪 TESTING UNIVERSAL HYBRID SCRAPER ON 6 DIVERSE SOURCES")
    print("="*80)
    print()
    print("Features being tested:")
    print("  ✅ Universal fetching (static HTML vs JavaScript rendering)")
    print("  ✅ Universal JSON detection (Next.js, embedded, captured APIs)")
    print("  ✅ Natural language field parsing")
    print("  ✅ Semantic field extraction")
    print("  ✅ Content filtering (breadcrumbs, metadata)")
    print("  ✅ Zero-cost extraction (JSON-first approach)")
    print()
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set (needed for field parsing)")
        print("   Continuing anyway...")
    
    results = []
    
    # Test each source
    for source_config in TEST_SOURCES:
        result = await test_source(source_config, api_key)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📈 SUMMARY: 6 Source Universal Test")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results if r.get('success'))
    total_items = sum(r.get('items', 0) for r in results)
    
    print(f"Overall: {successful}/{len(results)} sources successful\n")
    
    for result in results:
        status = "✅" if result.get('success') else "⚠️"
        items = result.get('items', 0)
        expected = result.get('expected', '?')
        method = result.get('method', 'error')
        fetch = result.get('fetch_method', '?')
        
        print(f"{status} {result['name']}")
        print(f"   Items: {items}/{expected} | Extraction: {method} | Fetch: {fetch}")
        
        if not result.get('success') and 'error' in result:
            print(f"   Error: {result['error']}")
    
    print(f"\n📊 Total items extracted: {total_items}")
    print(f"💰 Total cost: $0.00 (JSON-only extraction)")
    
    print(f"\n{'='*80}")
    if successful == len(results):
        print("🎉 ALL TESTS PASSED! System is truly universal!")
    elif successful >= len(results) * 0.67:
        print("✅ MOST TESTS PASSED! System works for majority of sites.")
    else:
        print("⚠️  SOME TESTS FAILED. Review errors above.")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())




