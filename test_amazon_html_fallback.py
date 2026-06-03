#!/usr/bin/env python3
"""
Test Amazon HTML fallback after JSON rejection
This validates the complete workflow: JSON detection → Quality validation → Rejection → HTML extraction
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
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from bs4 import BeautifulSoup

async def test_amazon_full_workflow():
    print("\n" + "="*100)
    print("🧪 TESTING: Amazon Complete Workflow (JSON → HTML Fallback)")
    print("="*100)
    print("\nThis test validates:")
    print("  1. ✅ JSON detection")
    print("  2. ✅ Quality validation (should reject analytics)")
    print("  3. ✅ HTML fallback extraction")
    print("  4. ✅ High-quality structured output")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Initialize
    pattern_gen = SemanticPatternGenerator(api_key=api_key)
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    json_detector = JSONDetector()
    semantic_extractor = SemanticExtractor()
    html_cleaner = SmartHTMLCleaner()
    dom_detector = DOMPatternDetector()
    
    url = "https://www.amazon.com/s?k=laptop"
    fields_nl = "Get product title, price, and rating"
    
    # Step 1: Fetch
    print("📥 Step 1: Fetch")
    print("-" * 100)
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes via {result.get('fetch_method')}")
    print()
    
    # Step 2: Parse fields
    print("📝 Step 2: Parse Fields")
    print("-" * 100)
    try:
        fields = await pattern_gen._parse_natural_language_fields(fields_nl, html[:3000])
    except:
        fields = pattern_gen._extract_fields_from_text(fields_nl)
    print(f"✅ Fields: {fields}")
    print()
    
    # Step 3: JSON Detection
    print("🔍 Step 3: JSON Detection & Quality Validation")
    print("-" * 100)
    json_detection_result = json_detector.detect_and_extract(
        html=html,
        url=url,
        captured_json=result.get('json_data', [])
    )
    
    if json_detection_result['json_found']:
        print(f"✅ Found JSON from: {', '.join(json_detection_result['sources'])}")
        
        extracted_items = json_detector.extract_from_json(
            json_data=json_detection_result['data'],
            fields=fields
        )
        print(f"✅ Extracted {len(extracted_items)} items from JSON")
        
        if extracted_items and len(extracted_items) >= 3:
            is_sufficient = json_detector.is_json_sufficient(
                json_results=json_detection_result,
                fields=fields
            )
            
            if is_sufficient:
                print(f"✅ JSON accepted - using JSON data")
                print(f"\n❌ TEST FAILED: JSON should have been rejected!")
                return
            else:
                print(f"✅ JSON rejected - quality validation worked!")
                print(f"   → Falling back to HTML extraction")
    print()
    
    # Step 4: HTML Fallback
    print("🌐 Step 4: HTML Extraction")
    print("-" * 100)
    
    cleaned_result = html_cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned: {cleaned_result['original_size']:,} → {cleaned_result['cleaned_size']:,} bytes")
    
    dom_patterns = dom_detector.detect_patterns(cleaned_html)
    best_pattern = dom_patterns.get('best_pattern')
    
    if best_pattern:
        print(f"✅ Pattern: {best_pattern.get('type')} (confidence: {best_pattern.get('confidence', 0):.0%})")
        print(f"   Selector: {best_pattern.get('selector')}")
        print(f"   Count: {best_pattern.get('count')} items")
        
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        selector = best_pattern.get('selector', '')
        containers = soup.select(selector) if selector else []
        print(f"✅ Containers: {len(containers)}")
        
        if containers:
            print(f"🤖 Generating pattern with LLM...")
            container_samples = []
            for container in containers[:3]:
                container_samples.append({
                    'tag': container.name,
                    'classes': ' '.join(container.get('class', [])),
                    'text_preview': container.get_text()[:100]
                })
            
            try:
                pattern = await pattern_gen.generate_pattern(
                    html_sample=cleaned_html[:5000],
                    fields=fields,
                    repeating_containers=container_samples
                )
                print(f"✅ Generated pattern")
                
                extracted_items = semantic_extractor.extract(
                    html=cleaned_html,
                    semantic_pattern=pattern,
                    containers=containers
                )
                print(f"✅ Extracted {len(extracted_items)} items via HTML")
                print()
                
                if extracted_items and len(extracted_items) >= 10:
                    print("="*100)
                    print("📊 SUCCESS!")
                    print("="*100)
                    print(f"✅ Extracted {len(extracted_items)} high-quality products")
                    print(f"\nSample item:")
                    for key, value in list(extracted_items[0].items())[:5]:
                        print(f"  • {key}: {str(value)[:80]}")
                else:
                    print("⚠️  Extracted items but less than expected")
            except Exception as e:
                print(f"❌ Pattern generation/extraction failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"⚠️  No patterns detected")
    
    print()
    print("="*100)

if __name__ == "__main__":
    asyncio.run(test_amazon_full_workflow())




