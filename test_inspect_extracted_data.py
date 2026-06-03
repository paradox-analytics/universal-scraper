#!/usr/bin/env python3
"""
Inspect actual extracted data from QA tests
Shows what fields and values were captured from each source
"""
import asyncio
import os
import sys
from pathlib import Path
import json

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


async def inspect_source(url: str, fields_nl: str, name: str):
    """Extract and inspect data from a single source"""
    print("\n" + "="*100)
    print(f"🔍 INSPECTING: {name}")
    print("="*100)
    print(f"URL: {url}")
    print(f"Fields: {fields_nl}")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # Initialize
    pattern_gen = SemanticPatternGenerator(api_key=api_key)
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    json_detector = JSONDetector()
    semantic_extractor = SemanticExtractor()
    html_cleaner = SmartHTMLCleaner()
    dom_detector = DOMPatternDetector()
    
    # Fetch
    result = await fetcher.fetch(url)
    html = result['html']
    
    # Parse fields
    try:
        fields = await pattern_gen._parse_natural_language_fields(fields_nl, html[:3000])
    except:
        fields = pattern_gen._extract_fields_from_text(fields_nl)
    
    print(f"📋 Parsed fields: {fields}")
    print()
    
    # JSON Detection
    json_detection_result = json_detector.detect_and_extract(
        html=html,
        url=url,
        captured_json=result.get('json_data', [])
    )
    
    extracted_items = []
    method = None
    
    if json_detection_result['json_found']:
        all_json = json_detection_result['data']
        extracted_items = json_detector.extract_from_json(json_data=all_json, fields=fields)
        
        if extracted_items and len(extracted_items) >= 3:
            is_sufficient = json_detector.is_json_sufficient(
                json_results=json_detection_result,
                fields=fields
            )
            
            if is_sufficient:
                method = "JSON"
                print(f"✅ Using JSON extraction ({len(extracted_items)} items)")
            else:
                print(f"⚠️  JSON rejected (quality validation)")
                extracted_items = []
    
    # HTML Fallback
    if not extracted_items or len(extracted_items) < 3:
        print(f"🌐 Using HTML extraction")
        
        cleaned_result = html_cleaner.clean(html)
        cleaned_html = cleaned_result['html']
        
        dom_patterns = dom_detector.detect_patterns(cleaned_html)
        best_pattern = dom_patterns.get('best_pattern')
        
        if best_pattern:
            selector = best_pattern.get('selector', '')
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            containers = soup.select(selector) if selector else []
            
            if containers:
                container_samples = []
                for container in containers[:5]:
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
                    
                    extracted_items = semantic_extractor.extract(
                        html=cleaned_html,
                        semantic_pattern=pattern,
                        containers=containers
                    )
                    method = "HTML"
                    print(f"✅ Extracted {len(extracted_items)} items via HTML")
                except Exception as e:
                    print(f"❌ HTML extraction failed: {e}")
    
    print()
    print("="*100)
    print(f"📊 EXTRACTED DATA - {method} Method")
    print("="*100)
    
    if not extracted_items:
        print("❌ NO DATA EXTRACTED")
        return
    
    print(f"\nTotal items: {len(extracted_items)}")
    print()
    
    # Show first 3 items in detail
    for i, item in enumerate(extracted_items[:3], 1):
        print(f"--- Item {i} ---")
        for key, value in item.items():
            if key != '_metadata':
                value_str = str(value)
                if len(value_str) > 80:
                    value_str = value_str[:77] + "..."
                print(f"  {key}: {value_str}")
        print()
    
    if len(extracted_items) > 3:
        print(f"... and {len(extracted_items) - 3} more items")
    
    # Analyze field coverage
    print()
    print("📈 Field Coverage Analysis:")
    all_fields = set()
    for item in extracted_items:
        all_fields.update(k for k in item.keys() if k != '_metadata')
    
    print(f"  Requested fields: {fields}")
    print(f"  Extracted fields: {sorted(all_fields)}")
    
    # Check for empty/null values
    empty_counts = {field: 0 for field in all_fields}
    for item in extracted_items:
        for field in all_fields:
            if field not in item or not item[field] or str(item[field]).strip() in ['', 'None', 'null', 'N/A']:
                empty_counts[field] += 1
    
    print(f"  Empty value rates:")
    for field, count in sorted(empty_counts.items()):
        rate = (count / len(extracted_items)) * 100
        status = "✅" if rate < 20 else ("⚠️" if rate < 50 else "❌")
        print(f"    {status} {field}: {rate:.1f}% empty ({count}/{len(extracted_items)})")
    
    print()


async def main():
    sources = [
        ("https://www.leafly.com/dispensary-info/seven-point/menu", 
         "Extract product name, price, and description for all products", 
         "Leafly (JS-heavy)"),
        
        ("https://www.amazon.com/s?k=laptop", 
         "Get product title, price, and rating", 
         "Amazon (E-commerce)"),
        
        ("https://www.ebay.com/sch/i.html?_nkw=macbook", 
         "Get item title, price, and condition", 
         "eBay (Auction)"),
        
        ("https://old.reddit.com/r/programming/", 
         "Extract post title, author, and upvotes", 
         "Reddit (Social)"),
        
        ("https://news.ycombinator.com/", 
         "Extract article title, points, and comments count", 
         "Hacker News (News)"),
    ]
    
    for url, fields, name in sources:
        await inspect_source(url, fields, name)
        await asyncio.sleep(2)  # Brief pause between tests


if __name__ == "__main__":
    asyncio.run(main())




