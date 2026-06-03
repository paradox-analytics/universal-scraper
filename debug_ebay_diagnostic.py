#!/usr/bin/env python3
"""
eBay Diagnostic Script - Deep Analysis of Extraction Failure

This script performs a comprehensive analysis of why eBay extraction is failing:
1. Fetches eBay with Camoufox
2. Saves the raw HTML
3. Analyzes JSON sources
4. Shows HTML structure analysis results
5. Shows AI-generated code
6. Manually inspects HTML for product elements
"""

import asyncio
import json
import os
from pathlib import Path
from bs4 import BeautifulSoup

# Add the project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def main():
    print("="*80)
    print("🔬 eBay DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()
    
    # Configuration
    url = "https://www.ebay.com/sch/i.html?_nkw=laptop"
    fields = ["title", "price", "condition", "shipping"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        return
    
    # Initialize scraper with Camoufox
    print("🦊 Initializing scraper with Camoufox...")
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False,  # Single page only
        log_level=20  # INFO level
    )
    
    try:
        # ==========================================
        # STEP 1: Fetch with Camoufox
        # ==========================================
        print("\n" + "="*80)
        print("STEP 1: Fetching eBay with Camoufox")
        print("="*80)
        
        # Fetch HTML
        fetch_result = await scraper.html_fetcher.fetch(url)
        html = fetch_result['html']
        captured_json = fetch_result.get('apis', [])
        
        print(f"✅ Fetched: {len(html):,} bytes")
        print(f"📦 Captured {len(captured_json)} API requests")
        
        # Save raw HTML
        html_file = "ebay_debug_raw.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"💾 Saved raw HTML to: {html_file}")
        
        # ==========================================
        # STEP 2: Analyze JSON Sources
        # ==========================================
        print("\n" + "="*80)
        print("STEP 2: JSON Source Analysis")
        print("="*80)
        
        json_results = scraper.json_detector.detect_and_extract(html, url, captured_json=captured_json)
        
        print(f"\n📊 JSON Detection Summary:")
        print(f"   • Type of json_results: {type(json_results)}")
        print(f"   • Keys: {json_results.keys() if isinstance(json_results, dict) else 'N/A'}")
        
        # Handle both dict and list responses
        if isinstance(json_results, dict):
            sources = json_results.get('sources', [])
            data = json_results.get('data', [])
        else:
            sources = []
            data = json_results if isinstance(json_results, list) else []
        
        print(f"   • Total sources found: {len(sources)}")
        print(f"   • Total items extracted from JSON: {len(data)}")
        
        if sources:
            print("\n🔍 JSON Sources Found:")
            for i, source_name in enumerate(sources, 1):
                print(f"   {i}. {source_name}")
        
        if data:
            print(f"\n🔍 JSON Data Details:")
            print(f"   Total data items: {len(data)}")
            for i, item in enumerate(data[:3], 1):  # First 3 items
                print(f"\n   Data Item {i}:")
                print(f"      Type: {type(item)}")
                print(f"      Size: {len(str(item))} bytes")
                
                # Check if it's actual product data or config
                if isinstance(item, dict):
                    keys = list(item.keys())[:10]
                    print(f"      Keys: {keys}")
                    
                    # Check for product-like keys
                    product_keys = ['title', 'price', 'name', 'product', 'item', 'listing']
                    has_product_data = any(k.lower() in str(keys).lower() for k in product_keys)
                    print(f"      Looks like product data: {has_product_data}")
                
                # Save data to file
                data_file = f"ebay_debug_json_data_{i}.json"
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(item, f, indent=2)
                print(f"      💾 Saved to: {data_file}")
        
        # ==========================================
        # STEP 3: HTML Structure Analysis
        # ==========================================
        print("\n" + "="*80)
        print("STEP 3: HTML Structure Analysis (LLM-based)")
        print("="*80)
        
        # Clean HTML first (as scraper does)
        clean_result = scraper.html_cleaner.clean(html)
        cleaned_html = clean_result['html']
        print(f"🧹 Cleaned HTML: {len(cleaned_html):,} bytes ({clean_result['reduction_percent']:.1f}% reduction)")
        print(f"   Original: {clean_result['original_size']:,} bytes")
        print(f"   Cleaned: {clean_result['cleaned_size']:,} bytes")
        
        # Save cleaned HTML
        cleaned_file = "ebay_debug_cleaned.html"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_html)
        print(f"💾 Saved cleaned HTML to: {cleaned_file}")
        
        # Check if cleaned HTML is suspiciously small
        if len(cleaned_html) < 1000:
            print(f"\n⚠️  WARNING: Cleaned HTML is only {len(cleaned_html)} bytes!")
            print(f"   This likely means the HTML cleaner removed all product data.")
            print(f"   Cleaned content: {repr(cleaned_html[:500])}")
        
        # Run structure analysis
        if scraper.html_structure_analyzer:
            print("\n🤖 Running LLM structure analysis...")
            structure_analysis = await scraper.html_structure_analyzer.analyze(url, cleaned_html)
            
            print(f"\n📋 Structure Analysis Results:")
            print(f"   • Repeating Element: {structure_analysis.get('repeating_element', 'N/A')}")
            print(f"   • Element Type: {structure_analysis.get('element_type', 'N/A')}")
            print(f"   • Data Location: {structure_analysis.get('data_location', 'N/A')}")
            print(f"   • Confidence: {structure_analysis.get('confidence', 0)}")
            
            if structure_analysis.get('key_selectors'):
                print(f"\n   🎯 Key Selectors:")
                for selector in structure_analysis['key_selectors'][:5]:
                    print(f"      • {selector}")
            
            if structure_analysis.get('field_mappings'):
                print(f"\n   🗺️  Field Mappings:")
                for field, mapping in list(structure_analysis['field_mappings'].items())[:5]:
                    print(f"      • {field}: {mapping}")
            
            # Save structure analysis
            structure_file = "ebay_debug_structure_analysis.json"
            with open(structure_file, 'w', encoding='utf-8') as f:
                json.dump(structure_analysis, f, indent=2)
            print(f"\n   💾 Saved structure analysis to: {structure_file}")
        
        # ==========================================
        # STEP 4: AI Code Generation
        # ==========================================
        print("\n" + "="*80)
        print("STEP 4: AI Code Generation Analysis")
        print("="*80)
        
        print("\n🤖 Generating extraction code with AI (3 iterations)...")
        
        # Generate structural hash
        struct_hash = scraper.hash_generator.generate(cleaned_html, url)
        
        # Check cache
        cached_code = scraper.code_cache.get(struct_hash, fields)
        if cached_code:
            print("✅ Code found in cache")
            code = cached_code
        else:
            print("❌ No cached code, generating new...")
            
            # Run structure analysis if not already done
            if not scraper.html_structure_analyzer:
                structure_analysis = None
            else:
                structure_analysis = await scraper.html_structure_analyzer.analyze(url, cleaned_html)
            
            # Generate code with iterations
            code_result = await scraper.ai_generator.generate_extraction_code(
                cleaned_html=cleaned_html,
                fields=fields,
                url=url,
                structure_analysis=structure_analysis,
                max_iterations=3
            )
            
            code = code_result['code']
            print(f"\n📝 Code generated after {code_result.get('iterations', 1)} iteration(s)")
            print(f"   Success: {code_result.get('success', False)}")
            print(f"   Items extracted during generation: {code_result.get('items_extracted', 0)}")
        
        # Save generated code
        code_file = "ebay_debug_generated_code.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"💾 Saved generated code to: {code_file}")
        
        print(f"\n📄 Generated Code Preview (first 50 lines):")
        print("-" * 80)
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines[:50], 1):
            print(f"{i:3d} | {line}")
        if len(code_lines) > 50:
            print(f"... ({len(code_lines) - 50} more lines)")
        print("-" * 80)
        
        # Execute the generated code
        print("\n⚡ Executing generated code...")
        try:
            soup = BeautifulSoup(cleaned_html, 'lxml')
            exec_globals = {'soup': soup, 'items': []}
            exec(code, exec_globals)
            extracted_items = exec_globals.get('items', [])
            print(f"✅ Code executed: {len(extracted_items)} items extracted")
            
            if extracted_items:
                print(f"\n📋 First extracted item:")
                print(json.dumps(extracted_items[0], indent=2))
            else:
                print("❌ No items extracted by generated code")
        except Exception as e:
            print(f"❌ Code execution failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ==========================================
        # STEP 5: Manual HTML Inspection
        # ==========================================
        print("\n" + "="*80)
        print("STEP 5: Manual HTML Inspection for Product Elements")
        print("="*80)
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Common eBay product selectors to try
        selectors_to_try = [
            ('li.s-item', 'Product list item'),
            ('.srp-results li', 'Search results list item'),
            ('.s-item', 'Product item class'),
            ('div[class*="s-item"]', 'Product item (fuzzy match)'),
            ('li[data-item-id]', 'Item with data-item-id'),
            ('[class*="item"][class*="card"]', 'Item card'),
        ]
        
        print("\n🔍 Trying common eBay selectors:")
        for selector, description in selectors_to_try:
            elements = soup.select(selector)
            print(f"\n   {description}: {selector}")
            print(f"      Found: {len(elements)} elements")
            
            if elements:
                # Show first element's structure
                first = elements[0]
                print(f"      First element preview:")
                print(f"         Tag: <{first.name}>")
                print(f"         Classes: {first.get('class', [])}")
                print(f"         ID: {first.get('id', 'N/A')}")
                print(f"         Attributes: {list(first.attrs.keys())[:10]}")
                print(f"         Text length: {len(first.get_text(strip=True))} chars")
                
                # Save first element HTML
                element_file = f"ebay_debug_element_{selector.replace(' ', '_').replace('[', '').replace(']', '').replace('*', 'star').replace('=', '')[:30]}.html"
                with open(element_file, 'w', encoding='utf-8') as f:
                    f.write(str(first.prettify()))
                print(f"         💾 Saved first element to: {element_file}")
        
        # Check for specific eBay patterns
        print("\n🔍 Checking for eBay-specific patterns:")
        
        # Look for price elements
        price_elements = soup.find_all(class_=lambda x: x and 's-item__price' in x if x else False)
        print(f"   • Elements with 's-item__price': {len(price_elements)}")
        
        # Look for title elements
        title_elements = soup.find_all(class_=lambda x: x and 's-item__title' in x if x else False)
        print(f"   • Elements with 's-item__title': {len(title_elements)}")
        
        # Look for data attributes
        data_attrs = [tag for tag in soup.find_all() if any(attr.startswith('data-') for attr in tag.attrs)]
        print(f"   • Elements with data-* attributes: {len(data_attrs)}")
        if data_attrs:
            print(f"      Sample data attributes: {list(data_attrs[0].attrs.keys())[:5]}")
        
        # Look for JSON in script tags
        script_tags = soup.find_all('script', type='application/json')
        print(f"   • <script type='application/json'> tags: {len(script_tags)}")
        
        script_ld_tags = soup.find_all('script', type='application/ld+json')
        print(f"   • <script type='application/ld+json'> tags: {len(script_ld_tags)}")
        
        # ==========================================
        # STEP 6: Summary & Recommendations
        # ==========================================
        print("\n" + "="*80)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*80)
        
        print("\n🔍 Key Findings:")
        print(f"   ✅ HTML fetched: {len(html):,} bytes")
        print(f"   📦 JSON sources found: {len(sources)}")
        print(f"   📦 Items from JSON: {len(data)}")
        print(f"   🧹 HTML cleaned: {len(cleaned_html):,} bytes")
        print(f"   🤖 AI code generated: {len(code)} chars")
        print(f"   ⚡ Items extracted by AI code: {len(extracted_items) if 'extracted_items' in locals() else 0}")
        
        print("\n💡 Next Steps:")
        print("   1. Review the saved HTML files to understand eBay's structure")
        print("   2. Check if JSON sources contain product data")
        print("   3. Review the structure analysis to see if LLM identified correct elements")
        print("   4. Review generated code to see what selectors it's using")
        print("   5. Check manual inspection results to find the actual product elements")
        
        print("\n📁 Files created:")
        for file in [
            "ebay_debug_raw.html",
            "ebay_debug_cleaned.html",
            "ebay_debug_structure_analysis.json",
            "ebay_debug_generated_code.py"
        ]:
            if Path(file).exists():
                size = Path(file).stat().st_size
                print(f"   • {file} ({size:,} bytes)")
        
        # List any JSON source files
        for file in Path('.').glob('ebay_debug_json_source_*.json'):
            size = file.stat().st_size
            print(f"   • {file.name} ({size:,} bytes)")
        
        # List any element files
        for file in Path('.').glob('ebay_debug_element_*.html'):
            size = file.stat().st_size
            print(f"   • {file.name} ({size:,} bytes)")
        
    finally:
        scraper.close()
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())

