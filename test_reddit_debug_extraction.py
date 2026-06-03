#!/usr/bin/env python3
"""
Debug Reddit extraction to understand why 0 items are extracted
"""
import asyncio
import os
import sys
import time
import logging
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.scraper import UniversalScraper

# Set DEBUG logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    print("\n" + "="*80)
    print("🐛 REDDIT EXTRACTION DEBUG")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    url = "https://www.reddit.com/r/webscraping/"
    context = "Extract Reddit posts with title, author, upvotes, comments count"
    
    print(f"🧪 Testing: {url}")
    print(f"📋 Context: {context}\n")
    
    # Initialize scraper with DEBUG logging
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
        log_level=logging.DEBUG
    )
    
    # Disable pagination
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    
    print("🔍 Step 1: Fetching HTML...\n")
    fetch_result = await scraper.html_fetcher.fetch(url)
    html = fetch_result['html']
    captured_json = fetch_result.get('captured_json', [])
    
    print(f"\n✅ Fetched HTML: {len(html)} bytes")
    print(f"📦 Captured JSON: {len(captured_json)} blobs")
    
    # Save HTML for inspection
    with open('debug_reddit_html.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"💾 Saved HTML to: debug_reddit_html.html")
    
    # Check if HTML contains posts
    if 'reddit' in html.lower():
        print("✅ HTML contains 'reddit'")
    if 'post' in html.lower():
        print("✅ HTML contains 'post'")
    
    print("\n🔍 Step 2: Detecting JSON sources...\n")
    json_results = scraper.json_detector.detect_and_extract(html, url, captured_json=captured_json)
    
    print(f"JSON Found: {json_results['json_found']}")
    print(f"Sources: {json_results.get('sources', [])}")
    print(f"Data type: {type(json_results.get('data'))}")
    
    if json_results.get('data'):
        print(f"Data keys: {list(json_results['data'].keys()) if isinstance(json_results['data'], dict) else 'not a dict'}")
        
        # Save JSON data for inspection
        with open('debug_reddit_json.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"💾 Saved JSON results to: debug_reddit_json.json")
    
    # Try extraction from JSON
    if json_results['json_found']:
        print("\n🔍 Step 3: Trying JSON extraction...\n")
        items = scraper.json_detector.extract_from_json(json_results['data'], fields=[])
        print(f"Extracted items from JSON: {len(items)}")
        if items:
            print(f"First item keys: {list(items[0].keys())}")
            print(f"First item preview: {json.dumps(items[0], indent=2, default=str)[:500]}")
    
    # Try HTML extraction
    print("\n🔍 Step 4: Trying HTML extraction...\n")
    print("Cleaning HTML...")
    clean_result = scraper.html_cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"Cleaned HTML: {len(cleaned_html)} bytes (reduction: {clean_result['reduction_percent']:.1f}%)")
    
    # Save cleaned HTML
    with open('debug_reddit_cleaned.html', 'w', encoding='utf-8') as f:
        f.write(cleaned_html)
    print(f"💾 Saved cleaned HTML to: debug_reddit_cleaned.html")
    
    print("\nGenerating structural hash...")
    hash_result = scraper.hash_generator.generate_hash(cleaned_html)
    structure_hash = hash_result['hash']
    print(f"Hash: {structure_hash}")
    
    print("\nGenerating extraction code...")
    gen_result = scraper.ai_generator.generate_extraction_code(
        cleaned_html,
        fields=[],
        url=url,
        extraction_context=context
    )
    extraction_code = gen_result['code']
    
    print(f"Generated code: {len(extraction_code)} bytes")
    
    # Save generated code
    with open('debug_reddit_code.py', 'w', encoding='utf-8') as f:
        f.write(extraction_code)
    print(f"💾 Saved generated code to: debug_reddit_code.py")
    
    print("\nExecuting extraction code...")
    extracted_data = scraper._execute_extraction_code(extraction_code, html)
    
    print(f"\n✅ Extracted {len(extracted_data)} items")
    
    if extracted_data:
        print(f"First item keys: {list(extracted_data[0].keys())}")
        print(f"\nFirst item:")
        print(json.dumps(extracted_data[0], indent=2, default=str)[:500])
    
    print("\n" + "="*80 + "\n")
    
    scraper.close()

if __name__ == "__main__":
    asyncio.run(main())







