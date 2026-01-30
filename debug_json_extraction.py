import asyncio
import json
import os
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def debug_site(url: str, site_name: str, context: str, target_fields: list):
    """Debug JSON extraction for a single site"""
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING: {site_name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Target fields: {target_fields}")
    print(f"Context: {context}\n")
    
    # Step 1: Fetch the page
    print("📥 Step 1: Fetching page with browser...")
    fetcher = HybridFetcher(
        force_mode='browser',
        proxy_config=None,
        browser_timeout=30000,
        enable_cache=False
    )
    
    fetch_result = await fetcher.fetch(url)
    html = fetch_result['html']
    captured_json = fetch_result.get('captured_json', [])
    
    print(f"✅ Page loaded: {len(html)} bytes")
    print(f"✅ Captured {len(captured_json)} JSON blob(s)\n")
    
    # Step 2: Detect JSON sources
    print("📦 Step 2: Detecting all JSON sources...")
    json_detector = JSONDetector()
    json_results = json_detector.detect_and_extract(html, url, captured_json)
    
    sources = json_results.get('sources', [])
    data_list = json_results.get('data', [])
    
    print(f"✅ Found {len(sources)} JSON source(s)\n")
    
    # Step 3: Examine each source
    print(f"{'='*80}")
    print("📊 DETAILED SOURCE ANALYSIS")
    print(f"{'='*80}\n")
    
    for i, (source_name, source_data) in enumerate(zip(sources, data_list), 1):
        print(f"--- Source {i}: {source_name} ---")
        
        # Show source type
        if source_name.startswith('captured_json'):
            print("   Type: API Response (captured)")
        elif source_name.startswith('embedded'):
            print("   Type: Embedded JSON in HTML")
        else:
            print("   Type: Unknown")
        
        # Show data structure
        if isinstance(source_data, dict):
            print(f"   Structure: Dictionary with {len(source_data)} keys")
            print(f"   Keys: {list(source_data.keys())[:10]}")  # First 10 keys
            
            # Look for arrays
            arrays_found = []
            for key, value in source_data.items():
                if isinstance(value, list) and len(value) > 0:
                    arrays_found.append({
                        'key': key,
                        'length': len(value),
                        'first_item_type': type(value[0]).__name__,
                        'first_item_keys': list(value[0].keys())[:10] if isinstance(value[0], dict) else None
                    })
            
            if arrays_found:
                print(f"   ✅ Found {len(arrays_found)} array(s):")
                for arr in arrays_found[:3]:  # Show first 3
                    print(f"      • {arr['key']}: {arr['length']} items")
                    if arr['first_item_keys']:
                        print(f"        First item keys: {arr['first_item_keys']}")
            else:
                print("   ⚠️ No arrays found at top level")
                # Look deeper
                print("   🔍 Searching nested objects...")
                nested_arrays = find_nested_arrays(source_data, max_depth=3)
                if nested_arrays:
                    print(f"   ✅ Found {len(nested_arrays)} nested array(s):")
                    for path, arr_info in nested_arrays[:3]:
                        print(f"      • {path}: {arr_info['length']} items")
                        if arr_info.get('first_item_keys'):
                            print(f"        First item keys: {arr_info['first_item_keys']}")
                
        elif isinstance(source_data, list):
            print(f"   Structure: Array with {len(source_data)} items")
            if len(source_data) > 0:
                print(f"   First item type: {type(source_data[0]).__name__}")
                if isinstance(source_data[0], dict):
                    print(f"   First item keys: {list(source_data[0].keys())[:10]}")
        else:
            print(f"   Structure: {type(source_data).__name__}")
        
        # Try extraction
        print("   🎯 Attempting extraction...")
        # Wrap in list if needed
        json_input = [source_data] if isinstance(source_data, dict) else source_data
        extracted = json_detector.extract_from_json(json_input, target_fields)
        print(f"   Result: {len(extracted)} item(s) extracted")
        
        if len(extracted) > 0:
            print(f"   ✅ SUCCESS! Sample item:")
            print(f"      {json.dumps(extracted[0], indent=6)[:500]}")
        else:
            print("   ❌ FAILED - No items extracted")
        
        print()
    
    # Step 4: Summary
    print(f"{'='*80}")
    print("📋 SUMMARY")
    print(f"{'='*80}")
    total_extracted = 0
    for d in data_list:
        json_input = [d] if isinstance(d, dict) else d
        total_extracted += len(json_detector.extract_from_json(json_input, target_fields))
    print(f"Total items extracted across all sources: {total_extracted}")
    print(f"Total JSON sources available: {len(sources)}")
    print()

def find_nested_arrays(obj, path="", max_depth=3, current_depth=0):
    """Recursively find arrays in nested objects"""
    if current_depth >= max_depth:
        return []
    
    results = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            if isinstance(value, list) and len(value) > 0:
                arr_info = {
                    'length': len(value),
                    'first_item_type': type(value[0]).__name__
                }
                if isinstance(value[0], dict):
                    arr_info['first_item_keys'] = list(value[0].keys())[:10]
                results.append((new_path, arr_info))
            elif isinstance(value, dict):
                results.extend(find_nested_arrays(value, new_path, max_depth, current_depth + 1))
    
    return results

async def main():
    print("\n" + "="*80)
    print("🔬 JSON EXTRACTION DEBUG")
    print("="*80)
    print("This script will examine the captured JSON and show why extraction is failing.\n")
    
    # Test configurations
    tests = [
        {
            'site': 'Apify Homepage',
            'url': 'https://apify.com/',
            'context': 'Extract featured Actors/scrapers with their name, description, author, run count, and rating',
            'fields': ['name', 'description', 'author', 'run_count', 'rating']
        },
        {
            'site': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes, comments count, post URL, and timestamp',
            'fields': ['title', 'author', 'upvotes', 'comments_count', 'post_url', 'timestamp']
        }
    ]
    
    for test in tests:
        await debug_site(test['url'], test['site'], test['context'], test['fields'])
    
    print("\n" + "="*80)
    print("🏁 DEBUG COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

