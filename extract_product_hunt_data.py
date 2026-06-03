#!/usr/bin/env python3
"""
Product Hunt Data Extraction - Extract from data-* attributes
"""
import json
import re
from bs4 import BeautifulSoup

print("=" * 80)
print("🔍 PRODUCT HUNT DATA EXTRACTION - Analyzing data-* attrs")
print("=" * 80)
print()

# Read the saved HTML
with open("product_hunt_raw_debug.html", 'r', encoding='utf-8') as f:
    html = f.read()

print(f"📄 HTML size: {len(html):,} bytes")
print()

# Extract all data-* attribute values that look like JSON
print("🔍 Searching for JSON in data-* attributes...")

soup = BeautifulSoup(html, 'html.parser')

# Find all elements with data attributes
json_blobs = []
for element in soup.find_all():
    if not hasattr(element, 'attrs') or not element.attrs:
        continue
    
    for attr, value in element.attrs.items():
        if not attr.startswith('data-'):
            continue
            
        if not isinstance(value, str):
            continue
            
        # Check if value looks like JSON
        value = value.strip()
        if (value.startswith('{') and value.endswith('}')) or \
           (value.startswith('[') and value.endswith(']')):
                try:
                    parsed = json.loads(value)
                    json_blobs.append({
                        'attribute': attr,
                        'tag': element.name,
                        'data': parsed,
                        'size': len(value)
                    })
                    print(f"✅ Found JSON in <{element.name} {attr}>: {len(value):,} chars")
                except json.JSONDecodeError:
                    pass

print(f"\n📊 Total JSON blobs found: {len(json_blobs)}")
print()

# Analyze the largest/most interesting blobs
if json_blobs:
    # Sort by size
    json_blobs.sort(key=lambda x: x['size'], reverse=True)
    
    print("📋 Top 5 JSON blobs by size:")
    for i, blob in enumerate(json_blobs[:5], 1):
        print(f"\n{i}. <{blob['tag']} {blob['attribute']}>:")
        print(f"   Size: {blob['size']:,} chars")
        
        data = blob['data']
        
        # Check if it's a list
        if isinstance(data, list):
            print(f"   Type: Array with {len(data)} items")
            if data and isinstance(data[0], dict):
                print(f"   Sample keys: {list(data[0].keys())[:10]}")
        elif isinstance(data, dict):
            print(f"   Type: Object")
            print(f"   Top keys: {list(data.keys())[:10]}")
            
            # Look for products/posts
            if 'data' in data:
                nested_data = data['data']
                if isinstance(nested_data, dict):
                    print(f"   data keys: {list(nested_data.keys())[:10]}")
    
    # Save the largest blob for analysis
    largest = json_blobs[0]
    output_file = "product_hunt_extracted_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(largest['data'], f, indent=2)
    print(f"\n💾 Saved largest blob to: {output_file}")
    
    # Analyze for product/post data
    print("\n" + "=" * 80)
    print("🔍 ANALYZING FOR PRODUCT DATA")
    print("=" * 80)
    
    # Check all blobs for product listings
    for blob in json_blobs:
        data = blob['data']
        products_found = str(data).count('"__typename":"Post"')
        if products_found > 5:
            print(f"\n✅ Found {products_found} Posts in <{blob['tag']} {blob['attribute']}>")
            print(f"   Size: {blob['size']:,} chars")
            
            # Try to navigate to the posts
            if isinstance(data, dict):
                # Common GraphQL patterns
                paths_to_check = [
                    ['data', 'posts', 'edges'],
                    ['data', 'home', 'sections'],
                    ['sections'],
                    ['edges'],
                    ['posts']
                ]
                
                for path in paths_to_check:
                    current = data
                    found = True
                    for key in path:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            found = False
                            break
                    
                    if found and isinstance(current, list) and len(current) > 0:
                        print(f"   📍 Found array at path: {' → '.join(path)}")
                        print(f"      {len(current)} items")
                        if isinstance(current[0], dict):
                            print(f"      Sample keys: {list(current[0].keys())[:15]}")

print("\n" + "=" * 80)
print("✅ Analysis complete")
print("=" * 80)
