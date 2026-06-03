#!/usr/bin/env python3
"""
Product Hunt Data Extractor - Extract from Next.js RSC Payload
"""
import json
import re

print("=" * 80)
print("🎯 PRODUCT HUNT - Complete Data Extraction")
print("=" * 80)
print()

# Read HTML
with open("product_hunt_raw_debug.html", 'r', encoding='utf-8') as f:
    html = f.read()

print(f"📄 HTML size: {len(html):,} bytes")
print()

# Strategy 1: Find the large JSON blob with product data
print("🔍 Strategy 1: Looking for inline JSON with products...")

# Find the position of the first Post
pos = html.find('"__typename":"Post"')
if pos > 0:
    print(f"✅ Found Post data at position {pos:,}")
    
    # Work backwards to find the start of the JSON object
    # Look for opening brace
    start = pos
    brace_count = 0
    for i in range(pos, max(0, pos - 10000), -1):
        if html[i] == '}':
            brace_count += 1
        elif html[i] == '{':
            if brace_count == 0:
                start = i
                break
            brace_count -= 1
    
    # Work forwards to find the end
    end = pos
    brace_count = 0
    for i in range(start, min(len(html), start + 100000)):
        if html[i] == '{':
            brace_count += 1
        elif html[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    print(f"   JSON range: {start:,} to {end:,} ({end-start:,} chars)")
    
    # Extract and parse
    json_str = html[start:end]
    
    try:
        data = json.loads(json_str)
        print(f"✅ Successfully parsed JSON!")
        print()
        
        # Save it
        with open("product_hunt_main_data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("💾 Saved to: product_hunt_main_data.json")
        print()
        
        # Analyze structure
        print("📊 Data Structure Analysis:")
        print(f"   Top-level keys: {list(data.keys())[:20]}")
        print()
        
        # Look for products using common patterns
        def find_products(obj, path=""):
            results = []
            
            if isinstance(obj, dict):
                # Check if this looks like a product
                if obj.get('__typename') == 'Post':
                    results.append((path, obj))
                
                # Recurse
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    results.extend(find_products(value, new_path))
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    results.extend(find_products(item, new_path))
            
            return results
        
        products = find_products(data)
        print(f"📦 Found {len(products)} products")
        
        if products:
            print()
            print("🎯 Sample Products:")
            for i, (path, product) in enumerate(products[:3], 1):
                print(f"\n{i}. {product.get('name', 'Unknown')}")
                print(f"   Tagline: {product.get('tagline', 'N/A')[:80]}")
                print(f"   Votes: {product.get('latestScore', 'N/A')}")
                print(f"   Path: {path[:100]}")
            
            # Show all paths where products are found
            print()
            print("📍 Product Locations:")
            paths_summary = {}
            for path, _ in products:
                # Get the path up to the array index
                base_path = re.sub(r'\[\d+\].*$', '[*]', path)
                paths_summary[base_path] = paths_summary.get(base_path, 0) + 1
            
            for path, count in sorted(paths_summary.items(), key=lambda x: -x[1]):
                print(f"   {path}: {count} products")
            
            # Extract just the products
            products_only = [p for _, p in products]
            with open("product_hunt_products_extracted.json", 'w', encoding='utf-8') as f:
                json.dump(products_only, f, indent=2)
            print()
            print(f"💾 Saved {len(products_only)} products to: product_hunt_products_extracted.json")
            
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(f"   Showing first 500 chars:")
        print(json_str[:500])

print()
print("=" * 80)
print("✅ Extraction complete")
print("=" * 80)
