#!/usr/bin/env python3
"""
Extract all Product Hunt product arrays
"""
import json
import re

html = open('product_hunt_raw_debug.html').read()

print("=" * 80)
print("🎯 EXTRACTING ALL PRODUCT HUNT ARRAYS")
print("=" * 80)
print()

# Find "items":[{"__typename":"Post"
pattern = r'"items":\s*\[{"__typename":"Post"'
matches = list(re.finditer(pattern, html))

print(f"Found {len(matches)} 'items' arrays with Posts")
print()

all_products = []

for idx, match in enumerate(matches, 1):
    start_pos = match.start()
    
    # Find the start of "items"
    items_start = html.rfind('"items"', start_pos - 100, start_pos + 10)
    
    if items_start == -1:
        continue
    
    # Now find the opening [ for the array
    array_start = html.find('[', items_start)
    
    # Find the matching closing ]
    bracket_count = 0
    array_end = array_start
    for i in range(array_start, min(len(html), array_start + 500000)):
        if html[i] == '[':
            bracket_count += 1
        elif html[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                array_end = i + 1
                break
    
    array_json = html[array_start:array_end]
    
    try:
        items = json.loads(array_json)
        print(f"Array {idx}:")
        print(f"  Position: {array_start:,}")
        print(f"  Size: {len(array_json):,} chars")
        print(f"  Items: {len(items)}")
        
        # Count Posts vs Ads
        posts = [item for item in items if item.get('__typename') == 'Post']
        ads = [item for item in items if item.get('__typename') == 'Ad']
        
        print(f"  Posts: {len(posts)}, Ads: {len(ads)}")
        
        if posts:
            print(f"  Sample: {posts[0].get('name', 'Unknown')}")
            all_products.extend(posts)
        
        print()
        
    except json.JSONDecodeError as e:
        print(f"Array {idx}: Failed to parse - {str(e)[:50]}")

print("=" * 80)
print(f"📦 TOTAL PRODUCTS EXTRACTED: {len(all_products)}")
print("=" * 80)

if all_products:
    # Save all products
    with open("product_hunt_all_products.json", 'w', encoding='utf-8') as f:
        json.dump(all_products, f, indent=2)
    print(f"💾 Saved to: product_hunt_all_products.json")
    
    # Show sample
    print()
    print("🎯 Sample Products:")
    for i, product in enumerate(all_products[:5], 1):
        print(f"\n{i}. {product.get('name')}")
        print(f"   {product.get('tagline', '')[:80]}")
        print(f"   ⬆️  {product.get('latestScore',0)} votes")
        print(f"   💬 {product.get('commentsCount', 0)} comments")
