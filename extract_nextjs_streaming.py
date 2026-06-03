#!/usr/bin/env python3
"""
Product Hunt Next.js Streaming Data Extraction
Extracts data from self.__next_f.push() calls
"""
import json
import re

print("=" * 80)
print("🔍 PRODUCT HUNT - Next.js Streaming Data Extraction")
print("=" * 80)
print()

# Read the HTML
with open("product_hunt_raw_debug.html", 'r', encoding='utf-8') as f:
    html = f.read()

print(f"📄 HTML size: {len(html):,} bytes")
print()

# Extract all self.__next_f.push() calls
print("🔍 Extracting self.__next_f.push() calls...")

pattern = r'self\.__next_f\.push\((.*?)\)\s*(?:</script>|$)'
matches = re.findall(pattern, html, re.DOTALL)

print(f"✅ Found {len(matches)} push() calls")
print()

# Parse each one
all_data = []
for i, match in enumerate(matches, 1):
    try:
        # The match is a JSON array like: [1,"some data"]
        parsed = json.loads(match)
        all_data.append(parsed)
        
        if isinstance(parsed, list) and len(parsed) >= 2:
            chunk_id = parsed[0]
            chunk_data = parsed[1]
            
            # Check if chunk_data contains product/post info
            if isinstance(chunk_data, str):
                post_count = chunk_data.count('"__typename":"Post"')
                if post_count > 0:
                    print(f"✅ Chunk {i} (ID: {chunk_id}): Contains {post_count} Posts")
                    print(f"   Data size: {len(chunk_data):,} chars")
                    
                    # Save this chunk
                    output_file = f"product_hunt_chunk_{chunk_id}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(chunk_data)
                    print(f"   💾 Saved to: {output_file}")
    except json.JSONDecodeError as e:
        print(f"⚠️  Chunk {i}: Failed to parse - {str(e)[:50]}")

# Save all data
output_file = "product_hunt_all_chunks.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2)
print(f"\n💾 Saved all chunks to: {output_file}")

# Now try to find and parse the actual GraphQL data
print("\n" + "=" * 80)
print("🔍 SEARCHING FOR GRAPHQL DATA")
print("=" * 80)
print()

for i, data in enumerate(all_data, 1):
    if isinstance(data, list) and len(data) >= 2:
        chunk_data = data[1]
        
        if isinstance(chunk_data, str):
            # Try to find JSON objects within the string
            # Next.js streaming format often has escaped JSON
            unescaped = chunk_data.replace('\\\\', '\\')
            
            # Look for GraphQL response pattern
            if '"edges"' in unescaped and '("typename":"Post"' in unescaped or '__typename\\":\\"Post' in unescaped:
                print(f"📦 Chunk {i}: Contains GraphQL edges structure")
                
                # Try to extract the JSON
                # Pattern: look for objects that start with { and contain edges
                json_pattern = r'(\{[^{}]*"edges"[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                potential_json = re.findall(json_pattern, unescaped)
                
                if potential_json:
                    print(f"   Found {len(potential_json)} potential JSON objects")

print("\n" + "=" * 80)
print("✅ Extraction complete")
print("=" * 80)
