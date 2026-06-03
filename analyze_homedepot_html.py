#!/usr/bin/env python3
"""
Extract and analyze product data from captured Home Depot HTML
"""
from pathlib import Path
import json
import re
from bs4 import BeautifulSoup

# Find the HTML sample
html_files = list(Path('html_samples').glob('*.html'))
if not html_files:
    print("❌ No HTML samples found")
    exit(1)

html_file = html_files[0]
print(f"📄 Analyzing: {html_file.name}")
print(f"   Size: {html_file.stat().st_size:,} bytes\n")

html = html_file.read_text()
soup = BeautifulSoup(html, 'html.parser')

# Extract JSON-LD structured data
print("="*80)
print("📊 JSON-LD STRUCTURED DATA")
print("="*80)

json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
print(f"\nFound {len(json_ld_scripts)} JSON-LD scripts:\n")

product_data = {}

for idx, script in enumerate(json_ld_scripts, 1):
    script_id = script.get('id', 'unknown')
    print(f"{idx}. Script ID: {script_id}")
    
    try:
        data = json.loads(script.string)
        print(f"   Type: {data.get('@type', 'unknown')}")
        
        # Extract key fields
        if data.get('@type') == 'Product':
            product_data['json_ld'] = data
            print(f"   ✅ PRODUCT DATA FOUND!")
            print(f"   Name: {data.get('name', 'N/A')}")
            print(f"   SKU: {data.get('sku', 'N/A')}")
            print(f"   Brand: {data.get('brand', {}).get('name', 'N/A')}")
            
            # Price info
            offers = data.get('offers', {})
            if offers:
                print(f"   Price: ${offers.get('price', 'N/A')}")
                print(f"   Currency: {offers.get('priceCurrency', 'N/A')}")
                print(f"   Availability: {offers.get('availability', 'N/A')}")
            
            # Rating
            rating = data.get('aggregateRating', {})
            if rating:
                print(f"   Rating: {rating.get('ratingValue', 'N/A')}/5")
                print(f"   Reviews: {rating.get('reviewCount', 'N/A')}")
        
        print()
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️  Failed to parse JSON: {e}\n")

# Look for other JSON data sources
print("="*80)
print("🔍 OTHER JSON DATA SOURCES")
print("="*80)

# Check for __NEXT_DATA__
if '__NEXT_DATA__' in html:
    print("\n✅ Found __NEXT_DATA__ (Next.js)")
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            next_data = json.loads(match.group(1))
            print(f"   Keys: {list(next_data.keys())}")
            product_data['next_data'] = next_data
        except:
            pass

# Check for window.digitalData
if 'window.digitalData' in html:
    print("\n✅ Found window.digitalData")
    match = re.search(r'window\.digitalData\s*=\s*({.*?});', html, re.DOTALL)
    if match:
        try:
            digital_data = json.loads(match.group(1))
            print(f"   Keys: {list(digital_data.keys())}")
            product_data['digital_data'] = digital_data
        except:
            pass

# Summary
print("\n" + "="*80)
print("📈 EXTRACTION SUMMARY")
print("="*80)

print(f"\nData Sources Found:")
for source in product_data.keys():
    print(f"   ✅ {source}")

if product_data:
    print(f"\n🎯 RECOMMENDED EXTRACTION METHOD: JSON-LD")
    print(f"   Reason: Structured product data available in application/ld+json")
    print(f"   Reliability: HIGH")
    print(f"   Speed: FAST (no need for complex parsing)")
    
    # Save extracted data
    output_file = Path('html_samples') / 'extracted_product_data.json'
    with open(output_file, 'w') as f:
        json.dump(product_data, f, indent=2)
    
    print(f"\n💾 Extracted data saved to: {output_file.name}")
else:
    print(f"\n⚠️  No structured product data found")
    print(f"   Fallback to HTML parsing required")

print("\n" + "="*80)
