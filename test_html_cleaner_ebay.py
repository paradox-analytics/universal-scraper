#!/usr/bin/env python3
"""
Quick test to see why HTML cleaner is removing all eBay content
"""

from pathlib import Path
from universal_scraper.core.html_cleaner import SmartHTMLCleaner

# Load the raw eBay HTML
html_file = "ebay_debug_raw.html"
if not Path(html_file).exists():
    print(f"❌ {html_file} not found. Run debug_ebay_diagnostic.py first.")
    exit(1)

with open(html_file, 'r', encoding='utf-8') as f:
    html = f.read()

print(f"📄 Original HTML: {len(html):,} bytes")
print()

# Test the cleaner
cleaner = SmartHTMLCleaner()
result = cleaner.clean(html)

cleaned_html = result['html']

print(f"🧹 Cleaned HTML: {len(cleaned_html):,} bytes")
print(f"   Reduction: {result['reduction_percent']:.1f}%")
print()

if len(cleaned_html) < 1000:
    print(f"⚠️  WARNING: Cleaned HTML is suspiciously small!")
    print(f"\n📋 Full cleaned content:")
    print("="*80)
    print(cleaned_html)
    print("="*80)
else:
    print(f"✅ Cleaned HTML looks reasonable")
    print(f"\n📋 First 2000 chars:")
    print("="*80)
    print(cleaned_html[:2000])
    print("="*80)

# Save cleaned HTML
with open('ebay_cleaned_test.html', 'w', encoding='utf-8') as f:
    f.write(cleaned_html)
print(f"\n💾 Saved cleaned HTML to: ebay_cleaned_test.html")







