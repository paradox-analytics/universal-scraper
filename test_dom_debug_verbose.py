#!/usr/bin/env python3
"""Verbose DOM pattern detection test"""

from universal_scraper.core.dom_pattern_detector import DOMPatternDetector

# Load eBay HTML
with open('ebay_debug_raw.html', 'r', encoding='utf-8') as f:
    html = f.read()

print(f"HTML size: {len(html):,} bytes\n")

detector = DOMPatternDetector()
patterns = detector.detect_patterns(html)

print("="*80)
print("📊 ALL ELEMENT SIGNATURES (top 20):")
print("="*80)
for i, sig in enumerate(patterns['element_signatures'][:20], 1):
    print(f"{i:2d}. {sig['signature']:50s} | count={sig['count']:4d} | text_len={sig['text_length']:6d}")

print("\n" + "="*80)
print("📊 DATA ATTRIBUTE PATTERNS:")
print("="*80)
for i, dat in enumerate(patterns['data_attributes'][:10], 1):
    print(f"{i:2d}. {dat['signature']:50s} | count={dat['count']:4d}")

print("\n" + "="*80)
print("📊 BEST PATTERN SELECTED:")
print("="*80)
if patterns['best_pattern']:
    bp = patterns['best_pattern']
    for key, value in bp.items():
        if key != 'sample':
            print(f"   {key}: {value}")
else:
    print("   None found!")

# Manually check for li.s-card
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'lxml')
li_s_cards = soup.find_all('li', class_='s-card')
print("\n" + "="*80)
print(f"✅ Manual check: Found {len(li_s_cards)} li.s-card elements")
print("="*80)







