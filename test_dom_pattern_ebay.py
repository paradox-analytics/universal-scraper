#!/usr/bin/env python3
"""
Quick test: Does DOM pattern detection find eBay's s-card pattern?
"""

from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from universal_scraper.core.smart_html_sampler import SmartHTMLSampler

# Load the eBay HTML we fetched earlier
with open('ebay_debug_raw.html', 'r', encoding='utf-8') as f:
    html = f.read()

print("="*80)
print("🔬 Testing DOM Pattern Detection on eBay")
print("="*80)
print(f"📄 HTML size: {len(html):,} bytes\n")

# Test DOM pattern detector
detector = DOMPatternDetector()
patterns = detector.detect_patterns(html)

print("📊 RESULTS:")
print(f"   Confidence: {patterns['confidence']:.2f}")
print(f"   Best pattern: {patterns['best_pattern']}")
print()

if patterns['best_pattern']:
    bp = patterns['best_pattern']
    print(f"✅ FOUND PATTERN:")
    print(f"   Type: {bp['type']}")
    print(f"   Selector: {bp['selector']}")
    print(f"   Count: {bp['count']}")
    print(f"   Confidence: {bp['confidence']:.2f}")
    print(f"   Extraction hint: {bp.get('extraction_hint', 'N/A')}")
    print()
    
    # Show sample
    print(f"📋 Sample element (first 500 chars):")
    print(bp.get('sample', 'N/A')[:500])
    print()

# Test smart HTML sampler
print("="*80)
print("🔬 Testing Smart HTML Sampler")
print("="*80)

sampler = SmartHTMLSampler(max_chars=8000)
smart_sample = sampler.sample_html(html, patterns)

print(f"📊 Smart sample size: {len(smart_sample):,} chars")
print(f"   Reduction: {(1 - len(smart_sample)/len(html))*100:.1f}%")
print()
print("📋 First 1000 chars of smart sample:")
print("="*80)
print(smart_sample[:1000])
print("="*80)

# Check if it found s-card
if 's-card' in str(patterns['best_pattern']):
    print("\n🎉 SUCCESS: Found eBay's s-card pattern!")
else:
    print("\n⚠️  Did not find s-card pattern")
    print("   Top patterns found:")
    for sig in patterns['element_signatures'][:5]:
        print(f"      - {sig['signature']}: {sig['count']} occurrences")







