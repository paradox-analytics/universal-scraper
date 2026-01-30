#!/usr/bin/env python3
"""
Product Hunt Diagnostic Analysis
Investigate why Product Hunt extraction is failing (0 items extracted)
"""

import asyncio
import json
import os
from pathlib import Path
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector


async def main():
    print("="*80)
    print("🔬 PRODUCT HUNT DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()
    
    # Configuration
    url = "https://www.producthunt.com/"
    fields = ["name", "tagline", "upvotes"]
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        return

    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=True,
            headless=True,
            enable_auto_pagination=False,
            extraction_context="Extract product listings with name, tagline, and upvotes"
        )
        
        print(f"🎯 Analyzing: {url}")
        print(f"📋 Fields: {', '.join(fields)}")
        print()

        # STEP 1: Fetch HTML
        print("="*80)
        print("STEP 1: Fetching HTML with Camoufox")
        print("="*80)
        fetch_result = await scraper.html_fetcher.fetch(url, fetch_mode='browser', use_camoufox=True)
        html = fetch_result['html']
        captured_json = fetch_result.get('captured_json', [])
        
        print(f"✅ HTML fetched: {len(html):,} bytes")
        
        # Check for common anti-bot/auth indicators
        html_lower = html.lower()
        auth_indicators = ['login', 'sign in', 'sign up', 'create account']
        bot_indicators = ['cloudflare', 'captcha', 'access denied', 'unusual traffic']
        
        print(f"\n📊 Page Analysis:")
        print(f"   • Contains auth keywords: {any(ind in html_lower for ind in auth_indicators)}")
        print(f"   • Contains bot detection: {any(ind in html_lower for ind in bot_indicators)}")
        
        # Save raw HTML
        raw_file = "product_hunt_debug_raw.html"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"💾 Saved raw HTML to: {raw_file}")
        print()

        # STEP 2: JSON Detection
        print("="*80)
        print("STEP 2: JSON Detection")
        print("="*80)
        json_results = scraper.json_detector.detect_and_extract(html, url, captured_json=captured_json)
        
        sources = json_results.get('sources', [])
        data = json_results.get('data', [])

        print(f"\n📊 JSON Detection Summary:")
        print(f"   • Captured API JSON: {len([j for j in sources if 'api' in j])}")
        print(f"   • Embedded JSON: {len([j for j in sources if 'embedded-json' in j])}")
        print(f"   • JSON-LD: {len([j for j in sources if 'json-ld' in j])}")
        print(f"   • Total sources: {len(sources)}")
        print(f"   • Total items extracted from JSON: {len(data)}")
        
        if sources:
            print("\n🔍 JSON Sources Found:")
            for i, source_name in enumerate(sources[:5], 1):
                print(f"   Source {i}: {source_name}")
                source_data = data[i-1] if i-1 < len(data) else {}
                print(f"      Size: {len(str(source_data))} bytes")
                
                # Save source data
                source_file = f"product_hunt_debug_json_source_{i}.json"
                with open(source_file, 'w', encoding='utf-8') as f:
                    json.dump(source_data, f, indent=2)
                print(f"      💾 Saved to: {source_file}")
        print()

        # STEP 3: HTML Cleaning & Structure Analysis
        print("="*80)
        print("STEP 3: HTML Cleaning & Structure Analysis")
        print("="*80)
        
        clean_result = scraper.html_cleaner.clean(html)
        cleaned_html = clean_result['html']
        print(f"🧹 Cleaned HTML: {len(cleaned_html):,} bytes ({clean_result['reduction_percent']:.1f}% reduction)")
        
        # Save cleaned HTML
        cleaned_file = "product_hunt_debug_cleaned.html"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_html)
        print(f"💾 Saved cleaned HTML to: {cleaned_file}")
        
        # DOM pattern detection
        print("\n📊 DOM Pattern Detection:")
        dom_detector = DOMPatternDetector()
        patterns = dom_detector.detect_patterns(cleaned_html)
        
        if patterns['best_pattern']:
            bp = patterns['best_pattern']
            print(f"   ✅ Best pattern found:")
            print(f"      • Selector: {bp['selector']}")
            print(f"      • Count: {bp['count']}")
            print(f"      • Type: {bp['type']}")
            print(f"      • Confidence: {bp['confidence']:.2f}")
        else:
            print(f"   ⚠️ No clear pattern detected")
            print(f"\n   Top element signatures:")
            for i, sig in enumerate(patterns['element_signatures'][:10], 1):
                print(f"      {i}. {sig['signature']}: {sig['count']} occurrences")
        print()

        # STEP 4: Manual HTML Inspection
        print("="*80)
        print("STEP 4: Manual HTML Inspection")
        print("="*80)
        soup = BeautifulSoup(cleaned_html, 'lxml')
        
        # Look for common product listing patterns
        potential_selectors = [
            'article',
            'div[data-test*="product"]',
            'div[class*="product"]',
            'div[class*="item"]',
            'div[class*="post"]',
            'a[href*="/posts/"]',  # Product Hunt specific
        ]
        
        print(f"🔍 Testing potential selectors:")
        for selector in potential_selectors:
            try:
                elements = soup.select(selector)
                print(f"   • {selector}: {len(elements)} elements")
                if elements and len(elements) > 0:
                    print(f"      Sample: {str(elements[0])[:200]}...")
            except Exception as e:
                print(f"   • {selector}: Error - {e}")
        
        # Check for React/Next.js indicators
        react_indicators = ['__NEXT_DATA__', '_next', 'react', '__nuxt', '__REACT_QUERY_STATE__']
        found_indicators = [ind for ind in react_indicators if ind in html]
        if found_indicators:
            print(f"\n📦 React/Next.js Detected:")
            print(f"   Found indicators: {', '.join(found_indicators)}")
            print(f"   → Product Hunt likely uses client-side rendering")
            print(f"   → Data may be in window.__NEXT_DATA__ or similar")
        
        # Check __NEXT_DATA__
        if '__NEXT_DATA__' in html:
            print(f"\n🔍 Extracting __NEXT_DATA__...")
            import re
            next_data_match = re.search(r'__NEXT_DATA__\s*=\s*({.+?})</script>', html, re.DOTALL)
            if next_data_match:
                try:
                    next_data = json.loads(next_data_match.group(1))
                    print(f"   ✅ Found __NEXT_DATA__: {len(str(next_data))} bytes")
                    
                    # Save it
                    next_data_file = "product_hunt_next_data.json"
                    with open(next_data_file, 'w', encoding='utf-8') as f:
                        json.dump(next_data, f, indent=2)
                    print(f"   💾 Saved to: {next_data_file}")
                    
                    # Look for posts data
                    posts_found = str(next_data).count('"name"') + str(next_data).count('"title"')
                    print(f"   📊 Potential products in data: ~{posts_found} (rough estimate)")
                except Exception as e:
                    print(f"   ❌ Failed to parse __NEXT_DATA__: {e}")
        
        print(f"\n{'='*80}")
        print("✅ DIAGNOSTIC COMPLETE")
        print("="*80)
        
        print(f"\n📋 FINDINGS SUMMARY:")
        print(f"   1. HTML fetched: {len(html):,} bytes")
        print(f"   2. JSON sources found: {len(sources)}")
        print(f"   3. DOM pattern detected: {'Yes' if patterns['best_pattern'] else 'No'}")
        print(f"   4. React/Next.js detected: {'Yes' if found_indicators else 'No'}")
        
        print(f"\n💡 LIKELY ISSUE:")
        if '__NEXT_DATA__' in html:
            print(f"   Product Hunt uses Next.js with server-side data in __NEXT_DATA__")
            print(f"   → Solution: Extract and parse __NEXT_DATA__ JSON")
        elif any(ind in html_lower for ind in auth_indicators):
            print(f"   Product Hunt may require authentication")
            print(f"   → Solution: Handle auth wall gracefully or document as requiring login")
        elif any(ind in html_lower for ind in bot_indicators):
            print(f"   Bot detection active")
            print(f"   → Solution: Enhanced anti-detection or residential proxies")
        else:
            print(f"   Unknown issue - needs manual inspection of saved HTML files")

    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







