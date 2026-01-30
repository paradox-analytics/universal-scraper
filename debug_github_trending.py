#!/usr/bin/env python3
"""
Diagnostic script for GitHub Trending extraction issues
"""

import asyncio
import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from bs4 import BeautifulSoup

async def main():
    print("="*80)
    print("🔬 GitHub Trending DIAGNOSTIC")
    print("="*80)
    print()
    
    url = "https://github.com/trending"
    fields = ["repository", "description", "stars", "language"]
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        return

    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=False,  # Use regular Playwright
            headless=True,
            enable_auto_pagination=False,
            extraction_context="Extract trending GitHub repositories"
        )
        
        print(f"🎯 Target: {url}")
        print(f"📋 Fields: {', '.join(fields)}")
        print()

        # STEP 1: Fetch HTML
        print("="*80)
        print("STEP 1: Fetching HTML")
        print("="*80)
        fetch_result = await scraper.html_fetcher.fetch(url)
        html = fetch_result['html']
        print(f"✅ Fetched: {len(html):,} bytes")
        
        # Save raw HTML
        raw_file = "github_trending_raw.html"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"💾 Saved to: {raw_file}")
        print()

        # STEP 2: DOM Pattern Detection
        print("="*80)
        print("STEP 2: DOM Pattern Detection")
        print("="*80)
        detector = DOMPatternDetector()
        patterns = detector.detect_patterns(html)
        
        if patterns['best_pattern']:
            bp = patterns['best_pattern']
            print(f"✅ BEST PATTERN FOUND:")
            print(f"   • Type: {bp['type']}")
            print(f"   • Selector: {bp['selector']}")
            print(f"   • Count: {bp['count']}")
            print(f"   • Confidence: {bp['confidence']:.2f}")
            print(f"\n📋 Sample (first 500 chars):")
            print(bp['sample'][:500])
        else:
            print("❌ No pattern found")
        print()

        # STEP 3: Manual HTML Inspection
        print("="*80)
        print("STEP 3: Manual HTML Inspection")
        print("="*80)
        soup = BeautifulSoup(html, 'lxml')
        
        # Look for common GitHub patterns
        patterns_to_check = [
            ('article.Box-row', 'GitHub trending repo articles'),
            ('div.Box-row', 'Box row divs'),
            ('h2.h3', 'Repository titles'),
            ('[data-hovercard-type="repository"]', 'Repository hovercards'),
            ('article', 'Generic articles'),
        ]
        
        for selector, description in patterns_to_check:
            elements = soup.select(selector)
            print(f"🔍 {selector} ({description}): {len(elements)} found")
            if elements:
                first = elements[0]
                print(f"   Sample (first 300 chars):")
                print(f"   {str(first)[:300]}")
                print()
        
        # STEP 4: Run Full Scrape
        print("="*80)
        print("STEP 4: Full Scrape Test")
        print("="*80)
        result = await scraper.scrape(url, fields)
        
        print(f"\n✅ RESULTS:")
        print(f"   • Items: {len(result['data'])}")
        print(f"   • Source: {result['extraction_source']}")
        print(f"   • Time: {result['total_time']:.1f}s")
        print(f"   • Cached: {result.get('code_cached', False)}")
        
        if result['data']:
            print(f"\n📋 Sample (first 3):")
            for i, item in enumerate(result['data'][:3], 1):
                print(f"\n   Item {i}:")
                for k, v in item.items():
                    print(f"      • {k}: {v}")
        else:
            print("❌ No items extracted")
        
        print("\n" + "="*80)
        print("✅ DIAGNOSTIC COMPLETE")
        print("="*80)

    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if scraper:
            await scraper.close()

if __name__ == '__main__':
    asyncio.run(main())
