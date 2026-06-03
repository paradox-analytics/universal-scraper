#!/usr/bin/env python3
"""
Test Metacritic HTML fetching and inspection
"""
import asyncio
import os
import sys
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from bs4 import BeautifulSoup

async def test_metacritic_html():
    """Test fetching and inspecting Metacritic HTML"""
    url = "https://www.metacritic.com/pictures/worst-movies-of-2025/"
    
    print(f"🔍 Testing Metacritic HTML fetching...")
    print(f"   URL: {url}")
    print()
    
    # Create fetcher
    fetcher = HybridFetcher(
        headless=True,
        browser_timeout=120000,  # 120 seconds
        enable_cache=False
    )
    
    try:
        print("📥 Fetching page...")
        result = await fetcher.fetch(
            url=url,
            scroll_to_bottom=False,
            wait_for_selector=None
        )
        
        html = result.get('html', '')
        print(f"✅ HTML fetched: {len(html)} bytes")
        print()
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for numbered sections (#15, #14, etc.)
        print("🔍 Searching for numbered list items...")
        
        # Method 1: Look for headings with #15, #14, etc.
        numbered_headings = soup.find_all(['h2', 'h3', 'h4'], string=lambda text: text and ('#15' in text or '#14' in text or '#13' in text))
        print(f"   Found {len(numbered_headings)} headings with #15/#14/#13")
        
        # Method 2: Look for sections with "Metascore"
        metascore_sections = soup.find_all(string=lambda text: text and 'Metascore' in text)
        print(f"   Found {len(metascore_sections)} text occurrences of 'Metascore'")
        
        # Method 3: Look for movie titles (common patterns)
        # Check for common movie title patterns
        title_patterns = [
            'Old Guy',
            'Regretting You',
            'Modi',
            'Smurfs',
            'Juliet & Romeo',
            'The Electric State',
            'War of the Worlds'
        ]
        
        found_titles = []
        for pattern in title_patterns:
            if pattern in html:
                found_titles.append(pattern)
        
        print(f"   Found {len(found_titles)}/{len(title_patterns)} expected movie titles in HTML")
        if found_titles:
            print(f"   Titles found: {', '.join(found_titles)}")
        
        # Method 4: Look for numbered sections in HTML structure
        # Check for patterns like "#15:", "#14:", etc.
        import re
        numbered_pattern = r'#\d+[:]'
        numbered_matches = re.findall(numbered_pattern, html)
        unique_numbers = set(numbered_matches)
        print(f"   Found {len(unique_numbers)} unique numbered patterns: {sorted(unique_numbers)[:10]}")
        
        # Save HTML sample for inspection
        print("\n💾 Saving HTML sample...")
        cleaned_html = SmartHTMLCleaner().clean(html)['html']
        
        # Save first 100KB for inspection
        with open('metacritic_html_sample.html', 'w', encoding='utf-8') as f:
            f.write(cleaned_html[:100000])
        print("   ✅ Saved: metacritic_html_sample.html")
        
        # Save raw HTML sample
        with open('metacritic_raw_html_sample.html', 'w', encoding='utf-8') as f:
            f.write(html[:100000])
        print("   ✅ Saved: metacritic_raw_html_sample.html")
        
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"HTML size: {len(html)} bytes")
        print(f"Numbered headings: {len(numbered_headings)}")
        print(f"Metascore mentions: {len(metascore_sections)}")
        print(f"Movie titles found: {len(found_titles)}/{len(title_patterns)}")
        print(f"Numbered patterns: {len(unique_numbers)}")
        
        if len(unique_numbers) >= 15:
            print("\n✅ HTML contains numbered list items - extraction should work")
        else:
            print("\n⚠️  WARNING: HTML may not contain all list items")
            print("   This could indicate:")
            print("   - Page requires JavaScript to load content")
            print("   - Content is loaded dynamically")
            print("   - Page is blocking automated browsers")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(test_metacritic_html())




