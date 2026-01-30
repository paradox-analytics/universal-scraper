#!/usr/bin/env python3
"""Debug GitHub stars extraction"""

import asyncio
import os
from pathlib import Path
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    scraper = UniversalScraper(api_key=api_key, use_camoufox=False, headless=True, enable_auto_pagination=False)
    
    try:
        # Get HTML
        fetch_result = await scraper.html_fetcher.fetch("https://github.com/trending")
        html = fetch_result['html']
        
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.select('article.Box-row')
        
        if articles:
            first = articles[0]
            print("="*80)
            print("🔍 INSPECTING STAR METADATA AREA")
            print("="*80)
            print()
            
            # Look for the metadata section (f6 class typically contains stars/language)
            metadata = first.select('.f6.color-fg-muted.mt-2')
            if metadata:
                print(f"📊 Found {len(metadata)} .f6.color-fg-muted.mt-2 sections")
                for i, meta in enumerate(metadata, 1):
                    print(f"\n--- Metadata Section {i} ---")
                    print(meta.prettify()[:1000])
                    print("\n   All links in this section:")
                    for link in meta.select('a'):
                        print(f"      href={link.get('href')}: {link.text.strip()}")
            
            # Try to find stargazers link
            print("\n" + "="*80)
            print("🌟 LOOKING FOR STARGAZERS LINK")
            print("="*80)
            stargazer_links = first.select('a[href*="stargazers"]')
            if stargazer_links:
                print(f"\n✅ Found {len(stargazer_links)} stargazers links")
                for link in stargazer_links:
                    print(f"\n   Link: {link.get('href')}")
                    print(f"   Text: '{link.text.strip()}'")
                    print(f"   Parent: {link.parent.name}")
                    # Check for adjacent spans
                    next_elem = link.find_next_sibling()
                    if next_elem:
                        print(f"   Next sibling: {next_elem.name} - '{next_elem.text.strip()}'")
            else:
                print("❌ No stargazers links found")
                print("\nAll links with 'star' in href or text:")
                for link in first.select('a'):
                    if 'star' in link.get('href', '').lower() or 'star' in link.text.lower():
                        print(f"   {link.get('href')}: {link.text.strip()}")
    finally:
        await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







