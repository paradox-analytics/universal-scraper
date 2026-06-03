#!/usr/bin/env python3
"""
Debug GitHub extraction to see why description/stars/language are null
"""

import asyncio
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper
from bs4 import BeautifulSoup


async def main():
    print("="*80)
    print("🔬 GitHub Extraction Debugging")
    print("="*80)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        return
    
    url = "https://github.com/trending"
    fields = ["repository", "description", "stars", "language"]
    
    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            use_camoufox=True,
            headless=True,
            enable_auto_pagination=False
        )
        
        print("🚀 Fetching and analyzing HTML...")
        
        # Get the HTML
        fetch_result = await scraper.html_fetcher.fetch(url)
        html = fetch_result['html']
        print(f"✅ Fetched: {len(html):,} bytes")
        print()
        
        # Parse and inspect structure
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.select('article.Box-row')
        print(f"📊 Found {len(articles)} article.Box-row elements")
        print()
        
        if articles:
            print("🔍 INSPECTING FIRST ARTICLE:")
            print("="*80)
            first = articles[0]
            
            # Show the structure
            print("\n📌 Repository:")
            repo_elem = first.select_one('h2 a')
            if repo_elem:
                print(f"   Selector: h2 a")
                print(f"   Text: {repo_elem.text.strip()}")
                print(f"   Found: ✅")
            else:
                print("   Not found: ❌")
            
            print("\n📌 Description:")
            # Try multiple selectors
            desc_selectors = ['p', '.col-9', 'p.col-9', 'div p']
            for sel in desc_selectors:
                desc = first.select_one(sel)
                if desc and desc.text.strip():
                    print(f"   Selector: {sel}")
                    print(f"   Text: {desc.text.strip()[:100]}")
                    print(f"   Found: ✅")
                    break
            else:
                print("   Not found in any selector: ❌")
                print("   Available p tags:")
                for i, p in enumerate(first.select('p')[:3], 1):
                    print(f"      {i}. {p.get('class')}: {p.text.strip()[:60]}")
            
            print("\n📌 Stars:")
            # Try multiple selectors
            stars_selectors = [
                'svg.octicon-star + span',
                'a[href*="stargazers"]',
                'span.d-inline-block.float-sm-right',
                '.f6.color-fg-muted.mt-2 a'
            ]
            for sel in stars_selectors:
                stars = first.select_one(sel)
                if stars and stars.text.strip():
                    print(f"   Selector: {sel}")
                    print(f"   Text: {stars.text.strip()}")
                    print(f"   Found: ✅")
                    break
            else:
                print("   Not found in any selector: ❌")
                print("   Looking for star-related elements:")
                for elem in first.select('a[href*="stargazers"], span:contains("star")'):
                    print(f"      {elem.name}.{elem.get('class')}: {elem.text.strip()[:60]}")
            
            print("\n📌 Language:")
            # Try multiple selectors
            lang_selectors = [
                'span[itemprop="programmingLanguage"]',
                'span.d-inline-block.ml-0.mr-3',
                '.f6.color-fg-muted.mt-2 span'
            ]
            for sel in lang_selectors:
                lang = first.select_one(sel)
                if lang and lang.text.strip():
                    print(f"   Selector: {sel}")
                    print(f"   Text: {lang.text.strip()}")
                    print(f"   Found: ✅")
                    break
            else:
                print("   Not found in any selector: ❌")
                print("   Available spans in metadata:")
                for span in first.select('.f6 span')[:5]:
                    print(f"      {span.get('class')}: {span.text.strip()[:60]}")
            
            print("\n" + "="*80)
            print("\n📝 Full Article HTML (first 2000 chars):")
            print(str(first)[:2000])
            print("\n...")
        
    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







