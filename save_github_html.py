#!/usr/bin/env python3
"""
Save GitHub Trending HTML for inspection
"""

import asyncio
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.hybrid_fetcher import HybridFetcher


async def main():
    fetcher = HybridFetcher(proxy_config=None)
    
    try:
        result = await fetcher.fetch("https://github.com/trending")
        html = result['html']
        
        with open("github_trending_raw.html", 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Saved {len(html):,} bytes to github_trending_raw.html")
        
        # Quick analysis
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        
        # Look for common patterns
        articles = soup.find_all('article')
        print(f"\n📊 Found {len(articles)} <article> elements")
        
        # Check for specific classes
        for cls in ['Box-row', 'repo-list', 'explore-content']:
            elements = soup.select(f'.{cls}')
            print(f"   • .{cls}: {len(elements)} elements")
        
        # Check for h2 tags (repo names)
        h2_tags = soup.find_all('h2')
        print(f"\n📊 Found {len(h2_tags)} <h2> tags")
        if h2_tags:
            print(f"   Sample h2: {h2_tags[0].get_text(strip=True)[:100]}")
    
    finally:
        await fetcher.close()


if __name__ == '__main__':
    asyncio.run(main())

