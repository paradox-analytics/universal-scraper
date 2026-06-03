#!/usr/bin/env python3
"""
Analyze the raw HTML to see how many articles are actually on the page
"""
import asyncio
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher


async def analyze_html():
    """Analyze how many articles are in the HTML"""
    print("\n" + "="*100)
    print("🔍 RAW HTML ANALYSIS - How many articles exist?")
    print("="*100)
    print()
    
    url = "https://news.ycombinator.com/"
    
    # Fetch HTML
    print("📥 Fetching HTML...")
    fetcher = HybridFetcher(
        proxy_config=None,
        enable_cache=False,
        headless=True,
        use_camoufox=False
    )
    
    fetch_result = await fetcher.fetch(url)
    html = fetch_result['html']
    print(f"✅ Fetched {len(html):,} bytes")
    print()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all article rows in HackerNews format
    # HN uses <tr class="athing"> for article rows
    article_rows = soup.find_all('tr', class_='athing')
    
    print(f"📊 Found {len(article_rows)} article rows (<tr class='athing'>)")
    print()
    
    # Extract titles
    print("="*100)
    print("📋 ALL ARTICLES ON PAGE")
    print("="*100)
    
    articles = []
    for i, row in enumerate(article_rows, 1):
        # Find title link
        title_span = row.find('span', class_='titleline')
        if title_span:
            title_link = title_span.find('a')
            if title_link:
                title = title_link.get_text(strip=True)
                articles.append(title)
                print(f"{i:2}. {title}")
    
    print()
    print(f"Total articles: {len(articles)}")
    print()
    
    # Compare with what extractors got
    print("="*100)
    print("📊 COMPARISON")
    print("="*100)
    print(f"Articles in HTML:  {len(articles)}")
    print(f"ScrapeGraphAI got: 30")
    print(f"Our DirectLLM got: 23")
    print()
    
    if len(articles) == 30:
        print("✅ ScrapeGraphAI matched the HTML exactly (30/30)")
        print("⚠️  Our DirectLLM missed 7 articles")
        print()
        print("Conclusion: We need to improve our LLM prompt to be more comprehensive")
    elif len(articles) > 30:
        print(f"🤔 HTML has MORE articles ({len(articles)}) than both extractors found")
        print("   Both extractors might be filtering or the page has duplicates")
    else:
        print(f"🤔 HTML has FEWER articles ({len(articles)}) than extractors reported")
        print("   Extractors might be finding additional content")
    
    print()
    
    # Show what the 7 missing items might be
    if len(articles) >= 30:
        print("="*100)
        print("🔍 LIKELY MISSING ARTICLES (items 24-30)")
        print("="*100)
        for i in range(23, min(30, len(articles))):
            print(f"{i+1}. {articles[i]}")
        print()
        print("These are the articles our LLM likely missed")
    
    print()


if __name__ == "__main__":
    asyncio.run(analyze_html())



