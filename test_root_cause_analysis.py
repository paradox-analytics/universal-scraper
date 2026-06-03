#!/usr/bin/env python3
"""
Root cause analysis: Why are we only extracting 23/30 items?
Investigate HTML cleaning, structure, and what the LLM sees
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
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def investigate_root_cause():
    """Investigate why we miss 7 items"""
    print("\n" + "="*100)
    print("🔬 ROOT CAUSE ANALYSIS - Why 23 instead of 30?")
    print("="*100)
    print()
    
    url = "https://news.ycombinator.com/"
    
    # Step 1: Fetch HTML
    print("📥 Step 1: Fetching HTML...")
    fetcher = HybridFetcher(
        proxy_config=None,
        enable_cache=False,
        headless=True,
        use_camoufox=False
    )
    
    fetch_result = await fetcher.fetch(url)
    raw_html = fetch_result['html']
    print(f"✅ Fetched {len(raw_html):,} bytes")
    
    # Count items in raw HTML
    soup_raw = BeautifulSoup(raw_html, 'html.parser')
    raw_articles = soup_raw.find_all('tr', class_='athing')
    print(f"   Found {len(raw_articles)} articles in raw HTML")
    print()
    
    # Step 2: Clean HTML
    print("🧹 Step 2: Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(raw_html)
    cleaned_html = clean_result['html']
    print(f"✅ Cleaned: {len(raw_html):,} → {len(cleaned_html):,} bytes ({clean_result['reduction_percent']:.1f}% reduction)")
    
    # Count items in cleaned HTML
    soup_clean = BeautifulSoup(cleaned_html, 'html.parser')
    clean_articles = soup_clean.find_all('tr', class_='athing')
    print(f"   Found {len(clean_articles)} articles in cleaned HTML")
    print()
    
    if len(clean_articles) < len(raw_articles):
        print("⚠️  WARNING: HTML cleaning removed some articles!")
        print(f"   Lost {len(raw_articles) - len(clean_articles)} articles during cleaning")
        print()
    else:
        print("✅ HTML cleaning preserved all articles")
        print()
    
    # Step 3: Analyze structure of all 30 items
    print("="*100)
    print("🔍 Step 3: Analyzing structure of all 30 items")
    print("="*100)
    print()
    
    print("Checking if items 24-30 are structurally different...")
    print()
    
    # Extract structure info for each item
    for i, article in enumerate(clean_articles, 1):
        rank_marker = f"Item #{i}"
        
        # Get title
        title_span = article.find('span', class_='titleline')
        title = title_span.find('a').get_text(strip=True) if title_span and title_span.find('a') else "NO TITLE"
        
        # Check for next sibling row (contains points/comments)
        next_row = article.find_next_sibling('tr')
        has_subtext = False
        points = None
        comments = None
        
        if next_row:
            subtext = next_row.find('td', class_='subtext')
            if subtext:
                has_subtext = True
                # Try to extract points
                score = subtext.find('span', class_='score')
                if score:
                    points_text = score.get_text(strip=True)
                    points = points_text
                
                # Try to extract comments
                comment_links = subtext.find_all('a')
                for link in comment_links:
                    link_text = link.get_text(strip=True)
                    if 'comment' in link_text.lower():
                        comments = link_text
                        break
        
        if i <= 5 or i >= 24:  # Show first 5 and last 7
            status = "✅" if has_subtext else "❌"
            print(f"{i:2}. {status} {title[:60]:<60}")
            if not has_subtext:
                print(f"     ⚠️  MISSING SUBTEXT ROW (no points/comments data)")
            elif not points:
                print(f"     ⚠️  Missing points")
            elif not comments:
                print(f"     ⚠️  Missing comments")
        elif i == 6:
            print("    ... (items 6-23) ...")
    
    print()
    
    # Step 4: Check cleaned HTML length and content
    print("="*100)
    print("📏 Step 4: Analyzing cleaned HTML size")
    print("="*100)
    print()
    
    # Count tokens approximately
    token_estimate = len(cleaned_html) / 4
    print(f"Cleaned HTML: {len(cleaned_html):,} bytes (~{token_estimate:,.0f} tokens)")
    print(f"Our chunk size: 25,000 tokens")
    print(f"Chunks needed: {token_estimate / 25000:.1f}")
    print()
    
    if token_estimate > 25000:
        print("⚠️  WARNING: HTML is larger than our chunk size!")
        print("   This might cause truncation")
        print()
    else:
        print("✅ HTML fits in a single chunk")
        print()
    
    # Step 5: Check what articles are at the end
    print("="*100)
    print("🎯 Step 5: Items 24-30 (The ones we miss)")
    print("="*100)
    print()
    
    for i in range(23, min(30, len(clean_articles))):
        article = clean_articles[i]
        
        # Get title
        title_span = article.find('span', class_='titleline')
        title = title_span.find('a').get_text(strip=True) if title_span and title_span.find('a') else "NO TITLE"
        
        # Get position in HTML
        article_html = str(article)
        position_in_html = cleaned_html.find(article_html)
        position_percent = (position_in_html / len(cleaned_html)) * 100
        
        print(f"{i+1}. {title}")
        print(f"    Position: {position_in_html:,} bytes ({position_percent:.1f}% into HTML)")
        print()
    
    # Step 6: Save cleaned HTML for inspection
    print("="*100)
    print("💾 Step 6: Saving files for manual inspection")
    print("="*100)
    print()
    
    with open('debug_raw_html.html', 'w') as f:
        f.write(raw_html)
    print("✅ Saved raw HTML to: debug_raw_html.html")
    
    with open('debug_cleaned_html.html', 'w') as f:
        f.write(cleaned_html)
    print("✅ Saved cleaned HTML to: debug_cleaned_html.html")
    
    # Extract just the articles portion
    articles_html = ""
    for article in clean_articles:
        articles_html += str(article) + "\n"
        next_row = article.find_next_sibling('tr')
        if next_row:
            articles_html += str(next_row) + "\n"
    
    with open('debug_articles_only.html', 'w') as f:
        f.write(articles_html)
    print("✅ Saved articles-only HTML to: debug_articles_only.html")
    
    print()
    
    # Step 7: Hypothesis
    print("="*100)
    print("🤔 HYPOTHESIS")
    print("="*100)
    print()
    
    if len(clean_articles) < 30:
        print(f"❌ HTML cleaning removed {30 - len(clean_articles)} articles")
        print("   ROOT CAUSE: HTML cleaner is too aggressive")
        print("   FIX: Adjust HTML cleaner to preserve article content")
    elif token_estimate > 25000:
        print("⚠️  HTML is being truncated due to chunk size")
        print("   ROOT CAUSE: Chunking cuts off bottom items")
        print("   FIX: Increase chunk size or improve chunking logic")
    else:
        print("🤔 All 30 articles are in the cleaned HTML")
        print("   Items are within chunk size")
        print("   ROOT CAUSE: LLM stops extracting early (model behavior)")
        print()
        print("   Possible reasons:")
        print("   1. LLM prioritizes 'interesting' items (high points/comments)")
        print("   2. LLM has implicit length limits on output")
        print("   3. LLM interprets 'main content' as top stories only")
        print()
        print("   FIX OPTIONS:")
        print("   a) Use GPT-4 (better at comprehensive extraction)")
        print("   b) Explicitly tell LLM to extract items 1-30")
        print("   c) Process in smaller chunks with overlap")
        print("   d) Accept 77% coverage (still excellent)")
    
    print()


if __name__ == "__main__":
    asyncio.run(investigate_root_cause())



