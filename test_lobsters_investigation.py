#!/usr/bin/env python3
"""
Investigate exactly what's happening with Lobsters extraction
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
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor


async def investigate_lobsters():
    """Deep dive into Lobsters extraction issue"""
    
    print("\n" + "="*100)
    print("🔍 LOBSTERS INVESTIGATION - Finding the Root Cause")
    print("="*100)
    print()
    
    url = "https://lobste.rs/"
    
    # Step 1: Fetch and analyze raw HTML
    print("="*100)
    print("📥 STEP 1: Fetch HTML and Analyze Structure")
    print("="*100)
    print()
    
    fetcher = HybridFetcher(proxy_config=None, enable_cache=False, headless=True, use_camoufox=False)
    fetch_result = await fetcher.fetch(url)
    raw_html = fetch_result['html']
    
    print(f"✅ Fetched {len(raw_html):,} bytes")
    print()
    
    # Parse and look for story structure
    soup = BeautifulSoup(raw_html, 'html.parser')
    
    print("Looking for story elements...")
    stories = soup.find_all('li', class_='story')
    
    if not stories:
        # Try alternative selectors
        stories = soup.find_all('div', class_='story')
    
    print(f"Found {len(stories)} story elements")
    print()
    
    if stories:
        print("📊 Analyzing first 3 stories:")
        print("-" * 100)
        
        for i, story in enumerate(stories[:3], 1):
            print(f"\nStory #{i}:")
            
            # Look for title
            title_elem = story.find('span', class_='link') or story.find('a', class_='u-url')
            title = title_elem.get_text(strip=True) if title_elem else "Not found"
            print(f"  Title: {title[:70]}")
            
            # Look for points/score
            score_elem = story.find('span', class_='score') or story.find('div', class_='score')
            score = score_elem.get_text(strip=True) if score_elem else "Not found"
            print(f"  Score element: {score}")
            
            # Look for comments
            comments_elem = story.find('span', class_='comments_label') or story.find('a', string=lambda x: x and 'comment' in x.lower())
            comments = comments_elem.get_text(strip=True) if comments_elem else "Not found"
            print(f"  Comments element: {comments}")
            
            # Show raw HTML snippet
            print(f"\n  Raw HTML snippet:")
            story_html = str(story)[:300]
            print(f"  {story_html}...")
            print()
    
    # Step 2: Test with current prompt (asking for "points")
    print("="*100)
    print("📝 STEP 2: Test Current Prompt (asking for 'points')")
    print("="*100)
    print()
    
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(raw_html)
    
    extractor = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=True
    )
    
    print("Extracting with fields: ['title', 'points', 'comments']")
    items_with_points = await extractor.extract(
        clean_result['html'],
        ['title', 'points', 'comments'],
        context="Extract all story listings"
    )
    
    print(f"✅ Extracted {len(items_with_points)} items")
    
    if items_with_points:
        # Analyze completeness
        points_count = sum(1 for item in items_with_points if item.get('points') not in [None, '', 'N/A'])
        comments_count = sum(1 for item in items_with_points if item.get('comments') not in [None, '', 'N/A'])
        
        print(f"\nField coverage:")
        print(f"  • Title: {len(items_with_points)}/{len(items_with_points)} (100%)")
        print(f"  • Points: {points_count}/{len(items_with_points)} ({points_count/len(items_with_points)*100:.0f}%)")
        print(f"  • Comments: {comments_count}/{len(items_with_points)} ({comments_count/len(items_with_points)*100:.0f}%)")
        
        print(f"\nSample items:")
        for i, item in enumerate(items_with_points[:3], 1):
            points_val = item.get('points', 'MISSING')
            comments_val = item.get('comments', 'MISSING')
            print(f"  {i}. {item.get('title', 'N/A')[:50]}")
            print(f"     points={points_val}, comments={comments_val}")
    print()
    
    # Step 3: Test with alternative field name (asking for "score")
    print("="*100)
    print("📝 STEP 3: Test Alternative Prompt (asking for 'score' instead of 'points')")
    print("="*100)
    print()
    
    print("Extracting with fields: ['title', 'score', 'comments']")
    items_with_score = await extractor.extract(
        clean_result['html'],
        ['title', 'score', 'comments'],
        context="Extract all story listings"
    )
    
    print(f"✅ Extracted {len(items_with_score)} items")
    
    if items_with_score:
        # Analyze completeness
        score_count = sum(1 for item in items_with_score if item.get('score') not in [None, '', 'N/A'])
        comments_count = sum(1 for item in items_with_score if item.get('comments') not in [None, '', 'N/A'])
        
        print(f"\nField coverage:")
        print(f"  • Title: {len(items_with_score)}/{len(items_with_score)} (100%)")
        print(f"  • Score: {score_count}/{len(items_with_score)} ({score_count/len(items_with_score)*100:.0f}%)")
        print(f"  • Comments: {comments_count}/{len(items_with_score)} ({comments_count/len(items_with_score)*100:.0f}%)")
        
        print(f"\nSample items:")
        for i, item in enumerate(items_with_score[:3], 1):
            score_val = item.get('score', 'MISSING')
            comments_val = item.get('comments', 'MISSING')
            print(f"  {i}. {item.get('title', 'N/A')[:50]}")
            print(f"     score={score_val}, comments={comments_val}")
    print()
    
    # Step 4: Test with more explicit prompt
    print("="*100)
    print("📝 STEP 4: Test Explicit Prompt (mention 'score' or 'points')")
    print("="*100)
    print()
    
    print("Extracting with fields: ['title', 'points', 'comments']")
    print("Context: Extract all story listings. The vote count might be called 'score' or 'points'.")
    
    items_explicit = await extractor.extract(
        clean_result['html'],
        ['title', 'points', 'comments'],
        context="Extract all story listings. The vote count might be called 'score' or 'points' - extract it as 'points'."
    )
    
    print(f"✅ Extracted {len(items_explicit)} items")
    
    if items_explicit:
        # Analyze completeness
        points_count = sum(1 for item in items_explicit if item.get('points') not in [None, '', 'N/A'])
        comments_count = sum(1 for item in items_explicit if item.get('comments') not in [None, '', 'N/A'])
        
        print(f"\nField coverage:")
        print(f"  • Title: {len(items_explicit)}/{len(items_explicit)} (100%)")
        print(f"  • Points: {points_count}/{len(items_explicit)} ({points_count/len(items_explicit)*100:.0f}%)")
        print(f"  • Comments: {comments_count}/{len(items_explicit)} ({comments_count/len(items_explicit)*100:.0f}%)")
        
        print(f"\nSample items:")
        for i, item in enumerate(items_explicit[:3], 1):
            points_val = item.get('points', 'MISSING')
            comments_val = item.get('comments', 'MISSING')
            print(f"  {i}. {item.get('title', 'N/A')[:50]}")
            print(f"     points={points_val}, comments={comments_val}")
    print()
    
    # Summary
    print("="*100)
    print("🎯 DIAGNOSIS")
    print("="*100)
    print()
    
    print("Completeness Comparison:")
    print(f"  • With 'points' field:          {points_count if 'items_with_points' in locals() else 0} items have points")
    print(f"  • With 'score' field:           {score_count if 'items_with_score' in locals() else 0} items have score")
    print(f"  • With explicit prompt hint:    {points_count if 'items_explicit' in locals() else 0} items have points")
    print()
    
    if items_with_score and items_with_score:
        score_coverage = sum(1 for item in items_with_score if item.get('score') not in [None, '', 'N/A']) / len(items_with_score) * 100
        points_coverage = sum(1 for item in items_with_points if item.get('points') not in [None, '', 'N/A']) / len(items_with_points) * 100
        
        if score_coverage > points_coverage + 10:
            print("🔴 ROOT CAUSE: Field name mismatch!")
            print(f"   • Asking for 'score' gives {score_coverage:.0f}% coverage")
            print(f"   • Asking for 'points' gives {points_coverage:.0f}% coverage")
            print()
            print("💡 SOLUTION: Use 'score' instead of 'points' for Lobsters")
            print("   OR implement field synonyms/mapping")
        elif points_coverage > 80:
            print("✅ 'points' field name works fine!")
            print(f"   Coverage: {points_coverage:.0f}%")
        else:
            print("⚠️  Neither field name gives good coverage")
            print("   Might be an HTML structure issue")
    
    print()


if __name__ == "__main__":
    asyncio.run(investigate_lobsters())



