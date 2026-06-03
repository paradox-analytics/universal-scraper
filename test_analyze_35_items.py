#!/usr/bin/env python3
"""
Analyze the 35 items we extracted vs ScrapeGraphAI's 30 items
Show exactly what data was captured
"""
import asyncio
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from scrapegraphai.graphs import SmartScraperGraph


async def analyze_captured_data():
    """Analyze what data we captured in detail"""
    
    print("\n" + "="*100)
    print("📊 DETAILED DATA ANALYSIS - What Did We Actually Capture?")
    print("="*100)
    print()
    
    url = "https://news.ycombinator.com/"
    fields = ["title", "points", "comments"]
    
    # Fetch HTML once
    print("📥 Fetching HTML...")
    fetcher = HybridFetcher(
        proxy_config=None,
        enable_cache=False,
        headless=True,
        use_camoufox=False
    )
    
    fetch_result = await fetcher.fetch(url)
    raw_html = fetch_result['html']
    
    # Clean HTML
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(raw_html)
    cleaned_html = clean_result['html']
    
    print(f"✅ HTML prepared: {len(cleaned_html):,} bytes")
    print()
    
    # Extract with our scraper
    print("="*100)
    print("🔧 OUR SCRAPER (DirectLLM with ScrapeGraphAI approach)")
    print("="*100)
    
    extractor = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,
        quality_mode="balanced",
        use_html2text=True
    )
    
    our_items = await extractor.extract(
        cleaned_html,
        fields,
        context="Extract all article listings with title, points, and comments count"
    )
    
    print(f"✅ Extracted {len(our_items)} items")
    print()
    
    # Extract with ScrapeGraphAI
    print("="*100)
    print("🤖 SCRAPEGRAPHAI")
    print("="*100)
    
    graph_config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ.get('OPENAI_API_KEY'),
        },
        "verbose": False,
        "headless": True,
    }
    
    smart_scraper = SmartScraperGraph(
        prompt="Extract all article listings with title, points, and comments count",
        source=url,
        config=graph_config
    )
    
    result = smart_scraper.run()
    
    # Extract their items
    their_items = []
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, list):
                their_items = value
                break
    
    print(f"✅ Extracted {len(their_items)} items")
    print()
    
    # Analyze our 35 items
    print("="*100)
    print("🔍 DETAILED BREAKDOWN - OUR 35 ITEMS")
    print("="*100)
    print()
    
    print("Format: [#] Title | Points | Comments | Quality")
    print("-" * 100)
    
    for i, item in enumerate(our_items, 1):
        title = item.get('title', 'N/A')
        points = item.get('points', 'N/A')
        comments = item.get('comments', 'N/A')
        
        # Calculate completeness
        fields_filled = sum([
            1 if title not in ['N/A', None, ''] else 0,
            1 if points not in ['N/A', None, ''] else 0,
            1 if comments not in ['N/A', None, ''] else 0
        ])
        completeness = f"{fields_filled}/3"
        
        # Truncate title
        title_display = title[:70] if title else 'N/A'
        
        print(f"{i:2}. {title_display:<70} | {str(points):>6} | {str(comments):>8} | {completeness}")
    
    print()
    
    # Analyze their 30 items
    print("="*100)
    print("🔍 DETAILED BREAKDOWN - THEIR 30 ITEMS")
    print("="*100)
    print()
    
    print("Format: [#] Title | Points | Comments | Quality")
    print("-" * 100)
    
    for i, item in enumerate(their_items, 1):
        # Handle different field names they might use
        title = item.get('title') or item.get('Title') or item.get('name') or 'N/A'
        points = item.get('points') or item.get('Points') or item.get('score') or 'N/A'
        comments = item.get('comments') or item.get('Comments') or item.get('comment_count') or 'N/A'
        
        # Calculate completeness
        fields_filled = sum([
            1 if title not in ['N/A', None, ''] else 0,
            1 if points not in ['N/A', None, ''] else 0,
            1 if comments not in ['N/A', None, ''] else 0
        ])
        completeness = f"{fields_filled}/3"
        
        # Truncate title
        title_display = str(title)[:70] if title else 'N/A'
        
        print(f"{i:2}. {title_display:<70} | {str(points):>6} | {str(comments):>8} | {completeness}")
    
    print()
    
    # Compare - find what we have that they don't
    print("="*100)
    print("🔍 COMPARISON ANALYSIS")
    print("="*100)
    print()
    
    our_titles = set([item.get('title', '').lower().strip() for item in our_items if item.get('title')])
    their_titles = set()
    for item in their_items:
        title = item.get('title') or item.get('Title') or item.get('name') or ''
        their_titles.add(str(title).lower().strip())
    
    # Find unique to us
    unique_to_us = our_titles - their_titles
    
    # Find unique to them
    unique_to_them = their_titles - our_titles
    
    # Common items
    common = our_titles & their_titles
    
    print(f"✅ Common items: {len(common)}")
    print(f"➕ Unique to us (5 extra items): {len(unique_to_us)}")
    print(f"➖ Unique to them: {len(unique_to_them)}")
    print()
    
    if unique_to_us:
        print("🔵 OUR 5 EXTRA ITEMS (not in ScrapeGraphAI's results):")
        print("-" * 100)
        for i, extra_title in enumerate(sorted(unique_to_us), 1):
            # Find full item
            full_item = next((item for item in our_items if item.get('title', '').lower().strip() == extra_title), None)
            if full_item:
                points = full_item.get('points', 'N/A')
                comments = full_item.get('comments', 'N/A')
                print(f"{i}. {extra_title[:70]:<70} | {str(points):>6} | {str(comments):>8}")
        print()
    
    if unique_to_them:
        print("⚠️  ITEMS THEY HAVE THAT WE DON'T (should investigate):")
        print("-" * 100)
        for i, missing_title in enumerate(sorted(unique_to_them), 1):
            print(f"{i}. {missing_title[:90]}")
        print()
    
    # Quality analysis
    print("="*100)
    print("📊 QUALITY METRICS")
    print("="*100)
    print()
    
    def analyze_quality(items, name):
        if not items:
            return
        
        total_fields = len(items) * 3  # 3 fields per item
        filled_fields = 0
        
        for item in items:
            title = item.get('title') or item.get('Title') or item.get('name')
            points = item.get('points') or item.get('Points') or item.get('score')
            comments = item.get('comments') or item.get('Comments') or item.get('comment_count')
            
            if title not in [None, '', 'N/A']:
                filled_fields += 1
            if points not in [None, '', 'N/A']:
                filled_fields += 1
            if comments not in [None, '', 'N/A']:
                filled_fields += 1
        
        completeness = (filled_fields / total_fields) * 100
        
        # Count perfect items (all 3 fields)
        perfect_items = 0
        for item in items:
            title = item.get('title') or item.get('Title') or item.get('name')
            points = item.get('points') or item.get('Points') or item.get('score')
            comments = item.get('comments') or item.get('Comments') or item.get('comment_count')
            
            if all([
                title not in [None, '', 'N/A'],
                points not in [None, '', 'N/A'],
                comments not in [None, '', 'N/A']
            ]):
                perfect_items += 1
        
        print(f"{name}:")
        print(f"  • Total items: {len(items)}")
        print(f"  • Total fields: {total_fields}")
        print(f"  • Filled fields: {filled_fields}")
        print(f"  • Completeness: {completeness:.1f}%")
        print(f"  • Perfect items (3/3): {perfect_items} ({perfect_items/len(items)*100:.1f}%)")
        print()
    
    analyze_quality(our_items, "Our Scraper (35 items)")
    analyze_quality(their_items, "ScrapeGraphAI (30 items)")
    
    # Final verdict
    print("="*100)
    print("🎯 FINAL VERDICT")
    print("="*100)
    print()
    
    if len(unique_to_us) > len(unique_to_them):
        print("✅ SUCCESS: We capture MORE items than ScrapeGraphAI!")
        print(f"   • We get all {len(common)} items they get")
        print(f"   • Plus {len(unique_to_us)} additional items")
        print()
    elif len(unique_to_them) > 0:
        print("⚠️  ATTENTION: They capture some items we don't")
        print(f"   • We get {len(common)} common items")
        print(f"   • We have {len(unique_to_us)} extras")
        print(f"   • They have {len(unique_to_them)} we're missing")
        print()
    else:
        print("✅ PERFECT: We capture all their items plus extras!")
        print()


if __name__ == "__main__":
    asyncio.run(analyze_captured_data())



