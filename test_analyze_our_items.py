#!/usr/bin/env python3
"""
Analyze the items we extracted - show exactly what data was captured
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


async def analyze_our_data():
    """Analyze what data we captured in detail"""
    
    print("\n" + "="*100)
    print("📊 DETAILED DATA ANALYSIS - Our 35-Item Extraction")
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
    print("🔧 EXTRACTING WITH OUR SCRAPER")
    print("="*100)
    print()
    
    extractor = DirectLLMExtractor(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        max_tokens_per_chunk=4000,  # ScrapeGraphAI-style small chunks
        quality_mode="balanced",     # 33% threshold
        use_html2text=True           # HTML-to-text conversion
    )
    
    our_items = await extractor.extract(
        cleaned_html,
        fields,
        context="Extract all article listings with title, points, and comments count"
    )
    
    print(f"✅ Extracted {len(our_items)} items")
    print()
    
    # Analyze our items in detail
    print("="*100)
    print(f"🔍 ALL {len(our_items)} ITEMS - DETAILED BREAKDOWN")
    print("="*100)
    print()
    
    print(f"{'#':<3} {'Title':<65} {'Points':>8} {'Comments':>10} {'Quality':>8}")
    print("-" * 100)
    
    for i, item in enumerate(our_items, 1):
        title = item.get('title', 'N/A')
        points = item.get('points', 'N/A')
        comments = item.get('comments', 'N/A')
        
        # Calculate completeness
        fields_filled = 0
        if title not in ['N/A', None, '']:
            fields_filled += 1
        if points not in ['N/A', None, '']:
            fields_filled += 1
        if comments not in ['N/A', None, '']:
            fields_filled += 1
        
        completeness = f"{fields_filled}/3"
        
        # Truncate title
        title_display = str(title)[:65] if title else 'N/A'
        
        # Format values
        points_str = str(points) if points not in [None, 'N/A', ''] else '-'
        comments_str = str(comments) if comments not in [None, 'N/A', ''] else '-'
        
        print(f"{i:<3} {title_display:<65} {points_str:>8} {comments_str:>10} {completeness:>8}")
    
    print()
    
    # Quality analysis
    print("="*100)
    print("📊 QUALITY METRICS")
    print("="*100)
    print()
    
    total_items = len(our_items)
    total_fields = total_items * 3
    
    filled_fields = 0
    perfect_items = 0  # All 3 fields
    good_items = 0     # 2/3 fields
    partial_items = 0  # 1/3 fields
    
    for item in our_items:
        title = item.get('title')
        points = item.get('points')
        comments = item.get('comments')
        
        item_filled = 0
        
        if title not in [None, '', 'N/A']:
            filled_fields += 1
            item_filled += 1
        if points not in [None, '', 'N/A']:
            filled_fields += 1
            item_filled += 1
        if comments not in [None, '', 'N/A']:
            filled_fields += 1
            item_filled += 1
        
        if item_filled == 3:
            perfect_items += 1
        elif item_filled == 2:
            good_items += 1
        elif item_filled == 1:
            partial_items += 1
    
    completeness = (filled_fields / total_fields) * 100
    
    print(f"Total Items:        {total_items}")
    print(f"Total Fields:       {total_fields} (3 per item)")
    print(f"Filled Fields:      {filled_fields}")
    print(f"Overall Completeness: {completeness:.1f}%")
    print()
    print("Item Quality Distribution:")
    print(f"  • Perfect (3/3 fields):  {perfect_items:2} items ({perfect_items/total_items*100:5.1f}%)")
    print(f"  • Good (2/3 fields):     {good_items:2} items ({good_items/total_items*100:5.1f}%)")
    print(f"  • Partial (1/3 fields):  {partial_items:2} items ({partial_items/total_items*100:5.1f}%)")
    print()
    
    # Field-specific analysis
    print("="*100)
    print("📊 FIELD-SPECIFIC ANALYSIS")
    print("="*100)
    print()
    
    title_count = sum(1 for item in our_items if item.get('title') not in [None, '', 'N/A'])
    points_count = sum(1 for item in our_items if item.get('points') not in [None, '', 'N/A'])
    comments_count = sum(1 for item in our_items if item.get('comments') not in [None, '', 'N/A'])
    
    print(f"Field Coverage:")
    print(f"  • Title:    {title_count}/{total_items} ({title_count/total_items*100:.1f}%)")
    print(f"  • Points:   {points_count}/{total_items} ({points_count/total_items*100:.1f}%)")
    print(f"  • Comments: {comments_count}/{total_items} ({comments_count/total_items*100:.1f}%)")
    print()
    
    # Show items with missing data
    incomplete_items = [item for item in our_items 
                       if not all([
                           item.get('title') not in [None, '', 'N/A'],
                           item.get('points') not in [None, '', 'N/A'],
                           item.get('comments') not in [None, '', 'N/A']
                       ])]
    
    if incomplete_items:
        print("="*100)
        print(f"⚠️  INCOMPLETE ITEMS ({len(incomplete_items)} items)")
        print("="*100)
        print()
        print("These items are missing some fields:")
        print()
        
        for i, item in enumerate(incomplete_items, 1):
            title = item.get('title', 'N/A')
            points = item.get('points', 'N/A')
            comments = item.get('comments', 'N/A')
            
            missing = []
            if title in [None, '', 'N/A']:
                missing.append('title')
            if points in [None, '', 'N/A']:
                missing.append('points')
            if comments in [None, '', 'N/A']:
                missing.append('comments')
            
            title_display = str(title)[:70] if title not in [None, '', 'N/A'] else '(no title)'
            
            print(f"{i}. {title_display}")
            print(f"   Missing: {', '.join(missing)}")
            print()
    
    # Final verdict
    print("="*100)
    print("🎯 SUMMARY")
    print("="*100)
    print()
    
    print(f"✅ Successfully extracted {total_items} items")
    print(f"✅ Overall data completeness: {completeness:.1f}%")
    print(f"✅ {perfect_items} items have all 3 fields ({perfect_items/total_items*100:.1f}%)")
    
    if len(incomplete_items) > 0:
        print(f"⚠️  {len(incomplete_items)} items have partial data (still valuable!)")
    
    print()
    print("💡 Comparison to ScrapeGraphAI:")
    print("   • They extract: 30 items with ~100% completeness")
    print(f"   • We extract: {total_items} items with {completeness:.1f}% completeness")
    print(f"   • We get {total_items - 30} MORE items (though some have partial data)")
    print()
    
    if total_items >= 30:
        print("🎉 SUCCESS: We match or exceed ScrapeGraphAI's quantity!")
        if completeness >= 90:
            print("🎉 QUALITY: Our data completeness is excellent!")
        else:
            print("💡 NOTE: Some extra items have partial data - this is expected with lenient filtering")
    
    print()


if __name__ == "__main__":
    asyncio.run(analyze_our_data())



