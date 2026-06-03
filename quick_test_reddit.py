"""Quick test to verify Reddit scraping works"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper

async def test():
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        extraction_context="Extract Reddit posts with title, author, upvotes, comments",
        fetch_mode="browser",
        headless=True,
        enable_llm_pagination=False
    )
    
    # Disable pagination
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    
    try:
        result = await scraper.scrape(
            "https://www.reddit.com/r/webscraping/",
            fields=["title", "author", "upvotes", "comments_count"],
            wait_for_selector="shreddit-post"
        )
        
        # Extract actual data (scraper returns dict with 'data' key)
        if isinstance(result, dict) and 'data' in result:
            result_list = result['data']
        elif isinstance(result, list):
            result_list = result
        else:
            result_list = []
        
        print(f"\n✅ SUCCESS: Extracted {len(result_list)} items")
        
        if result_list:
            print("\n📋 Sample items:")
            for i, item in enumerate(result_list[:3], 1):
                title = item.get('title', 'N/A')
                title_display = title[:60] if title else 'N/A'
                print(f"\n  {i}. {title_display}...")
                print(f"     Author: {item.get('author', 'N/A')}")
        
        return len(result_list)
    
    finally:
        scraper.close()

if __name__ == "__main__":
    count = asyncio.run(test())
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {count} items extracted")
    print(f"{'='*60}\n")

