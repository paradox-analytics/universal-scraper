"""Debug: Check if custom elements survive cleaning"""
import asyncio
import os
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper

async def test():
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        extraction_context="Extract Reddit posts",
        fetch_mode="browser",
        headless=True
    )
    
    try:
        # Fetch and clean HTML
        fetch_result = await scraper.html_fetcher.fetch(
            "https://www.reddit.com/r/webscraping/",
            wait_for_selector="shreddit-post"
        )
        
        raw_html = fetch_result['html']
        
        # Clean it
        clean_result = scraper.html_cleaner.clean(raw_html)
        cleaned_html = clean_result['html']
        
        # Check for custom elements in both
        print(f"\n{'='*60}")
        print("RAW HTML:")
        print(f"  Size: {len(raw_html):,} bytes")
        raw_custom = set(re.findall(r'<([a-z]+-[a-z-]+)', raw_html))
        print(f"  Custom elements found: {list(raw_custom)[:10]}")
        print(f"  'shreddit-post' count: {raw_html.count('shreddit-post')}")
        
        print(f"\n{'='*60}")
        print("CLEANED HTML:")
        print(f"  Size: {len(cleaned_html):,} bytes")
        cleaned_custom = set(re.findall(r'<([a-z]+-[a-z-]+)', cleaned_html))
        print(f"  Custom elements found: {list(cleaned_custom)[:10]}")
        print(f"  'shreddit-post' count: {cleaned_html.count('shreddit-post')}")
        
        # Show sample of cleaned HTML around shreddit-post
        if 'shreddit-post' in cleaned_html:
            pos = cleaned_html.find('shreddit-post')
            sample = cleaned_html[max(0, pos-200):pos+500]
            print(f"\n{'='*60}")
            print("SAMPLE around 'shreddit-post':")
            print(sample[:700])
        
        print(f"\n{'='*60}\n")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    asyncio.run(test())







