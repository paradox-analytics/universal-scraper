#!/usr/bin/env python3
"""
Test if smaller chunk sizes (like ScrapeGraphAI) help extract more items
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


async def test_chunk_sizes():
    """Test different chunk sizes to see if smaller chunks help"""
    
    print("\n" + "="*100)
    print("🧪 Chunk Size Experiment")
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
    
    print(f"✅ HTML prepared: {len(cleaned_html):,} bytes (~{len(cleaned_html)/4:.0f} tokens)")
    print()
    
    # Test different chunk sizes
    chunk_sizes = [
        (4000, "ScrapeGraphAI size"),
        (8000, "Medium"),
        (25000, "Our current size"),
    ]
    
    for chunk_size, description in chunk_sizes:
        print("=" * 100)
        print(f"🧪 Testing chunk_size={chunk_size:,} tokens ({description})")
        print("=" * 100)
        
        extractor = DirectLLMExtractor(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4o-mini",
            max_tokens_per_chunk=chunk_size,
            quality_mode="balanced",
            use_html2text=True
        )
        
        items = await extractor.extract(
            cleaned_html,
            fields,
            context="Extract all article listings with title, points, and comments count"
        )
        
        print(f"✅ Extracted {len(items)} items with chunk_size={chunk_size:,}")
        print()
        
        if items:
            print(f"   Sample (first 3):")
            for i, item in enumerate(items[:3], 1):
                print(f"   {i}. {item.get('title', 'N/A')[:60]}")
            print()
    
    print("=" * 100)
    print("🎯 CONCLUSION")
    print("=" * 100)
    print("Target: 30 items (what ScrapeGraphAI gets)")
    print()


if __name__ == "__main__":
    asyncio.run(test_chunk_sizes())



