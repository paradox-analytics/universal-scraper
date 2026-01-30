#!/usr/bin/env python3
"""Local test of scraper with real website"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up Python path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import scraper
from universal_scraper.core.scraper import UniversalScraper

# Test URL
TEST_URL = "https://www.metacritic.com/pictures/december-2025-movie-preview-avatar-kill-bill-marty-supreme/"

async def test_scrape():
    """Test scraping the Metacritic page"""
    
    print("=" * 80)
    print("LOCAL SCRAPER TEST")
    print("=" * 80)
    print(f"\nTesting URL: {TEST_URL}\n")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("   The scraper will run but LLM extraction may be limited")
        print("   Set OPENAI_API_KEY environment variable for full functionality\n")
    else:
        print(f"✓ OpenAI API key found: {api_key[:10]}...{api_key[-4:]}\n")
    
    # Fields to extract from the movie preview page
    fields = [
        "movie_title",
        "director",
        "release_date",
        "metascore",
        "description",
        "cast",
        "genre"
    ]
    
    print(f"Fields to extract: {', '.join(fields)}\n")
    print("-" * 80)
    print("Starting scrape...\n")
    
    try:
        # Initialize scraper (same config as main.py)
        scraper = UniversalScraper(
            api_key=api_key if api_key else "test-key-will-fail-gracefully",
            use_camoufox=True,
            fetch_mode='browser',  # Use browser mode for JS sites
            browser_timeout=120000,  # 2 minutes timeout
            use_direct_llm=True,
            enable_cache=True,  # Enable cache for local testing
            log_level=logging.INFO
        )
        
        # Scrape the URL
        result = await scraper.scrape(
            url=TEST_URL,
            fields=fields
        )
        
        print("\n" + "=" * 80)
        print("SCRAPE RESULTS")
        print("=" * 80)
        
        # Extract data from result (scraper returns dict with 'data' key)
        items = result.get('data', []) if isinstance(result, dict) else result if isinstance(result, list) else []
        
        if items:
            print(f"\n✓ Successfully extracted {len(items)} items\n")
            
            # Display first 5 items as sample
            display_count = min(5, len(items))
            print(f"Displaying first {display_count} items:\n")
            
            for i, item in enumerate(items[:display_count], 1):
                print(f"\n--- Item {i} ---")
                for key, value in item.items():
                    if key.startswith('_'):
                        continue  # Skip metadata for now
                    if value:
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 150:
                            value = value[:150] + "..."
                        print(f"  {key}: {value}")
            
            if len(items) > display_count:
                print(f"\n... and {len(items) - display_count} more items")
            
            # Save to file
            output_file = script_dir / "test_output.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {output_file}")
            print(f"\n✓ Test completed successfully!")
            print(f"✓ Total items extracted: {len(items)}")
            
        else:
            print("\n⚠️  No results returned")
            print("   This may indicate the scraper couldn't extract data")
            print("   Check logs above for details")
            print(f"   Result type: {type(result)}")
            print(f"   Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_scrape())

