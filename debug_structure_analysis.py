"""
Debug script to see what the HTML structure analyzer returns for failing sources
"""

import asyncio
import os
import json
from universal_scraper import UniversalScraper

# Test sources that are failing
FAILING_SOURCES = {
    'ebay': {
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'context': 'Extract eBay product listings with title, price, shipping, condition'
    },
    'metacritic': {
        'url': 'https://www.metacritic.com/browse/game/',
        'context': 'Extract game listings with title, platform, score, release date'
    }
}

async def debug_source(name, config):
    """Debug a single source to see structure analysis"""
    print("\n" + "="*80)
    print(f"🔍 DEBUGGING: {name.upper()}")
    print("="*80)
    print(f"URL: {config['url']}")
    print(f"Context: {config['context']}")
    print()
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        enable_cache=False,  # Disable cache to get fresh analysis
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=config['context']  # Pass context here, not in scrape()
    )
    
    try:
        # Scrape (this will trigger structure analysis)
        result = await scraper.scrape(
            config['url'],
            fields=[]
        )
        
        print(f"\n📊 Results:")
        print(f"   Items extracted: {len(result.get('data', []))}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(result.get('data', [])) > 0:
            print(f"\n   First item:")
            print(f"   {json.dumps(result['data'][0], indent=2)}")
        else:
            print("\n   ❌ No items extracted")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()
        
    print("\n" + "="*80)

async def main():
    """Run diagnostics on all failing sources"""
    print("\n" + "="*80)
    print("🧪 HTML STRUCTURE ANALYZER DIAGNOSTICS")
    print("="*80)
    print("\nThis will show what the LLM structure analyzer identifies for failing sources.")
    print("Look for the 'HTML Structure Analysis' logs to see what it detects.\n")
    
    for name, config in FAILING_SOURCES.items():
        await debug_source(name, config)
        print("\n⏳ Waiting 5 seconds before next source...")
        await asyncio.sleep(5)
    
    print("\n" + "="*80)
    print("✅ Diagnostics complete")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())

