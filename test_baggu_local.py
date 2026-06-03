"""
Test Baggu scraping locally with the same inputs as Apify
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper

async def main():
    # Use the same inputs as Apify
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    url = "https://baggu.com/collections/crescent-bags"
    fields = ["title", "price", "color", "product detail url"]
    
    print(f"🔍 Testing Baggu scraping:")
    print(f"   URL: {url}")
    print(f"   Fields: {fields}")
    print()
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=120000,
        use_direct_llm=True,
        enable_auto_pagination=False,
        log_level=20  # INFO level
    )
    
    try:
        result = await scraper.scrape(url, fields)
        
        items = result.get('data', [])
        print(f"\n{'='*80}")
        print(f"📊 RESULTS")
        print(f"{'='*80}")
        print(f"   Total items extracted: {len(items)}")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Success: {result.get('success', False)}")
        
        if items:
            print(f"\n   First item keys: {list(items[0].keys())}")
            print(f"\n   First item:")
            print(json.dumps(items[0], indent=2, default=str))
            
            # Check field normalization
            print(f"\n   Field normalization check:")
            for item in items[:3]:
                print(f"      Item keys: {list(item.keys())}")
                for key in item.keys():
                    if 'url' in key.lower() or 'detail' in key.lower():
                        print(f"         - {key}: {item[key]}")
        
        # Save results
        output_file = "baggu_local_test_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'result': result,
                'items': items,
                'total_items': len(items)
            }, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())







