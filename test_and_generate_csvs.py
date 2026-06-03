"""
Test Fixes and Generate CSVs for Approval

This will test the new fixes and generate CSV files showing the corrected data.
You can then compare these to the old CSVs to confirm the fixes work.
"""

import asyncio
import os
import time
import logging
import csv
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_and_save(name, url, context, output_file):
    """Test a site and save results to CSV"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Context: {context}")
    print(f"Output: {output_file}")
    print(f"\n⏱️  Scraping with NEW FIXES...\n")
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY environment variable set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return None
    
    start = time.time()
    
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,  # Single page test
        extraction_context=context,
        enable_context_validation=True,
    )
    
    try:
        result = await scraper.scrape(url, fields=[])  # Auto-extract
        elapsed = time.time() - start
        
        data = result.get('data', [])
        metadata = result.get('metadata', {})
        source = metadata.get('extraction_source', 'unknown')
        
        print(f"\n📊 RESULTS:")
        print(f"   Items: {len(data)}")
        print(f"   Source: {source}")
        print(f"   Time: {elapsed:.1f}s")
        
        if len(data) > 0:
            # Save to CSV
            if data:
                keys = list(data[0].keys())
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(data)
                
                print(f"   ✅ Saved {len(data)} items to {output_file}")
                
                # Show sample
                print(f"\n📝 Sample (first 2 items):")
                for i, item in enumerate(data[:2]):
                    print(f"\n   Item {i+1}:")
                    for k, v in list(item.items())[:5]:  # First 5 fields
                        val_str = str(v)[:80]
                        print(f"      {k}: {val_str}")
            else:
                print(f"   ⚠️  No data to save")
        else:
            print(f"   ❌ 0 items extracted")
        
        print(f"\n{'='*80}")
        
        return {
            'name': name,
            'items': len(data),
            'source': source,
            'time': elapsed,
            'success': len(data) > 0
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        scraper.close()

async def main():
    print("\n" + "="*80)
    print("🔬 TESTING NEW FIXES - Generate CSVs for Approval")
    print("="*80)
    print("\nThis will test all fixes and generate new CSV files:")
    print("  - reddit_sample_FIXED.csv")
    print("  - apify_sample_FIXED.csv")
    print("  - metacritic_sample_FIXED.csv")
    print("  - ebay_sample_FIXED.csv")
    print("\nYou can then compare these to the old CSV files to confirm the fixes work.")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ ERROR: No OPENAI_API_KEY environment variable set")
        print("\nTo set it, run:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\nOr add it to your .bashrc/.zshrc for persistence.")
        return
    
    print(f"\n✅ API Key found: {api_key[:20]}...")
    print(f"\n⏱️  Starting tests (this will take 5-10 minutes)...\n")
    
    tests = [
        {
            'name': "Reddit",
            'url': "https://www.reddit.com/r/webscraping/",
            'context': "Extract Reddit posts with title, author, upvotes, comments count",
            'output': 'reddit_sample_FIXED.csv'
        },
        {
            'name': "Apify",
            'url': "https://apify.com/",
            'context': "Extract Apify actors/scrapers with name, description, author, rating",
            'output': 'apify_sample_FIXED.csv'
        },
        {
            'name': "Metacritic",
            'url': "https://www.metacritic.com/browse/game/all/all/current-year/",
            'context': "Extract video game listings with title, platform, release date, Metascore",
            'output': 'metacritic_sample_FIXED.csv'
        },
        {
            'name': "eBay",
            'url': "https://www.ebay.com/b/Apple-Laptops/111422/bn_320025",
            'context': "Extract Apple laptop listings with title, price, condition, seller",
            'output': 'ebay_sample_FIXED.csv'
        }
    ]
    
    results = []
    for test in tests:
        result = await test_and_save(
            test['name'],
            test['url'],
            test['context'],
            test['output']
        )
        if result:
            results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    if not results:
        print("\n❌ No tests completed successfully")
        print("   Check the API key and try again")
        return
    
    success_count = sum(1 for r in results if r['success'])
    total_items = sum(r['items'] for r in results)
    
    print(f"\nSuccess Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    print(f"Total Items: {total_items}\n")
    
    print("RESULTS:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['name']}: {r['items']} items from {r['source']} ({r['time']:.1f}s)")
    
    print("\n" + "="*80)
    print("📁 CSV FILES GENERATED")
    print("="*80)
    print("\nYou can now compare the old vs new CSV files:")
    print("  OLD (before fixes):")
    print("    - reddit_sample.csv")
    print("    - apify_sample.csv")
    print("    - metacritic_sample.csv")
    print("    - ebay_sample.csv")
    print("\n  NEW (after fixes):")
    print("    - reddit_sample_FIXED.csv")
    print("    - apify_sample_FIXED.csv")
    print("    - metacritic_sample_FIXED.csv")
    print("    - ebay_sample_FIXED.csv")
    
    print("\n" + "="*80)
    print("🎯 APPROVAL CHECKLIST")
    print("="*80)
    print("\nCheck each CSV file:")
    print("  ✓ Reddit: Should have posts (not app config)")
    print("  ✓ Apify: Should have actors (not JS libraries)")
    print("  ✓ Metacritic: Should have games (not GDPR config)")
    print("  ✓ eBay: Should have laptops (not UI actions)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())








