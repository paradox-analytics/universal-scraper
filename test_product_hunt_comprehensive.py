import asyncio
import json
import os
import logging
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_comprehensive():
    # User provided credentials
    openai_key = "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"
    
    # Web Unlocker config
    # Format: host:port:username:password
    # brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1
    web_unblocker_api_key = "brd.superproxy.io:33335:brd-customer-hl_803e8195-zone-web_unlocker1:t8mhp1qev1i1"
    web_unblocker_zone = "web_unlocker1"

    url = "https://www.producthunt.com/categories/vibe-coding"
    fields = ["title", "author", "date", "post"]
    # Simpler target
    target = "Product cards with title, author, and description"
    
    print(f"🚀 Starting comprehensive test for {url}")
    print(f"Target: {target}")
    print(f"Fields: {fields}")
    print(f"Using Web Unlocker: {web_unblocker_api_key[:20]}...")
    
    # Initialize scraper with Web Unlocker settings
    # Note: We pass web_unblocker_api_key directly as the formatted string expected by CamoufoxFetcher
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True,
        web_unblocker_api_key=web_unblocker_api_key,
        web_unblocker_zone=web_unblocker_zone
    )
    
    try:
        # Fetch HTML directly for debugging
        print("   Fetching HTML for inspection...")
        fetch_result = await scraper.html_fetcher.fetch(url)
        html_content = fetch_result['html']
        
        with open('debug_product_hunt.html', 'w') as f:
            f.write(html_content)
        print(f"   HTML saved to debug_product_hunt.html ({len(html_content)} chars)")

        # Run extraction
        result = await scraper.scrape(
            url=url,
            fields=fields
        )
        
        # Analyze results
        data = result.get('data', [])
        count = len(data)
        print(f"\n📊 Extraction Results:")
        print(f"   Total Items: {count}")
        print(f"   Expected: ~43")
        
        # Check for garbage
        garbage_count = 0
        replacement_char_count = 0
        
        for i, item in enumerate(data):
            item_str = json.dumps(item)
            if '\ufffd' in item_str:
                replacement_char_count += 1
                print(f"   ⚠️  Item {i+1} contains replacement characters!")
            
            # Basic quality check (empty fields)
            empty_fields = [k for k, v in item.items() if not v]
            if len(empty_fields) > len(fields) / 2:
                print(f"   ⚠️  Item {i+1} has mostly empty fields: {empty_fields}")
                garbage_count += 1

        print(f"\n   Garbage/Empty Items: {garbage_count}")
        print(f"   Corrupted Items (\\ufffd): {replacement_char_count}")
        
        if count >= 40 and garbage_count == 0 and replacement_char_count == 0:
            print("\n✅ TEST PASSED: High quality data extracted.")
        else:
            print("\n❌ TEST FAILED: Issues detected.")
            
        # Save results for inspection
        with open('comprehensive_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("   Results saved to comprehensive_test_results.json")

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_comprehensive())
