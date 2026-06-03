import asyncio
import json
import logging
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_json_extraction():
    # User provided credentials
    openai_key = "REDACTED_OPENAI_KEY_1"
    
    # Web Unlocker config
    web_unblocker_api_key = "brd.superproxy.io:33335:brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS"
    web_unblocker_zone = "web_unlocker1"

    url = "https://www.producthunt.com/categories/vibe-coding"
    fields = ["title", "author", "date", "post"]
    
    print(f"🚀 Testing JSON extraction for {url}")
    print(f"Fields: {fields}")
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=openai_key,
        use_camoufox=True,
        web_unblocker_api_key=web_unblocker_api_key,
        web_unblocker_zone=web_unblocker_zone
    )
    
    try:
        # Fetch HTML
        print("\n📥 Fetching HTML...")
        fetch_result = await scraper.html_fetcher.fetch(url)
        html_content = fetch_result['html']
        print(f"✅ Fetched {len(html_content)} chars")
        
        # Test JSON detection
        print("\n🔍 Testing JSON detection...")
        json_result = scraper.json_detector.detect_and_extract(html_content, url)
        
        print(f"\n📊 JSON Detection Results:")
        print(f"  json_found: {json_result.get('json_found', False)}")
        print(f"  sources: {json_result.get('sources', [])}")
        print(f"  data items: {len(json_result.get('data', []))}")
        
        # Check each data item
        for i, item in enumerate(json_result.get('data', [])):
            print(f"\n  Data item {i+1}:")
            print(f"    Framework: {item.get('_framework', 'unknown')}")
            if '_data' in item:
                data = item['_data']
                print(f"    Data type: {type(data)}")
                if isinstance(data, list):
                    print(f"    Data length: {len(data)}")
                    if data and isinstance(data[0], dict):
                        print(f"    First item keys: {list(data[0].keys())[:10]}")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_json_extraction())
