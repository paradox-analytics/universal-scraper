import asyncio
import json
import os
import logging
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.json_quality_validator import JSONQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reproduce_issue():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    # Mimic frontend settings: OpenAI, Unlocker, Bright Data enabled
    # In api/main.py, 'use_camoufox' defaults to True.
    # Web Unblocker is usually handled via specific fetcher or proxy settings.
    # We'll use the default UniversalScraper which should use CamoufoxFetcher by default.
    
    url = "https://www.producthunt.com/categories/v"
    fields = ["title", "author", "date", "post"]
    target = "Products"
    
    print(f"🚀 Starting reproduction for {url}")
    print(f"Target: {target}")
    print(f"Fields: {fields}")
    
    scraper = UniversalScraper(api_key=api_key, use_camoufox=True)
    
    try:
        # 1. Fetch HTML
        print("\n1. Fetching HTML...")
        fetch_result = await scraper.html_fetcher.fetch(url)
        html = fetch_result.get('html', '')
        print(f"   HTML Length: {len(html)}")
        
        # Check for replacement characters in HTML
        REPLACEMENT_CHAR = '\ufffd'
        if REPLACEMENT_CHAR in html:
            print(f"   ⚠️  WARNING: Found replacement characters (\\ufffd) in HTML!")
            count = html.count(REPLACEMENT_CHAR)
            print(f"   Count: {count}")
            start_idx = html.find(REPLACEMENT_CHAR)
            print(f"   Sample: {html[start_idx:start_idx+50]}")
        else:
            print("   ✅ No replacement characters found in HTML.")

        # 2. Detect JSON
        print("\n2. Detecting JSON...")
        # We'll manually call detect_and_extract to inspect the raw results
        json_results = scraper.json_detector.detect_and_extract(html, url)
        
        print(f"   JSON Found: {json_results.get('json_found')}")
        print(f"   Sources: {json_results.get('sources')}")
        
        # Inspect extracted data for garbage
        data = json_results.get('data', [])
        print(f"   Extracted {len(data)} items.")
        
        validator = JSONQualityValidator()
        
        for i, item in enumerate(data):
            print(f"\n   Item {i+1} ({item.get('_framework', 'unknown')}):")
            # Check for garbage in values
            is_garbage = False
            data_content = item.get('_data', {})
            
            def check_garbage(obj):
                found_garbage = False
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str) and '\ufffd' in value:
                            print(f"     ❌ GARBAGE DETECTED in key '{key}': {value[:50]}...")
                            found_garbage = True
                        elif isinstance(value, (dict, list)):
                            if check_garbage(value):
                                found_garbage = True
                elif isinstance(obj, list):
                    for sub_item in obj:
                        if check_garbage(sub_item):
                            found_garbage = True
                return found_garbage

            is_garbage = check_garbage(data_content)
            
            # Run validator check
            is_high_quality = validator.is_high_quality_value(item.get('_data'))
            print(f"     Validator says high quality? {is_high_quality}")
            
            if is_garbage and is_high_quality:
                print("     🚨 CRITICAL FAILURE: Validator passed garbage data!")
                
    except Exception as e:
        logger.error(f"Error during reproduction: {e}", exc_info=True)
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(reproduce_issue())
