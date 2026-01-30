import asyncio
import os
import json
import logging
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.json_detector import JSONDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_ph():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return
    fetcher = CamoufoxFetcher()
    cleaner = SmartHTMLCleaner()
    detector = JSONDetector()
    
    url = "https://www.producthunt.com/"
    print(f"Fetching {url}...")
    
    result = await fetcher.fetch(url)
    html = result.get('html', '')
    
    print(f"Raw HTML length: {len(html)}")
    
    # Check for garbage in raw HTML
    garbage_escapes = ["\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014", "\\u0015"]
    found_garbage = [g for g in garbage_escapes if g in html]
    print(f"Garbage escapes found in raw HTML: {found_garbage}")
    
    # Test Cleaner
    print("\nCleaning HTML...")
    cleaned_result = cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"Cleaned HTML length: {len(cleaned_html)}")
    
    found_garbage_cleaned = [g for g in garbage_escapes if g in cleaned_html]
    print(f"Garbage escapes found in cleaned HTML: {found_garbage_cleaned}")
    
    # Test Detector
    print("\nDetecting JSON...")
    json_result = detector.detect_and_extract(html, url)
    
    print(f"JSON found: {json_result.get('json_found', False)}")
    print(f"Sources: {json_result.get('sources', [])}")
    
    # Save results
    with open("ph_debug_results.json", "w") as f:
        json.dump({
            "garbage_in_raw": found_garbage,
            "garbage_in_cleaned": found_garbage_cleaned,
            "json_sources": json_result.get('sources', []),
            "json_data_sample": json_result.get('data', [])[:2]
        }, f, indent=2)
    
    print("\nResults saved to ph_debug_results.json")

if __name__ == "__main__":
    asyncio.run(debug_ph())
