#!/usr/bin/env python3
"""
Test Chewy.com Universal Extraction
Verifies that the universal scraper works on Chewy category pages
"""
import asyncio
import json
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    print("=" * 80)
    print("🧪 TEST: Chewy.com Universal Extraction")
    print("=" * 80)
    
    # 1. Fetch the page
    url = "https://www.chewy.com/b/wet-food-389"
    print(f"\n1️⃣  Fetching {url}...")
    
    fetcher = HybridFetcher(
        headless=True,
        use_camoufox=True,
        force_mode='browser'  # Chewy usually requires browser
    )
    
    try:
        result = await fetcher.fetch(url)
        html = result['html']
        print(f"✅ Fetched {len(html):,} bytes")
        
        # 2. Detect and Extract JSON
        print("\n2️⃣  Detecting JSON...")
        detector = JSONDetector()
        json_results = detector.detect_and_extract(html, url)
        
        print(f"📊 Detection Results:")
        print(f"   JSON Found: {json_results['json_found']}")
        print(f"   Sources: {json_results['sources']}")
        print(f"   Data Objects: {len(json_results['data'])}")
        
        # 3. Extract Products
        print("\n3️⃣  Extracting Products...")
        fields = ["name", "price", "rating", "reviewCount", "image"]
        
        # Use the detector's extraction logic
        extracted = detector.extract_from_json(json_results['data'], fields)
        
        print(f"📦 Extracted {len(extracted)} items")
        
        if extracted:
            print("\n🎯 Sample Items:")
            for i, item in enumerate(extracted[:5], 1):
                print(f"\n{i}. {item.get('name', 'Unknown')}")
                print(f"   Price: {item.get('price', 'N/A')}")
                print(f"   Rating: {item.get('rating', 'N/A')}")
                print(f"   Reviews: {item.get('reviewCount', 'N/A')}")
                
            # Validation
            if len(extracted) >= 5:
                print("\n✅ TEST PASSED: Successfully extracted products from Chewy!")
            else:
                print("\n⚠️  TEST WARNING: Extracted fewer items than expected")
        else:
            print("\n❌ TEST FAILED: No items extracted from JSON")
            print("   (This might be expected if Chewy doesn't use embedded JSON for products)")
            
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())
