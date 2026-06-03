#!/usr/bin/env python3
"""
Test Product Hunt Universal Extraction
Verifies that the enhanced JSONDetector correctly extracts products from Next.js 13+ RSC payload
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
    print("🧪 TEST: Product Hunt Universal Extraction")
    print("=" * 80)
    
    # 1. Fetch the page
    print("\n1️⃣  Fetching Product Hunt...")
    fetcher = HybridFetcher(
        headless=True,
        use_camoufox=True,
        force_mode='browser'  # Force browser to ensure we get JS-rendered content
    )
    
    try:
        url = "https://www.producthunt.com/"
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
        
        # Verify we found inline JSON
        has_inline = any('inline-json' in s or 'rsc_payload' in str(d) for s in json_results['sources'] for d in json_results['data'])
        if has_inline:
            print("✅ SUCCESS: Detected inline JSON / RSC payload")
        else:
            print("❌ FAILURE: Did not detect inline JSON")
            
        # 3. Extract Products
        print("\n3️⃣  Extracting Products...")
        fields = ["name", "tagline", "votesCount", "commentsCount", "slug"]
        
        # Use the detector's sufficiency check logic which does the extraction
        extracted = detector.extract_from_json(json_results['data'], fields)
        
        print(f"📦 Extracted {len(extracted)} items")
        
        if extracted:
            print("\n🎯 Sample Items:")
            for i, item in enumerate(extracted[:5], 1):
                print(f"\n{i}. {item.get('name', 'Unknown')}")
                print(f"   Tagline: {item.get('tagline', 'N/A')}")
                print(f"   Votes: {item.get('votesCount', item.get('latestScore', 'N/A'))}")
                print(f"   Slug: {item.get('slug', 'N/A')}")
                
            # Validation
            assert len(extracted) >= 10, f"Should extract at least 10 products, found {len(extracted)}"
            assert extracted[0].get('name'), "First item should have a name"
            
            print("\n✅ TEST PASSED: Successfully extracted products from Product Hunt!")
        else:
            print("\n❌ TEST FAILED: No items extracted")
            
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())
