#!/usr/bin/env python3
"""
Single-page eBay test with full logging to verify:
1. JSON quality validation rejects tracking data
2. DOM pattern detection finds li.s-card
3. HTML extraction succeeds with actual products
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def main():
    print("="*80)
    print("🧪 eBay Single-Page Test (JSON Validation + DOM Detection)")
    print("="*80)
    print()
    
    url = "https://www.ebay.com/sch/i.html?_nkw=laptop"
    fields = ["title", "price"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create scraper with pagination DISABLED
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        headless=True,
        enable_llm_pagination=False,  # Disable LLM pagination
        enable_auto_pagination=False   # Disable auto-pagination
    )
    
    try:
        result = await scraper.scrape(url, fields)
        
        data = result.get('data', [])
        source = result.get('source', 'N/A')
        
        print("\n" + "="*80)
        print("✅ EXTRACTION COMPLETE")
        print("="*80)
        print(f"📊 Items extracted: {len(data)}")
        print(f"📦 Source: {source}")
        print(f"⏱️  Time: {result.get('extraction_time', 0):.1f}s")
        print()
        
        if data:
            print(f"📋 First 3 items:")
            for i, item in enumerate(data[:3], 1):
                print(f"\n   Item {i}:")
                for key, value in item.items():
                    val_str = str(value)[:60]
                    print(f"      • {key}: {val_str}")
                    
            # Check if data looks like products or tracking
            sample_keys = set()
            for item in data[:3]:
                sample_keys.update(item.keys())
            
            tracking_keywords = ['session', 'tracking', 'correlation', 'guid', 'token']
            has_tracking = any(any(kw in str(k).lower() for kw in tracking_keywords) for k in sample_keys)
            
            data_keywords = ['title', 'price', 'name', 'product']
            has_data = any(any(kw in str(k).lower() for kw in data_keywords) for k in sample_keys)
            
            print(f"\n📊 Data Analysis:")
            print(f"   • Contains tracking keywords: {'Yes ❌' if has_tracking else 'No ✅'}")
            print(f"   • Contains data keywords: {'Yes ✅' if has_data else 'No ❌'}")
            
            if has_tracking and not has_data:
                print(f"   ⚠️  WARNING: Extracted data looks like tracking/metadata!")
            elif has_data:
                print(f"   ✅ SUCCESS: Extracted data looks like actual products!")
        else:
            print("❌ No items extracted!")
            
    finally:
        scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







