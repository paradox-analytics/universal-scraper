#!/usr/bin/env python3
"""
End-to-end test: eBay extraction with new DOM pattern detection
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def main():
    print("="*80)
    print("🧪 eBay Extraction Test with DOM Pattern Detection")
    print("="*80)
    print()
    
    # Configuration
    url = "https://www.ebay.com/sch/i.html?_nkw=laptop"
    fields = ["title", "price", "condition", "shipping"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create scraper with Camoufox
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=True,
        headless=True,
        enable_llm_pagination=False
    )
    
    try:
        print(f"🎯 Scraping: {url}")
        print(f"📋 Fields: {', '.join(fields)}")
        print()
        
        result = await scraper.scrape(url, fields)
        
        data = result.get('data', [])
        print("="*80)
        print("✅ EXTRACTION COMPLETE")
        print("="*80)
        print(f"📊 Items extracted: {len(data)}")
        print(f"📦 Extraction source: {result.get('source', 'N/A')}")
        print(f"⏱️  Total time: {result.get('extraction_time', 0):.1f}s")
        print()
        
        if data:
            print("📋 Sample items (first 3):")
            for i, item in enumerate(data[:3], 1):
                print(f"\n   Item {i}:")
                for key, value in item.items():
                    print(f"      • {key}: {value}")
        else:
            print("❌ No items extracted!")
            
    finally:
        scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







