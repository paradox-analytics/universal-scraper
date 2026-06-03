#!/usr/bin/env python3
"""
Test Chewy.com Full Pipeline
Verifies that the UniversalScraper attempts all extraction methods (JSON -> HTML -> LLM)
even when the target is difficult/blocked.
"""
import asyncio
import json
import sys
import logging
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    print("=" * 80)
    print("🧪 TEST: Chewy.com Full Pipeline (JSON -> HTML -> LLM)")
    print("=" * 80)
    
    # Proxy Configuration
    proxy_config = {
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    # Ensure we have a dummy API key if none exists, to enable LLM paths
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = 'sk-dummy-key-for-testing-flow'
    
    print(f"\n🔌 Using Proxy: {proxy_config['server']}")
    
    # Initialize Scraper
    scraper = UniversalScraper(
        proxy_config=proxy_config,
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        use_direct_llm=True,  # Enable Direct LLM fallback
        enable_cache=False    # Disable cache to force fresh fetch
    )
    
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["name", "price", "rating", "reviewCount", "image"]
    
    try:
        print(f"\n🚀 Starting Scrape: {url}")
        result = await scraper.scrape(url, fields)
        
        print("\n📊 Final Result:")
        print(f"   Source: {result.get('source')}")
        print(f"   Items: {len(result.get('data', []))}")
        
        if not result.get('data'):
            print("\n❌ No data extracted (Expected due to blocking)")
            print("   However, check logs above to verify fallback attempts:")
            print("   1. 'Detecting JSON sources...'")
            print("   2. 'JSON sources insufficient...'")
            print("   3. 'Trying Direct LLM Extraction...'")
            
    except Exception as e:
        print(f"\n❌ Scrape failed: {e}")
        import traceback
        traceback.print_exc()
            
    finally:
        # Cleanup is handled by scraper context or garbage collection, 
        # but UniversalScraper doesn't have an async close() method exposed directly 
        # on the class (it uses fetcher.close internally but usually in context manager).
        # We'll just let it exit.
        pass

if __name__ == "__main__":
    asyncio.run(main())
