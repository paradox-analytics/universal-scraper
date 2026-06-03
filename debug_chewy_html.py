#!/usr/bin/env python3
"""
Debug Chewy HTML content
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def main():
    fetcher = HybridFetcher(
        headless=True,
        use_camoufox=True,
        force_mode='browser'
    )
    
    try:
        url = "https://www.chewy.com/b/wet-food-389"
        result = await fetcher.fetch(url)
        html = result['html']
        print(f"HTML Length: {len(html)}")
        print("-" * 40)
        print(html)
        print("-" * 40)
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())
