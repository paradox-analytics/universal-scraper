#!/usr/bin/env python3
"""
Simple Field Mapper test without Camoufox (to avoid library bugs)
Tests semantic field understanding on GitHub Trending
"""

import asyncio
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import sys
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper


async def main():
    print("="*80)
    print("🧪 FIELD MAPPER TEST - GitHub Trending (without Camoufox)")
    print("="*80)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        return
    
    url = "https://github.com/trending"
    fields = ["repository", "description", "stars", "language"]
    
    print(f"🎯 URL: {url}")
    print(f"📋 Fields: {', '.join(fields)}")
    print()
    
    scraper = None
    try:
        scraper = UniversalScraper(
            api_key=api_key,
            model_name="gpt-4o-mini",
            use_camoufox=False,  # Use regular Playwright instead
            headless=True,
            enable_auto_pagination=False
        )
        
        print("✅ Field Mapper enabled (semantic understanding)")
        print("🚀 Scraping...")
        print()
        
        result = await scraper.scrape(url, fields)
        
        print()
        print("="*80)
        print("✅ RESULTS")
        print("="*80)
        print(f"📊 Items: {len(result['data'])}")
        print(f"⏱️  Time: {result.get('total_time', 0):.1f}s")
        print()
        
        if result['data']:
            # Check repository field specifically (the problem field)
            repos_found = sum(1 for item in result['data'] if item.get('repository'))
            repo_success = (repos_found / len(result['data'])) * 100
            
            complete = sum(
                1 for item in result['data']
                if all(item.get(f) for f in fields)
            )
            quality = (complete / len(result['data'])) * 100
            
            print(f"📈 Overall Quality: {quality:.0f}% ({complete}/{len(result['data'])} complete)")
            print(f"🎯 Repository Field: {repo_success:.0f}% ({repos_found}/{len(result['data'])} found)")
            print()
            
            if repo_success >= 80:
                print("✅ SUCCESS! Field Mapper dramatically improved accuracy")
                print("   (Was 0% before semantic mapping)")
            elif repo_success >= 50:
                print("⚠️  PARTIAL: Some improvement")
            else:
                print("❌ Field Mapper didn't help")
            
            print()
            print("Sample (first 2):")
            for i, item in enumerate(result['data'][:2], 1):
                print(f"\n   Item {i}:")
                for k, v in item.items():
                    status = "✅" if v else "❌"
                    print(f"      {status} {k}: {str(v)[:60]}")
        
        else:
            print("❌ No items extracted")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if scraper:
            await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())







