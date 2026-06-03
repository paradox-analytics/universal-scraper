#!/usr/bin/env python3
"""
Test Enhanced DOM Pattern Detector on Stack Overflow

Verify that content-based scoring correctly identifies data containers
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║        Testing Enhanced DOM Pattern Detector (Content-Based)              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Site: Stack Overflow
Previous Issue: Detected li.h:bg-black-150 (800 UI elements) ❌
Expected: Should detect div.s-post-summary (15 questions) ✅

Changes:
✅ Content-based scoring (text density, semantic HTML, structure)
✅ Frequency penalty (>100 instances heavily penalized)
✅ No keyword ontology (universal for all sites)
✅ Reverted to GPT-4o-mini (cost savings)
    """)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        print("\n🔍 Scraping Stack Overflow...")
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes', 'answers', 'views']
        )
        
        items = result.get('data', [])
        
        print(f"\n📊 Results:")
        print(f"   Items Extracted: {len(items)}")
        
        if items:
            # Calculate quality
            total_fields = len(items) * 4
            filled_fields = sum(
                1 for item in items 
                for v in item.values() 
                if v is not None and v != ''
            )
            quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
            
            print(f"   Quality: {quality:.0f}%")
            print(f"\n   Sample Items:")
            for i, item in enumerate(items[:3], 1):
                null_count = sum(1 for v in item.values() if v is None or v == '')
                print(f"   {i}. {item}")
                if null_count > 0:
                    null_fields = [k for k, v in item.items() if v is None or v == '']
                    print(f"      ⚠️  Null fields: {', '.join(null_fields)}")
            
            # Success criteria
            if len(items) >= 10 and quality >= 70:
                print(f"\n✅ SUCCESS!")
                print(f"   • Extracted {len(items)} items (expected 15+)")
                print(f"   • Quality: {quality:.0f}% (expected 70%+)")
                print(f"\n🎉 Content-based DOM detection WORKS!")
                return True
            elif len(items) > 0:
                print(f"\n⚠️  PARTIAL SUCCESS")
                print(f"   • Extracted {len(items)} items")
                print(f"   • Quality: {quality:.0f}%")
                print(f"   • Need to improve field extraction")
                return False
            else:
                print(f"\n❌ FAILED - 0 items")
                return False
        else:
            print(f"   ❌ 0 items extracted")
            print(f"\n❌ Content-based scoring did not fix the issue")
            print(f"   Check logs above for detected pattern")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await scraper.close()

if __name__ == '__main__':
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)






