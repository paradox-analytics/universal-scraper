#!/usr/bin/env python3
"""
Test CSS Bug Fix and Enhanced Bot Detection

Tests:
1. CSS selector escaping (Stack Overflow with `:` in class names)
2. Enhanced anti-bot detection (Etsy 403 bypass)
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              Testing CSS Fix + Enhanced Bot Detection                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        # Test 1: CSS Bug Fix (Stack Overflow)
        print("\n" + "="*80)
        print("Test 1: CSS Selector Bug Fix (Stack Overflow)")
        print("="*80)
        print("Issue: Class names with `:` (e.g., `h:bg-black-150`) treated as pseudo-classes")
        print("Fix: Escape special characters in CSS selectors")
        print()
        
        result1 = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes', 'answers', 'views']
        )
        
        items1 = result1.get('data', [])
        print(f"\n✅ Result: {len(items1)} items extracted")
        
        if items1:
            # Calculate quality
            total_fields = len(items1) * 4
            filled_fields = sum(
                1 for item in items1 
                for v in item.values() 
                if v is not None and v != ''
            )
            quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
            
            print(f"   Quality: {quality:.0f}%")
            print(f"   First item: {items1[0]}")
            
            if len(items1) >= 10 and quality >= 70:
                print("\n   🎉 CSS BUG FIX: SUCCESS!")
            elif len(items1) > 0:
                print("\n   ⚠️  Partial success (some data extracted)")
            else:
                print("\n   ❌ CSS bug may still exist")
        else:
            print("   ❌ No items extracted - CSS bug may still exist")
        
        # Test 2: Enhanced Bot Detection (Etsy)
        print("\n\n" + "="*80)
        print("Test 2: Enhanced Bot Detection (Etsy)")
        print("="*80)
        print("Issue: 403 Forbidden, CAPTCHA protection")
        print("Fix: Advanced fingerprinting, Bezier mouse movements, Canvas noise")
        print()
        
        result2 = await scraper.scrape(
            url='https://www.etsy.com/search?q=vintage+jewelry',
            fields=['title', 'price', 'shop']
        )
        
        items2 = result2.get('data', [])
        print(f"\n✅ Result: {len(items2)} items extracted")
        
        if items2:
            total_fields = len(items2) * 3
            filled_fields = sum(
                1 for item in items2 
                for v in item.values() 
                if v is not None and v != ''
            )
            quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
            
            print(f"   Quality: {quality:.0f}%")
            print(f"   First item: {items2[0]}")
            
            if len(items2) >= 10 and quality >= 70:
                print("\n   🎉 BOT DETECTION FIX: SUCCESS!")
            elif len(items2) > 0:
                print("\n   ⚠️  Partial success (bypassed some detection)")
            else:
                print("\n   ❌ Still blocked by anti-bot")
        else:
            print("   ⚠️  Still blocked by Etsy (may need proxies for this site)")
        
        # Summary
        print("\n\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        stackoverflow_success = len(items1) >= 10
        etsy_success = len(items2) > 0
        
        print(f"\n✅ CSS Bug Fix (Stack Overflow): {'SUCCESS' if stackoverflow_success else 'FAILED'}")
        print(f"{'✅' if etsy_success else '⚠️ '} Bot Detection (Etsy): {'SUCCESS' if etsy_success else 'STILL BLOCKED (may need proxies)'}")
        
        if stackoverflow_success and etsy_success:
            print("\n🎉 Both fixes working! Ready to re-test all 10 sites.")
        elif stackoverflow_success:
            print("\n✅ CSS bug fixed. Etsy still blocked (very strict anti-bot).")
        else:
            print("\n⚠️  Some issues remain. Check logs above.")
        
    finally:
        await scraper.close()

if __name__ == '__main__':
    asyncio.run(main())






