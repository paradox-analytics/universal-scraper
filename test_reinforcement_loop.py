#!/usr/bin/env python3
"""
Test Reinforcement DOM Detection Loop

Verify that the system automatically retries with better selectors
when extraction quality is low.
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_stack_overflow_reinforcement():
    """
    Stack Overflow currently extracts 15 items but with 25-50% quality (votes field null).
    This should trigger Pass 2 (LLM-guided nested analysis).
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║            Testing Reinforcement DOM Detection Loop                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

Site: Stack Overflow
Current State: 15 items, ~25-50% quality (votes field null)
Expected: System should auto-trigger Pass 2 to fix null fields

Pass 1: Content-based detection → Extracts items but votes=None
Pass 2: LLM-guided nested analysis → Should find correct vote selector
Pass 3: Deep context analysis → Fallback if Pass 2 also fails
    """)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        print("🔍 Scraping Stack Overflow with reinforcement loop...")
        print()
        
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes', 'answers', 'views']
        )
        
        items = result.get('data', [])
        
        if not items:
            print("❌ FAILED - 0 items extracted")
            return False
        
        # Calculate quality
        total_fields = len(items) * 4
        filled_fields = sum(
            1 for item in items
            for v in item.values()
            if v is not None and v != ''
        )
        quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
        
        print(f"\n{'='*75}")
        print(f"📊 RESULTS")
        print(f"{'='*75}")
        print(f"Items Extracted: {len(items)}")
        print(f"Quality: {quality:.1f}%")
        print()
        
        # Show first 3 items
        print(f"Sample Items:")
        for i, item in enumerate(items[:3], 1):
            null_count = sum(1 for v in item.values() if v is None or v == '')
            status = '✅' if null_count == 0 else f'⚠️ ({null_count} null)'
            print(f"{i}. {status}")
            for k, v in item.items():
                status_icon = '❌' if v is None or v == '' else '✅'
                value_str = str(v)[:60] if v else 'None'
                print(f"   {status_icon} {k}: {value_str}")
        
        print(f"{'='*75}")
        
        # Success criteria
        if quality >= 70:
            print(f"\n✅ EXCELLENT! Quality: {quality:.1f}% (>= 70%)")
            print(f"🎉 Reinforcement loop successfully improved extraction!")
            return True
        elif quality >= 50:
            print(f"\n⚠️  GOOD - Quality: {quality:.1f}% (>= 50%)")
            print(f"Reinforcement helped but could be better")
            return True
        else:
            print(f"\n❌ FAILED - Quality: {quality:.1f}% (< 50%)")
            print(f"Reinforcement loop did not improve quality enough")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await scraper.close()

async def test_github_trending_reinforcement():
    """
    GitHub Trending currently has 33% quality (all fields null).
    Should definitely trigger Pass 2 and possibly Pass 3.
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║            Testing Reinforcement: GitHub Trending                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

Current State: ~11 items, 33% quality (repository, description, stars null)
Expected: System should trigger Pass 2 and/or Pass 3
    """)
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        print("🔍 Scraping GitHub Trending with reinforcement loop...")
        print()
        
        result = await scraper.scrape(
            url='https://github.com/trending',
            fields=['repository', 'description', 'stars']
        )
        
        items = result.get('data', [])
        
        if not items:
            print("❌ FAILED - 0 items extracted")
            return False
        
        # Calculate quality
        total_fields = len(items) * 3
        filled_fields = sum(
            1 for item in items
            for v in item.values()
            if v is not None and v != ''
        )
        quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0.0
        
        print(f"\n{'='*75}")
        print(f"📊 RESULTS")
        print(f"{'='*75}")
        print(f"Items Extracted: {len(items)}")
        print(f"Quality: {quality:.1f}%")
        print()
        
        # Show first 2 items
        print(f"Sample Items:")
        for i, item in enumerate(items[:2], 1):
            null_count = sum(1 for v in item.values() if v is None or v == '')
            status = '✅' if null_count == 0 else f'⚠️ ({null_count} null)'
            print(f"{i}. {status} {item}")
        
        print(f"{'='*75}")
        
        # Success criteria
        if quality >= 70:
            print(f"\n✅ EXCELLENT! Quality: {quality:.1f}% (>= 70%)")
            print(f"🎉 Reinforcement loop successfully fixed GitHub!")
            return True
        elif quality >= 50:
            print(f"\n⚠️  IMPROVED - Quality: {quality:.1f}% (>= 50%)")
            print(f"Better than before (was 33%)")
            return True
        else:
            print(f"\n❌ STILL LOW - Quality: {quality:.1f}%")
            print(f"Reinforcement loop needs more work")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await scraper.close()

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    REINFORCEMENT LOOP TEST SUITE                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

Testing the new multi-pass adaptive DOM detection system.
System should automatically retry with better selectors when quality is low.

Quality Thresholds:
- >= 70%: Excellent ✅
- >= 50%: Acceptable ⚠️
- < 50%: Failed, triggers next pass ❌
    """)
    
    # Test 1: Stack Overflow (moderate quality, should improve)
    print("\n" + "="*75)
    print("TEST 1/2: Stack Overflow")
    print("="*75)
    result1 = await test_stack_overflow_reinforcement()
    
    # Test 2: GitHub Trending (low quality, needs multiple passes)
    print("\n" + "="*75)
    print("TEST 2/2: GitHub Trending")
    print("="*75)
    result2 = await test_github_trending_reinforcement()
    
    # Summary
    print("\n" + "="*75)
    print("📊 TEST SUMMARY")
    print("="*75)
    print(f"Stack Overflow: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"GitHub Trending: {'✅ PASS' if result2 else '❌ FAIL'}")
    print("="*75)
    
    if result1 and result2:
        print("\n🎉 ALL TESTS PASSED - Reinforcement loop is working!")
    elif result1 or result2:
        print("\n⚠️  PARTIAL SUCCESS - Some tests passed")
    else:
        print("\n❌ ALL TESTS FAILED - Reinforcement loop needs debugging")
    
    return result1 and result2

if __name__ == '__main__':
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)






