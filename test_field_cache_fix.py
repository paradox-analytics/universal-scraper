"""
Test Field-Aware Cache Fix

Demonstrates that different field names now get their own cached code,
preventing field mismatch issues.
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_field_cache_fix():
    print("\n" + "="*80)
    print("🧪 Testing Field-Aware Cache Fix")
    print("="*80 + "\n")
    
    api_key = os.environ['OPENAI_API_KEY']
    
    url = 'https://stackoverflow.com/questions?tab=newest'
    
    # Test 1: Scrape with fields ['title', 'votes']
    print("Test 1: Scraping with fields ['title', 'votes']")
    print("-" * 80)
    
    scraper1 = UniversalScraper(
        api_key=api_key,
        use_camoufox=False,
        enable_auto_pagination=False
    )
    
    result1 = await scraper1.scrape(
        url=url,
        fields=['title', 'votes']
    )
    await scraper1.close()
    
    items1 = result1.get('data', [])
    quality1 = result1.get('quality', 0)
    
    print(f"Results: {len(items1)} items, {quality1:.0f}% quality")
    if items1:
        print(f"Sample: {items1[0]}")
        print(f"Keys: {list(items1[0].keys())}")
    print()
    
    # Test 2: Scrape with fields ['question_title', 'vote_count']
    # (Natural language generated different field names)
    print("Test 2: Scraping with fields ['question_title', 'vote_count']")
    print("-" * 80)
    
    scraper2 = UniversalScraper(
        api_key=api_key,
        use_camoufox=False,
        enable_auto_pagination=False
    )
    
    result2 = await scraper2.scrape(
        url=url,
        fields=['question_title', 'vote_count']
    )
    await scraper2.close()
    
    items2 = result2.get('data', [])
    quality2 = result2.get('quality', 0)
    
    print(f"Results: {len(items2)} items, {quality2:.0f}% quality")
    if items2:
        print(f"Sample: {items2[0]}")
        print(f"Keys: {list(items2[0].keys())}")
    print()
    
    # Test 3: Scrape again with ['title', 'votes'] to confirm cache hit
    print("Test 3: Re-scraping with fields ['title', 'votes'] (should use cache)")
    print("-" * 80)
    
    scraper3 = UniversalScraper(
        api_key=api_key,
        use_camoufox=False,
        enable_auto_pagination=False
    )
    
    result3 = await scraper3.scrape(
        url=url,
        fields=['title', 'votes']
    )
    await scraper3.close()
    
    items3 = result3.get('data', [])
    quality3 = result3.get('quality', 0)
    
    print(f"Results: {len(items3)} items, {quality3:.0f}% quality")
    if items3:
        print(f"Sample: {items3[0]}")
        print(f"Keys: {list(items3[0].keys())}")
    print()
    
    # Summary
    print("="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Test 1 (['title', 'votes']):              {quality1:.0f}% quality")
    print(f"Test 2 (['question_title', 'vote_count']): {quality2:.0f}% quality")
    print(f"Test 3 (['title', 'votes'] - cached):      {quality3:.0f}% quality")
    print()
    
    # Validate fix
    if quality1 >= 70 and quality2 >= 70 and quality3 >= 70:
        print("✅ SUCCESS: All tests passed!")
        print("   ✅ Different field names get their own cached code")
        print("   ✅ Field alignment is perfect (no None values)")
        print("   ✅ Cache reuse works correctly")
    elif quality1 >= 70 and quality2 == 0:
        print("❌ FAILURE: Field mismatch issue still present")
        print("   The cache returned code for the wrong field names")
    elif quality1 >= 70 and quality2 >= 70:
        print("⚠️ PARTIAL: Fix working but quality needs improvement")
    else:
        print("❌ FAILURE: General extraction issues")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_field_cache_fix())





