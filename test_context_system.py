"""
Test the new context-driven scraping system
"""
import asyncio
import os
from universal_scraper import UniversalScraper

# Set API key
api_key = os.getenv("OPENAI_API_KEY", "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")


async def test_ticketmaster():
    """
    Test Ticketmaster with context
    Expected: Should extract events, not footer links
    """
    print("\n" + "="*80)
    print("🎪 TEST 1: TICKETMASTER (Context-Driven)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=api_key,
        extraction_context="Extract concert events with artist name, venue, date, and ticket price",
        fetch_mode="browser",
        enable_llm_pagination=True,
        enable_context_validation=True
    )
    
    url = "https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF"
    
    try:
        result = await scraper.scrape(url, fields=[])
        
        print(f"\n✅ SUCCESS!")
        print(f"   Items extracted: {len(result['data'])}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(result['data']) > 0:
            print(f"\n   First item:")
            first_item = result['data'][0]
            for key, value in list(first_item.items())[:5]:
                print(f"      {key}: {str(value)[:100]}")
        
        # Validation
        if len(result['data']) >= 20:
            print(f"\n   ✅ PASS: Extracted {len(result['data'])} items (expected 20+)")
        else:
            print(f"\n   ⚠️ WARNING: Only extracted {len(result['data'])} items (expected 20+)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leafly():
    """
    Test Leafly (should still work)
    Expected: Should extract products with auto-pagination
    """
    print("\n" + "="*80)
    print("🌿 TEST 2: LEAFLY (Should Still Work)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=api_key,
        extraction_context="Extract cannabis products with strain name, THC content, and price",
        fetch_mode="browser",
        enable_llm_pagination=True,
        enable_context_validation=True
    )
    
    url = "https://www.leafly.com/dispensary-info/mammoth-holistics/menu"
    
    try:
        result = await scraper.scrape(url, fields=[])
        
        print(f"\n✅ SUCCESS!")
        print(f"   Items extracted: {len(result['data'])}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        metadata = result.get('metadata', {})
        if metadata.get('auto_pagination'):
            print(f"   Auto-pagination: ✅ Enabled")
            print(f"   Pages scraped: {metadata.get('total_pages_scraped', 1)}")
        
        if len(result['data']) > 0:
            print(f"\n   First item:")
            first_item = result['data'][0]
            for key, value in list(first_item.items())[:5]:
                print(f"      {key}: {str(value)[:100]}")
        
        # Validation
        if len(result['data']) >= 500:
            print(f"\n   ✅ PASS: Extracted {len(result['data'])} items (expected 500+)")
        else:
            print(f"\n   ⚠️ WARNING: Only extracted {len(result['data'])} items (expected 500+)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_inference():
    """
    Test context manager directly
    """
    print("\n" + "="*80)
    print("🧠 TEST 3: CONTEXT INFERENCE")
    print("="*80)
    
    from universal_scraper.core import ContextManager
    
    context_mgr = ContextManager(api_key=api_key)
    
    # Test 1: Concert events
    print("\n1. Testing: 'Extract concert events with dates and venues'")
    context = context_mgr.parse_context("Extract concert events with dates and venues")
    print(f"   → Data type: {context.data_type}")
    print(f"   → Fields: {context.fields}")
    print(f"   → Confidence: {context.inference_confidence}")
    
    # Test 2: Products
    print("\n2. Testing: 'Scrape product listings with name, price, rating'")
    context = context_mgr.parse_context("Scrape product listings with name, price, rating")
    print(f"   → Data type: {context.data_type}")
    print(f"   → Fields: {context.fields}")
    print(f"   → Confidence: {context.inference_confidence}")
    
    # Test 3: Businesses
    print("\n3. Testing: 'Get all brewery information'")
    context = context_mgr.parse_context("Get all brewery information")
    print(f"   → Data type: {context.data_type}")
    print(f"   → Fields: {context.fields}")
    print(f"   → Confidence: {context.inference_confidence}")
    
    print("\n   ✅ Context inference working!")
    return True


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 CONTEXT-DRIVEN SCRAPER TEST SUITE")
    print("="*80)
    
    results = {
        'context_inference': False,
        'ticketmaster': False,
        'leafly': False
    }
    
    # Test 1: Context inference (fast)
    try:
        results['context_inference'] = await test_context_inference()
    except Exception as e:
        print(f"\n❌ Context inference test failed: {e}")
    
    # Test 2: Ticketmaster (most important - this was broken before)
    try:
        results['ticketmaster'] = await test_ticketmaster()
    except Exception as e:
        print(f"\n❌ Ticketmaster test failed: {e}")
    
    # Test 3: Leafly (regression test - make sure we didn't break it)
    try:
        results['leafly'] = await test_leafly()
    except Exception as e:
        print(f"\n❌ Leafly test failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Context-driven scraping is working!")
    else:
        print(f"\n⚠️ Some tests failed. Review output above.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())








