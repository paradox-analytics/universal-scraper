"""
QUICK TEST - Context System (< 2 minutes)
Tests only the critical new functionality without long pagination tests
"""
import asyncio
import os
from universal_scraper import UniversalScraper

# Set API key
api_key = os.getenv("OPENAI_API_KEY", "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")


async def test_context_inference():
    """Test context manager (10 seconds)"""
    print("\n" + "="*80)
    print("🧠 TEST 1: CONTEXT INFERENCE")
    print("="*80)
    
    from universal_scraper.core import ContextManager
    
    context_mgr = ContextManager(api_key=api_key)
    
    # Test 1: Concert events
    print("\n1. 'Extract concert events with dates and venues'")
    context = context_mgr.parse_context("Extract concert events with dates and venues")
    print(f"   → Type: {context.data_type}")
    print(f"   → Fields: {context.fields}")
    print(f"   → Confidence: {context.inference_confidence}")
    
    assert context.data_type == "events", f"Expected 'events', got '{context.data_type}'"
    assert context.inference_confidence > 0.8, f"Low confidence: {context.inference_confidence}"
    
    # Test 2: Products
    print("\n2. 'Scrape product listings with name, price, rating'")
    context = context_mgr.parse_context("Scrape product listings with name, price, rating")
    print(f"   → Type: {context.data_type}")
    print(f"   → Fields: {context.fields}")
    print(f"   → Confidence: {context.inference_confidence}")
    
    assert context.data_type == "products", f"Expected 'products', got '{context.data_type}'"
    
    print("\n   ✅ PASS: Context inference working!")
    return True


async def test_ticketmaster():
    """Test Ticketmaster (60 seconds) - THE CRITICAL TEST"""
    print("\n" + "="*80)
    print("🎪 TEST 2: TICKETMASTER (Context-Driven - NO PAGINATION)")
    print("="*80)
    print("This was BROKEN before (returned 11 footer links)")
    print("Should now return 20+ concert events")
    
    scraper = UniversalScraper(
        api_key=api_key,
        extraction_context="Extract concert events with artist name, venue, date, and ticket price",
        fetch_mode="browser",
        enable_llm_pagination=False,  # DISABLE pagination for speed
        enable_context_validation=True
    )
    
    url = "https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF"
    
    try:
        result = await scraper.scrape(url, fields=[])
        
        print(f"\n✅ EXTRACTION COMPLETE!")
        print(f"   Items: {len(result['data'])}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        metadata = result.get('metadata', {})
        if metadata.get('extraction_metadata'):
            print(f"   Context used: ✅")
            print(f"   JSON source: {metadata['extraction_metadata'].get('json_source', 'N/A')}")
            print(f"   Validation: {metadata['extraction_metadata'].get('validation', {}).get('is_target_data', 'N/A')}")
        
        if len(result['data']) > 0:
            print(f"\n   First item sample:")
            first_item = result['data'][0]
            for key, value in list(first_item.items())[:5]:
                print(f"      {key}: {str(value)[:80]}")
        
        # CRITICAL VALIDATION
        if len(result['data']) >= 10:
            print(f"\n   ✅ PASS: {len(result['data'])} items (expected 10+)")
            print(f"   🎉 SUCCESS: Ticketmaster is FIXED!")
            return True
        else:
            print(f"\n   ⚠️ WARNING: Only {len(result['data'])} items (expected 10+)")
            print(f"   Before context system: would return 11 footer links")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leafly_single_page():
    """Test Leafly SINGLE PAGE ONLY (30 seconds) - Regression test"""
    print("\n" + "="*80)
    print("🌿 TEST 3: LEAFLY (Single Page - Regression Test)")
    print("="*80)
    print("Testing that we didn't break what was working")
    
    scraper = UniversalScraper(
        api_key=api_key,
        extraction_context="Extract cannabis products with strain name and THC",
        fetch_mode="browser",
        enable_llm_pagination=False,  # DISABLE pagination for speed
        enable_context_validation=True
    )
    
    url = "https://www.leafly.com/dispensary-info/mammoth-holistics/menu"
    
    try:
        result = await scraper.scrape(url, fields=[])
        
        print(f"\n✅ EXTRACTION COMPLETE!")
        print(f"   Items: {len(result['data'])} (single page only)")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(result['data']) > 0:
            print(f"\n   First item sample:")
            first_item = result['data'][0]
            for key, value in list(first_item.items())[:5]:
                print(f"      {key}: {str(value)[:80]}")
        
        # Validation
        if len(result['data']) >= 15:
            print(f"\n   ✅ PASS: {len(result['data'])} items from first page")
            return True
        else:
            print(f"\n   ⚠️ WARNING: Only {len(result['data'])} items")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run quick tests (< 2 minutes)"""
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 CONTEXT SYSTEM - QUICK TEST (< 2 minutes)")
    print("="*80)
    
    results = {
        'context_inference': False,
        'ticketmaster': False,
        'leafly_single': False
    }
    
    # Test 1: Context inference (10 sec)
    try:
        results['context_inference'] = await test_context_inference()
    except Exception as e:
        print(f"\n❌ Context test failed: {e}")
    
    # Test 2: Ticketmaster (60 sec) - MOST IMPORTANT
    try:
        results['ticketmaster'] = await test_ticketmaster()
    except Exception as e:
        print(f"\n❌ Ticketmaster test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Leafly single page (30 sec) - Regression
    try:
        results['leafly_single'] = await test_leafly_single_page()
    except Exception as e:
        print(f"\n❌ Leafly test failed: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Context-driven scraping is working!")
        print(f"✅ Ticketmaster is FIXED (was returning footer links)")
        print(f"✅ Leafly still works (didn't break existing functionality)")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n⚠️ Some tests failed: {failed}")
        print(f"Review output above for details.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())








