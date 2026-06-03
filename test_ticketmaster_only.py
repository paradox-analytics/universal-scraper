"""
Test ONLY Ticketmaster - identify slowdown
"""
import asyncio
import os
import time
from universal_scraper import UniversalScraper

api_key = os.getenv("OPENAI_API_KEY", "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")


async def test_ticketmaster_with_context():
    """Test Ticketmaster WITH context system"""
    print("\n" + "="*80)
    print("🎪 TEST: TICKETMASTER WITH CONTEXT SYSTEM")
    print("="*80)
    print("URL: https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF")
    
    start = time.time()
    
    print("\n⏱️  Step 1: Initializing scraper...")
    init_start = time.time()
    scraper = UniversalScraper(
        api_key=api_key,
        extraction_context="Extract concert events with artist name, venue, date",
        fetch_mode="browser",
        enable_llm_pagination=False,  # Disabled for speed
        enable_context_validation=True
    )
    print(f"   Done in {time.time() - init_start:.1f}s")
    
    url = "https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF"
    
    print("\n⏱️  Step 2: Starting scrape...")
    scrape_start = time.time()
    
    try:
        result = await scraper.scrape(url, fields=[])
        scrape_time = time.time() - scrape_start
        
        print(f"\n✅ COMPLETE in {scrape_time:.1f}s")
        print(f"   Items extracted: {len(result['data'])}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        metadata = result.get('metadata', {})
        if 'extraction_metadata' in metadata:
            em = metadata['extraction_metadata']
            print(f"   JSON source used: {em.get('json_source', 'N/A')}")
            print(f"   Validation passed: {em.get('validation', {}).get('is_target_data', 'N/A')}")
        
        if len(result['data']) > 0:
            print(f"\n   Sample item:")
            first = result['data'][0]
            for k, v in list(first.items())[:3]:
                print(f"      {k}: {str(v)[:60]}")
        
        total_time = time.time() - start
        print(f"\n⏱️  TOTAL TIME: {total_time:.1f}s")
        print(f"   Init: {time.time() - init_start:.1f}s")
        print(f"   Scrape: {scrape_time:.1f}s")
        
        if total_time > 120:
            print(f"\n⚠️  WARNING: Took over 2 minutes! Something is slow.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED after {time.time() - scrape_start:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ticketmaster_without_context():
    """Test Ticketmaster WITHOUT context system (baseline)"""
    print("\n" + "="*80)
    print("🎪 BASELINE: TICKETMASTER WITHOUT CONTEXT SYSTEM")
    print("="*80)
    
    start = time.time()
    
    print("\n⏱️  Step 1: Initializing scraper...")
    init_start = time.time()
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=None,  # NO CONTEXT
        enable_context_validation=False  # NO VALIDATION
    )
    print(f"   Done in {time.time() - init_start:.1f}s")
    
    url = "https://www.ticketmaster.com/discover/concerts?classificationId=KnvZfZ7vAvF"
    
    print("\n⏱️  Step 2: Starting scrape...")
    scrape_start = time.time()
    
    try:
        result = await scraper.scrape(url, fields=[])
        scrape_time = time.time() - scrape_start
        
        print(f"\n✅ COMPLETE in {scrape_time:.1f}s")
        print(f"   Items extracted: {len(result['data'])}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(result['data']) > 0:
            print(f"\n   Sample item:")
            first = result['data'][0]
            for k, v in list(first.items())[:3]:
                print(f"      {k}: {str(v)[:60]}")
        
        total_time = time.time() - start
        print(f"\n⏱️  TOTAL TIME: {total_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED after {time.time() - scrape_start:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run comparison test"""
    print("\n" + "="*80)
    print("⚡ TICKETMASTER PERFORMANCE TEST")
    print("="*80)
    print("Comparing WITH vs WITHOUT context system")
    
    # Test 1: WITH context
    print("\n\n" + "="*80)
    print("TEST 1/2: WITH CONTEXT SYSTEM")
    print("="*80)
    with_context_success = await test_ticketmaster_with_context()
    
    # Test 2: WITHOUT context (baseline)
    print("\n\n" + "="*80)
    print("TEST 2/2: WITHOUT CONTEXT SYSTEM (BASELINE)")
    print("="*80)
    without_context_success = await test_ticketmaster_without_context()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"   WITH context: {'✅ PASS' if with_context_success else '❌ FAIL'}")
    print(f"   WITHOUT context: {'✅ PASS' if without_context_success else '❌ FAIL'}")
    print("\n   Compare the times above to see if context system adds overhead.")
    
    if not with_context_success:
        print("\n⚠️  Context system may have introduced a bug.")
    
    return with_context_success


if __name__ == "__main__":
    asyncio.run(main())








