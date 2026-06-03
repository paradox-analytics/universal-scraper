"""
Test Camoufox integration with the universal scraper
Tests on Reddit and eBay (previously failing sites)
"""

import asyncio
import os
from universal_scraper import UniversalScraper


async def test_reddit_camoufox():
    """Test Reddit with Camoufox (was working without proxy)"""
    print("\n" + "="*80)
    print("🦊 Testing REDDIT with Camoufox")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context="Extract Reddit posts with title, author, upvotes, comments count",
        use_camoufox=True,  # Use Camoufox!
        browser_timeout=60000
    )
    
    try:
        result = await scraper.scrape(
            "https://www.reddit.com/r/webscraping/",
            fields=[],
            wait_for_selector="shreddit-post"
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"   Items extracted: {len(result.get('data', []))}")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Time: {result.get('elapsed_time', 0):.1f}s")
        
        if len(result.get('data', [])) > 0:
            print(f"\n📋 Sample (first 2 items):")
            for i, item in enumerate(result['data'][:2], 1):
                title = item.get('title', 'N/A')
                author = item.get('author', 'N/A')
                print(f"   {i}. {title[:60]}...")
                print(f"      Author: {author}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return None
    finally:
        scraper.close()


async def test_ebay_camoufox():
    """Test eBay with Camoufox (was completely failing)"""
    print("\n" + "="*80)
    print("🦊 Testing EBAY with Camoufox")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context="Extract eBay product listings with title, price, shipping, condition",
        use_camoufox=True,  # Use Camoufox!
        browser_timeout=60000
    )
    
    try:
        result = await scraper.scrape(
            "https://www.ebay.com/sch/i.html?_nkw=laptop",
            fields=[]
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"   Items extracted: {len(result.get('data', []))}")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Time: {result.get('elapsed_time', 0):.1f}s")
        
        if len(result.get('data', [])) > 0:
            print(f"\n📋 Sample (first 2 items):")
            for i, item in enumerate(result['data'][:2], 1):
                title = item.get('title', 'N/A')
                price = item.get('price', 'N/A')
                print(f"   {i}. {title[:60] if title else 'N/A'}...")
                print(f"      Price: {price}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        scraper.close()


async def main():
    """Run all Camoufox tests"""
    print("\n" + "="*80)
    print("🦊 CAMOUFOX INTEGRATION TESTS")
    print("="*80)
    print("\nTesting Camoufox integration on previously problematic sites:")
    print("  • Reddit (worked without proxy, failed with proxy)")
    print("  • eBay (completely failed)")
    print("\n")
    
    # Test Reddit
    reddit_result = await test_reddit_camoufox()
    
    # Wait a bit between tests
    print("\n⏳ Waiting 5 seconds before next test...")
    await asyncio.sleep(5)
    
    # Test eBay
    ebay_result = await test_ebay_camoufox()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    reddit_status = "✅ WORKING" if reddit_result and len(reddit_result.get('data', [])) > 0 else "❌ FAILED"
    ebay_status = "✅ WORKING" if ebay_result and len(ebay_result.get('data', [])) > 0 else "❌ FAILED"
    
    print(f"\n🦊 Camoufox Results:")
    print(f"   Reddit: {reddit_status} ({len(reddit_result.get('data', [])) if reddit_result else 0} items)")
    print(f"   eBay: {ebay_status} ({len(ebay_result.get('data', [])) if ebay_result else 0} items)")
    
    if reddit_status == "✅ WORKING" and ebay_status == "✅ WORKING":
        print("\n🎉 ALL TESTS PASSED! Camoufox is working great!")
    elif reddit_status == "✅ WORKING" or ebay_status == "✅ WORKING":
        print("\n⚠️ PARTIAL SUCCESS - Some sites working with Camoufox")
    else:
        print("\n❌ ALL TESTS FAILED - Check errors above")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())







