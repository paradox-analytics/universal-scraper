"""
Simple single-page Camoufox test - NO pagination
"""

import asyncio
import os
from universal_scraper import UniversalScraper


async def test_reddit():
    """Test Reddit with Camoufox - single page only"""
    print("\n" + "="*80)
    print("🦊 Testing REDDIT with Camoufox (single page)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        fetch_mode="browser",
        enable_llm_pagination=False,  # Disable pagination
        extraction_context="Extract Reddit posts with title, author, upvotes, comments count",
        use_camoufox=True,
        browser_timeout=60000
    )
    
    # EXPLICITLY disable pagination detectors
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
        scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
    
    print("⚠️  Pagination DISABLED")
    
    try:
        result = await scraper.scrape(
            "https://www.reddit.com/r/webscraping/",
            fields=[],
            wait_for_selector="shreddit-post"
        )
        
        items = result.get('data', [])
        print(f"\n✅ SUCCESS!")
        print(f"   Items: {len(items)}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(items) > 0:
            print(f"\n📋 Sample (first 3):")
            for i, item in enumerate(items[:3], 1):
                title = item.get('title', 'N/A')
                author = item.get('author', 'N/A')
                print(f"   {i}. {title[:60] if title else 'N/A'}...")
                print(f"      by {author}")
        
        return len(items)
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        scraper.close()


async def test_ebay():
    """Test eBay with Camoufox - single page only"""
    print("\n" + "="*80)
    print("🦊 Testing EBAY with Camoufox (single page)")
    print("="*80)
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        fetch_mode="browser",
        enable_llm_pagination=False,  # Disable pagination
        extraction_context="Extract eBay product listings with title, price, shipping",
        use_camoufox=True,
        browser_timeout=60000
    )
    
    # EXPLICITLY disable pagination detectors
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
        scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
    
    print("⚠️  Pagination DISABLED")
    
    try:
        result = await scraper.scrape(
            "https://www.ebay.com/sch/i.html?_nkw=laptop",
            fields=[]
        )
        
        items = result.get('data', [])
        print(f"\n✅ SUCCESS!")
        print(f"   Items: {len(items)}")
        print(f"   Source: {result.get('source', 'unknown')}")
        
        if len(items) > 0:
            print(f"\n📋 Sample (first 3):")
            for i, item in enumerate(items[:3], 1):
                title = item.get('title', 'N/A')
                price = item.get('price', 'N/A')
                print(f"   {i}. {title[:60] if title else 'N/A'}...")
                print(f"      ${price}")
        
        return len(items)
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        scraper.close()


async def main():
    """Run simple single-page tests"""
    print("\n" + "="*80)
    print("🦊 CAMOUFOX SIMPLE SINGLE-PAGE TESTS")
    print("="*80)
    print("\nTesting Camoufox on single pages (pagination disabled):\n")
    
    # Test Reddit
    reddit_count = await test_reddit()
    
    # Wait between tests
    print("\n⏳ Waiting 5 seconds...\n")
    await asyncio.sleep(5)
    
    # Test eBay
    ebay_count = await test_ebay()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    reddit_status = "✅" if reddit_count > 10 else "❌"
    ebay_status = "✅" if ebay_count > 10 else "❌"
    
    print(f"\n🦊 Camoufox Results (single page only):")
    print(f"   Reddit: {reddit_status} {reddit_count} items")
    print(f"   eBay: {ebay_status} {ebay_count} items")
    
    if reddit_count > 10 and ebay_count > 10:
        print("\n🎉 SUCCESS! Camoufox is working great!")
    elif reddit_count > 10 or ebay_count > 10:
        print("\n⚠️ PARTIAL SUCCESS")
    else:
        print("\n❌ TESTS FAILED")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())







