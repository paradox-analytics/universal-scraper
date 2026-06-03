"""
Test Camoufox + Apify Proxies - The ultimate combination
"""

import asyncio
import os
from universal_scraper import UniversalScraper


async def test_ebay_with_proxy():
    """Test eBay with Camoufox + Apify proxy"""
    print("\n" + "="*80)
    print("🦊 Testing EBAY with Camoufox + Apify Proxy (single page)")
    print("="*80)
    
    # Apify proxy configuration
    proxy_config = {
        'server': 'http://proxy.apify.com:8000',
        'username': 'auto',
        'password': os.getenv('APIFY_PROXY_PASSWORD', 'your_password_here')
    }
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        proxy_config=proxy_config,  # Using Apify proxy
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context="Extract eBay product listings with title, price, shipping",
        use_camoufox=True,  # Using Camoufox
        browser_timeout=120000  # 2 minutes for proxy warmup
    )
    
    # Disable pagination
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
        scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
    
    print("⚠️  Pagination DISABLED")
    print("🔐 Using Apify residential proxy")
    print("🦊 Using Camoufox browser")
    
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
        else:
            print(f"\n⚠️  No items extracted - eBay might be heavily blocking")
        
        return len(items)
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        scraper.close()


async def test_reddit_with_proxy():
    """Test Reddit with Camoufox + Apify proxy"""
    print("\n" + "="*80)
    print("🦊 Testing REDDIT with Camoufox + Apify Proxy (single page)")
    print("="*80)
    
    # Apify proxy configuration
    proxy_config = {
        'server': 'http://proxy.apify.com:8000',
        'username': 'auto',
        'password': os.getenv('APIFY_PROXY_PASSWORD', 'your_password_here')
    }
    
    scraper = UniversalScraper(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        proxy_config=proxy_config,  # Using Apify proxy
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context="Extract Reddit posts with title, author, upvotes, comments count",
        use_camoufox=True,  # Using Camoufox
        browser_timeout=120000  # 2 minutes for proxy warmup
    )
    
    # Disable pagination
    if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
        scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
    if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
        scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
    
    print("⚠️  Pagination DISABLED")
    print("🔐 Using Apify residential proxy")
    print("🦊 Using Camoufox browser")
    
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


async def main():
    """Test Camoufox + Proxies combo"""
    print("\n" + "="*80)
    print("🦊 CAMOUFOX + APIFY PROXY TESTS")
    print("="*80)
    print("\nTesting the ultimate combination:")
    print("  • Camoufox (advanced anti-detection)")
    print("  • Apify residential proxies")
    print("  • Single page extraction\n")
    
    # Check for Apify password
    if not os.getenv('APIFY_PROXY_PASSWORD'):
        print("⚠️  WARNING: APIFY_PROXY_PASSWORD not set!")
        print("   Set it with: export APIFY_PROXY_PASSWORD='your_password'\n")
    
    # Test Reddit first (should work)
    reddit_count = await test_reddit_with_proxy()
    
    print("\n⏳ Waiting 10 seconds before next test...\n")
    await asyncio.sleep(10)
    
    # Test eBay (challenging)
    ebay_count = await test_ebay_with_proxy()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY: Camoufox + Proxy")
    print("="*80)
    
    reddit_status = "✅" if reddit_count > 10 else "❌"
    ebay_status = "✅" if ebay_count > 10 else "❌"
    
    print(f"\n🦊 Camoufox + Apify Proxy Results:")
    print(f"   Reddit: {reddit_status} {reddit_count} items")
    print(f"   eBay: {ebay_status} {ebay_count} items")
    
    print("\n📈 Comparison vs No Proxy:")
    print(f"   Reddit (no proxy): 62 items")
    print(f"   Reddit (with proxy): {reddit_count} items")
    print(f"   eBay (no proxy): 0 items")
    print(f"   eBay (with proxy): {ebay_count} items")
    
    if reddit_count > 10 and ebay_count > 10:
        print("\n🎉 BOTH WORKING! Camoufox + Proxy is the winning combo!")
    elif reddit_count > 10:
        print("\n⚠️ Reddit works, but eBay still challenging")
        print("   → eBay may need longer wait times or different approach")
    else:
        print("\n❌ Proxy may be causing issues - check credentials")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())







