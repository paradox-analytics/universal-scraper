#!/usr/bin/env python3
"""
Quick local test for Chewy.com - Single page only
Tests field extraction fixes (title, rating, review count, price)
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.scraper import UniversalScraper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Quick test - single page only"""
    
    # Web Unblocker credentials
    web_unblocker_api_key = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"
    
    # Get OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Test URL and fields
    url = "https://www.chewy.com/b/wet-food-389"
    fields = ["title", "rating", "review count", "product url"]
    
    logger.info("="*80)
    logger.info("🧪 QUICK TEST: Chewy.com - Single Page Only")
    logger.info("="*80)
    logger.info(f"   URL: {url}")
    logger.info(f"   Fields: {', '.join(fields)}")
    logger.info("="*80)
    logger.info("")
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=openai_api_key,
        proxy_config=None,
        headless=True,
        use_camoufox=True,
        fetch_mode='browser',
        browser_timeout=60000,  # 60s timeout
        use_direct_llm=True,
        enable_cache=False,
        web_unblocker_api_key=web_unblocker_api_key,
        web_unblocker_zone="web_unlocker1",
        log_level=logging.INFO
    )
    
    # CRITICAL: Limit to 1 page only
    scraper._max_pages_limit = 1
    logger.info(f"📄 Pagination limit: {scraper._max_pages_limit} page (single page only)")
    
    # Reduce Web Unblocker timeout
    if hasattr(scraper, 'html_fetcher') and scraper.html_fetcher and hasattr(scraper.html_fetcher, 'web_unblocker_fetcher') and scraper.html_fetcher.web_unblocker_fetcher:
        scraper.html_fetcher.web_unblocker_fetcher.timeout = 60
    
    try:
        logger.info(f"🚀 Starting scrape (single page only)...")
        result = await scraper.scrape(url, fields)
        
        # Display results
        logger.info("")
        logger.info("="*80)
        logger.info("📊 RESULTS")
        logger.info("="*80)
        
        items = result.get('data', [])
        logger.info(f"   Total items extracted: {len(items)}")
        
        if items:
            # Field coverage analysis
            logger.info("")
            logger.info("   Field Coverage:")
            all_fields_found = set()
            for item in items:
                all_fields_found.update(item.keys())
            
            requested_fields = set(fields)
            missing_fields = requested_fields - all_fields_found
            found_fields = requested_fields & all_fields_found
            
            logger.info(f"   ✅ Found: {', '.join(sorted(found_fields))}")
            if missing_fields:
                logger.warning(f"   ❌ Missing: {', '.join(sorted(missing_fields))}")
            else:
                logger.info(f"   ✅ All fields present!")
            
            # Show first 3 items
            logger.info("")
            logger.info("   First 3 items:")
            for i, item in enumerate(items[:3], 1):
                logger.info(f"   {i}. {json.dumps(item, indent=4)}")
            
            # URL analysis
            logger.info("")
            logger.info("   URL Analysis:")
            unique_urls = set()
            product_urls = []
            for item in items:
                url_val = item.get('product url') or item.get('productUrl') or item.get('url') or item.get('href') or item.get('_url')
                if url_val:
                    # Handle case where URL might be a dict (extract actual URL)
                    if isinstance(url_val, dict):
                        # Try to find URL in dict values or data attributes
                        url_str = None
                        for key, value in url_val.items():
                            if isinstance(value, str) and ('http' in value or 'chewy.com' in value):
                                url_str = value
                                break
                            elif key in ['href', 'url', 'link', 'productUrl']:
                                url_str = value
                                break
                        if url_str:
                            url_val = url_str
                        else:
                            # If no URL found, skip this item
                            continue
                    
                    if isinstance(url_val, str):
                        unique_urls.add(url_val)
                        product_urls.append(url_val)
            logger.info(f"   Unique product URLs: {len(unique_urls)}")
            logger.info(f"   Items with URLs: {len(product_urls)}/{len(items)}")
            if len(unique_urls) == 1:
                logger.warning(f"   ⚠️ All items share same URL: {list(unique_urls)[0]}")
            else:
                logger.info(f"   ✅ Multiple unique URLs found")
                logger.info(f"   Sample URLs:")
                for i, url_val in enumerate(list(unique_urls)[:3], 1):
                    logger.info(f"      {i}. {url_val}")
            
            # Quality evaluation against expected data
            logger.info("")
            logger.info("   Quality Evaluation:")
            logger.info("   Expected from screenshot:")
            logger.info("      - Product 1: 'Fancy Feast Gems Mousse...' - Rating: 4.4, Reviews: 2,391")
            logger.info("      - Product 2: 'Sheba Perfect Portions Grain-Free...' - Rating: 4.6, Reviews: 1,655")
            logger.info("      - Product 3: 'Sheba Perfect Portions Chicken...' - Rating: 4.6, Reviews: 721")
            logger.info("      - Product 4: 'Iams Perfect Portions Healthy Adult...' - Rating: 4.3, Reviews: 1,557")
            logger.info("")
            
            # Check if we have matching titles
            titles_found = []
            for item in items[:10]:  # Check first 10
                title = item.get('title', '')
                if title:
                    titles_found.append(title)
                    # Check for expected keywords
                    if 'Fancy Feast' in title or 'Sheba' in title or 'Iams' in title:
                        logger.info(f"   ✅ Found expected brand: {title[:60]}...")
            
            # Check rating ranges
            ratings = [item.get('rating') for item in items if item.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                logger.info(f"   Average rating: {avg_rating:.2f} (expected ~4.4-4.6)")
            
            # Check review counts
            review_counts = [item.get('review count') for item in items if item.get('review count')]
            if review_counts:
                avg_reviews = sum(review_counts) / len(review_counts)
                logger.info(f"   Average review count: {avg_reviews:.0f} (expected hundreds to thousands)")
            
            # Check product URLs
            if product_urls:
                chewy_urls = [url for url in product_urls if 'chewy.com' in str(url)]
                logger.info(f"   Product URLs with chewy.com domain: {len(chewy_urls)}/{len(product_urls)}")
                if len(chewy_urls) > 0:
                    logger.info(f"   ✅ Product URLs look valid")
                else:
                    logger.warning(f"   ⚠️ No chewy.com URLs found in product URLs")
        else:
            logger.error("   ❌ No items extracted!")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Scrape failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())

