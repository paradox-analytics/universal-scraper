"""
Test IMDB Top 250 Movies with both our scraper and ScrapeGraphAI
"""
import asyncio
import os
import json
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from scrapegraphai.graphs import SmartScraperGraph


async def test_our_scraper():
    """Test with our DirectLLM scraper"""
    print("\n" + "="*80)
    print("🔍 TESTING OUR SCRAPER ON IMDB TOP 250")
    print("="*80)
    
    url = "https://www.imdb.com/chart/top/"
    fields = ["title", "year", "rating", "director"]
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    # Fetch HTML
    print(f"\n📥 Fetching: {url}")
    fetcher = HybridFetcher(
        headless=True,
        use_camoufox=False,
        enable_cache=False
    )
    
    result = await fetcher.fetch(
        url,
        wait_for_selector=".ipc-metadata-list-summary-item",
        scroll_to_bottom=True
    )
    
    if not result or 'html' not in result:
        print("❌ Failed to fetch HTML")
        return None
    
    html = result['html']
    print(f"   ✓ Fetched {len(html):,} bytes")
    
    # Clean HTML
    print("\n🧹 Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"   ✓ Reduced by {clean_result['reduction_percent']:.1f}%")
    
    # Extract with DirectLLM
    print(f"\n🤖 Extracting with DirectLLM...")
    extractor = DirectLLMExtractor(
        api_key=api_key,
        model_name="gpt-4o-mini",
        quality_mode="balanced",
        use_html2text=True
    )
    
    items = await extractor.extract(
        cleaned_html,
        fields=fields,
        context="Extract top-rated movies with title, year, IMDB rating (0-10), and director"
    )
    
    print(f"\n📊 RESULTS:")
    print(f"   Items extracted: {len(items)}")
    
    if items:
        # Filter items with valid ratings (0-10 for IMDB)
        valid_items = [
            item for item in items
            if item.get('rating') and 
            isinstance(item.get('rating'), (int, float)) and 
            0 <= item.get('rating') <= 10
        ]
        
        total_fields = len(valid_items) * len(fields)
        filled_fields = sum(
            1 for item in valid_items
            for field in fields
            if item.get(field)
        )
        completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        print(f"   Valid items (with rating 0-10): {len(valid_items)}")
        print(f"   Completeness: {completeness:.1f}%")
        
        print(f"\n📝 First 10 items:")
        for i, item in enumerate(valid_items[:10], 1):
            print(f"\n   {i}. {item.get('title', 'N/A')}")
            print(f"      Year: {item.get('year', 'N/A')}")
            print(f"      Rating: {item.get('rating', 'N/A')}")
            print(f"      Director: {item.get('director', 'N/A')}")
        
        return valid_items
    else:
        print("   ⚠️  No items extracted!")
        return []


def test_scrapegraphai():
    """Test with ScrapeGraphAI"""
    print("\n" + "="*80)
    print("🔍 TESTING SCRAPEGRAPHAI ON IMDB TOP 250")
    print("="*80)
    
    url = "https://www.imdb.com/chart/top/"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    graph_config = {
        "llm": {
            "api_key": api_key,
            "model": "openai/gpt-4o-mini"
        },
        "verbose": False,
        "headless": True
    }
    
    smart_scraper_graph = SmartScraperGraph(
        prompt="Extract top 250 movies with their title, year, IMDB rating, and director",
        source=url,
        config=graph_config
    )
    
    print(f"\n📥 Running ScrapeGraphAI on: {url}")
    print(f"   This may take 30-60 seconds...")
    
    result = smart_scraper_graph.run()
    
    print(f"\n📊 RESULTS:")
    
    # Parse result
    items = None
    if isinstance(result, dict):
        for key in ['movies', 'items', 'results', 'data', 'films']:
            if key in result and isinstance(result[key], list):
                items = result[key]
                break
        
        if not items and any(isinstance(v, list) for v in result.values()):
            items = [v for v in result.values() if isinstance(v, list)][0]
        else:
            items = [result] if result else []
    elif isinstance(result, list):
        items = result
    else:
        items = []
    
    print(f"   Items extracted: {len(items) if items else 0}")
    
    if items and len(items) > 0:
        print(f"\n📝 First 5 items:")
        for i, item in enumerate(items[:5], 1):
            if isinstance(item, dict):
                title = item.get('title') or item.get('name') or 'N/A'
                year = item.get('year') or 'N/A'
                rating = item.get('rating') or item.get('score') or 'N/A'
                director = item.get('director') or 'N/A'
                
                print(f"\n   {i}. {title}")
                print(f"      Year: {year}")
                print(f"      Rating: {rating}")
                print(f"      Director: {director}")
    else:
        print("   ⚠️  No items extracted!")
    
    return items


async def main():
    """Run comparison test"""
    print("\n" + "🎬"*40)
    print("IMDB TOP 250 COMPARISON TEST")
    print("🎬"*40)
    
    # Test our scraper
    our_items = await test_our_scraper()
    
    # Test ScrapeGraphAI
    scrapegraphai_items = test_scrapegraphai()
    
    # Compare
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    print(f"\nOur Scraper:      {len(our_items) if our_items else 0} items")
    print(f"ScrapeGraphAI:    {len(scrapegraphai_items) if scrapegraphai_items else 0} items")
    
    if our_items and scrapegraphai_items:
        diff = len(our_items) - len(scrapegraphai_items)
        if diff > 0:
            print(f"\n🟢 We extracted {diff} MORE items (+{diff/len(scrapegraphai_items)*100:.0f}%)")
        elif diff < 0:
            print(f"\n🔴 We extracted {abs(diff)} FEWER items ({diff/len(scrapegraphai_items)*100:.0f}%)")
        else:
            print(f"\n🏆 Same number of items!")


if __name__ == "__main__":
    asyncio.run(main())



