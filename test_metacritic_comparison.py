"""
Test Metacritic extraction with both our scraper and ScrapeGraphAI
"""
import asyncio
import os
import json
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner

# ScrapeGraphAI test
try:
    from scrapegraphai.graphs import SmartScraperGraph
    SCRAPEGRAPHAI_AVAILABLE = True
except ImportError:
    SCRAPEGRAPHAI_AVAILABLE = False
    print("⚠️  ScrapeGraphAI not available")


async def test_our_scraper():
    """Test with our DirectLLM scraper"""
    print("\n" + "="*80)
    print("🔍 TESTING OUR SCRAPER (DirectLLM + Langchain)")
    print("="*80)
    
    url = "https://www.metacritic.com/browse/game/all/all/current-year/"
    fields = ["name", "description", "score"]
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    # Fetch HTML
    print(f"\n📥 Fetching: {url}")
    fetcher = HybridFetcher(
        headless=True,
        use_camoufox=False,  # Use Playwright for speed
        enable_cache=False
    )
    
    result = await fetcher.fetch(
        url,
        wait_for_selector=".c-finderProductCard",  # Wait for game cards
        scroll_to_bottom=True  # Ensure all content loaded
    )
    
    if not result or 'html' not in result:
        print("❌ Failed to fetch HTML")
        return None
    
    html = result['html']
    print(f"   ✓ Fetched {len(html):,} bytes via {result.get('fetch_method', 'unknown')}")
    
    # Clean HTML
    print("\n🧹 Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"   ✓ Reduced by {clean_result['reduction_percent']:.1f}%")
    print(f"   ✓ Cleaned: {len(cleaned_html):,} bytes")
    
    # Extract with DirectLLM
    print(f"\n🤖 Extracting with DirectLLM...")
    print(f"   Fields: {fields}")
    print(f"   Quality mode: balanced")
    
    extractor = DirectLLMExtractor(
        api_key=api_key,
        model_name="gpt-4o-mini",
        quality_mode="balanced",
        use_html2text=True  # Using Langchain transformer
    )
    
    items = await extractor.extract(
        cleaned_html,
        fields=fields,
        context="Extract video game listings with name, description, and score/rating"
    )
    
    print(f"\n📊 RESULTS:")
    print(f"   Items extracted: {len(items)}")
    
    if items:
        # Calculate completeness
        total_fields = len(items) * len(fields)
        filled_fields = sum(
            1 for item in items 
            for field in fields 
            if item.get(field) and str(item.get(field)).strip()
        )
        completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        print(f"   Completeness: {completeness:.1f}%")
        
        # Show first 3 items
        print(f"\n📝 First 3 items:")
        for i, item in enumerate(items[:3], 1):
            print(f"\n   {i}. {item.get('name', 'N/A')}")
            print(f"      Score: {item.get('score', 'N/A')}")
            desc = item.get('description', 'N/A')
            if desc and len(str(desc)) > 60:
                desc = str(desc)[:60] + "..."
            print(f"      Description: {desc}")
    else:
        print("   ⚠️  No items extracted!")
    
    return items


def test_scrapegraphai():
    """Test with ScrapeGraphAI"""
    if not SCRAPEGRAPHAI_AVAILABLE:
        return None
    
    print("\n" + "="*80)
    print("🔍 TESTING SCRAPEGRAPHAI")
    print("="*80)
    
    url = "https://www.metacritic.com/browse/game/all/all/current-year/"
    
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
    
    # Create the SmartScraperGraph instance
    smart_scraper_graph = SmartScraperGraph(
        prompt="Extract all video games with their name, description, and score/rating",
        source=url,
        config=graph_config
    )
    
    print(f"\n📥 Running ScrapeGraphAI on: {url}")
    result = smart_scraper_graph.run()
    
    print(f"\n📊 RESULTS:")
    
    if isinstance(result, dict):
        # Try to find the items in various possible keys
        items = None
        for key in ['games', 'items', 'results', 'data']:
            if key in result and isinstance(result[key], list):
                items = result[key]
                break
        
        if not items and isinstance(result, dict):
            # Maybe the result itself is structured data
            print(f"   Result keys: {list(result.keys())}")
            if any(isinstance(v, list) for v in result.values()):
                items = [v for v in result.values() if isinstance(v, list)][0]
            else:
                items = [result]  # Single item
    elif isinstance(result, list):
        items = result
    else:
        items = []
    
    print(f"   Items extracted: {len(items) if items else 0}")
    
    if items:
        print(f"\n📝 First 3 items:")
        for i, item in enumerate(items[:3], 1):
            print(f"\n   {i}. {item}")
    else:
        print("   ⚠️  No items extracted!")
        print(f"   Raw result: {result}")
    
    return items


async def main():
    """Run comparison test"""
    print("\n" + "🎮"*40)
    print("METACRITIC COMPARISON TEST")
    print("🎮"*40)
    
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
    
    # Save results
    results = {
        "url": "https://www.metacritic.com/browse/game/all/all/current-year/",
        "our_scraper": {
            "count": len(our_items) if our_items else 0,
            "items": our_items[:5] if our_items else []  # First 5
        },
        "scrapegraphai": {
            "count": len(scrapegraphai_items) if scrapegraphai_items else 0,
            "items": scrapegraphai_items[:5] if scrapegraphai_items else []  # First 5
        }
    }
    
    with open("metacritic_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: metacritic_comparison_results.json")


if __name__ == "__main__":
    asyncio.run(main())



