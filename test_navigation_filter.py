"""
Test navigation filtering on Metacritic
"""
import asyncio
import os
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from bs4 import BeautifulSoup


async def test_navigation_filtering():
    """Test with updated HTML cleaner that removes nav elements"""
    print("\n" + "="*80)
    print("🔍 TESTING NAVIGATION FILTERING ON METACRITIC")
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
        use_camoufox=False,
        enable_cache=False
    )
    
    result = await fetcher.fetch(
        url,
        wait_for_selector=".c-finderProductCard",
        scroll_to_bottom=True
    )
    
    if not result or 'html' not in result:
        print("❌ Failed to fetch HTML")
        return None
    
    html = result['html']
    print(f"   ✓ Fetched {len(html):,} bytes")
    
    # Analyze BEFORE cleaning
    print(f"\n📊 BEFORE Cleaning:")
    soup_before = BeautifulSoup(html, 'html.parser')
    nav_count = len(soup_before.find_all('nav'))
    header_count = len(soup_before.find_all('header'))
    footer_count = len(soup_before.find_all('footer'))
    aside_count = len(soup_before.find_all('aside'))
    game_cards = len(soup_before.select('.c-finderProductCard'))
    
    print(f"   <nav> tags: {nav_count}")
    print(f"   <header> tags: {header_count}")
    print(f"   <footer> tags: {footer_count}")
    print(f"   <aside> tags: {aside_count}")
    print(f"   Game cards: {game_cards}")
    
    # Clean HTML with NEW cleaner (removes nav elements)
    print(f"\n🧹 Cleaning HTML with navigation filtering...")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"   ✓ Reduced by {clean_result['reduction_percent']:.1f}%")
    print(f"   ✓ Cleaned: {len(cleaned_html):,} bytes")
    
    # Analyze AFTER cleaning
    print(f"\n📊 AFTER Cleaning:")
    soup_after = BeautifulSoup(cleaned_html, 'html.parser')
    nav_count_after = len(soup_after.find_all('nav'))
    header_count_after = len(soup_after.find_all('header'))
    footer_count_after = len(soup_after.find_all('footer'))
    aside_count_after = len(soup_after.find_all('aside'))
    game_cards_after = len(soup_after.select('.c-finderProductCard'))
    
    print(f"   <nav> tags: {nav_count_after} (removed: {nav_count - nav_count_after})")
    print(f"   <header> tags: {header_count_after} (removed: {header_count - header_count_after})")
    print(f"   <footer> tags: {footer_count_after} (removed: {footer_count - footer_count_after})")
    print(f"   <aside> tags: {aside_count_after} (removed: {aside_count - aside_count_after})")
    print(f"   Game cards: {game_cards_after} (preserved: {game_cards_after}/{game_cards})")
    
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
        context="Extract video game listings with name, description, and Metascore rating"
    )
    
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"   Total items: {len(items)}")
    
    # Separate items with and without scores
    items_with_scores = [
        item for item in items
        if item.get('score') and 
        isinstance(item.get('score'), (int, float))
    ]
    items_without_scores = [
        item for item in items
        if not item.get('score') or
        not isinstance(item.get('score'), (int, float))
    ]
    
    print(f"   Items WITH scores: {len(items_with_scores)}")
    print(f"   Items WITHOUT scores: {len(items_without_scores)}")
    
    if items_without_scores:
        print(f"\n⚠️  Navigation items still present:")
        for item in items_without_scores[:5]:
            print(f"      - {item.get('name', 'N/A')}")
    else:
        print(f"\n✅ No navigation items! All extracted items have scores.")
    
    # Calculate completeness for items with scores
    if items_with_scores:
        total_fields = len(items_with_scores) * len(fields)
        filled_fields = sum(
            1 for item in items_with_scores
            for field in fields
            if item.get(field)
        )
        completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        print(f"   Completeness (with scores): {completeness:.1f}%")
    
    # Show first 5 items
    print(f"\n📝 First 5 items:")
    for i, item in enumerate(items_with_scores[:5], 1):
        name = item.get('name', 'N/A')
        score = item.get('score', 'N/A')
        desc = item.get('description', 'N/A')
        if desc and len(str(desc)) > 60:
            desc = str(desc)[:60] + "..."
        
        print(f"\n   {i}. {name}")
        print(f"      Score: {score}")
        print(f"      Description: {desc}")
    
    # Comparison
    print(f"\n" + "="*80)
    print(f"📊 IMPROVEMENT SUMMARY")
    print(f"="*80)
    print(f"BEFORE (with old cleaner):")
    print(f"  - Total items: 45")
    print(f"  - Valid items: 33")
    print(f"  - Navigation items: 11")
    print(f"\nAFTER (with navigation filtering):")
    print(f"  - Total items: {len(items)}")
    print(f"  - Valid items: {len(items_with_scores)}")
    print(f"  - Navigation items: {len(items_without_scores)}")
    print(f"\nImprovement: {11 - len(items_without_scores)} fewer navigation items!")
    
    return items_with_scores


if __name__ == "__main__":
    asyncio.run(test_navigation_filtering())



