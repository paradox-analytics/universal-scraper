"""
Diagnose Metacritic HTML structure (no API key needed)
"""
import asyncio
from bs4 import BeautifulSoup
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def analyze_metacritic():
    """Analyze Metacritic HTML structure"""
    print("\n" + "="*80)
    print("🔍 ANALYZING METACRITIC HTML STRUCTURE")
    print("="*80)
    
    url = "https://www.metacritic.com/browse/game/all/all/current-year/"
    
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
        return
    
    html = result['html']
    print(f"   ✓ Fetched {len(html):,} bytes")
    
    # Save raw HTML for inspection
    with open("metacritic_raw.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✓ Saved to: metacritic_raw.html")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    
    # Analyze structure
    print(f"\n📊 HTML STRUCTURE ANALYSIS:")
    
    # Look for game cards
    game_cards = soup.select(".c-finderProductCard")
    print(f"\n1. Game Cards (.c-finderProductCard):")
    print(f"   Found: {len(game_cards)} cards")
    
    if game_cards:
        print(f"\n   First card structure:")
        first_card = game_cards[0]
        
        # Try to find name
        name_elem = first_card.select_one(".c-finderProductCard_title")
        if name_elem:
            print(f"   ✓ Name element: .c-finderProductCard_title")
            print(f"     Text: {name_elem.get_text(strip=True)[:60]}...")
        else:
            print(f"   ❌ No name element found")
            print(f"     Trying alternatives...")
            name_alt = first_card.select_one("h3") or first_card.select_one(".title")
            if name_alt:
                print(f"     Found in: {name_alt.name}.{name_alt.get('class')}")
                print(f"     Text: {name_alt.get_text(strip=True)[:60]}...")
        
        # Try to find score
        score_elem = first_card.select_one(".c-siteReviewScore")
        if score_elem:
            print(f"   ✓ Score element: .c-siteReviewScore")
            print(f"     Text: {score_elem.get_text(strip=True)}")
        else:
            print(f"   ❌ No score element found")
            print(f"     Trying alternatives...")
            score_alt = first_card.select_one(".score") or first_card.select_one("[class*='score']")
            if score_alt:
                print(f"     Found in: {score_alt.name}.{score_alt.get('class')}")
                print(f"     Text: {score_alt.get_text(strip=True)}")
        
        # Try to find description
        desc_elem = first_card.select_one(".c-finderProductCard_description")
        if desc_elem:
            print(f"   ✓ Description element: .c-finderProductCard_description")
            print(f"     Text: {desc_elem.get_text(strip=True)[:60]}...")
        else:
            print(f"   ❌ No description element found")
            print(f"     Trying alternatives...")
            desc_alt = first_card.select_one(".description") or first_card.select_one("p")
            if desc_alt:
                print(f"     Found in: {desc_alt.name}.{desc_alt.get('class')}")
                print(f"     Text: {desc_alt.get_text(strip=True)[:60]}...")
        
        print(f"\n   Full first card HTML (first 500 chars):")
        print(f"   {str(first_card)[:500]}...")
    
    # Check if it's a SPA (Single Page App) that loads data via JS
    print(f"\n2. JavaScript/SPA Detection:")
    script_tags = soup.find_all('script')
    print(f"   Script tags: {len(script_tags)}")
    
    # Look for data in script tags
    for script in script_tags[:5]:  # Check first 5
        content = script.string or ""
        if '__NEXT_DATA__' in content or 'window.__' in content:
            print(f"   ✓ Found embedded data in script tag")
            if len(content) > 100:
                print(f"     Sample: {content[:100]}...")
            break
    
    # Try other common selectors
    print(f"\n3. Alternative Selectors:")
    alternatives = [
        (".product-card", "product-card"),
        (".game-card", "game-card"),
        ("[class*='product']", "product (partial match)"),
        ("[class*='game']", "game (partial match)"),
        ("[class*='card']", "card (partial match)"),
        ("article", "article tag"),
        (".item", "item class"),
    ]
    
    for selector, desc in alternatives:
        elements = soup.select(selector)
        if elements:
            print(f"   ✓ {desc}: {len(elements)} found")
        else:
            print(f"   ❌ {desc}: none found")
    
    # Clean HTML and analyze
    print(f"\n4. Cleaned HTML Analysis:")
    cleaner = SmartHTMLCleaner()
    clean_result = cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"   Original: {len(html):,} bytes")
    print(f"   Cleaned: {len(cleaned_html):,} bytes")
    print(f"   Reduction: {clean_result['reduction_percent']:.1f}%")
    
    # Save cleaned HTML
    with open("metacritic_cleaned.html", "w", encoding="utf-8") as f:
        f.write(cleaned_html)
    print(f"   ✓ Saved to: metacritic_cleaned.html")
    
    # Check if important content was removed
    soup_cleaned = BeautifulSoup(cleaned_html, 'lxml')
    cards_after = soup_cleaned.select(".c-finderProductCard")
    print(f"   Cards after cleaning: {len(cards_after)}")
    
    if len(cards_after) < len(game_cards):
        print(f"   ⚠️  WARNING: Cleaning removed {len(game_cards) - len(cards_after)} cards!")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("="*80)
    
    if len(game_cards) == 0:
        print("\n⚠️  NO GAME CARDS FOUND")
        print("   This site likely loads content dynamically with JavaScript")
        print("   Recommendations:")
        print("   1. Increase wait time or wait for different selector")
        print("   2. Check if content is in embedded JSON (check metacritic_raw.html)")
        print("   3. Site might have bot detection (try with residential proxies)")
    elif len(game_cards) > 0 and len(cards_after) < len(game_cards):
        print("\n⚠️  CLEANING REMOVES CONTENT")
        print("   Recommendation: Adjust HTML cleaning to preserve game cards")
    else:
        print(f"\n✅ FOUND {len(game_cards)} GAME CARDS")
        print("   Recommendation: Extract data from these cards")
        print(f"   Selectors to use:")
        print(f"   - Container: .c-finderProductCard")
        print(f"   - Name: .c-finderProductCard_title")
        print(f"   - Score: .c-siteReviewScore")
        print(f"   - Description: .c-finderProductCard_description")


if __name__ == "__main__":
    asyncio.run(analyze_metacritic())

