"""
Test Metacritic extraction with our scraper only
"""
import asyncio
import os
import json
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def test_metacritic():
    """Test with our DirectLLM scraper on pre-fetched HTML"""
    print("\n" + "="*80)
    print("🔍 TESTING METACRITIC EXTRACTION")
    print("="*80)
    
    url = "https://www.metacritic.com/browse/game/all/all/current-year/"
    fields = ["name", "description", "score"]
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None
    
    # Read the already-fetched cleaned HTML
    print(f"\n📄 Reading pre-fetched HTML...")
    with open("metacritic_cleaned.html", "r", encoding="utf-8") as f:
        cleaned_html = f.read()
    print(f"   ✓ Loaded {len(cleaned_html):,} bytes")
    
    # Extract with DirectLLM
    print(f"\n🤖 Extracting with DirectLLM...")
    print(f"   Fields: {fields}")
    print(f"   Quality mode: balanced")
    
    extractor = DirectLLMExtractor(
        api_key=api_key,
        model_name="gpt-4o-mini",
        quality_mode="balanced",
        use_html2text=True
    )
    
    items = await extractor.extract(
        cleaned_html,
        fields=fields,
        context="Extract video game listings with name, description, and Metascore rating (0-100 scale)"
    )
    
    print(f"\n📊 RESULTS:")
    print(f"   Items extracted: {len(items)}")
    
    if items:
        # Calculate completeness
        total_fields = len(items) * len(fields)
        filled_fields = sum(
            1 for item in items 
            for field in fields 
            if item.get(field) and str(item.get(field)).strip() and str(item.get(field)).lower() not in ['n/a', 'none', 'null']
        )
        completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        print(f"   Completeness: {completeness:.1f}%")
        
        # Show ALL items
        print(f"\n📝 ALL {len(items)} EXTRACTED ITEMS:")
        print("="*80)
        for i, item in enumerate(items, 1):
            name = item.get('name', 'N/A')
            score = item.get('score', 'N/A')
            desc = item.get('description', 'N/A')
            
            # Truncate description
            if desc and len(str(desc)) > 80:
                desc = str(desc)[:80] + "..."
            
            print(f"\n{i}. {name}")
            print(f"   Score: {score}")
            print(f"   Description: {desc}")
        
        # Save results
        with open("metacritic_results.json", "w") as f:
            json.dump(items, f, indent=2, default=str)
        print(f"\n💾 Saved to: metacritic_results.json")
        
    else:
        print("   ⚠️  No items extracted!")
    
    return items


if __name__ == "__main__":
    asyncio.run(test_metacritic())



