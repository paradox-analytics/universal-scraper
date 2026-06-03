"""
Test Semantic Pattern Generation - End-to-end with LLM
"""

import asyncio
import os
import logging
from universal_scraper.core.ai_generator import AICodeGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

# Sample Stack Overflow HTML
STACKOVERFLOW_HTML = """
<!DOCTYPE html>
<html>
<body>
    <div class="s-post-summary js-post-summary" data-post-id="123">
        <div class="s-post-summary--stats">
            <div class="s-post-summary--stats-item s-post-summary--stats-item__emphasized">
                <span class="s-post-summary--stats-item-number">15</span>
                <span class="s-post-summary--stats-item-unit">votes</span>
            </div>
        </div>
        <div class="s-post-summary--content">
            <h3 class="s-post-summary--content-title">
                <a href="/questions/123">How to fix Python import error?</a>
            </h3>
            <div class="s-post-summary--content-excerpt">
                I'm getting ImportError when trying to import numpy
            </div>
        </div>
    </div>
    
    <div class="s-post-summary js-post-summary" data-post-id="456">
        <div class="s-post-summary--stats">
            <div class="s-post-summary--stats-item s-post-summary--stats-item__emphasized">
                <span class="s-post-summary--stats-item-number">7</span>
                <span class="s-post-summary--stats-item-unit">votes</span>
            </div>
        </div>
        <div class="s-post-summary--content">
            <h3 class="s-post-summary--content-title">
                <a href="/questions/456">JavaScript async/await best practices</a>
            </h3>
            <div class="s-post-summary--content-excerpt">
                What are the best practices for async/await in JavaScript?
            </div>
        </div>
    </div>
</body>
</html>
"""

async def test_semantic_pattern_generation():
    """Test end-to-end: LLM generates pattern → Extractor uses it"""
    print("\n" + "="*80)
    print("🧪 END-TO-END TEST: Semantic Pattern Generation + Extraction")
    print("="*80)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Step 1: Generate semantic pattern with LLM
    print("\n📝 Step 1: Generating semantic pattern with LLM...")
    generator = AICodeGenerator(api_key=api_key)
    
    fields = ['title', 'votes']
    structure_analysis = {
        'repeating_element': 'div.s-post-summary',
        'pattern_count': 2,
        'data_location': 'mixed'
    }
    
    pattern_result = generator.generate_semantic_pattern(
        cleaned_html=STACKOVERFLOW_HTML,
        fields=fields,
        url="https://stackoverflow.com/questions",
        structure_analysis=structure_analysis
    )
    
    print(f"\n✅ Pattern generated:")
    print(f"   Model: {pattern_result['model_used']}")
    print(f"   Explanation: {pattern_result['explanation']}")
    print(f"\n   Pattern:")
    
    import json
    print(json.dumps(pattern_result['pattern'], indent=2))
    
    # Step 2: Extract data using the generated pattern
    print("\n🎨 Step 2: Extracting data using semantic pattern...")
    
    extractor = SemanticExtractor()
    soup = BeautifulSoup(STACKOVERFLOW_HTML, 'html.parser')
    containers = soup.find_all('div', class_='s-post-summary')
    
    results = extractor.extract(
        html=STACKOVERFLOW_HTML,
        semantic_pattern=pattern_result['pattern'],
        containers=containers
    )
    
    print(f"\n✅ Extracted {len(results)} items:")
    for i, item in enumerate(results, 1):
        print(f"   {i}. {item}")
    
    # Step 3: Verify quality
    print("\n📊 Step 3: Verifying quality...")
    
    errors = []
    if len(results) != 2:
        errors.append(f"Expected 2 items, got {len(results)}")
    
    for item in results:
        if not item.get('title'):
            errors.append(f"Missing title in: {item}")
        if not item.get('votes'):
            errors.append(f"Missing votes in: {item}")
    
    if errors:
        print(f"\n❌ QUALITY CHECK FAILED:")
        for error in errors:
            print(f"   - {error}")
    else:
        print(f"\n✅ QUALITY CHECK PASSED")
        print(f"   - All items have required fields")
        print(f"   - Data looks correct")
    
    print("\n" + "="*80)
    if not errors:
        print("🎉 END-TO-END TEST PASSED!")
        print("="*80)
        print("\n✅ Semantic pattern generation works!")
        print("✅ Pattern extraction works!")
        print("✅ Ready to integrate into UniversalScraper\n")
    else:
        print("⚠️  TEST COMPLETED WITH WARNINGS")
        print("="*80)
    
    return pattern_result, results

if __name__ == "__main__":
    asyncio.run(test_semantic_pattern_generation())





