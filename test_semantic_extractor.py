"""
Test Semantic Extractor - Verify it works before full integration
"""

import asyncio
import logging
from bs4 import BeautifulSoup
from universal_scraper.core.semantic_extractor import SemanticExtractor

logging.basicConfig(level=logging.INFO)

# Sample HTML (Stack Overflow-like structure)
SAMPLE_HTML = """
<html>
<body>
    <div class="s-post-summary js-post-summary">
        <div class="s-post-summary--stats">
            <div class="s-post-summary--stats-item s-post-summary--stats-item__emphasized">
                <span class="s-post-summary--stats-item-number">5</span>
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
    
    <div class="s-post-summary js-post-summary">
        <div class="s-post-summary--stats">
            <div class="s-post-summary--stats-item s-post-summary--stats-item__emphasized">
                <span class="s-post-summary--stats-item-number">12</span>
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

# Sample e-commerce HTML
ECOMMERCE_HTML = """
<html>
<body>
    <div class="product-card">
        <h2>Laptop Computer</h2>
        <span class="price">$1,299.99</span>
        <div class="rating" data-rating="4.5">★★★★☆</div>
    </div>
    
    <div class="product-card">
        <h2>Wireless Mouse</h2>
        <span class="price">$29.99</span>
        <div class="rating" data-rating="4.8">★★★★★</div>
    </div>
</body>
</html>
"""

def test_stackoverflow_pattern():
    """Test semantic extraction with Stack Overflow pattern"""
    print("\n" + "="*80)
    print("🧪 TEST 1: Stack Overflow Pattern")
    print("="*80)
    
    # Semantic pattern (what AI would generate)
    semantic_pattern = {
        "title": {
            "primary": {
                "type": "heading",
                "position": "first"
            },
            "fallbacks": [
                {"type": "link_text"},
                {"type": "bold_text", "min_length": 10}
            ],
            "validation": {
                "not_empty": True,
                "min_length": 5
            }
        },
        "votes": {
            "primary": {
                "type": "number",
                "pattern": r'\d+'
            },
            "fallbacks": [
                {"type": "attribute", "name": "data-votes"}
            ]
        }
    }
    
    # Parse HTML and find containers
    soup = BeautifulSoup(SAMPLE_HTML, 'html.parser')
    containers = soup.find_all('div', class_='s-post-summary')
    
    # Extract using semantic patterns
    extractor = SemanticExtractor()
    results = extractor.extract(SAMPLE_HTML, semantic_pattern, containers)
    
    print(f"\n📊 Results: {len(results)} items extracted")
    for i, item in enumerate(results, 1):
        print(f"   {i}. {item}")
    
    # Verify
    assert len(results) == 2, f"Expected 2 items, got {len(results)}"
    assert results[0]['title'], "First item should have title"
    assert results[0]['votes'], "First item should have votes"
    print("\n✅ Stack Overflow pattern test PASSED")
    
    return results

def test_ecommerce_pattern():
    """Test semantic extraction with e-commerce pattern"""
    print("\n" + "="*80)
    print("🧪 TEST 2: E-commerce Pattern")
    print("="*80)
    
    # Semantic pattern
    semantic_pattern = {
        "product_name": {
            "primary": {
                "type": "heading",
                "position": "first"
            },
            "fallbacks": [
                {"type": "bold_text"}
            ]
        },
        "price": {
            "primary": {
                "type": "currency",
                "symbols": ["$", "€"]
            },
            "fallbacks": [
                {"type": "attribute", "name": "data-price"}
            ]
        },
        "rating": {
            "primary": {
                "type": "attribute",
                "name": "data-rating"
            },
            "fallbacks": [
                {"type": "number", "pattern": r'\d\.\d'}
            ]
        }
    }
    
    # Parse HTML and find containers
    soup = BeautifulSoup(ECOMMERCE_HTML, 'html.parser')
    containers = soup.find_all('div', class_='product-card')
    
    # Extract
    extractor = SemanticExtractor()
    results = extractor.extract(ECOMMERCE_HTML, semantic_pattern, containers)
    
    print(f"\n📊 Results: {len(results)} items extracted")
    for i, item in enumerate(results, 1):
        print(f"   {i}. {item}")
    
    # Verify
    assert len(results) == 2, f"Expected 2 items, got {len(results)}"
    assert '$' in results[0]['price'], "Price should contain $"
    assert results[0]['rating'], "Should have rating"
    print("\n✅ E-commerce pattern test PASSED")
    
    return results

def test_fallback_mechanism():
    """Test fallback strategies work"""
    print("\n" + "="*80)
    print("🧪 TEST 3: Fallback Mechanism")
    print("="*80)
    
    # HTML where primary strategy fails
    html = """
    <div class="item">
        <strong>This is bold text</strong>
        <span class="price">$99.99</span>
    </div>
    """
    
    # Pattern where heading (primary) doesn't exist, falls back to bold
    semantic_pattern = {
        "title": {
            "primary": {
                "type": "heading"  # Won't find h1-h6
            },
            "fallbacks": [
                {"type": "bold_text"}  # Should find <strong>
            ]
        }
    }
    
    soup = BeautifulSoup(html, 'html.parser')
    containers = [soup.find('div', class_='item')]
    
    extractor = SemanticExtractor()
    results = extractor.extract(html, semantic_pattern, containers)
    
    print(f"\n📊 Results: {len(results)} items extracted")
    for i, item in enumerate(results, 1):
        print(f"   {i}. {item}")
    
    assert len(results) == 1, f"Expected 1 item, got {len(results)}"
    assert results[0]['title'] == "This is bold text", "Should use fallback"
    print("\n✅ Fallback mechanism test PASSED")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎨 SEMANTIC EXTRACTOR TEST SUITE")
    print("="*80)
    print("\nTesting semantic pattern extraction WITHOUT LLM or exec()...")
    
    try:
        test_stackoverflow_pattern()
        test_ecommerce_pattern()
        test_fallback_mechanism()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - Semantic Extractor Working!")
        print("="*80)
        print("\n✅ Ready to integrate into UniversalScraper")
        print("✅ This replaces brittle CSS code generation")
        print("✅ Patterns are resilient to layout changes\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()





