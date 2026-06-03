"""
Test Leafly scraping with hybrid system and natural language fields
"""

import asyncio
import os
import sys
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.html_fetcher import HTMLFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector


async def test_leafly_scrape():
    """Test Leafly dispensary menu scraping"""
    
    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    print("="*80)
    print("🧪 TESTING LEAFLY HYBRID SCRAPER")
    print("="*80)
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    html_fetcher = HTMLFetcher()
    html_cleaner = SmartHTMLCleaner()
    dom_detector = DOMPatternDetector()
    embedding_gen = StructuralEmbedding()
    pattern_gen = SemanticPatternGenerator(api_key=api_key)
    semantic_extractor = SemanticExtractor()
    
    # Test URL
    url = "https://www.leafly.com/dispensary-info/seven-point/menu"
    
    # Natural language input (what user typed)
    natural_language_fields = "Extract the product name, price and description for all products"
    
    print(f"🌐 URL: {url}")
    print(f"📝 Natural Language: '{natural_language_fields}'")
    print()
    
    # Step 1: Fetch HTML
    print("-"*80)
    print("STEP 1: FETCHING HTML")
    print("-"*80)
    result = html_fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes")
    print()
    
    # Step 2: Clean HTML
    print("-"*80)
    print("STEP 2: CLEANING HTML")
    print("-"*80)
    clean_result = html_cleaner.clean(html)
    cleaned_html = clean_result['html']
    print(f"✅ Cleaned to {len(cleaned_html):,} bytes")
    print()
    
    # Step 3: Parse natural language to fields
    print("-"*80)
    print("STEP 3: PARSING NATURAL LANGUAGE")
    print("-"*80)
    print(f"Input: '{natural_language_fields}'")
    
    # Call the natural language parser
    parsed_fields = await pattern_gen._parse_natural_language_fields(
        natural_language_fields,
        cleaned_html[:3000]  # Sample for context
    )
    
    print(f"✅ Parsed fields: {parsed_fields}")
    print()
    
    # Step 4: Detect containers
    print("-"*80)
    print("STEP 4: DETECTING REPEATING CONTAINERS")
    print("-"*80)
    dom_patterns = dom_detector.detect_patterns(cleaned_html)
    containers = dom_patterns.get('repeating_containers', [])
    print(f"✅ Found {len(containers)} repeating container types")
    
    if containers:
        print("\nTop 3 containers:")
        for i, container in enumerate(containers[:3], 1):
            print(f"  {i}. Pattern: {container.get('pattern', 'N/A')}")
            print(f"     Count: {container.get('count', 0)}")
            print(f"     Signature: {str(container.get('signature', ''))[:100]}...")
    print()
    
    # Step 5: Generate semantic pattern
    print("-"*80)
    print("STEP 5: GENERATING SEMANTIC PATTERN")
    print("-"*80)
    print(f"Fields to extract: {parsed_fields}")
    
    # Simplify containers for serialization
    container_info = None
    if containers:
        container_info = [{
            'pattern': c.get('pattern'),
            'count': c.get('count')
        } for c in containers[:5]]
    
    pattern = await pattern_gen.generate_pattern(
        html_sample=cleaned_html[:15000],
        fields=parsed_fields,
        context=f"Extract data from a cannabis dispensary menu page at {url}",
        repeating_containers=container_info
    )
    
    print(f"✅ Generated pattern with {len(pattern)} fields")
    print("\nGenerated Pattern:")
    print(json.dumps(pattern, indent=2))
    print()
    
    # Step 6: Extract data
    print("-"*80)
    print("STEP 6: EXTRACTING DATA")
    print("-"*80)
    
    # Get actual container elements
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Try multiple container strategies
    container_selectors = [
        # Product-specific
        {'class_': lambda x: x and 'product' in str(x).lower()},
        {'class_': lambda x: x and 'item' in str(x).lower()},
        {'class_': lambda x: x and 'menu' in str(x).lower()},
        {'class_': lambda x: x and 'card' in str(x).lower()},
        # Generic
        {'name': 'article'},
        {'name': 'li'},
    ]
    
    all_containers = []
    for selector in container_selectors:
        found = soup.find_all(**selector)
        if found:
            print(f"  Found {len(found)} containers with {selector}")
            all_containers.extend(found[:20])  # Limit to first 20
            if len(all_containers) >= 20:
                break
    
    if not all_containers:
        print("  ⚠️  No containers found, using body")
        all_containers = [soup.find('body')] if soup.find('body') else None
    
    print(f"\nExtracting from {len(all_containers) if all_containers else 0} containers...")
    
    extracted_data = semantic_extractor.extract(
        html=html,
        semantic_pattern=pattern,
        containers=all_containers
    )
    
    print(f"✅ Extracted {len(extracted_data)} items")
    print()
    
    # Step 7: Display results
    print("-"*80)
    print("STEP 7: RESULTS")
    print("-"*80)
    
    if extracted_data:
        print(f"\n📊 Extracted {len(extracted_data)} items:\n")
        for i, item in enumerate(extracted_data[:10], 1):
            print(f"Item {i}:")
            for key, value in item.items():
                if value and value != 'null':
                    print(f"  {key}: {value}")
            print()
        
        if len(extracted_data) > 10:
            print(f"... and {len(extracted_data) - 10} more items")
    else:
        print("❌ No data extracted")
        print("\nDebugging info:")
        print(f"- Containers used: {len(all_containers) if all_containers else 0}")
        print(f"- Pattern fields: {list(pattern.keys())}")
        
        # Show a sample of the HTML to understand structure
        print("\nFirst 2000 chars of cleaned HTML:")
        print(cleaned_html[:2000])
    
    print()
    print("="*80)
    print("🏁 TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(test_leafly_scrape())




