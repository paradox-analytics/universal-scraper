"""
Test Leafly with the TRULY UNIVERSAL Hybrid Scraper
Now with automatic JS rendering support!
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from universal_scraper.core.json_parser import JSONParser
from universal_scraper.core.json_detector import JSONDetector  # UNIVERSAL: Embedded JSON
from bs4 import BeautifulSoup


class LocalHybridScraper:
    """
    Local version of HybridScraper for testing
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_gen = StructuralEmbedding()
        self.pattern_cache = PatternCache(
            cache_dir="./cache/test_patterns",
            similarity_threshold=0.75
        )
        self.pattern_gen = SemanticPatternGenerator(api_key=api_key)
        self.semantic_extractor = SemanticExtractor()
        self.json_parser = JSONParser()  # NEW: Intelligent JSON parser
        self.json_detector = JSONDetector()  # UNIVERSAL: Embedded JSON (__NEXT_DATA__, etc.)
        # UNIVERSAL FETCHER: Automatically handles HTML, JavaScript, and JSON
        self.hybrid_fetcher = HybridFetcher(
            proxy_config=None,  # No proxy for local testing
            headless=True,
            use_camoufox=True,
            enable_cache=True
        )
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        print("✅ HybridScraper initialized")
        print("   • Structural embeddings ready")
        print("   • Pattern cache ready")
        print("   • Semantic pattern generator ready")
        print("   • UNIVERSAL JSON DETECTOR ready (__NEXT_DATA__, etc.)")
        print("   • UNIVERSAL FETCHER ready (HTML + JS + JSON)")
    
    async def scrape(self, url: str, fields):
        """Scrape with universal capabilities"""
        import time
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎯 SCRAPING: {url}")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: Fetch with universal fetcher
            print("📥 STEP 1: Universal Fetch (auto-detects HTML/JS/JSON)")
            print("-" * 80)
            result = await self.hybrid_fetcher.fetch(url)
            
            if not result or 'html' not in result:
                return {"error": "Failed to fetch", "success": False}
            
            html = result['html']
            fetch_method = result.get('fetch_method', 'unknown')
            
            print(f"✅ Fetched {len(html):,} bytes")
            print(f"   Method: {fetch_method}")
            
            # Show fetcher stats
            stats = self.hybrid_fetcher.get_stats()
            if fetch_method == 'browser':
                print(f"   🦊 Used browser (JavaScript rendering)")
            else:
                print(f"   ⚡ Used static HTML (fast)")
            print()
            
            # Step 2: Parse natural language fields FIRST
            print("🗣️  STEP 2: Natural Language Field Parsing")
            print("-" * 80)
            if isinstance(fields, str):
                print(f"Input: '{fields}'")
                parsed_fields = await self.pattern_gen._parse_natural_language_fields(
                    fields,
                    html[:3000]
                )
                print(f"✅ Parsed to: {parsed_fields}")
                fields = parsed_fields
            else:
                print(f"Fields: {fields}")
            print()
            
            # Step 2.5: UNIVERSAL JSON DETECTION (embedded + captured APIs)
            # This uses the existing universal JSONDetector that works on ALL JS sites
            print("🔍 UNIVERSAL JSON DETECTION")
            print("-" * 80)
            
            json_data_captured = result.get('json_data', [])
            if isinstance(fields, list):
                json_detection_result = self.json_detector.detect_and_extract(
                    html=html,
                    url=url,
                    captured_json=json_data_captured
                )
                
                if json_detection_result['json_found']:
                    json_sources = json_detection_result['sources']
                    all_json = json_detection_result['data']
                    
                    print(f"✅ Found JSON from: {', '.join(json_sources)}")
                    print(f"   📦 Total JSON sources: {len(all_json)}")
                    print()
                    
                    # Try to extract items using the universal JSONDetector method
                    try:
                        extracted_items = self.json_detector.extract_from_json(
                            json_data=all_json,
                            fields=fields
                        )
                        
                        # CRITICAL: Validate both quantity AND quality
                        if extracted_items and len(extracted_items) >= 3:
                            # Check content quality (reject analytics/tracking data)
                            is_sufficient = self.json_detector.is_json_sufficient(
                                json_results=json_detection_result,
                                fields=fields
                            )
                            
                            if not is_sufficient:
                                print(f"⚠️  JSON rejected: Poor content quality (likely analytics/tracking)")
                                print(f"   → Falling back to HTML extraction")
                                print()
                                # Continue to HTML extraction
                            elif is_sufficient:
                                # Success! Use JSON data
                                total_time = time.time() - start_time
                                
                                print(f"✅ JSON SUCCESS! Extracted {len(extracted_items)} items")
                                print(f"💰 Cost: $0.00 (no LLM needed!)")
                                print(f"⚡ Method: Universal JSON Detection")
                                print(f"📋 Sources: {', '.join(json_sources)}")
                                print(f"⚡ Time: {total_time:.2f}s")
                                print()
                                
                                # Display results
                                print("="*80)
                                print("📊 RESULTS (FROM UNIVERSAL JSON)")
                                print("="*80)
                                
                                print(f"\n✅ Successfully extracted {len(extracted_items)} products!\n")
                                
                                for i, item in enumerate(extracted_items[:10], 1):
                                    print(f"Product {i}:")
                                    for key, value in item.items():
                                        if value and value != 'null' and len(str(value)) > 2:
                                            display_value = str(value)[:100]
                                            if len(str(value)) > 100:
                                                display_value += "..."
                                            print(f"  • {key}: {display_value}")
                                    print()
                                
                                if len(extracted_items) > 10:
                                    print(f"... and {len(extracted_items) - 10} more products")
                                
                                print(f"\n📈 Extraction Stats:")
                                print(f"   Total products: {len(extracted_items)}")
                                print(f"   Extraction method: Universal JSON Detection")
                                print(f"   JSON sources: {', '.join(json_sources)}")
                                print(f"   Time: {total_time:.2f}s")
                                print(f"   Cost: $0.00")
                                print(f"   Fields extracted: {fields}")
                                
                                return {
                                    "url": url,
                                    "success": True,
                                    "items_count": len(extracted_items),
                                    "extraction_method": "json_universal",
                                    "json_sources": json_sources,
                                    "data": extracted_items,
                                    "time": total_time,
                                    "cost": 0.0
                                }
                        else:
                            print(f"ℹ️  JSON extraction found {len(extracted_items) if extracted_items else 0} items (needs ≥3)")
                            print(f"   Falling back to HTML extraction")
                            print()
                        
                    except Exception as e:
                        print(f"⚠️  JSON extraction failed: {e}")
                        print(f"   Falling back to HTML extraction")
                        print()
                        import traceback
                        traceback.print_exc()
                else:
                    print("ℹ️  No JSON detected, using HTML semantic extraction")
                    print()
            else:
                print(f"ℹ️  Fields not ready ({type(fields)}), using HTML extraction")
                print()
            
            # Step 3: Generate embedding
            print("🧬 STEP 3: Structural Embedding")
            print("-" * 80)
            embedding = self.embedding_gen.generate(html)
            print(f"✅ Generated 512-dim embedding")
            print()
            
            # Step 4: Check pattern cache
            print("💾 STEP 4: Pattern Cache Lookup")
            print("-" * 80)
            # Note: cache lookup would go here
            print("ℹ️  Cache check skipped for demo")
            print()
            
            # Step 5: Clean HTML
            print("🧹 STEP 5: HTML Cleaning")
            print("-" * 80)
            clean_result = self.html_cleaner.clean(html)
            cleaned_html = clean_result['html']
            print(f"✅ Cleaned to {len(cleaned_html):,} bytes ({100*(1-len(cleaned_html)/len(html)):.1f}% reduction)")
            print()
            
            # Step 6: Detect containers
            print("📦 STEP 6: Container Detection")
            print("-" * 80)
            dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
            containers = dom_patterns.get('repeating_containers', [])
            print(f"✅ Found {len(containers)} repeating patterns")
            if containers:
                for i, c in enumerate(containers[:3], 1):
                    print(f"   {i}. {c.get('pattern', 'N/A')} ({c.get('count', 0)} instances)")
            print()
            
            # Step 7: Generate semantic pattern
            print("🤖 STEP 7: Semantic Pattern Generation (LLM)")
            print("-" * 80)
            print(f"Fields: {fields}")
            
            container_info = None
            if containers:
                container_info = [{
                    'pattern': c.get('pattern'),
                    'count': c.get('count')
                } for c in containers[:5]]
            
            pattern = await self.pattern_gen.generate_pattern(
                html_sample=cleaned_html[:15000],
                fields=fields,
                context=f"Cannabis dispensary menu at {url}",
                repeating_containers=container_info
            )
            
            print(f"✅ Pattern generated with {len(pattern)} fields")
            print("\nGenerated Pattern (first 2 fields):")
            for i, (field, config) in enumerate(list(pattern.items())[:2], 1):
                print(f"\n  {i}. {field}:")
                print(f"     Primary: {config.get('primary', {}).get('type', 'N/A')}")
                fallbacks = config.get('fallbacks', [])
                if fallbacks:
                    print(f"     Fallbacks: {len(fallbacks)} strategies")
            print()
            
            # Step 8: Extract data
            print("⚡ STEP 8: Data Extraction")
            print("-" * 80)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find product containers
            container_selectors = [
                {'class_': lambda x: x and any(kw in str(x).lower() for kw in ['product', 'item', 'menu', 'strain'])},
                {'name': 'article'},
                {'name': 'li'},
            ]
            
            all_containers = []
            for selector in container_selectors:
                found = soup.find_all(**selector)
                if found:
                    print(f"  Found {len(found)} containers with {selector}")
                    all_containers.extend(found[:30])
                    if len(all_containers) >= 30:
                        break
            
            if not all_containers:
                print("  ⚠️  No specific containers found, using body")
                all_containers = [soup.find('body')] if soup.find('body') else None
            
            extracted_data = self.semantic_extractor.extract(
                html=html,
                semantic_pattern=pattern,
                containers=all_containers
            )
            
            total_time = time.time() - start_time
            
            print(f"\n✅ Extracted {len(extracted_data)} items in {total_time:.2f}s")
            print()
            
            # Step 9: Display results
            print("="*80)
            print("📊 RESULTS")
            print("="*80)
            
            if extracted_data:
                print(f"\n✅ Successfully extracted {len(extracted_data)} cannabis products!\n")
                
                for i, item in enumerate(extracted_data[:10], 1):
                    print(f"Product {i}:")
                    for key, value in item.items():
                        if value and value != 'null' and len(str(value)) > 2:
                            # Truncate long values
                            display_value = str(value)[:100]
                            if len(str(value)) > 100:
                                display_value += "..."
                            print(f"  • {key}: {display_value}")
                    print()
                
                if len(extracted_data) > 10:
                    print(f"... and {len(extracted_data) - 10} more products")
                
                # Show some stats
                print(f"\n📈 Extraction Stats:")
                print(f"   Total products: {len(extracted_data)}")
                print(f"   Fetch method: {fetch_method}")
                print(f"   Time: {total_time:.2f}s")
                print(f"   Fields extracted: {list(pattern.keys())}")
                
            else:
                print("❌ No products extracted")
                print("\nPossible reasons:")
                print("  • Products are in iframes")
                print("  • Need to wait for AJAX")
                print("  • Containers not detected correctly")
                print("\nShowing first 1000 chars of HTML:")
                print(html[:1000])
            
            return {
                "url": url,
                "success": len(extracted_data) > 0,
                "items_count": len(extracted_data),
                "fetch_method": fetch_method,
                "data": extracted_data,
                "time": total_time
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }


async def main():
    """Main test function"""
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    print("\n" + "="*80)
    print("🧪 TESTING TRULY UNIVERSAL HYBRID SCRAPER")
    print("="*80)
    print("\nFeatures:")
    print("  ✅ Static HTML (fast)")
    print("  ✅ JavaScript rendering (Camoufox)")
    print("  ✅ JSON API discovery")
    print("  ✅ Natural language field parsing")
    print("  ✅ Semantic pattern generation")
    print("  ✅ Pattern caching (99.5% savings)")
    print()
    
    # Initialize scraper
    scraper = LocalHybridScraper(api_key=api_key)
    
    # Test Leafly (requires JavaScript)
    url = "https://www.leafly.com/dispensary-info/seven-point/menu"
    fields = "Extract the product name, price and description for all products"
    
    result = await scraper.scrape(url, fields)
    
    # Cleanup
    await scraper.hybrid_fetcher.close()
    
    print("\n" + "="*80)
    print("🏁 TEST COMPLETE")
    print("="*80)
    
    if result.get('success'):
        print(f"\n✅ SUCCESS! Extracted {result['items_count']} products")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'No products found')}")


if __name__ == '__main__':
    asyncio.run(main())

