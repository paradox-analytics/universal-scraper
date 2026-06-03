#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for Universal Hybrid Scraper

Tests extraction quality across diverse website types:
- E-commerce (Amazon, eBay)
- Social/Forums (Reddit, Hacker News)
- JS-heavy (Leafly, Product Hunt)
- News/Content sites
- Static HTML sites

Success criteria:
- High-quality structured data extraction
- Proper JSON quality validation
- Appropriate HTML fallback
- Universal capability across all site types
"""
import asyncio
import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Any

# Add project root to sys.path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector
from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector


class ComprehensiveQA:
    """Comprehensive QA testing for universal scraper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pattern_gen = SemanticPatternGenerator(api_key=api_key)
        self.hybrid_fetcher = HybridFetcher(
            proxy_config=None,
            headless=True,
            use_camoufox=True,
            enable_cache=False  # Disable cache for fresh tests
        )
        self.json_detector = JSONDetector()
        self.embedding_gen = StructuralEmbedding()
        self.pattern_cache = PatternCache(cache_dir="./test_cache", similarity_threshold=0.75)
        self.semantic_extractor = SemanticExtractor()
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        # Test sources with expected behavior
        self.test_sources = [
            {
                "name": "Leafly (JS-heavy, Cannabis)",
                "url": "https://www.leafly.com/dispensary-info/seven-point/menu",
                "fields": "Extract product name, price, and description for all products",
                "expected_method": "json",
                "expected_min_items": 10,
                "expected_quality": "high",
                "site_type": "JS-heavy e-commerce"
            },
            {
                "name": "Amazon (E-commerce)",
                "url": "https://www.amazon.com/s?k=laptop",
                "fields": "Get product title, price, and rating",
                "expected_method": "html_fallback",  # JSON should be rejected
                "expected_min_items": 10,
                "expected_quality": "high",
                "site_type": "Mixed HTML/JS"
            },
            {
                "name": "eBay (Auction)",
                "url": "https://www.ebay.com/sch/i.html?_nkw=macbook",
                "fields": "Get item title, price, and condition",
                "expected_method": "html_fallback",  # JSON should be rejected
                "expected_min_items": 10,
                "expected_quality": "high",
                "site_type": "Mixed HTML/JS"
            },
            {
                "name": "Reddit (Social)",
                "url": "https://old.reddit.com/r/programming/",
                "fields": "Extract post title, author, and upvotes",
                "expected_method": "html",
                "expected_min_items": 15,
                "expected_quality": "high",
                "site_type": "Static HTML"
            },
            {
                "name": "Hacker News (News)",
                "url": "https://news.ycombinator.com/",
                "fields": "Extract article title, points, and comments count",
                "expected_method": "html",
                "expected_min_items": 20,
                "expected_quality": "high",
                "site_type": "Static HTML"
            },
            {
                "name": "Product Hunt (Tech)",
                "url": "https://www.producthunt.com/",
                "fields": "Get product name, tagline, and upvotes",
                "expected_method": "json",
                "expected_min_items": 5,
                "expected_quality": "high",
                "site_type": "Modern JS/React"
            }
        ]
        
        self.results = []
    
    async def test_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single source with comprehensive quality checks"""
        print("\n" + "="*100)
        print(f"🧪 TESTING: {source['name']}")
        print("="*100)
        print(f"URL: {source['url']}")
        print(f"Type: {source['site_type']}")
        print(f"Fields: {source['fields']}")
        print(f"Expected: {source['expected_min_items']}+ items via {source['expected_method']}")
        print()
        
        start_time = time.time()
        result = {
            "name": source['name'],
            "url": source['url'],
            "site_type": source['site_type'],
            "expected_method": source['expected_method'],
            "expected_min_items": source['expected_min_items'],
            "success": False,
            "items_extracted": 0,
            "extraction_method": None,
            "json_quality_score": None,
            "json_rejected": False,
            "data_quality": "unknown",
            "issues": [],
            "time": 0
        }
        
        try:
            # Step 1: Fetch
            print("📥 Step 1: Universal Fetch")
            print("-" * 100)
            fetch_result = await self.hybrid_fetcher.fetch(source['url'])
            
            if not fetch_result or 'html' not in fetch_result:
                result['issues'].append("Failed to fetch HTML")
                return result
            
            html = fetch_result['html']
            fetch_method = fetch_result.get('fetch_method', 'unknown')
            print(f"✅ Fetched {len(html):,} bytes via {fetch_method}")
            print()
            
            # Step 2: Parse fields
            print("📝 Step 2: Parse Natural Language Fields")
            print("-" * 100)
            try:
                fields = await self.pattern_gen._parse_natural_language_fields(
                    source['fields'],
                    html[:3000]
                )
                print(f"✅ Parsed to: {fields}")
            except Exception as e:
                print(f"⚠️  Failed to parse with LLM: {e}")
                fields = self.pattern_gen._extract_fields_from_text(source['fields'])
                print(f"✅ Parsed to (fallback): {fields}")
            print()
            
            # Step 3: JSON Detection & Quality Validation
            print("🔍 Step 3: Universal JSON Detection & Quality Validation")
            print("-" * 100)
            
            json_data_captured = fetch_result.get('json_data', [])
            json_detection_result = self.json_detector.detect_and_extract(
                html=html,
                url=source['url'],
                captured_json=json_data_captured
            )
            
            extracted_items = []
            
            if json_detection_result['json_found']:
                json_sources = json_detection_result['sources']
                all_json = json_detection_result['data']
                
                print(f"✅ Found JSON from: {', '.join(json_sources)}")
                print(f"📦 Total JSON sources: {len(all_json)}")
                print()
                
                # Extract from JSON
                print("🔬 Extracting items from JSON...")
                extracted_items = self.json_detector.extract_from_json(
                    json_data=all_json,
                    fields=fields
                )
                print(f"   Extracted {len(extracted_items)} items")
                
                if extracted_items and len(extracted_items) >= 3:
                    # Show sample
                    sample = extracted_items[0]
                    print(f"   Sample item: {list(sample.keys())}")
                    for key, value in list(sample.items())[:3]:
                        if key != '_metadata':
                            print(f"      • {key}: {str(value)[:60]}")
                    print()
                    
                    # Quality validation
                    print("🎯 Validating content quality...")
                    is_sufficient = self.json_detector.is_json_sufficient(
                        json_results=json_detection_result,
                        fields=fields
                    )
                    
                    if is_sufficient:
                        print(f"   ✅ JSON ACCEPTED - High quality data")
                        result['extraction_method'] = 'json'
                        result['items_extracted'] = len(extracted_items)
                        result['json_rejected'] = False
                    else:
                        print(f"   ⚠️  JSON REJECTED - Low quality (analytics/tracking)")
                        print(f"   → Falling back to HTML extraction")
                        result['json_rejected'] = True
                        extracted_items = []  # Clear for HTML fallback
                else:
                    print(f"   ℹ️  Only {len(extracted_items)} items (needs ≥3)")
                    extracted_items = []
            else:
                print("ℹ️  No JSON detected")
            print()
            
            # Step 4: HTML Fallback (if JSON rejected or insufficient)
            if not extracted_items or len(extracted_items) < 3:
                print("🌐 Step 4: HTML Extraction (Semantic + LLM)")
                print("-" * 100)
                
                # Clean HTML
                cleaned_result = self.html_cleaner.clean(html)
                cleaned_html = cleaned_result['html']
                print(f"✅ Cleaned HTML: {cleaned_result['original_size']:,} → {cleaned_result['cleaned_size']:,} bytes ({cleaned_result['reduction_percent']:.1f}% reduction)")
                
                # Detect patterns
                dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
                best_pattern = dom_patterns.get('best_pattern')
                
                if best_pattern:
                    print(f"✅ Found pattern: {best_pattern.get('type')} (confidence: {best_pattern.get('confidence', 0):.0%})")
                    print(f"   Selector: {best_pattern.get('selector')}")
                    print(f"   Count: {best_pattern.get('count')} items")
                    
                    # Get containers using the detected pattern
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(cleaned_html, 'html.parser')
                    selector = best_pattern.get('selector', '')
                    containers = soup.select(selector) if selector else []
                    print(f"✅ Extracted {len(containers)} containers")
                else:
                    print(f"⚠️  No patterns detected")
                    containers = []
                
                if containers:
                    # Generate pattern with LLM
                    print(f"🤖 Generating semantic pattern with LLM...")
                    try:
                        # Create serializable container representations
                        container_samples = []
                        for container in containers[:3]:
                            container_samples.append({
                                'tag': container.name,
                                'classes': ' '.join(container.get('class', [])),
                                'text_preview': container.get_text()[:100]
                            })
                        
                        pattern = await self.pattern_gen.generate_pattern(
                            html_sample=cleaned_html[:5000],
                            fields=fields,
                            repeating_containers=container_samples
                        )
                        print(f"✅ Generated pattern for {len(pattern.get('fields', []))} fields")
                        
                        # Extract with semantic extractor
                        extracted_items = self.semantic_extractor.extract(
                            html=cleaned_html,
                            semantic_pattern=pattern,
                            containers=containers
                        )
                        print(f"✅ Extracted {len(extracted_items)} items via HTML")
                        result['extraction_method'] = 'html'
                        result['items_extracted'] = len(extracted_items)
                    except Exception as e:
                        print(f"❌ HTML extraction failed: {e}")
                        result['issues'].append(f"HTML extraction error: {str(e)}")
                else:
                    print(f"⚠️  No repeating containers found")
                    result['issues'].append("No repeating patterns detected")
                print()
            
            # Step 5: Quality Assessment
            print("📊 Step 5: Data Quality Assessment")
            print("-" * 100)
            
            if extracted_items and len(extracted_items) >= source['expected_min_items']:
                result['success'] = True
                result['data_quality'] = self._assess_data_quality(extracted_items, fields)
                print(f"✅ SUCCESS: {len(extracted_items)} items extracted")
                print(f"   Quality: {result['data_quality']}")
                print(f"   Method: {result['extraction_method']}")
            elif extracted_items:
                result['data_quality'] = self._assess_data_quality(extracted_items, fields)
                print(f"⚠️  PARTIAL: {len(extracted_items)} items (expected {source['expected_min_items']}+)")
                print(f"   Quality: {result['data_quality']}")
                result['issues'].append(f"Insufficient items: {len(extracted_items)}/{source['expected_min_items']}")
            else:
                result['data_quality'] = "failed"
                print(f"❌ FAILED: No items extracted")
                result['issues'].append("Zero items extracted")
            
            result['time'] = time.time() - start_time
            
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            result['issues'].append(f"Exception: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _assess_data_quality(self, items: List[Dict], fields: List[str]) -> str:
        """Assess quality of extracted data"""
        if not items:
            return "failed"
        
        # Check field coverage
        sample = items[0]
        matched_fields = sum(1 for f in fields if f in sample)
        coverage = matched_fields / len(fields) if fields else 0
        
        # Check for analytics garbage
        has_garbage = False
        for item in items[:3]:
            for key, value in item.items():
                if key == '_metadata':
                    continue
                value_str = str(value).lower()
                if any(kw in value_str for kw in ['_optimistic_', 'operationid', 'traceid', 'correlation']):
                    has_garbage = True
                    break
        
        if has_garbage:
            return "low (analytics garbage)"
        elif coverage >= 0.8 and len(items) >= 10:
            return "excellent"
        elif coverage >= 0.6 and len(items) >= 5:
            return "good"
        elif coverage >= 0.4:
            return "fair"
        else:
            return "poor"
    
    async def run_all_tests(self):
        """Run all QA tests"""
        print("\n" + "="*100)
        print("🚀 COMPREHENSIVE QA TEST SUITE - UNIVERSAL HYBRID SCRAPER")
        print("="*100)
        print(f"Testing {len(self.test_sources)} diverse data sources")
        print()
        
        for source in self.test_sources:
            result = await self.test_source(source)
            self.results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Summary report
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary report"""
        print("\n" + "="*100)
        print("📈 COMPREHENSIVE QA SUMMARY")
        print("="*100)
        print()
        
        successful = sum(1 for r in self.results if r['success'])
        partial = sum(1 for r in self.results if r['items_extracted'] > 0 and not r['success'])
        failed = sum(1 for r in self.results if r['items_extracted'] == 0)
        
        print(f"Overall Results:")
        print(f"  ✅ Success: {successful}/{len(self.results)}")
        print(f"  ⚠️  Partial: {partial}/{len(self.results)}")
        print(f"  ❌ Failed: {failed}/{len(self.results)}")
        print()
        
        print("Detailed Results:")
        print("-" * 100)
        
        for result in self.results:
            status_icon = "✅" if result['success'] else ("⚠️" if result['items_extracted'] > 0 else "❌")
            print(f"{status_icon} {result['name']}")
            print(f"   Type: {result['site_type']}")
            print(f"   Items: {result['items_extracted']}/{result['expected_min_items']} (expected)")
            print(f"   Method: {result['extraction_method'] or 'none'}")
            print(f"   Quality: {result['data_quality']}")
            if result['json_rejected']:
                print(f"   ⚠️  JSON rejected (quality validation)")
            if result['issues']:
                print(f"   Issues: {', '.join(result['issues'])}")
            print(f"   Time: {result['time']:.1f}s")
            print()
        
        print("="*100)
        print("🎯 QA ASSESSMENT")
        print("="*100)
        
        if successful == len(self.results):
            print("✅ PASS: All tests successful!")
            print("   System is ready for production deployment.")
        elif successful + partial >= len(self.results) * 0.8:
            print("⚠️  ACCEPTABLE: 80%+ tests passed")
            print("   Review partial failures and iterate if needed.")
        else:
            print("❌ FAIL: Too many failures")
            print("   System needs iteration and improvement before deployment.")
            print()
            print("Recommended Actions:")
            
            # Analyze failures
            json_issues = sum(1 for r in self.results if r['json_rejected'])
            html_issues = sum(1 for r in self.results if r['extraction_method'] == 'html' and not r['success'])
            
            if json_issues > 0:
                print(f"  • JSON quality validation rejected {json_issues} sources")
                print(f"    → Review: Is HTML fallback working properly?")
            
            if html_issues > 0:
                print(f"  • HTML extraction failed on {html_issues} sources")
                print(f"    → Review: Pattern generation and semantic extraction")
            
            no_extraction = [r for r in self.results if r['items_extracted'] == 0]
            if no_extraction:
                print(f"  • Zero items extracted on: {', '.join([r['name'] for r in no_extraction])}")
                print(f"    → Review: DOM detection and container identification")
        
        print()
        print("="*100)


async def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    qa = ComprehensiveQA(api_key=api_key)
    await qa.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

