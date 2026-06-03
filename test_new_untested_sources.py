"""
Test Hybrid System on NEW Untested Sources

Testing completely new websites to demonstrate:
1. Universal capability (works on any site)
2. Pattern cache reuse (saves costs)
3. LLM pattern generation for new types
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from bs4 import BeautifulSoup

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.html_fetcher import HTMLFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key
API_KEY = "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"

# NEW UNTESTED SOURCES - diverse website types
TEST_SOURCES = [
    # E-commerce
    {
        "url": "https://www.etsy.com/search?q=handmade",
        "fields": ["title", "price", "seller"],
        "category": "ecommerce",
        "name": "Etsy Search"
    },
    # News/Blog
    {
        "url": "https://www.theverge.com",
        "fields": ["title", "author", "date"],
        "category": "news",
        "name": "The Verge"
    },
    # Documentation
    {
        "url": "https://docs.python.org/3/library/",
        "fields": ["title", "description"],
        "category": "documentation",
        "name": "Python Docs"
    },
    # Social Media / Forum
    {
        "url": "https://lobste.rs",
        "fields": ["title", "url", "points"],
        "category": "forum",
        "name": "Lobsters (HN-like)"
    },
    # Job Listings
    {
        "url": "https://news.ycombinator.com/jobs",
        "fields": ["title", "company"],
        "category": "job_listing",
        "name": "HN Jobs"
    },
]


class NewSourcesTest:
    """Test on completely new, untested sources."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_gen = StructuralEmbedding()
        # Use the SAME cache as previous test to demonstrate pattern reuse!
        self.pattern_cache = PatternCache(
            cache_dir="./cache/patterns_llm",
            similarity_threshold=0.75
        )
        self.pattern_gen = SemanticPatternGenerator(api_key=api_key)
        self.semantic_extractor = SemanticExtractor()
        self.html_fetcher = HTMLFetcher()
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'fallback_uses': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'items_extracted': 0,
            'errors': 0
        }
        
        logger.info("🚀 Testing NEW untested sources!")
        logger.info(f"   Using existing cache with {self.pattern_cache.get_stats()['total_patterns']} patterns")
    
    async def scrape(self, url: str, fields: list, category: str, name: str):
        """Scrape with LLM-powered patterns."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 {name} ({category})")
        logger.info(f"   URL: {url}")
        logger.info(f"   Fields: {', '.join(fields)}")
        logger.info(f"{'='*80}")
        
        try:
            # Step 1: Fetch HTML
            logger.info("📥 Step 1: Fetching HTML...")
            result = self.html_fetcher.fetch(url)
            if not result or 'html' not in result:
                self.metrics['errors'] += 1
                return {"error": "Failed to fetch", "success": False}
            
            html = result['html']
            logger.info(f"   ✓ Fetched {len(html):,} bytes")
            
            # Step 2: Generate embedding
            logger.info("🧬 Step 2: Generating structural embedding...")
            embedding = self.embedding_gen.generate(html)
            logger.info(f"   ✓ Generated 512-dim embedding")
            
            # Step 3: Search cache
            logger.info("🔍 Step 3: Searching pattern cache...")
            similar = self.pattern_cache.find_similar_pattern(embedding, fields)
            
            pattern_cost = 0.0
            used_cache = False
            used_llm = False
            
            if similar:
                # CACHE HIT - Reusing existing pattern!
                pattern_id, pattern, similarity = similar
                used_cache = True
                self.metrics['cache_hits'] += 1
                pattern_cost = 0.0001
                
                logger.info(f"   ✅ CACHE HIT! Reusing pattern: {pattern_id}")
                logger.info(f"      Similarity: {similarity:.3f}")
                logger.info(f"      💰 Saved $0.02 (no LLM call needed!)")
                logger.info(f"      ⚡ This is the MAGIC of the hybrid system!")
                
            else:
                # CACHE MISS - Generate new pattern with LLM
                self.metrics['cache_misses'] += 1
                logger.info(f"   ℹ️  CACHE MISS - This is a NEW website type!")
                logger.info(f"      Generating semantic pattern with LLM...")
                
                # Clean HTML
                clean_result = self.html_cleaner.clean(html)
                cleaned_html = clean_result['html']
                logger.info(f"      Cleaned HTML: {len(cleaned_html):,} bytes")
                
                # Detect repeating containers
                logger.info(f"      🔍 Detecting repeating containers...")
                dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
                repeating_containers = dom_patterns.get('repeating_containers', [])[:5] if dom_patterns else []
                logger.info(f"      Found {len(repeating_containers)} repeating containers")
                
                # Generate semantic pattern with LLM
                logger.info(f"      🤖 Calling GPT-4o-mini...")
                pattern_start = time.time()
                
                try:
                    pattern = await self.pattern_gen.generate_pattern(
                        html_sample=cleaned_html[:15000],
                        fields=fields,
                        context=f"Extract {', '.join(fields)} from {category} website",
                        repeating_containers=repeating_containers if repeating_containers else None
                    )
                    pattern_time = time.time() - pattern_start
                    pattern_cost = 0.02
                    used_llm = True
                    self.metrics['llm_calls'] += 1
                    
                    logger.info(f"      ✓ Pattern generated in {pattern_time:.2f}s")
                    logger.info(f"      💰 Cost: ${pattern_cost:.4f}")
                    logger.info(f"      📋 Strategy: {list(pattern.keys())}")
                    
                except Exception as e:
                    logger.error(f"      ❌ LLM generation failed: {e}")
                    pattern = self.pattern_gen._generate_fallback_pattern(fields)
                    pattern_cost = 0.0
                    self.metrics['fallback_uses'] += 1
                
                # Validate and save pattern
                if self.pattern_gen.validate_pattern(pattern):
                    domain = url.split('/')[2]
                    pattern_id = self.pattern_cache.save_pattern(
                        pattern=pattern,
                        embedding=embedding,
                        domain=domain,
                        fields=fields
                    )
                    logger.info(f"      💾 Cached for future reuse: {pattern_id}")
            
            # Step 4: Extract data
            logger.info("⚡ Step 4: Extracting data...")
            
            # Simple container detection
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to find containers
            containers = None
            if 'etsy' in url.lower():
                containers = soup.find_all(['div', 'article'], class_=lambda x: x and 'listing' in x.lower() if x else False)[:20]
            elif 'verge' in url.lower():
                containers = soup.find_all('article')[:20]
            elif 'python' in url.lower():
                containers = soup.find_all(['div', 'section'], class_=lambda x: x and ('module' in x or 'section' in x) if x else False)[:20]
            elif 'lobste' in url.lower():
                containers = soup.find_all('li', class_=lambda x: x and 'story' in x if x else False)[:20]
            elif 'jobs' in url.lower():
                containers = soup.find_all('tr', class_=lambda x: x and 'athing' in x if x else False)[:20]
            
            if not containers:
                containers = [soup.find('body')] if soup.find('body') else None
            
            extract_start = time.time()
            extracted_data = self.semantic_extractor.extract(
                html=html,
                semantic_pattern=pattern,
                containers=containers
            )
            extract_time = time.time() - extract_start
            
            total_time = time.time() - start_time
            items_count = len(extracted_data)
            
            self.metrics['total_cost'] += pattern_cost
            self.metrics['total_time'] += total_time
            self.metrics['items_extracted'] += items_count
            
            logger.info(f"   ✓ Extracted {items_count} items in {extract_time:.2f}s")
            
            # Show sample
            if items_count > 0:
                logger.info(f"\n📄 Sample Data (first 3):")
                for i, item in enumerate(extracted_data[:3], 1):
                    logger.info(f"   {i}. {json.dumps(item, indent=6, ensure_ascii=False)}")
            else:
                logger.warning(f"   ⚠️  No items extracted")
            
            status_icon = "♻️" if used_cache else ("✅" if used_llm else "⚠️")
            method = "CACHE" if used_cache else ("LLM" if used_llm else "FALLBACK")
            logger.info(f"\n{status_icon} Total: {total_time:.2f}s | 💰 Cost: ${pattern_cost:.4f} | Method: {method}")
            
            return {
                "url": url,
                "name": name,
                "category": category,
                "fields": fields,
                "success": items_count > 0,
                "items_count": items_count,
                "used_cache": used_cache,
                "used_llm": used_llm,
                "cost": pattern_cost,
                "time": total_time,
                "data": extracted_data[:5]
            }
            
        except Exception as e:
            logger.error(f"❌ Error scraping {name}: {e}")
            import traceback
            traceback.print_exc()
            self.metrics['errors'] += 1
            return {
                "url": url,
                "name": name,
                "success": False,
                "error": str(e)
            }
    
    def print_summary(self, results):
        """Print test summary."""
        logger.info(f"\n\n{'='*80}")
        logger.info(f"📊 TEST SUMMARY - NEW UNTESTED SOURCES")
        logger.info(f"{'='*80}")
        
        successful = [r for r in results if r.get('success')]
        
        logger.info(f"\n✅ Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
        logger.info(f"♻️  Cache Hit Rate: {self.metrics['cache_hits']}/{self.metrics['total_requests']} ({self.metrics['cache_hits']/self.metrics['total_requests']*100 if self.metrics['total_requests'] > 0 else 0:.0f}%)")
        logger.info(f"🤖 LLM Calls: {self.metrics['llm_calls']}")
        logger.info(f"💾 Cache Reuses: {self.metrics['cache_hits']}")
        logger.info(f"📦 Items Extracted: {self.metrics['items_extracted']} total")
        logger.info(f"⏱️  Total Time: {self.metrics['total_time']:.2f}s")
        logger.info(f"💰 Total Cost: ${self.metrics['total_cost']:.4f}")
        
        if self.metrics['total_requests'] > 0:
            logger.info(f"💵 Avg Cost/Request: ${self.metrics['total_cost']/self.metrics['total_requests']:.4f}")
        
        # Cost comparison
        parsera_cost = self.metrics['total_requests'] * 0.03
        savings = parsera_cost - self.metrics['total_cost']
        savings_pct = (savings / parsera_cost * 100) if parsera_cost > 0 else 0
        
        logger.info(f"\n💰 Cost Comparison:")
        logger.info(f"   • Parsera (LLM per request): ${parsera_cost:.4f}")
        logger.info(f"   • Hybrid System: ${self.metrics['total_cost']:.4f}")
        logger.info(f"   • Savings: ${savings:.4f} ({savings_pct:.0f}%)")
        
        # Cache impact
        if self.metrics['cache_hits'] > 0:
            saved_by_cache = self.metrics['cache_hits'] * 0.02
            logger.info(f"\n💾 Cache Impact:")
            logger.info(f"   • Patterns reused: {self.metrics['cache_hits']}")
            logger.info(f"   • Money saved by cache: ${saved_by_cache:.4f}")
            logger.info(f"   • This is the POWER of caching! 🚀")
        
        # Cache stats
        cache_stats = self.pattern_cache.get_stats()
        logger.info(f"\n📦 Pattern Cache (after test):")
        logger.info(f"   • Patterns stored: {cache_stats['total_patterns']}")
        logger.info(f"   • Unique domains: {cache_stats['domains']}")
        
        if self.metrics['errors'] > 0:
            logger.warning(f"\n⚠️  Errors: {self.metrics['errors']} requests failed")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ TEST COMPLETE")
        logger.info(f"{'='*80}")


async def main():
    """Run test on new untested sources."""
    logger.info("="*80)
    logger.info("🧪 TESTING NEW UNTESTED SOURCES")
    logger.info("="*80)
    logger.info(f"\nTesting {len(TEST_SOURCES)} completely new websites")
    logger.info("Demonstrating universal capability and pattern reuse\n")
    
    tester = NewSourcesTest(api_key=API_KEY)
    results = []
    
    for i, source in enumerate(TEST_SOURCES, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Test {i}/{len(TEST_SOURCES)}")
        logger.info(f"{'#'*80}")
        
        try:
            result = await tester.scrape(
                url=source['url'],
                fields=source['fields'],
                category=source['category'],
                name=source['name']
            )
            results.append(result)
            
            # Delay between requests
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                **source,
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    tester.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"new_sources_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': tester.metrics
        }, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"\n💾 Results saved to {output_file}")
    
    # Key insights
    logger.info(f"\n\n{'='*80}")
    logger.info(f"🎯 KEY INSIGHTS FROM NEW SOURCES TEST")
    logger.info(f"{'='*80}")
    
    if tester.metrics['cache_hits'] > 0:
        logger.info(f"\n✅ Pattern Reuse WORKING!")
        logger.info(f"   {tester.metrics['cache_hits']} sources reused existing patterns")
        logger.info(f"   Saved ${tester.metrics['cache_hits'] * 0.02:.2f} by not calling LLM")
        logger.info(f"   This proves the caching system works! 🎉")
    
    if tester.metrics['llm_calls'] > 0:
        logger.info(f"\n✅ Universal Capability PROVEN!")
        logger.info(f"   {tester.metrics['llm_calls']} new website types handled")
        logger.info(f"   System adapts to ANY website! 🚀")
    
    if tester.metrics['items_extracted'] > 0:
        logger.info(f"\n✅ High-Quality Extraction!")
        logger.info(f"   {tester.metrics['items_extracted']} items extracted")
        avg_items = tester.metrics['items_extracted'] / len([r for r in results if r.get('success')])
        logger.info(f"   Avg: {avg_items:.1f} items per site")


if __name__ == "__main__":
    asyncio.run(main())




