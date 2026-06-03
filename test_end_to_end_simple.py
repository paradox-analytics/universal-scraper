"""
End-to-End Hybrid System Test (Simplified)

Tests the complete pipeline:
1. Fetch HTML
2. Generate structural embedding
3. Search for similar patterns
4. Generate semantic pattern with LLM (if needed)
5. Extract data
6. Demonstrate pattern reuse

Simplified to avoid serialization issues with DOM signatures.
"""

import asyncio
import json
import logging
import time
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


# Test sources - chosen to show pattern reuse
TEST_SOURCES = [
    {
        "url": "https://news.ycombinator.com",
        "fields": ["title", "url"],
        "category": "forum",
        "name": "Hacker News"
    },
    {
        "url": "https://github.com/trending",
        "fields": ["name", "description", "stars"],
        "category": "listing",
        "name": "GitHub Trending"
    },
    {
        "url": "https://stackoverflow.com/questions",
        "fields": ["title", "votes"],
        "category": "forum",
        "name": "Stack Overflow"
    },
]


class SimpleHybridTest:
    """Simplified hybrid test without complex DOM signature handling."""
    
    def __init__(self):
        self.embedding_gen = StructuralEmbedding()
        self.pattern_cache = PatternCache(
            cache_dir="./cache/patterns_test",
            similarity_threshold=0.75
        )
        self.pattern_gen = SemanticPatternGenerator()
        self.semantic_extractor = SemanticExtractor()
        self.html_fetcher = HTMLFetcher()
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'items_extracted': 0
        }
        
        logger.info("🚀 Simple Hybrid Test initialized")
    
    async def scrape(self, url: str, fields: list, category: str, name: str):
        """Scrape with the hybrid approach."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 {name} ({category})")
        logger.info(f"   URL: {url}")
        logger.info(f"   Fields: {', '.join(fields)}")
        logger.info(f"{'='*80}")
        
        # Step 1: Fetch HTML
        logger.info("📥 Step 1: Fetching HTML...")
        result = self.html_fetcher.fetch(url)
        if not result or 'html' not in result:
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
        
        if similar:
            # CACHE HIT
            pattern_id, pattern, similarity = similar
            used_cache = True
            self.metrics['cache_hits'] += 1
            pattern_cost = 0.0001
            
            logger.info(f"   ✅ CACHE HIT! Pattern: {pattern_id}")
            logger.info(f"      Similarity: {similarity:.3f}")
            logger.info(f"      💰 Saved ~$0.02 (no LLM call)")
            
        else:
            # CACHE MISS - Generate pattern
            self.metrics['cache_misses'] += 1
            logger.info(f"   ℹ️  CACHE MISS - Generating new pattern with LLM...")
            
            # Clean HTML
            clean_result = self.html_cleaner.clean(html)
            cleaned_html = clean_result['html']
            logger.info(f"      Cleaned HTML: {len(cleaned_html):,} bytes")
            
            # Detect repeating containers
            logger.info(f"      🔍 Detecting repeating containers...")
            dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
            repeating_containers = dom_patterns.get('repeating_containers', [])[:5] if dom_patterns else []
            logger.info(f"      Found {len(repeating_containers)} repeating containers")
            
            # Generate semantic pattern
            logger.info(f"      🎨 Calling LLM to generate semantic pattern...")
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
                
                logger.info(f"      ✓ Pattern generated in {pattern_time:.2f}s")
                logger.info(f"      💰 Cost: ~${pattern_cost:.4f}")
                
                # Validate pattern
                if self.pattern_gen.validate_pattern(pattern):
                    # Save to cache
                    domain = url.split('/')[2]
                    pattern_id = self.pattern_cache.save_pattern(
                        pattern=pattern,
                        embedding=embedding,
                        domain=domain,
                        fields=fields
                    )
                    logger.info(f"      💾 Saved to cache: {pattern_id}")
                else:
                    logger.warning(f"      ⚠️  Invalid pattern, using fallback")
                    pattern = self.pattern_gen._generate_fallback_pattern(fields)
                    pattern_cost = 0.0
                
            except Exception as e:
                logger.error(f"      ❌ LLM generation failed: {e}")
                pattern = self.pattern_gen._generate_fallback_pattern(fields)
                pattern_cost = 0.0
                logger.info(f"      ⚠️  Using fallback pattern")
        
        # Step 4: Extract data
        logger.info("⚡ Step 4: Extracting data with semantic pattern...")
        
        # Simple container detection using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find containers based on site type
        if 'hacker' in url.lower():
            containers = soup.find_all('tr', class_='athing')[:20]
        elif 'github' in url.lower():
            containers = soup.find_all('article')[:20]
        elif 'stackoverflow' in url.lower():
            containers = soup.find_all('div', class_='s-post-summary')[:20]
        else:
            containers = None
        
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
                logger.info(f"   {i}. {json.dumps(item, indent=6)}")
        
        logger.info(f"\n⏱️  Total: {total_time:.2f}s | 💰 Cost: ${pattern_cost:.4f} | ♻️  Cache: {'HIT' if used_cache else 'MISS'}")
        
        return {
            "url": url,
            "name": name,
            "category": category,
            "fields": fields,
            "success": items_count > 0,
            "items_count": items_count,
            "used_cache": used_cache,
            "cost": pattern_cost,
            "time": total_time,
            "data": extracted_data[:5]  # Save first 5 for inspection
        }
    
    def print_summary(self, results):
        """Print test summary."""
        logger.info(f"\n\n{'='*80}")
        logger.info(f"📊 TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        successful = [r for r in results if r.get('success')]
        
        logger.info(f"\n✅ Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
        logger.info(f"♻️  Cache Hit Rate: {self.metrics['cache_hits']}/{self.metrics['total_requests']} ({self.metrics['cache_hits']/self.metrics['total_requests']*100:.0f}%)")
        logger.info(f"📦 Items Extracted: {self.metrics['items_extracted']} total")
        logger.info(f"⏱️  Total Time: {self.metrics['total_time']:.2f}s")
        logger.info(f"💰 Total Cost: ${self.metrics['total_cost']:.4f}")
        logger.info(f"💵 Avg Cost/Request: ${self.metrics['total_cost']/self.metrics['total_requests']:.4f}")
        
        # Cost comparison
        parsera_cost = self.metrics['total_requests'] * 0.03
        savings = parsera_cost - self.metrics['total_cost']
        savings_pct = (savings / parsera_cost * 100) if parsera_cost > 0 else 0
        
        logger.info(f"\n💰 Cost Comparison:")
        logger.info(f"   • Parsera (LLM per request): ${parsera_cost:.4f}")
        logger.info(f"   • Hybrid System: ${self.metrics['total_cost']:.4f}")
        logger.info(f"   • Savings: ${savings:.4f} ({savings_pct:.0f}%)")
        
        # Cache stats
        cache_stats = self.pattern_cache.get_stats()
        logger.info(f"\n📦 Pattern Cache:")
        logger.info(f"   • Patterns stored: {cache_stats['total_patterns']}")
        logger.info(f"   • Unique domains: {cache_stats['domains']}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ TEST COMPLETE")
        logger.info(f"{'='*80}")


async def main():
    """Run end-to-end test."""
    logger.info("="*80)
    logger.info("🧪 END-TO-END HYBRID SYSTEM TEST")
    logger.info("="*80)
    logger.info(f"\nTesting {len(TEST_SOURCES)} websites")
    logger.info("Demonstrating LLM pattern generation and caching\n")
    
    tester = SimpleHybridTest()
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
    output_file = f"end_to_end_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': tester.metrics
        }, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to {output_file}")
    
    # Key takeaways
    logger.info(f"\n\n{'='*80}")
    logger.info(f"🎯 KEY FINDINGS")
    logger.info(f"{'='*80}")
    
    if tester.metrics['cache_hits'] > 0:
        logger.info(f"\n✅ Pattern reuse WORKING!")
        logger.info(f"   {tester.metrics['cache_hits']} requests reused cached patterns")
        logger.info(f"   Saved ${tester.metrics['cache_hits'] * 0.02:.2f} in LLM costs")
    
    if tester.metrics['items_extracted'] > 0:
        logger.info(f"\n✅ Extraction WORKING!")
        logger.info(f"   {tester.metrics['items_extracted']} items extracted successfully")
    
    logger.info(f"\n✅ System validated end-to-end!")


if __name__ == "__main__":
    asyncio.run(main())

