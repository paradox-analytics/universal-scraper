"""
Test Hybrid System on Diverse Sources

Tests the complete hybrid pipeline with a variety of website types:
- E-commerce
- News/Blog
- Forum/Community
- Social Media
- Documentation
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


# Diverse test sources across different domains
TEST_SOURCES = [
    {
        "url": "https://news.ycombinator.com",
        "fields": ["title", "url", "points"],
        "category": "forum",
        "name": "Hacker News"
    },
    {
        "url": "https://www.producthunt.com",
        "fields": ["name", "description", "votes"],
        "category": "product_listing",
        "name": "Product Hunt"
    },
    {
        "url": "https://techcrunch.com",
        "fields": ["title", "description", "date"],
        "category": "news",
        "name": "TechCrunch"
    },
    {
        "url": "https://www.reddit.com/r/programming",
        "fields": ["title", "author", "upvotes"],
        "category": "forum",
        "name": "Reddit Programming"
    },
    {
        "url": "https://dev.to",
        "fields": ["title", "author", "reactions"],
        "category": "blog",
        "name": "Dev.to"
    },
]


class HybridSystemTest:
    """Test hybrid scraping system on diverse sources."""
    
    def __init__(self):
        # Check if API key is available
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if self.api_key:
            logger.info("✅ OpenAI API key found - will use LLM pattern generation")
        else:
            logger.warning("⚠️  No API key found - will use fallback patterns")
        
        self.embedding_gen = StructuralEmbedding()
        self.pattern_cache = PatternCache(
            cache_dir="./cache/patterns_diverse",
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
            'llm_calls': 0,
            'fallback_uses': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'items_extracted': 0,
            'errors': 0
        }
        
        logger.info("🚀 Hybrid System Test initialized")
    
    async def scrape(self, url: str, fields: list, category: str, name: str):
        """Scrape with the hybrid approach."""
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
                logger.info(f"   ℹ️  CACHE MISS - Generating new pattern...")
                
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
                if self.api_key:
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
                        used_llm = True
                        self.metrics['llm_calls'] += 1
                        
                        logger.info(f"      ✓ Pattern generated in {pattern_time:.2f}s")
                        logger.info(f"      💰 Cost: ~${pattern_cost:.4f}")
                        
                    except Exception as e:
                        logger.error(f"      ❌ LLM generation failed: {e}")
                        pattern = self.pattern_gen._generate_fallback_pattern(fields)
                        pattern_cost = 0.0
                        self.metrics['fallback_uses'] += 1
                        logger.info(f"      ⚠️  Using fallback pattern")
                else:
                    logger.info(f"      ⚠️  No API key - using fallback pattern")
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
                    logger.info(f"      💾 Saved to cache: {pattern_id}")
            
            # Step 4: Extract data
            logger.info("⚡ Step 4: Extracting data with semantic pattern...")
            
            # Simple container detection using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to find containers based on common patterns
            containers = None
            if 'hacker' in url.lower():
                containers = soup.find_all('tr', class_='athing')[:20]
            elif 'github' in url.lower():
                containers = soup.find_all('article')[:20]
            elif 'stackoverflow' in url.lower() or 'reddit' in url.lower():
                containers = soup.find_all(['div', 'article'], class_=lambda x: x and ('post' in x or 'summary' in x))[:20]
            elif 'producthunt' in url.lower():
                containers = soup.find_all(['div', 'article'], class_=lambda x: x and 'product' in x.lower() if x else False)[:20]
            elif 'techcrunch' in url.lower() or 'dev.to' in url.lower():
                containers = soup.find_all('article')[:20]
            
            if not containers:
                # Fallback: use body as single container
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
                logger.warning(f"   ⚠️  No items extracted - check pattern or containers")
            
            status_icon = "✅" if used_llm else "⚠️"
            method = "LLM" if used_llm else ("CACHE" if used_cache else "FALLBACK")
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
                "data": extracted_data[:5]  # Save first 5 for inspection
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
        logger.info(f"📊 TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        successful = [r for r in results if r.get('success')]
        
        logger.info(f"\n✅ Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
        logger.info(f"♻️  Cache Hit Rate: {self.metrics['cache_hits']}/{self.metrics['total_requests']} ({self.metrics['cache_hits']/self.metrics['total_requests']*100 if self.metrics['total_requests'] > 0 else 0:.0f}%)")
        logger.info(f"🤖 LLM Calls: {self.metrics['llm_calls']}")
        logger.info(f"📋 Fallback Uses: {self.metrics['fallback_uses']}")
        logger.info(f"📦 Items Extracted: {self.metrics['items_extracted']} total")
        logger.info(f"⏱️  Total Time: {self.metrics['total_time']:.2f}s")
        logger.info(f"💰 Total Cost: ${self.metrics['total_cost']:.4f}")
        
        if self.metrics['total_requests'] > 0:
            logger.info(f"💵 Avg Cost/Request: ${self.metrics['total_cost']/self.metrics['total_requests']:.4f}")
        
        # Cost comparison
        if self.api_key:
            parsera_cost = self.metrics['total_requests'] * 0.03
            savings = parsera_cost - self.metrics['total_cost']
            savings_pct = (savings / parsera_cost * 100) if parsera_cost > 0 else 0
            
            logger.info(f"\n💰 Cost Comparison (with LLM):")
            logger.info(f"   • Parsera (LLM per request): ${parsera_cost:.4f}")
            logger.info(f"   • Hybrid System: ${self.metrics['total_cost']:.4f}")
            logger.info(f"   • Savings: ${savings:.4f} ({savings_pct:.0f}%)")
        else:
            logger.info(f"\n💡 Using fallback patterns (no API key)")
            logger.info(f"   Set OPENAI_API_KEY to enable LLM pattern generation")
        
        # Cache stats
        cache_stats = self.pattern_cache.get_stats()
        logger.info(f"\n📦 Pattern Cache:")
        logger.info(f"   • Patterns stored: {cache_stats['total_patterns']}")
        logger.info(f"   • Unique domains: {cache_stats['domains']}")
        
        if self.metrics['errors'] > 0:
            logger.warning(f"\n⚠️  Errors: {self.metrics['errors']} requests failed")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ TEST COMPLETE")
        logger.info(f"{'='*80}")


async def main():
    """Run diverse sources test."""
    logger.info("="*80)
    logger.info("🧪 HYBRID SYSTEM TEST - DIVERSE SOURCES")
    logger.info("="*80)
    logger.info(f"\nTesting {len(TEST_SOURCES)} diverse websites")
    logger.info("Demonstrating universal extraction across different domains\n")
    
    # Check API key
    if os.environ.get('OPENAI_API_KEY'):
        logger.info("✅ OpenAI API key detected - will generate LLM patterns\n")
    else:
        logger.warning("⚠️  No OpenAI API key - will use fallback patterns")
        logger.info("   To enable LLM generation: export OPENAI_API_KEY='sk-...'\n")
    
    tester = HybridSystemTest()
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
    output_file = f"diverse_sources_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': tester.metrics
        }, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())




