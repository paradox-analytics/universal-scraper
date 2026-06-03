"""
Test Hybrid Solution POC - Structural Embeddings + Semantic Patterns

This tests the proof of concept for the universal + cacheable approach:
1. Generate structural embeddings from HTML
2. Find similar websites using vector similarity
3. Generate semantic patterns (LLM-based, but cached)
4. Extract data using semantic patterns (no LLM)

Expected Results:
- First request: Uses LLM to generate pattern (~15-35s, ~$0.02)
- Similar websites: Reuse cached pattern (~1-3s, ~$0.0001)
- Success rate: 90-95% on new websites
- Pattern reuse rate: 85%+ for similar sites
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any
import numpy as np

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.html_fetcher import HTMLFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test websites (diverse structural patterns)
TEST_WEBSITES = [
    {
        "url": "https://news.ycombinator.com",
        "fields": ["title", "points", "url"],
        "type": "forum",
        "description": "Hacker News - Forum/List layout"
    },
    {
        "url": "https://www.amazon.com/s?k=laptop",
        "fields": ["title", "price", "rating"],
        "type": "ecommerce",
        "description": "Amazon - E-commerce layout"
    },
    {
        "url": "https://www.ebay.com/sch/i.html?_nkw=laptop",
        "fields": ["title", "price", "condition"],
        "type": "ecommerce",
        "description": "eBay - E-commerce layout (similar to Amazon)"
    },
    {
        "url": "https://reddit.com/r/python",
        "fields": ["title", "upvotes", "url"],
        "type": "forum",
        "description": "Reddit - Forum/List layout (similar to HN)"
    },
    {
        "url": "https://github.com/trending",
        "fields": ["name", "description", "stars"],
        "type": "listing",
        "description": "GitHub - Repository listing"
    },
    {
        "url": "https://www.producthunt.com",
        "fields": ["name", "tagline", "upvotes"],
        "type": "listing",
        "description": "Product Hunt - Product listing"
    },
    {
        "url": "https://stackoverflow.com/questions",
        "fields": ["title", "votes", "answers"],
        "type": "forum",
        "description": "Stack Overflow - Q&A forum (similar to Reddit/HN)"
    },
    {
        "url": "https://www.etsy.com/search?q=handmade",
        "fields": ["title", "price", "seller"],
        "type": "ecommerce",
        "description": "Etsy - E-commerce (similar to Amazon/eBay)"
    },
    {
        "url": "https://www.imdb.com/chart/top",
        "fields": ["title", "rating", "year"],
        "type": "listing",
        "description": "IMDB - Movie listing"
    },
    {
        "url": "https://www.yelp.com/search?find_desc=restaurants",
        "fields": ["name", "rating", "price"],
        "type": "listing",
        "description": "Yelp - Business listing"
    }
]


class HybridScraperPOC:
    """
    Proof of Concept for Hybrid Solution.
    
    Demonstrates:
    - Structural embedding generation
    - Pattern caching and reuse
    - Semantic pattern generation
    - Semantic extraction
    """
    
    def __init__(self, api_key: str = None):
        """Initialize POC components."""
        self.embedding_generator = StructuralEmbedding(embedding_dim=512)
        self.pattern_generator = SemanticPatternGenerator(api_key=api_key)
        self.semantic_extractor = SemanticExtractor()
        self.pattern_cache = PatternCache(
            cache_dir="./cache/patterns_poc",
            similarity_threshold=0.85
        )
        self.html_fetcher = HTMLFetcher()
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        logger.info("🚀 Hybrid Scraper POC initialized")
    
    async def scrape(self, url: str, fields: List[str]) -> Dict[str, Any]:
        """
        Scrape a website using the hybrid approach.
        
        Args:
            url: Website URL
            fields: Fields to extract
            
        Returns:
            Dict with results and metadata
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Scraping: {url}")
        logger.info(f"📋 Fields: {', '.join(fields)}")
        
        # Step 1: Fetch HTML
        logger.info("\n📥 Step 1: Fetching HTML...")
        result = self.html_fetcher.fetch(url)
        if not result or 'html' not in result:
            return {"error": "Failed to fetch HTML", "url": url}
        
        html = result['html']
        
        fetch_time = time.time() - start_time
        logger.info(f"   ✓ Fetched HTML ({len(html)} bytes) in {fetch_time:.2f}s")
        
        # Step 2: Generate structural embedding
        logger.info("\n🧬 Step 2: Generating structural embedding...")
        embed_start = time.time()
        embedding = self.embedding_generator.generate(html)
        embed_time = time.time() - embed_start
        logger.info(f"   ✓ Generated embedding (dim={len(embedding)}) in {embed_time:.3f}s")
        
        # Step 3: Search for similar cached pattern
        logger.info("\n🔍 Step 3: Searching for similar patterns...")
        search_start = time.time()
        similar_pattern = self.pattern_cache.find_similar_pattern(embedding, fields)
        search_time = time.time() - search_start
        
        used_cache = False
        pattern_id = None
        similarity_score = 0.0
        
        if similar_pattern:
            pattern_id, pattern, similarity_score = similar_pattern
            used_cache = True
            logger.info(f"   ✅ Found cached pattern: {pattern_id} (similarity={similarity_score:.3f})")
            logger.info(f"   💰 Cost saved: ~$0.02 (no LLM call needed)")
        else:
            logger.info("   ℹ️  No similar pattern found")
            
            # Step 4: Generate new semantic pattern (LLM call)
            logger.info("\n🎨 Step 4: Generating new semantic pattern with LLM...")
            pattern_start = time.time()
            
            # Clean HTML for pattern generation
            cleaned_html = self.html_cleaner.clean(html)
            
            # Detect repeating containers (helps pattern generation)
            containers = self.dom_detector.detect_patterns(html)
            container_sigs = [str(c.get('signature', '')) for c in containers[:5]] if containers else None
            
            # Generate pattern
            pattern = await self.pattern_generator.generate_pattern(
                html_sample=cleaned_html[:15000],
                fields=fields,
                repeating_containers=container_sigs
            )
            
            pattern_time = time.time() - pattern_start
            logger.info(f"   ✓ Generated pattern in {pattern_time:.2f}s")
            logger.info(f"   💰 Cost: ~$0.02 (LLM call)")
            
            # Save pattern to cache
            domain = url.split('/')[2]
            pattern_id = self.pattern_cache.save_pattern(
                pattern=pattern,
                embedding=embedding,
                domain=domain,
                fields=fields
            )
            logger.info(f"   💾 Saved pattern: {pattern_id}")
        
        # Step 5: Extract data using semantic pattern (NO LLM!)
        logger.info("\n⚡ Step 5: Extracting data with semantic pattern...")
        extract_start = time.time()
        
        # Detect containers for extraction
        containers = self.dom_detector.detect_patterns(html)
        container_elements = []
        if containers:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            for container in containers[:50]:  # Limit to 50 items
                for elem in soup.find_all():
                    if self._matches_signature(elem, container['signature']):
                        container_elements.append(elem)
                        break
        
        # Extract using semantic pattern
        extracted_data = self.semantic_extractor.extract(
            html=html,
            semantic_pattern=pattern,
            containers=container_elements[:50] if container_elements else None
        )
        
        extract_time = time.time() - extract_start
        total_time = time.time() - start_time
        
        logger.info(f"   ✓ Extracted {len(extracted_data)} items in {extract_time:.2f}s")
        logger.info(f"\n⏱️  Total time: {total_time:.2f}s")
        logger.info(f"🎯 Used cached pattern: {used_cache}")
        
        # Return results with metadata
        return {
            "url": url,
            "fields": fields,
            "data": extracted_data,
            "metadata": {
                "total_time": total_time,
                "fetch_time": fetch_time,
                "embed_time": embed_time,
                "search_time": search_time,
                "extract_time": extract_time,
                "used_cache": used_cache,
                "pattern_id": pattern_id,
                "similarity_score": similarity_score,
                "items_extracted": len(extracted_data)
            }
        }
    
    def _matches_signature(self, element, signature: str) -> bool:
        """Check if element matches container signature."""
        elem_sig = element.name
        if element.has_attr('class'):
            elem_sig += '.' + '.'.join(sorted(element['class'][:3]))
        return elem_sig == signature


async def main():
    """Run POC test on diverse websites."""
    logger.info("="*80)
    logger.info("🧪 HYBRID SOLUTION POC - Testing Universal + Cacheable Scraping")
    logger.info("="*80)
    
    # Initialize scraper
    scraper = HybridScraperPOC()
    
    # Test all websites
    results = []
    for i, site in enumerate(TEST_WEBSITES, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# Test {i}/{len(TEST_WEBSITES)}: {site['description']}")
        logger.info(f"{'#'*80}")
        
        try:
            result = await scraper.scrape(site['url'], site['fields'])
            results.append({
                **site,
                **result
            })
        except Exception as e:
            logger.error(f"❌ Error scraping {site['url']}: {e}")
            results.append({
                **site,
                "error": str(e)
            })
    
    # Analyze results
    logger.info("\n\n" + "="*80)
    logger.info("📊 POC RESULTS ANALYSIS")
    logger.info("="*80)
    
    # Calculate metrics
    total_sites = len(results)
    successful = [r for r in results if 'data' in r and len(r['data']) > 0]
    failed = [r for r in results if 'error' in r or ('data' in r and len(r['data']) == 0)]
    
    used_cache = [r for r in successful if r.get('metadata', {}).get('used_cache')]
    generated_new = [r for r in successful if not r.get('metadata', {}).get('used_cache')]
    
    success_rate = (len(successful) / total_sites) * 100
    cache_rate = (len(used_cache) / len(successful)) * 100 if successful else 0
    
    avg_time_cached = np.mean([r['metadata']['total_time'] for r in used_cache]) if used_cache else 0
    avg_time_new = np.mean([r['metadata']['total_time'] for r in generated_new]) if generated_new else 0
    
    logger.info(f"\n📈 Success Rate: {success_rate:.1f}% ({len(successful)}/{total_sites})")
    logger.info(f"♻️  Cache Reuse Rate: {cache_rate:.1f}% ({len(used_cache)}/{len(successful)})")
    logger.info(f"⏱️  Avg Time (Cached): {avg_time_cached:.2f}s")
    logger.info(f"⏱️  Avg Time (New Pattern): {avg_time_new:.2f}s")
    
    # Similarity analysis
    logger.info("\n🔍 Structural Similarity Analysis:")
    similarities = [r['metadata']['similarity_score'] for r in successful if r['metadata'].get('used_cache')]
    if similarities:
        logger.info(f"   • Min similarity: {min(similarities):.3f}")
        logger.info(f"   • Max similarity: {max(similarities):.3f}")
        logger.info(f"   • Avg similarity: {np.mean(similarities):.3f}")
    
    # Group by structural type
    logger.info("\n📂 Patterns by Type:")
    type_groups = {}
    for r in successful:
        site_type = r.get('type', 'unknown')
        if site_type not in type_groups:
            type_groups[site_type] = []
        type_groups[site_type].append(r)
    
    for site_type, sites in type_groups.items():
        cached = sum(1 for s in sites if s.get('metadata', {}).get('used_cache'))
        logger.info(f"   • {site_type}: {len(sites)} sites ({cached} reused patterns)")
    
    # Cost analysis
    logger.info("\n💰 Cost Analysis (estimated):")
    llm_calls = len(generated_new)
    cached_calls = len(used_cache)
    
    cost_new = llm_calls * 0.02  # $0.02 per LLM call
    cost_cached = cached_calls * 0.0001  # $0.0001 per cached extraction
    total_cost = cost_new + cost_cached
    
    logger.info(f"   • LLM calls: {llm_calls} × $0.02 = ${cost_new:.4f}")
    logger.info(f"   • Cached extractions: {cached_calls} × $0.0001 = ${cost_cached:.4f}")
    logger.info(f"   • Total cost: ${total_cost:.4f}")
    logger.info(f"   • Cost per request: ${total_cost/total_sites:.4f}")
    
    # Compare to alternatives
    logger.info("\n📊 Comparison to Alternatives (for these 10 requests):")
    logger.info(f"   • Parsera (LLM per request): ~${0.03 * total_sites:.2f}")
    logger.info(f"   • Our Current System: ~${0.005 * total_sites:.2f} (but fails on new sites)")
    logger.info(f"   • Hybrid Solution: ~${total_cost:.4f} ✅")
    
    # Pattern cache stats
    logger.info("\n📦 Pattern Cache Statistics:")
    cache_stats = scraper.pattern_cache.get_stats()
    logger.info(f"   • Total patterns: {cache_stats['total_patterns']}")
    logger.info(f"   • Unique domains: {cache_stats['domains']}")
    logger.info(f"   • Avg success rate: {cache_stats['avg_success_rate']:.2%}")
    
    # Show sample extracted data
    logger.info("\n📄 Sample Extracted Data:")
    for r in successful[:3]:
        logger.info(f"\n   {r['description']}:")
        sample_items = r['data'][:2]
        for item in sample_items:
            logger.info(f"      {json.dumps(item, indent=8)}")
    
    # Detailed results
    logger.info("\n\n" + "="*80)
    logger.info("📋 DETAILED RESULTS")
    logger.info("="*80)
    
    for i, r in enumerate(results, 1):
        status = "✅" if 'data' in r and len(r['data']) > 0 else "❌"
        cached = "♻️  (cached)" if r.get('metadata', {}).get('used_cache') else "🆕 (new pattern)"
        
        logger.info(f"\n{i}. {status} {r['description']}")
        logger.info(f"   URL: {r['url']}")
        logger.info(f"   Type: {r.get('type', 'unknown')} {cached if 'data' in r else ''}")
        
        if 'metadata' in r:
            m = r['metadata']
            logger.info(f"   Time: {m['total_time']:.2f}s (fetch={m['fetch_time']:.2f}s, extract={m['extract_time']:.2f}s)")
            logger.info(f"   Items: {m['items_extracted']}")
            if m.get('similarity_score'):
                logger.info(f"   Similarity: {m['similarity_score']:.3f}")
        
        if 'error' in r:
            logger.info(f"   Error: {r['error']}")
    
    # Summary
    logger.info("\n\n" + "="*80)
    logger.info("🎉 POC COMPLETE")
    logger.info("="*80)
    logger.info(f"\n✅ Success Rate: {success_rate:.1f}% (Target: 90-95%)")
    logger.info(f"♻️  Pattern Reuse: {cache_rate:.1f}% (Target: 85%)")
    logger.info(f"💰 Cost per Request: ${total_cost/total_sites:.4f} (Target: < $0.01)")
    
    if success_rate >= 90 and cache_rate >= 80:
        logger.info("\n🎯 POC SUCCESS! Ready for full implementation.")
    elif success_rate >= 70:
        logger.info("\n⚠️  POC partially successful. Needs refinement.")
    else:
        logger.info("\n❌ POC needs significant improvement.")
    
    # Save results to file
    with open('poc_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n💾 Results saved to poc_results.json")


if __name__ == "__main__":
    asyncio.run(main())

