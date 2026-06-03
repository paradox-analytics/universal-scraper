"""
Threshold Tuning Test - Find optimal similarity threshold for pattern reuse

Tests multiple thresholds (0.65, 0.70, 0.75, 0.80, 0.85) to find the best balance
between pattern reuse rate and false positive rate.
"""

import logging
import json
import numpy as np
from typing import Dict, List

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.html_fetcher import HTMLFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Extended test set with more diverse websites
TEST_WEBSITES = {
    "forum": [
        ("https://news.ycombinator.com", "Hacker News"),
        ("https://reddit.com/r/python", "Reddit"),
        ("https://stackoverflow.com/questions", "Stack Overflow"),
    ],
    "ecommerce": [
        ("https://www.amazon.com/s?k=laptop", "Amazon"),
        ("https://www.ebay.com/sch/i.html?_nkw=laptop", "eBay"),
    ],
    "listing": [
        ("https://github.com/trending", "GitHub Trending"),
        ("https://www.imdb.com/chart/top", "IMDB Top 250"),
    ]
}


def test_threshold(embeddings: Dict, threshold: float) -> Dict:
    """
    Test a specific threshold and calculate metrics.
    
    Returns dict with:
    - same_type_matches: Number of same-type pairs above threshold
    - diff_type_matches: Number of diff-type pairs above threshold
    - pattern_reuse_rate: Percentage of same-type pairs that would reuse patterns
    - false_positive_rate: Percentage of diff-type pairs incorrectly matched
    """
    embedding_gen = StructuralEmbedding()
    
    same_type_above = 0
    same_type_total = 0
    diff_type_above = 0
    diff_type_total = 0
    
    urls = list(embeddings.keys())
    
    for i, url1 in enumerate(urls):
        for j, url2 in enumerate(urls):
            if i >= j:
                continue
            
            emb1 = embeddings[url1]['embedding']
            emb2 = embeddings[url2]['embedding']
            type1 = embeddings[url1]['type']
            type2 = embeddings[url2]['type']
            
            similarity = embedding_gen.compute_similarity(emb1, emb2)
            same_type = type1 == type2
            
            if same_type:
                same_type_total += 1
                if similarity >= threshold:
                    same_type_above += 1
            else:
                diff_type_total += 1
                if similarity >= threshold:
                    diff_type_above += 1
    
    pattern_reuse_rate = (same_type_above / max(same_type_total, 1)) * 100
    false_positive_rate = (diff_type_above / max(diff_type_total, 1)) * 100
    
    return {
        'threshold': threshold,
        'same_type_matches': same_type_above,
        'same_type_total': same_type_total,
        'diff_type_matches': diff_type_above,
        'diff_type_total': diff_type_total,
        'pattern_reuse_rate': pattern_reuse_rate,
        'false_positive_rate': false_positive_rate,
        'score': pattern_reuse_rate - (false_positive_rate * 0.5)  # Penalize false positives
    }


def main():
    """Test multiple thresholds to find optimal value."""
    
    logger.info("="*80)
    logger.info("🎚️  THRESHOLD TUNING TEST")
    logger.info("="*80)
    
    # Initialize components
    embedding_gen = StructuralEmbedding(embedding_dim=512)
    fetcher = HTMLFetcher()
    
    # Generate embeddings for all websites
    logger.info("\n📥 Fetching websites and generating embeddings...")
    embeddings = {}
    
    for site_type, sites in TEST_WEBSITES.items():
        for url, name in sites:
            logger.info(f"  • {name}...")
            try:
                result = fetcher.fetch(url)
                if result and 'html' in result:
                    html = result['html']
                    embedding = embedding_gen.generate(html)
                    embeddings[url] = {
                        'type': site_type,
                        'name': name,
                        'embedding': embedding
                    }
                    logger.info(f"    ✓ Done ({len(html):,} bytes)")
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
    
    logger.info(f"\n✅ Generated {len(embeddings)} embeddings")
    
    # Test multiple thresholds
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING THRESHOLDS")
    logger.info("="*80)
    
    thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []
    
    for threshold in thresholds:
        logger.info(f"\n📊 Testing threshold = {threshold:.2f}")
        metrics = test_threshold(embeddings, threshold)
        results.append(metrics)
        
        logger.info(f"  • Same-type matches: {metrics['same_type_matches']}/{metrics['same_type_total']} "
                   f"({metrics['pattern_reuse_rate']:.1f}%)")
        logger.info(f"  • Diff-type matches: {metrics['diff_type_matches']}/{metrics['diff_type_total']} "
                   f"({metrics['false_positive_rate']:.1f}%)")
        logger.info(f"  • Score: {metrics['score']:.1f} (higher is better)")
    
    # Find optimal threshold
    logger.info("\n" + "="*80)
    logger.info("🎯 ANALYSIS & RECOMMENDATION")
    logger.info("="*80)
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    best = sorted_results[0]
    
    logger.info("\n📈 Results by Threshold:")
    logger.info("-"*80)
    logger.info(f"{'Threshold':<12} {'Reuse Rate':<15} {'False Pos':<15} {'Score':<10} {'Status'}")
    logger.info("-"*80)
    
    for r in results:
        indicator = "🏆 BEST" if r == best else "  "
        logger.info(f"{r['threshold']:.2f}         "
                   f"{r['pattern_reuse_rate']:>5.1f}%         "
                   f"{r['false_positive_rate']:>5.1f}%         "
                   f"{r['score']:>6.1f}     {indicator}")
    
    logger.info("\n" + "="*80)
    logger.info(f"🏆 OPTIMAL THRESHOLD: {best['threshold']:.2f}")
    logger.info("="*80)
    
    logger.info(f"\nAt threshold {best['threshold']:.2f}:")
    logger.info(f"  ✅ Pattern Reuse Rate: {best['pattern_reuse_rate']:.1f}%")
    logger.info(f"  ⚠️  False Positive Rate: {best['false_positive_rate']:.1f}%")
    logger.info(f"  📊 Overall Score: {best['score']:.1f}")
    
    # Provide recommendation
    logger.info("\n💡 RECOMMENDATION:")
    
    if best['pattern_reuse_rate'] >= 70:
        logger.info(f"  ✅ EXCELLENT: {best['threshold']:.2f} provides {best['pattern_reuse_rate']:.0f}% pattern reuse")
        logger.info(f"  Use this threshold in production for optimal cost savings.")
    elif best['pattern_reuse_rate'] >= 50:
        logger.info(f"  ✓ GOOD: {best['threshold']:.2f} provides {best['pattern_reuse_rate']:.0f}% pattern reuse")
        logger.info(f"  Consider using this threshold, or test with more websites to improve.")
    else:
        logger.info(f"  ⚠️  MODERATE: {best['threshold']:.2f} only provides {best['pattern_reuse_rate']:.0f}% pattern reuse")
        logger.info(f"  May need to improve embedding features further or collect more training data.")
    
    if best['false_positive_rate'] > 20:
        logger.info(f"\n  ⚠️  WARNING: False positive rate is {best['false_positive_rate']:.1f}%")
        logger.info(f"  Consider adding validation or using a higher threshold to reduce false matches.")
    
    # Cost projection
    logger.info("\n💰 COST PROJECTION (100,000 requests/month):")
    logger.info("-"*80)
    
    requests = 100000
    reuse_rate = best['pattern_reuse_rate'] / 100
    
    llm_calls = int(requests * (1 - reuse_rate))
    cached_calls = int(requests * reuse_rate)
    
    cost_llm = llm_calls * 0.02
    cost_cached = cached_calls * 0.0001
    total_cost = cost_llm + cost_cached
    cost_per_request = total_cost / requests
    
    logger.info(f"  • LLM calls: {llm_calls:,} × $0.02 = ${cost_llm:,.2f}")
    logger.info(f"  • Cached calls: {cached_calls:,} × $0.0001 = ${cost_cached:,.2f}")
    logger.info(f"  • Total cost: ${total_cost:,.2f}")
    logger.info(f"  • Cost per request: ${cost_per_request:.4f}")
    
    logger.info("\n  Compared to:")
    logger.info(f"    • Parsera: ${requests * 0.03:,.2f} (100% LLM)")
    logger.info(f"    • Savings: ${(requests * 0.03) - total_cost:,.2f} ({((requests * 0.03 - total_cost) / (requests * 0.03)) * 100:.0f}%)")
    
    # Save results
    output = {
        'test_date': '2025-11-16',
        'websites_tested': len(embeddings),
        'thresholds_tested': [r['threshold'] for r in results],
        'results': results,
        'optimal_threshold': best['threshold'],
        'optimal_reuse_rate': best['pattern_reuse_rate'],
        'optimal_false_positive_rate': best['false_positive_rate'],
        'recommendation': 'production_ready' if best['pattern_reuse_rate'] >= 70 else 'needs_improvement'
    }
    
    with open('threshold_tuning_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    logger.info("\n💾 Results saved to threshold_tuning_results.json")
    
    logger.info("\n" + "="*80)
    logger.info("✅ THRESHOLD TUNING COMPLETE")
    logger.info("="*80)
    
    # Final recommendation for code
    logger.info("\n📝 TO IMPLEMENT:")
    logger.info(f"""
Update pattern_cache.py:

    self.similarity_threshold = {best['threshold']:.2f}  # Optimized threshold

Or in UniversalScraper initialization:

    scraper = UniversalScraper(
        pattern_similarity_threshold={best['threshold']:.2f},  # Optimized
        enable_pattern_reuse=True
    )
""")


if __name__ == "__main__":
    main()




