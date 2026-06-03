"""
Simple test for Structural Embedding - No LLM required

This tests just the structural embedding generation and similarity matching.
Shows that similar websites (e.g., e-commerce sites) have similar embeddings.
"""

import logging
import json
import numpy as np
from typing import List, Dict

from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.html_fetcher import HTMLFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test websites grouped by type
TEST_WEBSITES = {
    "forum": [
        "https://news.ycombinator.com",
        "https://reddit.com/r/python",
        "https://stackoverflow.com/questions"
    ],
    "ecommerce": [
        "https://www.amazon.com/s?k=laptop",
        "https://www.ebay.com/sch/i.html?_nkw=laptop"
    ],
    "listing": [
        "https://github.com/trending",
        "https://www.imdb.com/chart/top"
    ]
}


def main():
    """Test structural embedding generation and similarity."""
    
    logger.info("="*80)
    logger.info("🧬 STRUCTURAL EMBEDDING TEST")
    logger.info("="*80)
    
    # Initialize components
    embedding_gen = StructuralEmbedding(embedding_dim=512)
    fetcher = HTMLFetcher()
    
    # Store embeddings
    all_embeddings: Dict[str, Dict] = {}
    
    # Generate embeddings for all websites
    for site_type, urls in TEST_WEBSITES.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"📂 Processing {site_type.upper()} websites")
        logger.info(f"{'='*80}")
        
        for url in urls:
            logger.info(f"\n🌐 {url}")
            
            try:
                # Fetch HTML
                result = fetcher.fetch(url)
                if not result or 'html' not in result:
                    logger.error(f"  ❌ Failed to fetch")
                    continue
                
                html = result['html']
                logger.info(f"  ✓ Fetched HTML ({len(html):,} bytes)")
                
                # Generate embedding
                embedding = embedding_gen.generate(html)
                logger.info(f"  ✓ Generated embedding (dim={len(embedding)})")
                
                # Store
                all_embeddings[url] = {
                    'type': site_type,
                    'embedding': embedding,
                    'url': url
                }
                
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
    
    logger.info(f"\n\n{'='*80}")
    logger.info(f"📊 SIMILARITY ANALYSIS")
    logger.info(f"={'='*80}")
    
    # Calculate all pairwise similarities
    urls = list(all_embeddings.keys())
    similarity_matrix = {}
    
    for i, url1 in enumerate(urls):
        for j, url2 in enumerate(urls):
            if i >= j:
                continue  # Skip diagonal and duplicate pairs
            
            emb1 = all_embeddings[url1]['embedding']
            emb2 = all_embeddings[url2]['embedding']
            
            similarity = embedding_gen.compute_similarity(emb1, emb2)
            
            key = f"{url1} <-> {url2}"
            similarity_matrix[key] = {
                'url1': url1,
                'url2': url2,
                'type1': all_embeddings[url1]['type'],
                'type2': all_embeddings[url2]['type'],
                'similarity': similarity,
                'same_type': all_embeddings[url1]['type'] == all_embeddings[url2]['type']
            }
    
    # Sort by similarity
    sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1]['similarity'], reverse=True)
    
    # Display results
    logger.info(f"\n🔝 TOP 10 MOST SIMILAR PAIRS:")
    logger.info(f"-"*80)
    
    for i, (key, data) in enumerate(sorted_pairs[:10], 1):
        same_type_indicator = "✅" if data['same_type'] else "  "
        url1_short = data['url1'].split('/')[2]
        url2_short = data['url2'].split('/')[2]
        
        logger.info(f"{i:2d}. {same_type_indicator} {data['similarity']:.3f} | "
                   f"{data['type1']:10s} | {url1_short:25s} <-> {url2_short}")
    
    logger.info(f"\n\n🔻 LEAST SIMILAR PAIRS:")
    logger.info(f"-"*80)
    
    for i, (key, data) in enumerate(sorted_pairs[-10:], 1):
        same_type_indicator = "❌" if not data['same_type'] else "  "
        url1_short = data['url1'].split('/')[2]
        url2_short = data['url2'].split('/')[2]
        
        logger.info(f"{i:2d}. {same_type_indicator} {data['similarity']:.3f} | "
                   f"{data['type1']:10s} | {url1_short:25s} <-> {url2_short}")
    
    # Group by type
    logger.info(f"\n\n📈 SIMILARITY BY TYPE:")
    logger.info(f"-"*80)
    
    same_type_similarities = [d['similarity'] for d in similarity_matrix.values() if d['same_type']]
    diff_type_similarities = [d['similarity'] for d in similarity_matrix.values() if not d['same_type']]
    
    if same_type_similarities:
        logger.info(f"Same Type (e.g., both e-commerce):")
        logger.info(f"  • Min:  {min(same_type_similarities):.3f}")
        logger.info(f"  • Max:  {max(same_type_similarities):.3f}")
        logger.info(f"  • Avg:  {np.mean(same_type_similarities):.3f}")
        logger.info(f"  • Count: {len(same_type_similarities)}")
    
    if diff_type_similarities:
        logger.info(f"\nDifferent Types (e.g., e-commerce vs forum):")
        logger.info(f"  • Min:  {min(diff_type_similarities):.3f}")
        logger.info(f"  • Max:  {max(diff_type_similarities):.3f}")
        logger.info(f"  • Avg:  {np.mean(diff_type_similarities):.3f}")
        logger.info(f"  • Count: {len(diff_type_similarities)}")
    
    # Calculate separation score
    if same_type_similarities and diff_type_similarities:
        avg_same = np.mean(same_type_similarities)
        avg_diff = np.mean(diff_type_similarities)
        separation = avg_same - avg_diff
        
        logger.info(f"\n🎯 SEPARATION SCORE: {separation:.3f}")
        logger.info(f"   (Higher is better - indicates embeddings cluster by type)")
        
        if separation > 0.15:
            logger.info(f"   ✅ EXCELLENT: Strong clustering by website type!")
        elif separation > 0.10:
            logger.info(f"   ✓ GOOD: Embeddings show clear type patterns")
        elif separation > 0.05:
            logger.info(f"   ⚠️  MODERATE: Some type clustering visible")
        else:
            logger.info(f"   ❌ POOR: Embeddings don't cluster well by type")
    
    # Threshold analysis
    logger.info(f"\n\n🎚️  THRESHOLD ANALYSIS:")
    logger.info(f"-"*80)
    logger.info(f"If we use similarity threshold = 0.85 for pattern reuse:")
    
    threshold = 0.85
    same_type_above = sum(1 for s in same_type_similarities if s >= threshold)
    diff_type_above = sum(1 for s in diff_type_similarities if s >= threshold)
    
    logger.info(f"  • Same-type pairs above threshold: {same_type_above}/{len(same_type_similarities)} "
               f"({same_type_above/max(len(same_type_similarities),1)*100:.1f}%)")
    logger.info(f"  • Diff-type pairs above threshold: {diff_type_above}/{len(diff_type_similarities)} "
               f"({diff_type_above/max(len(diff_type_similarities),1)*100:.1f}%)")
    
    if same_type_above > 0:
        logger.info(f"\n  ✅ Pattern reuse would work for similar websites!")
    else:
        logger.info(f"\n  ⚠️  May need to lower threshold for pattern reuse")
        # Find optimal threshold
        for test_threshold in [0.80, 0.75, 0.70, 0.65]:
            same_above = sum(1 for s in same_type_similarities if s >= test_threshold)
            if same_above > 0:
                logger.info(f"  💡 Try threshold = {test_threshold}: "
                           f"{same_above}/{len(same_type_similarities)} same-type pairs would match")
                break
    
    logger.info(f"\n\n{'='*80}")
    logger.info(f"✅ TEST COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nKey Findings:")
    logger.info(f"  • Successfully generated {len(all_embeddings)} structural embeddings")
    if same_type_similarities:
        logger.info(f"  • Similar sites have avg similarity: {np.mean(same_type_similarities):.3f}")
    if diff_type_similarities:
        logger.info(f"  • Different sites have avg similarity: {np.mean(diff_type_similarities):.3f}")
    logger.info(f"\nConclusion:")
    if separation > 0.10:
        logger.info(f"  ✅ Structural embeddings successfully cluster websites by type!")
        logger.info(f"  ✅ Pattern reuse across similar websites should work well!")
    else:
        logger.info(f"  ⚠️  Embeddings show some clustering but may need refinement")
    
    # Save results
    results = {
        'embeddings': {url: {'type': data['type']} for url, data in all_embeddings.items()},
        'similarity_matrix': {k: {**v, 'similarity': float(v['similarity'])} for k, v in similarity_matrix.items()},
        'stats': {
            'same_type_avg': float(np.mean(same_type_similarities)) if same_type_similarities else 0,
            'diff_type_avg': float(np.mean(diff_type_similarities)) if diff_type_similarities else 0,
            'separation': float(separation) if same_type_similarities and diff_type_similarities else 0
        }
    }
    
    with open('structural_embedding_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Results saved to structural_embedding_test_results.json")


if __name__ == "__main__":
    main()




