#!/usr/bin/env python3
"""
Local Testing Script for 3 URLs with Bright Data Proxies and Web Unblocker

Tests:
1. ProductHunt: https://www.producthunt.com/categories/vibe-coding
2. Metacritic: https://www.metacritic.com/pictures/worst-movies-of-2025/
3. Leafly: https://www.leafly.com/dispensary-info/the-grove---pahrump/menu

Uses:
- Bright Data Residential Proxies
- Bright Data Web Unblocker (fallback)
- OpenAI for extraction
"""

import json
import logging
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_three_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = "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"

# Bright Data Residential Proxy
BRIGHT_DATA_RESIDENTIAL = {
    'server': 'http://brd.superproxy.io:33335',
    'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
    'password': 'rs2mvj79xi2t'
}

# Bright Data Web Unblocker (proxy format: host,port,username,password)
WEB_UNBLOCKER_CREDENTIALS = "brd.superproxy.io,33335,brd-customer-hl_803e8195-zone-web_unlocker1,t8mhp1qev1i1"

# Test URLs with expected fields
TEST_URLS = [
    {
        'url': 'https://www.producthunt.com/categories/vibe-coding',
        'name': 'ProductHunt',
        'description': 'ProductHunt coding products category',
        'expected_fields': ['name', 'tagline', 'votes', 'comments', 'maker', 'image', 'url'],
        'extraction_context': 'Extract product listings with name, tagline, vote count, comment count, maker/creator information (if available in the data), image URL (construct from thumbnailImageUuid if needed: https://ph-files.imgix.net/{uuid}), and product URL. Note: maker and image may not be visible on category listing pages - extract what is available.'
    },
    {
        'url': 'https://www.metacritic.com/pictures/worst-movies-of-2025/',
        'name': 'Metacritic',
        'description': 'Metacritic worst movies of 2025',
        'expected_fields': ['title', 'year', 'score', 'platform', 'image', 'url', 'description'],
        'extraction_context': 'Extract movie listings with title, year, critic score, platform, image URL, movie URL, and description'
    },
    {
        'url': 'https://www.leafly.com/dispensary-info/the-grove---pahrump/menu',
        'name': 'Leafly',
        'description': 'Leafly dispensary menu',
        'expected_fields': ['name', 'type', 'price', 'thc', 'cbd', 'image', 'description', 'effects'],
        'extraction_context': 'Extract cannabis product listings with name, product type, price, THC percentage, CBD percentage, image URL, description, and effects'
    }
]


def analyze_results(result: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze extraction results and provide quality metrics with 90% threshold tracking"""
    # Get quality score from metadata if available (more accurate than field coverage average)
    metadata = result.get('metadata', {})
    scraper_quality = metadata.get('quality_score') or metadata.get('direct_llm_quality') or metadata.get('quality')
    
    analysis = {
        'success': len(result.get('data', [])) > 0,  # Success = items extracted
        'items_extracted': len(result.get('data', [])),
        'expected_fields': test_config['expected_fields'],
        'field_coverage': {},
        'quality_score': 0.0,
        'quality_from_scraper': scraper_quality,  # Track scraper's own quality metric
        'meets_90_percent_threshold': False,
        'issues': [],
        'cache_info': {},  # Track cache usage
        'missing_fields': [],
        'low_coverage_fields': []
    }
    
    # Extract cache information from metadata
    if metadata:
        analysis['cache_info'] = {
            'code_cached': metadata.get('code_cached'),
            'pattern_cached': metadata.get('pattern_cached'),
            'direct_llm_cached': metadata.get('direct_llm_cached'),
            'extraction_source': metadata.get('extraction_source', 'unknown'),
            'early_exit': metadata.get('early_exit', False)
        }
    
    if not result.get('data'):
        analysis['issues'].append('No data extracted')
        return analysis
    
    # Check field coverage
    all_fields_found = set()
    for item in result['data']:
        if isinstance(item, dict):
            all_fields_found.update(item.keys())
    
    for field in test_config['expected_fields']:
        found_count = sum(1 for item in result['data'] if isinstance(item, dict) and field in item and item[field] and str(item[field]).strip() not in ['None', 'null', ''])
        coverage = found_count / len(result['data']) if result['data'] else 0
        analysis['field_coverage'][field] = {
            'coverage': coverage,
            'items_with_field': found_count,
            'total_items': len(result['data'])
        }
        
        # Track missing or low coverage fields
        if coverage == 0:
            analysis['missing_fields'].append(field)
        elif coverage < 0.5:
            analysis['low_coverage_fields'].append(field)
    
    # Use scraper's quality score if available, otherwise calculate from field coverage
    if scraper_quality is not None:
        analysis['quality_score'] = scraper_quality / 100.0 if scraper_quality > 1.0 else scraper_quality
    else:
        # Calculate quality score (average field coverage)
        if analysis['field_coverage']:
            avg_coverage = sum(f['coverage'] for f in analysis['field_coverage'].values()) / len(analysis['field_coverage'])
            analysis['quality_score'] = avg_coverage
    
    # Check if meets 90% threshold
    analysis['meets_90_percent_threshold'] = analysis['quality_score'] >= 0.90
    
    # Identify issues with 90% threshold in mind
    if not analysis['meets_90_percent_threshold']:
        gap = 0.90 - analysis['quality_score']
        analysis['issues'].append(f"QUALITY BELOW 90% THRESHOLD: {analysis['quality_score']:.1%} (gap: {gap:.1%})")
    
    for field, coverage_data in analysis['field_coverage'].items():
        if coverage_data['coverage'] < 0.5:
            analysis['issues'].append(f"Low coverage for '{field}': {coverage_data['coverage']:.1%}")
    
    if analysis['missing_fields']:
        analysis['issues'].append(f"Missing fields: {', '.join(analysis['missing_fields'])}")
    
    if analysis['items_extracted'] == 0:
        analysis['issues'].append('No items extracted')
    elif analysis['items_extracted'] < 3:
        analysis['issues'].append(f'Very few items extracted: {analysis["items_extracted"]}')
    
    return analysis


async def test_url(test_config: Dict[str, Any], use_web_unblocker: bool = False) -> Dict[str, Any]:
    """Test a single URL"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {test_config['name']}")
    logger.info(f"URL: {test_config['url']}")
    logger.info(f"{'='*80}")
    
    # Configure proxy - use Web Unblocker if specified, otherwise residential
    proxy_config = None
    web_unblocker_key = None
    web_unblocker_zone = "web_unlocker1"
    
    if use_web_unblocker:
        logger.info("Using Bright Data Web Unblocker")
        # Web Unblocker uses proxy format but different zone
        # Parse the credentials
        parts = WEB_UNBLOCKER_CREDENTIALS.split(',')
        if len(parts) >= 4:
            proxy_config = {
                'server': f'http://{parts[0]}:{parts[1]}',
                'username': parts[2],
                'password': parts[3]
            }
            web_unblocker_key = WEB_UNBLOCKER_CREDENTIALS  # Pass as API key format
            web_unblocker_zone = parts[2].split('-zone-')[-1] if '-zone-' in parts[2] else "web_unlocker1"
    else:
        logger.info("Using Bright Data Residential Proxy")
        proxy_config = BRIGHT_DATA_RESIDENTIAL
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=OPENAI_API_KEY,
        model_name='gpt-4o-mini',  # Fast and cost-effective
        proxy_config=proxy_config,
        web_unblocker_api_key=web_unblocker_key,  # Web Unblocker credentials
        web_unblocker_zone=web_unblocker_zone,
        fetch_mode='hybrid',  # Auto-detect best method
        headless=True,
        browser_timeout=180000,  # 3 minutes for slow-loading pages
        use_camoufox=True,  # Better anti-detection
        use_direct_llm=True,  # Use Direct LLM extraction
        quality_mode='balanced',  # Balanced quality
        enable_cache=True,  # Enable caching for faster subsequent runs
        enable_warming=True,  # Warm session for better success
        enable_auto_pagination=False,  # DISABLE pagination - only test single pages
        extraction_context=test_config.get('extraction_context'),  # Context for better extraction
        enable_context_validation=True,  # Validate extracted data
        log_level=logging.INFO
    )
    
    try:
        start_time = time.time()
        
        # Scrape the URL
        logger.info(f"Starting scrape...")
        result = await scraper.scrape(
            url=test_config['url'],
            fields=test_config['expected_fields']
        )
        
        elapsed_time = time.time() - start_time
        
        # Analyze results
        analysis = analyze_results(result, test_config)
        
        # Prepare output
        output = {
            'test_config': {
                'name': test_config['name'],
                'url': test_config['url'],
                'description': test_config['description']
            },
            'result': result,
            'analysis': analysis,
            'performance': {
                'elapsed_time': elapsed_time,
                'items_per_second': analysis['items_extracted'] / elapsed_time if elapsed_time > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Log summary with quality tracking
        logger.info(f"\n{'='*80}")
        logger.info(f"Results for {test_config['name']}:")
        logger.info(f"  Success: {analysis['success']}")
        logger.info(f"  Items Extracted: {analysis['items_extracted']}")
        logger.info(f"  Quality Score: {analysis['quality_score']:.1%}")
        
        # Quality threshold check
        if analysis['meets_90_percent_threshold']:
            logger.info(f"  ✅ QUALITY: Meets 90% threshold")
        else:
            gap = 0.90 - analysis['quality_score']
            logger.error(f"  ❌ QUALITY: Below 90% threshold (gap: {gap:.1%})")
        
        logger.info(f"  Time: {elapsed_time:.2f}s")
        logger.info(f"  Items/sec: {output['performance']['items_per_second']:.2f}")
        
        # Cache usage tracking
        cache_info = analysis.get('cache_info', {})
        if cache_info:
            logger.info(f"\n  Cache Usage:")
            logger.info(f"    Extraction Source: {cache_info.get('extraction_source', 'unknown')}")
            logger.info(f"    Code Cached: {cache_info.get('code_cached', 'N/A')}")
            logger.info(f"    Pattern Cached: {cache_info.get('pattern_cached', 'N/A')}")
            logger.info(f"    Direct LLM Cached: {cache_info.get('direct_llm_cached', 'N/A')}")
            logger.info(f"    Early Exit: {cache_info.get('early_exit', False)}")
        
        # Field coverage details
        if analysis.get('missing_fields'):
            logger.warning(f"  Missing Fields: {', '.join(analysis['missing_fields'])}")
        if analysis.get('low_coverage_fields'):
            logger.warning(f"  Low Coverage Fields: {', '.join(analysis['low_coverage_fields'])}")
        
        if analysis['issues']:
            logger.warning(f"\n  Issues:")
            for issue in analysis['issues']:
                logger.warning(f"    - {issue}")
        
        # Show sample data
        if result.get('data'):
            logger.info(f"\n  Sample Data (first item):")
            sample = result['data'][0]
            for key, value in list(sample.items())[:5]:  # Show first 5 fields
                value_str = str(value)[:100] if value else 'None'
                logger.info(f"    {key}: {value_str}")
        
        return output
        
    except Exception as e:
        logger.error(f"Error scraping {test_config['name']}: {str(e)}", exc_info=True)
        return {
            'test_config': {
                'name': test_config['name'],
                'url': test_config['url'],
                'description': test_config['description']
            },
            'error': str(e),
            'success': False,
            'timestamp': datetime.now().isoformat()
        }
    finally:
        await scraper.close()


async def main():
    """Main test function"""
    logger.info("="*80)
    logger.info("Universal Scraper - Local Testing for 3 URLs")
    logger.info("="*80)
    logger.info(f"OpenAI API Key: {OPENAI_API_KEY[:20]}...")
    logger.info(f"Bright Data Residential: {BRIGHT_DATA_RESIDENTIAL['server']}")
    logger.info(f"Bright Data Web Unblocker: {WEB_UNBLOCKER_CREDENTIALS.split(',')[0]}")
    logger.info("="*80)
    
    all_results = []
    
    # Test each URL
    for i, test_config in enumerate(TEST_URLS, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"TEST {i}/{len(TEST_URLS)}: {test_config['name']}")
        logger.info(f"{'#'*80}")
        
        # Try with residential proxy first
        logger.info("\nAttempt 1: Using Residential Proxy")
        result = await test_url(test_config, use_web_unblocker=False)
        all_results.append(result)
        
        # If failed or quality below 90%, try with Web Unblocker
        analysis = result.get('analysis', {})
        if not analysis.get('success') or not analysis.get('meets_90_percent_threshold', False):
            logger.info("\nAttempt 2: Using Web Unblocker (fallback)")
            result_web = await test_url(test_config, use_web_unblocker=True)
            # Use the better result (prefer one that meets 90% threshold)
            analysis_web = result_web.get('analysis', {})
            current_meets_threshold = analysis.get('meets_90_percent_threshold', False)
            web_meets_threshold = analysis_web.get('meets_90_percent_threshold', False)
            
            if (web_meets_threshold and not current_meets_threshold) or \
               (analysis_web.get('quality_score', 0) > analysis.get('quality_score', 0) + 0.1):
                all_results[-1] = result_web
                logger.info("Web Unblocker produced better results, using that")
        
        # Small delay between tests
        if i < len(TEST_URLS):
            logger.info("\nWaiting 5 seconds before next test...")
            await asyncio.sleep(5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_three_urls_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    # Quality tracking summary
    quality_issues = []
    cache_reuse_summary = {}
    
    for result in all_results:
        name = result.get('test_config', {}).get('name', 'Unknown')
        analysis = result.get('analysis', {})
        success = analysis.get('success', False)
        items = analysis.get('items_extracted', 0)
        quality = analysis.get('quality_score', 0)
        meets_threshold = analysis.get('meets_90_percent_threshold', False)
        
        # Track quality issues
        if not meets_threshold:
            quality_issues.append({
                'site': name,
                'quality': quality,
                'gap': 0.90 - quality,
                'missing_fields': analysis.get('missing_fields', []),
                'low_coverage_fields': analysis.get('low_coverage_fields', [])
            })
        
        # Track cache reuse
        cache_info = analysis.get('cache_info', {})
        if cache_info.get('code_cached') or cache_info.get('pattern_cached') or cache_info.get('direct_llm_cached'):
            cache_reuse_summary[name] = cache_info
        
        status = "✅" if meets_threshold else "⚠️" if success else "❌"
        threshold_status = "✅" if meets_threshold else "❌"
        logger.info(f"{status} {name}: {items} items, {quality:.1%} quality {threshold_status} (90% threshold)")
    
    # Log fundamental issues
    logger.info(f"\n{'='*80}")
    logger.info("FUNDAMENTAL ISSUES TRACKING")
    logger.info(f"{'='*80}")
    
    if quality_issues:
        logger.error(f"\n❌ QUALITY BELOW 90% THRESHOLD ({len(quality_issues)} sites):")
        for issue in quality_issues:
            logger.error(f"  • {issue['site']}: {issue['quality']:.1%} (gap: {issue['gap']:.1%})")
            if issue['missing_fields']:
                logger.error(f"    Missing: {', '.join(issue['missing_fields'])}")
            if issue['low_coverage_fields']:
                logger.error(f"    Low coverage: {', '.join(issue['low_coverage_fields'])}")
    else:
        logger.info("✅ All sites meet 90% quality threshold")
    
    if cache_reuse_summary:
        logger.info(f"\n📚 CACHE REUSE PATTERNS ({len(cache_reuse_summary)} sites):")
        for site, cache_info in cache_reuse_summary.items():
            logger.info(f"  • {site}:")
            logger.info(f"    - Source: {cache_info.get('extraction_source', 'unknown')}")
            logger.info(f"    - Code cached: {cache_info.get('code_cached', 'N/A')}")
            logger.info(f"    - Pattern cached: {cache_info.get('pattern_cached', 'N/A')}")
            logger.info(f"    - Direct LLM cached: {cache_info.get('direct_llm_cached', 'N/A')}")
    else:
        logger.warning("⚠️ No cache reuse detected - patterns not being repurposed")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("="*80)


if __name__ == '__main__':
    asyncio.run(main())

