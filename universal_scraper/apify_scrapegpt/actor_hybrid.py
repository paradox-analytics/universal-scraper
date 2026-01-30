"""
Universal Web Scraper - Hybrid Apify Actor

Combines LLM-powered pattern generation with vector-based caching for:
- Universal capability (works on ANY website)
- Cost efficiency (99.5% savings on cached requests)
- No configuration needed
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# CRITICAL: Set up Python path FIRST before any project imports
# This ensures imports work both locally and in Apify
script_dir = Path(__file__).parent.absolute()

# In Apify: actor.py is at /usr/src/app/actor.py
# universal_scraper is at /usr/src/app/universal_scraper/
# So we need /usr/src/app in the path (which is script_dir)

# In local dev: actor_hybrid.py is at .../universal_scraper/apify/actor_hybrid.py
# universal_scraper is at .../universal_scraper/
# So we need .../universal-scraper/ in the path (which is script_dir.parent.parent)

# Detect environment and set project_root accordingly
if 'apify' in str(script_dir):
    # Local development
    project_root = script_dir.parent.parent
else:
    # Apify environment - actor.py is at /usr/src/app/
    project_root = script_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Handle both package and local imports
try:
    from apify import Actor
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    print("  Apify SDK not available, running in standalone mode")

# Import hybrid system components using absolute imports
from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.pattern_cache import PatternCache
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from universal_scraper.core.semantic_extractor import SemanticExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher  # Universal: handles HTML, JS, JSON
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.dom_pattern_detector import DOMPatternDetector
from universal_scraper.core.json_detector import JSONDetector  # UNIVERSAL: Embedded JSON extraction

from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridScraper:
    """
    Hybrid Universal Scraper for Apify
    
    Features:
    - LLM-powered semantic pattern generation
    - ChromaDB vector-based pattern caching
    - Structural embeddings for similarity matching
    - Works on ANY website without configuration
    """
    
    def __init__(
        self,
        api_key: str,
        cache_dir: str = "./storage/pattern_cache",
        proxy_config: Optional[Dict[str, str]] = None,
        headless: bool = True
    ):
        """Initialize hybrid scraper"""
        self.api_key = api_key
        self.proxy_config = proxy_config
        self.headless = headless
        self.embedding_gen = StructuralEmbedding()
        self.pattern_cache = PatternCache(
            cache_dir=cache_dir,
            similarity_threshold=0.75
        )
        self.pattern_gen = SemanticPatternGenerator(api_key=api_key)
        self.semantic_extractor = SemanticExtractor()
        self.json_detector = JSONDetector()  # UNIVERSAL: Embedded JSON detection (__NEXT_DATA__, etc.)
        # UNIVERSAL FETCHER: Automatically handles HTML, JavaScript, and JSON
        self.hybrid_fetcher = HybridFetcher(
            proxy_config=proxy_config,
            headless=headless,
            use_camoufox=True,  # Use Camoufox for best anti-detection
            enable_cache=True
        )
        self.html_cleaner = SmartHTMLCleaner()
        self.dom_detector = DOMPatternDetector()
        
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'total_cost': 0.0,
            'items_extracted': 0
        }
        
        logger.info(" Hybrid Scraper initialized")
        logger.info(f"   Pattern cache: {self.pattern_cache.get_stats()['total_patterns']} patterns ready")
    
    async def scrape(self, url: str, fields: List[str]) -> Dict[str, Any]:
        """
        Scrape a URL using the hybrid approach
        
        Args:
            url: URL to scrape
            fields: List of fields to extract
            
        Returns:
            Dictionary with extracted data and metadata
        """
        import time
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        logger.info(f" Scraping: {url}")
        logger.info(f"   Fields: {', '.join(fields)}")
        
        try:
            # Step 1: Fetch HTML (Universal: auto-detects static HTML vs JS rendering)
            logger.info(" Fetching HTML with universal fetcher...")
            result = await self.hybrid_fetcher.fetch(url)
            if not result or 'html' not in result:
                return {"error": "Failed to fetch", "success": False}
            
            html = result['html']
            fetch_method = result.get('fetch_method', 'unknown')
            logger.info(f"    Fetched {len(html):,} bytes via {fetch_method}")
            
            # Step 1.5: UNIVERSAL JSON DETECTION (embedded + captured APIs)
            # This is the existing universal approach that works on ALL JS sites
            logger.info(" UNIVERSAL JSON DETECTION...")
            
            json_data_captured = result.get('json_data', [])
            json_detection_result = self.json_detector.detect_and_extract(
                html=html,
                url=url,
                captured_json=json_data_captured
            )
            
            if json_detection_result['json_found']:
                json_sources = json_detection_result['sources']
                all_json = json_detection_result['data']
                
                logger.info(f"    Found JSON from: {', '.join(json_sources)}")
                logger.info(f"    Total JSON sources: {len(all_json)}")
                
                # Try to extract items using the universal JSONDetector method
                try:
                    extracted_items = self.json_detector.extract_from_json(
                        json_data=all_json,
                        fields=fields
                    )
                    
                    if extracted_items and len(extracted_items) >= 3:
                        # Success! Use JSON data
                        total_time = time.time() - start_time
                        logger.info(f"    JSON SUCCESS! Extracted {len(extracted_items)} items")
                        logger.info(f"    Cost: $0.00 (no LLM needed!)")
                        logger.info(f"    Method: Universal JSON Detection")
                        logger.info(f"    Sources: {', '.join(json_sources)}")
                        
                        self.metrics['items_extracted'] += len(extracted_items)
                        
                        return {
                            "url": url,
                            "success": True,
                            "items_count": len(extracted_items),
                            "used_cache": False,
                            "extraction_method": "json_universal",
                            "json_sources": json_sources,
                            "cost": 0.0,
                            "time": total_time,
                            "data": extracted_items
                        }
                    else:
                        logger.info(f"   ℹ  JSON extraction found {len(extracted_items) if extracted_items else 0} items (needs ≥3)")
                        
                except Exception as e:
                    logger.warning(f"     JSON extraction failed: {e}")
            else:
                logger.info("   ℹ  No JSON detected, using HTML semantic extraction")
            
            # Step 2: Generate embedding
            logger.info(" Generating structural embedding...")
            embedding = self.embedding_gen.generate(html)
            
            # Step 3: Search cache
            logger.info(" Searching pattern cache...")
            similar = self.pattern_cache.find_similar_pattern(embedding, fields)
            
            pattern_cost = 0.0
            used_cache = False
            
            if similar:
                # CACHE HIT
                pattern_id, pattern, similarity = similar
                used_cache = True
                self.metrics['cache_hits'] += 1
                pattern_cost = 0.0001
                
                logger.info(f"    CACHE HIT! Pattern: {pattern_id}")
                logger.info(f"      Similarity: {similarity:.3f}")
                logger.info(f"       Saved $0.02")
                
            else:
                # CACHE MISS - Generate new pattern
                self.metrics['cache_misses'] += 1
                self.metrics['llm_calls'] += 1
                logger.info(f"   ℹ  CACHE MISS - Generating new pattern...")
                
                # Clean HTML
                clean_result = self.html_cleaner.clean(html)
                cleaned_html = clean_result['html']
                
                # Detect repeating containers
                dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
                repeating_containers = dom_patterns.get('repeating_containers', [])[:5] if dom_patterns else []
                
                # Generate semantic pattern with LLM
                logger.info(f"       Calling LLM...")
                try:
                    pattern = await self.pattern_gen.generate_pattern(
                        html_sample=cleaned_html[:15000],
                        fields=fields,
                        context=f"Extract {', '.join(fields)} from this website",
                        repeating_containers=repeating_containers if repeating_containers else None
                    )
                    pattern_cost = 0.02
                    
                    logger.info(f"       Pattern generated")
                    logger.info(f"       Cost: $0.02")
                    
                except Exception as e:
                    logger.error(f"       LLM generation failed: {e}")
                    pattern = self.pattern_gen._generate_fallback_pattern(fields)
                    pattern_cost = 0.0
                
                # Save pattern
                if self.pattern_gen.validate_pattern(pattern):
                    domain = url.split('/')[2]
                    pattern_id = self.pattern_cache.save_pattern(
                        pattern=pattern,
                        embedding=embedding,
                        domain=domain,
                        fields=fields
                    )
                    logger.info(f"       Cached: {pattern_id}")
            
            # Step 4: Extract data
            logger.info(" Extracting data...")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Simple container detection
            containers = soup.find_all(['article', 'div', 'li', 'tr'], 
                                      class_=lambda x: x and any(keyword in str(x).lower() 
                                      for keyword in ['item', 'product', 'post', 'story', 'listing', 'card', 'entry']))[:20]
            
            if not containers:
                containers = [soup.find('body')] if soup.find('body') else None
            
            extracted_data = self.semantic_extractor.extract(
                html=html,
                semantic_pattern=pattern,
                containers=containers
            )
            
            total_time = time.time() - start_time
            items_count = len(extracted_data)
            
            self.metrics['total_cost'] += pattern_cost
            self.metrics['items_extracted'] += items_count
            
            logger.info(f"    Extracted {items_count} items in {total_time:.2f}s")
            logger.info(f"   {'' if used_cache else ''} Cost: ${pattern_cost:.4f}")
            
            return {
                "url": url,
                "success": items_count > 0,
                "items_count": items_count,
                "used_cache": used_cache,
                "extraction_method": "html_semantic",
                "cost": pattern_cost,
                "time": total_time,
                "data": extracted_data
            }
            
        except Exception as e:
            logger.error(f" Error scraping {url}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scraping metrics"""
        cache_stats = self.pattern_cache.get_stats()
        return {
            **self.metrics,
            'cache_hit_rate': self.metrics['cache_hits'] / self.metrics['total_requests'] if self.metrics['total_requests'] > 0 else 0,
            'avg_cost_per_request': self.metrics['total_cost'] / self.metrics['total_requests'] if self.metrics['total_requests'] > 0 else 0,
            'patterns_cached': cache_stats['total_patterns'],
            'unique_domains': cache_stats['domains']
        }


async def main():
    """Main actor entry point"""
    if not APIFY_AVAILABLE:
        logger.error(" Apify SDK not available")
        return
    
    async with Actor:
        logger.info(' Hybrid Universal Scraper Actor started')
        
        # Get input
        actor_input = await Actor.get_input() or {}
        
        # Extract configuration
        urls = [req['url'] for req in actor_input.get('startUrls', [])]
        fields = actor_input.get('fields', '')
        api_key = actor_input.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("=" * 80)
            logger.warning("⚠️  NO OPENAI API KEY PROVIDED")
            logger.warning("=" * 80)
            logger.warning("")
            logger.warning("This actor requires an OpenAI API key to function properly.")
            logger.warning("")
            logger.warning("To provide your API key:")
            logger.warning("  1. Go to your actor input configuration")
            logger.warning("  2. Add 'openaiApiKey' field with your OpenAI API key")
            logger.warning("     OR set OPENAI_API_KEY environment variable")
            logger.warning("")
            logger.warning("Get your API key from: https://platform.openai.com/api-keys")
            logger.warning("")
            logger.warning("=" * 80)
            
            # Push informational message to dataset
            await Actor.push_data({
                '_message': 'OpenAI API key required',
                '_error': 'No OpenAI API key provided. Please add "openaiApiKey" to actor input or set OPENAI_API_KEY environment variable.',
                '_instruction': 'Get your API key from https://platform.openai.com/api-keys',
                '_success': False,
                '_action_required': 'Add openaiApiKey to actor input'
            })
            
            # Save metadata explaining the issue
            await Actor.set_value('OUTPUT_METADATA', {
                'status': 'completed',
                'message': 'OpenAI API key required',
                'total_urls': 0,
                'successful': 0,
                'total_items': 0,
                'error': 'OpenAI API key is required. Please add "openaiApiKey" to actor input or set OPENAI_API_KEY environment variable.',
                'instruction': 'Get your API key from https://platform.openai.com/api-keys'
            })
            
            logger.info(" Actor completed (API key required for actual scraping)")
            return  # Return successfully without failing
        
        if not urls:
            logger.error(" No URLs provided")
            await Actor.fail("At least one URL is required in startUrls")
            return
        
        # Handle fields: can be string (natural language) or array (field names)
        if not fields:
            logger.warning("  No fields specified - will extract common fields")
            fields = "Extract title, description, and URL"
        elif isinstance(fields, str):
            logger.info(f" Natural language fields: '{fields[:100]}...'")
        elif isinstance(fields, list):
            logger.info(f" Field list: {', '.join(fields)}")
        else:
            logger.warning(f"  Unknown fields type: {type(fields)}, using default")
            fields = "Extract title, description, and URL"
        
        # Extract proxy and browser settings
        apify_proxy_config = actor_input.get('proxyConfiguration')
        headless = actor_input.get('headless', True)
        
        # Configure proxy for HybridFetcher
        proxy_config = None
        if apify_proxy_config and APIFY_AVAILABLE:
            try:
                proxy_configuration = await Actor.create_proxy_configuration(actor_proxy_input=apify_proxy_config)
                proxy_url = await proxy_configuration.new_url()
                if proxy_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(proxy_url)
                    proxy_config = {
                        'server': f'{parsed.scheme}://{parsed.hostname}:{parsed.port}',
                        'username': parsed.username or '',
                        'password': parsed.password or ''
                    }
                    logger.info(f'    Apify proxy configured: {proxy_config["server"]}')
            except Exception as e:
                logger.warning(f'    Failed to configure Apify proxy: {e}')
                logger.info(f'   Continuing without proxy...')
        
        logger.info(f" Configuration:")
        logger.info(f"   URLs: {len(urls)}")
        if isinstance(fields, list):
            logger.info(f"   Fields: {', '.join(fields)}")
        else:
            logger.info(f"   Fields: {fields}")
        logger.info(f"   Proxy: {'Enabled' if proxy_config else 'Disabled'}")
        logger.info(f"   Headless: {headless}")
        
        # Initialize hybrid scraper with full universal capabilities
        scraper = HybridScraper(
            api_key=api_key,
            proxy_config=proxy_config,
            headless=headless
        )
        
        # Scrape all URLs
        all_results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {i}/{len(urls)}")
            logger.info(f"{'='*80}")
            
            result = await scraper.scrape(url, fields)
            all_results.append(result)
            
            # Push data to dataset
            if result.get('success') and result.get('data'):
                for item in result['data']:
                    await Actor.push_data({
                        **item,
                        '_metadata': {
                            'source_url': url,
                            'used_cache': result.get('used_cache'),
                            'extraction_cost': result.get('cost')
                        }
                    })
            
            # Small delay between requests
            if i < len(urls):
                await asyncio.sleep(1)
        
        # Get metrics
        metrics = scraper.get_metrics()
        
        # Save metadata
        await Actor.set_value('OUTPUT_METADATA', {
            'total_urls': len(urls),
            'successful': sum(1 for r in all_results if r.get('success')),
            'total_items': metrics['items_extracted'],
            'cache_hits': metrics['cache_hits'],
            'cache_misses': metrics['cache_misses'],
            'llm_calls': metrics['llm_calls'],
            'total_cost': metrics['total_cost'],
            'avg_cost_per_request': metrics['avg_cost_per_request'],
            'cache_hit_rate': f"{metrics['cache_hit_rate']*100:.1f}%",
            'patterns_cached': metrics['patterns_cached'],
            'unique_domains': metrics['unique_domains']
        })
        
        # Print summary
        logger.info(f"\n\n{'='*80}")
        logger.info(f" SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f" Successful: {sum(1 for r in all_results if r.get('success'))}/{len(urls)}")
        logger.info(f" Items Extracted: {metrics['items_extracted']}")
        logger.info(f"  Cache Hit Rate: {metrics['cache_hit_rate']*100:.1f}%")
        logger.info(f" Total Cost: ${metrics['total_cost']:.4f}")
        logger.info(f" Avg Cost/Request: ${metrics['avg_cost_per_request']:.4f}")
        logger.info(f" Patterns Cached: {metrics['patterns_cached']}")
        logger.info(f"{'='*80}")
        
        logger.info(' Actor finished successfully')


if __name__ == '__main__':
    """Run actor"""
    if APIFY_AVAILABLE:
        asyncio.run(main())
    else:
        logger.error(" Apify SDK required")

