"""
Universal Web Scraper V2 - Direct LLM + Pattern Caching

The PROVEN solution:
- Direct LLM extraction (works universally)
- Pattern learning (enables caching)
- Unified caching (local & Apify)
- Cost optimization (99% savings)
"""

import os
import sys
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# CRITICAL: Set up Python path FIRST
script_dir = Path(__file__).parent.absolute()
if 'apify' in str(script_dir):
 project_root = script_dir.parent.parent
else:
 project_root = script_dir

if str(project_root) not in sys.path:
 sys.path.insert(0, str(project_root))

# Apify SDK
try:
 from apify import Actor
 APIFY_AVAILABLE = True
except ImportError:
 APIFY_AVAILABLE = False
 print(" Apify SDK not available, running in standalone mode")

# Import new architecture components
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector
from universal_scraper.core.html_cleaner import SmartHTMLCleaner
from universal_scraper.core.structural_embedding import StructuralEmbedding
from universal_scraper.core.unified_cache import UnifiedPatternCache
from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.pattern_learner import PatternLearner
from universal_scraper.core.semantic_pattern_generator import SemanticPatternGenerator
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniversalScraperV2:
 """
 Universal Scraper V2 - Direct LLM + Pattern Caching

 Flow:
 1. Fetch HTML (universal: static/JS/JSON)
 2. Try JSON extraction (quality-validated)
 3. If JSON fails, check pattern cache
 4. If cache miss, use DirectLLM extraction
 5. Learn pattern from LLM results
 6. Save pattern to cache
 7. Return data

 Subsequent requests with same structure: Use cached pattern ($0.00)
 """

 def __init__(
 self,
 api_key: str,
 proxy_config: Optional[Dict[str, str]] = None,
 headless: bool = True,
 force_local_cache: bool = False,
 quality_mode: str = "balanced" # DirectLLM quality mode
 ):
 """Initialize universal scraper V2"""
 self.api_key = api_key
 self.proxy_config = proxy_config
 self.headless = headless
 self.quality_mode = quality_mode

 # Core components
 self.hybrid_fetcher = HybridFetcher(
 proxy_config=proxy_config,
 headless=headless,
 use_camoufox=True,
 enable_cache=True
 )
 self.json_detector = JSONDetector()
 self.html_cleaner = SmartHTMLCleaner()
 self.embedding_gen = StructuralEmbedding()
 self.pattern_cache = UnifiedPatternCache(force_local=force_local_cache)
 self.llm_extractor = DirectLLMExtractor(
 api_key=api_key,
 quality_mode=quality_mode, # Use Langchain transformer + quality mode
 use_html2text=True # Enable HTML-to-text (uses Langchain now!)
 )
 self.pattern_learner = PatternLearner()
 self.pattern_gen = SemanticPatternGenerator(api_key=api_key) # For NL parsing

 # Metrics
 self.metrics = {
 'total_requests': 0,
 'json_extractions': 0,
 'cache_hits': 0,
 'cache_misses': 0,
 'llm_calls': 0,
 'patterns_learned': 0,
 'total_cost': 0.0
 }

 logger.info("="*80)
 logger.info(" Universal Scraper V2 Initialized")
 logger.info("="*80)
 logger.info(f" Cache: {self.pattern_cache.env}")
 logger.info(f" LLM Model: {self.llm_extractor.model_name}")
 logger.info(f" Fetcher: Camoufox (anti-detection)")
 logger.info("="*80)

 async def scrape(
 self,
 url: str,
 fields # Can be List[str] or str (natural language)
 ) -> Dict[str, Any]:
 """
 Universal scraping with LLM + caching

 Args:
 url: URL to scrape
 fields: Fields to extract (list or natural language string)

 Returns:
 Extraction results with metadata
 """
 import time
 start_time = time.time()

 self.metrics['total_requests'] += 1

 logger.info(f"\n{'='*80}")
 logger.info(f" Request #{self.metrics['total_requests']}: {url}")
 logger.info(f"{'='*80}")

 try:
 # Step 1: Fetch HTML (universal fetching)
 logger.info(" Step 1: Universal Fetching...")
 result = await self.hybrid_fetcher.fetch(url)

 if not result or 'html' not in result:
 return {"success": False, "error": "Failed to fetch"}

 html = result['html']
 fetch_method = result.get('fetch_method', 'unknown')
 logger.info(f" Fetched {len(html):,} bytes via {fetch_method}")

 # Step 2: Parse natural language fields if needed
 if isinstance(fields, str):
 logger.info(" Step 2: Parsing natural language fields...")
 parsed_fields = await self.pattern_gen._parse_natural_language_fields(
 fields,
 html[:3000]
 )
 fields_list = parsed_fields
 logger.info(f" Parsed to: {fields_list}")
 elif isinstance(fields, list):
 fields_list = fields
 logger.info(f" Fields: {fields_list}")
 else:
 logger.error(f" Invalid fields type: {type(fields)}")
 return {"success": False, "error": f"Invalid fields type: {type(fields)}"}

 # Step 3: Try JSON extraction (quality-validated)
 logger.info(" Step 3: Universal JSON Detection...")
 json_data_captured = result.get('json_data', [])
 json_detection_result = self.json_detector.detect_and_extract(
 html=html,
 url=url,
 captured_json=json_data_captured
 )

 if json_detection_result['json_found']:
 json_sources = json_detection_result['sources']
 all_json = json_detection_result['data']

 logger.info(f" Found JSON from: {', '.join(json_sources)}")

 try:
 extracted_items = self.json_detector.extract_from_json(
 json_data=all_json,
 fields=fields_list
 )

 # Quality validation
 if extracted_items and len(extracted_items) >= 3:
 is_sufficient = self.json_detector.is_json_sufficient(
 json_results=json_detection_result,
 fields=fields_list
 )

 if is_sufficient:
 total_time = time.time() - start_time
 self.metrics['json_extractions'] += 1

 logger.info(f" JSON SUCCESS! {len(extracted_items)} items")
 logger.info(f" Cost: $0.00")
 logger.info(f" Time: {total_time:.1f}s")

 return {
 "url": url,
 "success": True,
 "items_count": len(extracted_items),
 "extraction_method": "json_universal",
 "json_sources": json_sources,
 "used_cache": False,
 "cost": 0.0,
 "time": total_time,
 "data": extracted_items
 }
 else:
 logger.info(f" JSON quality too low, falling back to HTML")
 else:
 logger.info(f" ℹ JSON insufficient ({len(extracted_items) if extracted_items else 0} items)")

 except Exception as e:
 logger.warning(f" JSON extraction failed: {e}")
 else:
 logger.info(" ℹ No JSON detected")

 # Step 4: Clean HTML
 logger.info(" Step 4: Cleaning HTML...")
 cleaned_result = self.html_cleaner.clean(html)
 cleaned_html = cleaned_result['html']
 logger.info(f" {len(html):,} → {len(cleaned_html):,} bytes ({cleaned_result['reduction_percent']:.1f}% reduction)")

 # Step 5: Generate structural embedding
 logger.info(" Step 5: Generating structural embedding...")
 embedding = self.embedding_gen.generate(cleaned_html)
 embedding_hash = hashlib.md5(str(embedding).encode()).hexdigest()[:16]
 logger.info(f" Embedding: {embedding_hash}")

 # Step 6: Check pattern cache
 logger.info(" Step 6: Checking pattern cache...")
 domain = url.split('/')[2] if len(url.split('/')) > 2 else "unknown"
 cached_pattern_data = await self.pattern_cache.get_pattern(
 embedding_hash=embedding_hash,
 fields=fields_list,
 domain=domain
 )

 extraction_cost = 0.0
 used_cache = False
 extracted_items = []

 if cached_pattern_data:
 # CACHE HIT - Execute cached pattern
 self.metrics['cache_hits'] += 1
 used_cache = True
 extraction_cost = 0.0001 # Negligible

 logger.info(f" CACHE HIT!")
 logger.info(f" Saved ~$0.03")
 logger.info(" Step 7: Executing cached pattern...")

 cached_pattern = cached_pattern_data['pattern']

 # Execute pattern (simple CSS selector extraction)
 extracted_items = self._execute_cached_pattern(cleaned_html, cached_pattern)

 logger.info(f" Extracted {len(extracted_items)} items via cached pattern")

 else:
 # CACHE MISS - Use DirectLLM extraction + Learn pattern
 self.metrics['cache_misses'] += 1
 self.metrics['llm_calls'] += 1

 logger.info(f" CACHE MISS")
 logger.info(" Step 7: Direct LLM Extraction...")

 # Estimate cost
 estimated_cost = self.llm_extractor.estimate_cost(
 len(cleaned_html),
 len(fields_list)
 )
 logger.info(f" Estimated cost: ${estimated_cost:.4f}")

 # Extract with LLM
 extracted_items = await self.llm_extractor.extract(
 html=cleaned_html,
 fields=fields_list,
 context=f"Extract {', '.join(fields_list)} from {domain}"
 )

 extraction_cost = estimated_cost
 self.metrics['total_cost'] += extraction_cost

 logger.info(f" Extracted {len(extracted_items)} items")
 logger.info(f" Cost: ${extraction_cost:.4f}")

 # Step 8: Learn pattern from successful extraction
 if extracted_items and len(extracted_items) >= 3:
 logger.info(" Step 8: Learning pattern from LLM results...")

 learned_pattern = await self.pattern_learner.learn_pattern(
 html=cleaned_html,
 extracted_items=extracted_items,
 fields=fields_list
 )

 if learned_pattern:
 # Validate pattern
 is_valid = self.pattern_learner.validate_pattern(
 pattern=learned_pattern,
 html=cleaned_html,
 expected_items=extracted_items
 )

 if is_valid:
 # Save to cache
 logger.info(" Step 9: Saving pattern to cache...")
 await self.pattern_cache.save_pattern(
 embedding_hash=embedding_hash,
 fields=fields_list,
 pattern=learned_pattern,
 domain=domain,
 url=url
 )

 self.metrics['patterns_learned'] += 1
 logger.info(f" Pattern saved for future requests")
 logger.info(f" Next {domain} request will cost $0.00!")
 else:
 logger.warning(f" Pattern validation failed, not caching")
 else:
 logger.warning(f" Could not learn pattern")
 else:
 logger.warning(f" Too few items to learn pattern")

 # Return results
 total_time = time.time() - start_time

 logger.info(f"\n{'='*80}")
 logger.info(f" REQUEST COMPLETE")
 logger.info(f"{'='*80}")
 logger.info(f" Items: {len(extracted_items)}")
 logger.info(f" Method: {'Cached Pattern' if used_cache else 'Direct LLM'}")
 logger.info(f" Cost: ${extraction_cost:.4f}")
 logger.info(f" Time: {total_time:.1f}s")
 logger.info(f"{'='*80}\n")

 return {
 "url": url,
 "success": True,
 "items_count": len(extracted_items),
 "extraction_method": "cached_pattern" if used_cache else "direct_llm",
 "used_cache": used_cache,
 "cost": extraction_cost,
 "time": total_time,
 "data": extracted_items
 }

 except Exception as e:
 logger.error(f" Scrape failed for {url}: {e}")
 import traceback
 traceback.print_exc()
 return {"url": url, "success": False, "error": str(e)}

 def _execute_cached_pattern(
 self,
 html: str,
 pattern: Dict[str, Any]
 ) -> List[Dict[str, Any]]:
 """
 Execute a cached extraction pattern

 Args:
 html: Cleaned HTML
 pattern: Cached extraction pattern

 Returns:
 Extracted items
 """
 soup = BeautifulSoup(html, 'html.parser')

 container_selector = pattern.get('container_selector')
 if not container_selector:
 return []

 containers = soup.select(container_selector)
 if not containers:
 logger.warning(f" No containers found for: {container_selector}")
 return []

 items = []
 field_patterns = pattern.get('fields', {})

 for container in containers:
 item = {}

 for field, field_pattern in field_patterns.items():
 selector = field_pattern.get('selector')
 if not selector:
 continue

 element = container.select_one(selector)
 if element:
 value = element.get_text().strip()
 if value:
 item[field] = value

 if item:
 items.append(item)

 return items

 def get_metrics(self) -> Dict[str, Any]:
 """Get scraping metrics"""
 return self.metrics.copy()


async def main():
 """Main Apify actor entry point"""

 if APIFY_AVAILABLE:
 async with Actor:
 # Get input
 actor_input = await Actor.get_input() or {}

 # Extract parameters
 urls = actor_input.get('startUrls', [])
 fields = actor_input.get('fields', '')
 api_key = actor_input.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')

 # Get proxy config
 apify_proxy_config = actor_input.get('proxyConfiguration')
 headless = actor_input.get('headless', True)

 # Validate
 if not urls:
 logger.error(" No URLs provided")
 return

 if not api_key:
 logger.error(" No OpenAI API key provided")
 return

 # Parse URLs
 if isinstance(urls, list):
 url_list = [item.get('url') if isinstance(item, dict) else item for item in urls]
 else:
 url_list = [urls]

 # Configure proxy
 proxy_config = None
 if apify_proxy_config:
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
 logger.info(f' Apify proxy configured')
 except Exception as e:
 logger.warning(f' Failed to configure proxy: {e}')

 logger.info(f"\n{'='*80}")
 logger.info(f" CONFIGURATION")
 logger.info(f"{'='*80}")
 logger.info(f" URLs: {len(url_list)}")
 logger.info(f" Fields: {fields if isinstance(fields, str) else ', '.join(fields)}")
 logger.info(f" Proxy: {'Enabled' if proxy_config else 'Disabled'}")
 logger.info(f" Headless: {headless}")
 logger.info(f"{'='*80}\n")

 # Initialize scraper
 # Get DirectLLM parameters
 quality_mode = actor_input.get('directLLMQualityMode', 'balanced')

 scraper = UniversalScraperV2(
 api_key=api_key,
 proxy_config=proxy_config,
 headless=headless,
 force_local_cache=False, # Use Apify KV in production
 quality_mode=quality_mode # DirectLLM quality mode
 )

 # Process URLs
 for i, url in enumerate(url_list, 1):
 logger.info(f"\n{'#'*80}")
 logger.info(f"Processing {i}/{len(url_list)}")
 logger.info(f"{'#'*80}\n")

 result = await scraper.scrape(url, fields)

 # Push results to dataset
 if result['success'] and result.get('data'):
 await Actor.push_data(result['data'])

 # Brief pause between requests
 if i < len(url_list):
 await asyncio.sleep(1)

 # Print final metrics
 metrics = scraper.get_metrics()
 logger.info(f"\n{'='*80}")
 logger.info(f" FINAL METRICS")
 logger.info(f"{'='*80}")
 logger.info(f" Total requests: {metrics['total_requests']}")
 logger.info(f" JSON extractions: {metrics['json_extractions']}")
 logger.info(f" Cache hits: {metrics['cache_hits']}")
 logger.info(f" Cache misses: {metrics['cache_misses']}")
 logger.info(f" LLM calls: {metrics['llm_calls']}")
 logger.info(f" Patterns learned: {metrics['patterns_learned']}")
 logger.info(f" Total cost: ${metrics['total_cost']:.4f}")

 if metrics['total_requests'] > 0:
 cache_hit_rate = metrics['cache_hits'] / metrics['total_requests'] * 100
 avg_cost = metrics['total_cost'] / metrics['total_requests']
 logger.info(f" Cache hit rate: {cache_hit_rate:.1f}%")
 logger.info(f" Avg cost/request: ${avg_cost:.4f}")

 logger.info(f"{'='*80}\n")

 else:
 # Standalone mode (local testing)
 logger.info("Running in standalone mode (no Apify)")
 api_key = os.environ.get('OPENAI_API_KEY')

 if not api_key:
 logger.error(" OPENAI_API_KEY not set")
 return

 # Test with a simple URL
 scraper = UniversalScraperV2(
 api_key=api_key,
 force_local_cache=True # Use local cache for testing
 )

 result = await scraper.scrape(
 url="https://news.ycombinator.com/",
 fields="article_title, points, comments_count"
 )

 print(f"\n{'='*80}")
 print(f"Result: {result['success']}")
 print(f"Items: {result.get('items_count', 0)}")
 print(f"Method: {result.get('extraction_method')}")
 print(f"Cost: ${result.get('cost', 0):.4f}")
 print(f"{'='*80}\n")


if __name__ == "__main__":
 asyncio.run(main())
