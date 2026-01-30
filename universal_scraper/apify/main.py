"""
Universal Web Scraper - Apify Actor with Web Unblocker Support

Uses the main UniversalScraper with all improvements:
- Web Unblocker fallback for Kasada/Cloudflare
- JSON-first extraction
- Universal pagination detection
- Context-aware validation
- Direct LLM extraction
"""

import os
import sys
import logging
import asyncio
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

# Import main UniversalScraper (with all improvements)
from universal_scraper.core.scraper import UniversalScraper

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
 """Main Apify actor entry point"""

 if APIFY_AVAILABLE:
 async with Actor:
 logger.info("="*80)
 logger.info(" Universal Web Scraper - Apify Actor")
 logger.info("="*80)

 # Get input
 actor_input = await Actor.get_input() or {}

 # Extract parameters
 urls = actor_input.get('startUrls', [])
 fields = actor_input.get('fields', [])
 api_key = actor_input.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')

 # Get proxy config (Apify proxy)
 apify_proxy_config = actor_input.get('proxyConfiguration')
 headless = actor_input.get('headless', True)

 # Web Unblocker configuration (NEW)
 web_unblocker_api_key = actor_input.get('webUnblockerApiKey') or os.environ.get('BRIGHT_DATA_API_KEY')
 web_unblocker_zone = actor_input.get('webUnblockerZone', 'web_unlocker1')

 # External proxy configuration (NEW - for Bright Data, etc.)
 external_proxy_config = None
 if actor_input.get('useExternalProxy'):
 external_proxy_config = {
 'server': actor_input.get('externalProxyServer', ''),
 'username': actor_input.get('externalProxyUsername', ''),
 'password': actor_input.get('externalProxyPassword', '')
 }
 if not all([external_proxy_config['server'], external_proxy_config['username']]):
 logger.warning(" External proxy enabled but incomplete configuration")
 external_proxy_config = None

 # Validate
 if not urls:
 logger.error(" No URLs provided")
 await Actor.fail("At least one URL is required in startUrls")
 return

 if not api_key:
 logger.error(" No OpenAI API key provided")
 await Actor.fail("OpenAI API key is required. Set it in input or OPENAI_API_KEY environment variable.")
 return

 # Parse URLs
 if isinstance(urls, list):
 url_list = [item.get('url') if isinstance(item, dict) else item for item in urls]
 else:
 url_list = [urls]

 # Parse fields
 if isinstance(fields, str):
 fields_list = [f.strip() for f in fields.split(',')]
 elif isinstance(fields, list):
 fields_list = fields
 else:
 fields_list = []

 # Configure proxy (prioritize external proxy if provided, else Apify proxy)
 proxy_config = None

 if external_proxy_config:
 # Use external proxy (Bright Data, etc.)
 proxy_config = external_proxy_config
 logger.info(f" External proxy configured: {proxy_config['server']}")
 elif apify_proxy_config:
 # Use Apify proxy
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
 logger.info(f' Apify proxy configured: {proxy_config["server"]}')
 except Exception as e:
 logger.warning(f' Failed to configure Apify proxy: {e}')

 # Log configuration
 logger.info(f"\n{'='*80}")
 logger.info(f" CONFIGURATION")
 logger.info(f"{'='*80}")
 logger.info(f" URLs: {len(url_list)}")
 logger.info(f" Fields: {', '.join(fields_list) if fields_list else 'AUTO-EXTRACT'}")
 logger.info(f" Proxy: {'External' if external_proxy_config else 'Apify' if proxy_config else 'Disabled'}")
 logger.info(f" Web Unblocker: {'Enabled' if web_unblocker_api_key else 'Disabled'}")
 if web_unblocker_api_key:
 logger.info(f" Web Unblocker Zone: {web_unblocker_zone}")
 logger.info(f" Auto-Pagination: {'Enabled' if actor_input.get('enableAutoPagination', False) else 'Disabled'}")
 if actor_input.get('enableAutoPagination', False):
 max_pages_val = actor_input.get('maxPages', 0)
 if max_pages_val > 0:
 logger.info(f" Max Pages: {max_pages_val}")
 else:
 logger.info(f" Max Pages: All pages")
 logger.info(f" Headless: {headless}")
 logger.info(f"{'='*80}\n")

 # Get pagination settings
 enable_auto_pagination = actor_input.get('enableAutoPagination', False)
 max_pages = actor_input.get('maxPages', 0) # 0 or blank means all pages

 # Initialize UniversalScraper with all improvements
 scraper = UniversalScraper(
 api_key=api_key,
 proxy_config=proxy_config,
 headless=headless,
 use_camoufox=True,
 fetch_mode='browser', # Use browser mode for JS sites
 browser_timeout=120000, # 2 minutes timeout
 use_direct_llm=True,
 enable_cache=False, # Disable cache in Apify (use Apify KV if needed)
 enable_auto_pagination=enable_auto_pagination,
 web_unblocker_api_key=web_unblocker_api_key, # Web Unblocker support
 web_unblocker_zone=web_unblocker_zone, # Web Unblocker zone
 log_level=logging.INFO
 )

 # Store max_pages limit for use in scrape calls (only if auto-pagination is enabled)
 if enable_auto_pagination:
 scraper._max_pages_limit = max_pages if max_pages > 0 else None
 if max_pages > 0:
 logger.info(f" Max Pages Limit: {max_pages}")
 else:
 logger.info(f" Max Pages Limit: None (scrape all pages)")
 else:
 scraper._max_pages_limit = None

 # Process URLs
 total_items = 0
 successful_urls = 0
 failed_urls = 0

 for i, url in enumerate(url_list, 1):
 logger.info(f"\n{'#'*80}")
 logger.info(f"Processing {i}/{len(url_list)}: {url}")
 logger.info(f"{'#'*80}\n")

 try:
 result = await scraper.scrape(url, fields_list)

 # Check if we have data (even if success flag is missing)
 items = result.get('data', [])
 if items and len(items) > 0:
 total_items += len(items)
 successful_urls += 1

 # Build a mapping of normalized requested field names
 # This allows ANY requested field to be included, not just predefined schema fields
 requested_fields_map = {}
 for field in fields_list:
 normalized_field = field.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
 requested_fields_map[normalized_field] = field # Keep original for display

 # Push each item separately
 for item in items:
 # Normalize field names and values to match Apify schema
 # Create a clean, normalized item dict that matches DATASET_SCHEMA.json
 normalized_item = {}
 url_field_value = None

 # Schema mapping: map extracted fields to schema fields
 # Schema allows: title, name, price, rating, review_count, url, description, color, _url, _metadata
 schema_fields = {
 'title': 'title',
 'name': 'name',
 'price': 'price',
 'rating': 'rating',
 'review_count': 'review_count',
 'url': 'url', # Map product_detail_url, product_url, etc. to 'url'
 'description': 'description',
 'color': 'color'
 }

 # First pass: collect title and other fields for URL construction
 title_value = None
 for key, value in item.items():
 if key.lower() in ['title', 'name']:
 title_value = str(value) if value else None
 break

 # Second pass: normalize all fields
 for key, value in item.items():
 # Skip internal metadata fields (they'll be added separately)
 if key.startswith('_'):
 continue

 # Normalize key: "product detail url" → "product_detail_url", "est. market value" → "est_market_value"
 normalized_key = key.lower().replace(' ', '_').replace('-', '_').replace('.', '_')

 # Handle URL fields - map to schema 'url' field
 if normalized_key in ['url', 'product_url', 'product_detail_url', 'producturl', 'link', 'href', 'property_url']:
 # Convert product ID to URL if needed
 if isinstance(value, (int, float)):
 # Product ID - try to construct URL from title slug if available
 base_domain = '/'.join(url.split('/')[:3]) # Get https://baggu.com
 # Try to use title to construct slug-based URL
 title_slug = (title_value or normalized_item.get('title', '')).replace(' ', '-').lower()
 if title_slug:
 url_field_value = f"{base_domain}/products/{title_slug}"
 else:
 # Fallback: use product ID (may not work, but better than nothing)
 url_field_value = f"{base_domain}/products/{int(value)}"
 elif isinstance(value, str) and value.startswith('http'):
 url_field_value = value
 elif isinstance(value, str):
 # Relative URL - make absolute
 from urllib.parse import urljoin
 url_field_value = urljoin(url, value)

 # Map to schema 'url' field (not _url)
 if url_field_value:
 normalized_item['url'] = url_field_value
 continue

 # UNIVERSAL: Handle nested objects - extract string values from any object field
 if isinstance(value, dict):
 # Try to extract a meaningful string value from the object
 # Priority: field-specific keys (e.g., colorName for color), then generic keys
 extracted_value = None

 # Field-specific extraction patterns
 if normalized_key == 'color':
 extracted_value = value.get('colorName') or value.get('name') or value.get('value')
 elif normalized_key in ['variant', 'variation']:
 extracted_value = value.get('variantName') or value.get('name') or value.get('value')
 elif normalized_key in ['price', 'cost']:
 extracted_value = value.get('amount') or value.get('value') or value.get('price')
 elif normalized_key in ['url', 'link', 'href']:
 extracted_value = value.get('href') or value.get('url') or value.get('link')
 elif normalized_key in ['title', 'name']:
 extracted_value = value.get('title') or value.get('name') or value.get('label')

 # Generic fallback: try common keys
 if not extracted_value:
 extracted_value = (
 value.get('name') or
 value.get('value') or
 value.get('title') or
 value.get('label') or
 value.get('text')
 )

 # If we found a string value, use it; otherwise try to stringify the object
 if extracted_value:
 value = str(extracted_value)
 else:
 # Last resort: stringify the entire object (may not be ideal, but better than nothing)
 value = str(value)

 # Map fields to schema fields OR include if it's a requested field
 if normalized_key in schema_fields:
 schema_field = schema_fields[normalized_key]
 # Convert price from cents to dollars if it's a number
 if schema_field == 'price' and isinstance(value, (int, float)):
 if value > 1000:
 # Likely in cents, convert to dollars
 normalized_item[schema_field] = f"${value / 100:.2f}"
 else:
 # Already in dollars
 normalized_item[schema_field] = f"${value:.2f}"
 else:
 normalized_item[schema_field] = str(value) if value is not None else None
 elif normalized_key in requested_fields_map:
 # This is a requested field - include it even if not in predefined schema
 # Use the normalized key as the field name
 normalized_item[normalized_key] = str(value) if value is not None else None
 elif normalized_key == 'color':
 # Color field is in schema but not in schema_fields mapping
 normalized_item['color'] = str(value) if value is not None else None
 else:
 # Field not in schema and not requested - skip it
 # But log it for debugging
 pass

 # Use product URL if available, otherwise use page URL
 normalized_item['_url'] = url_field_value if url_field_value else url

 # Ensure 'url' field exists (for schema compliance)
 if 'url' not in normalized_item:
 normalized_item['url'] = normalized_item.get('_url', url)

 # Add metadata
 normalized_item['_metadata'] = {
 'fetch_method': result.get('fetch_method', 'unknown'),
 'extraction_source': result.get('source', 'unknown'),
 'execution_time': result.get('metadata', {}).get('execution_time', 0)
 }

 # Push normalized item (not original item)
 await Actor.push_data(normalized_item)

 logger.info(f" Extracted {len(items)} items from {url}")
 elif result.get('success') is False:
 failed_urls += 1
 error_msg = result.get('error', 'Unknown error')
 logger.warning(f" Failed to extract data from {url}: {error_msg}")

 # Push error to dataset for debugging
 await Actor.push_data({
 '_url': url,
 '_error': error_msg,
 '_success': False
 })
 else:
 # No data but no explicit failure - might be empty page
 logger.warning(f" No data extracted from {url} (may be empty page)")
 failed_urls += 1

 # Brief pause between requests
 if i < len(url_list):
 await asyncio.sleep(2)

 except Exception as e:
 failed_urls += 1
 logger.error(f" Error processing {url}: {e}")
 import traceback
 traceback.print_exc()

 # Push error to dataset
 await Actor.push_data({
 '_url': url,
 '_error': str(e),
 '_success': False
 })

 # Print final summary
 logger.info(f"\n{'='*80}")
 logger.info(f" FINAL SUMMARY")
 logger.info(f"{'='*80}")
 logger.info(f" Total URLs: {len(url_list)}")
 logger.info(f" Successful: {successful_urls}")
 logger.info(f" Failed: {failed_urls}")
 logger.info(f" Total items extracted: {total_items}")
 logger.info(f"{'='*80}\n")

 else:
 # Standalone mode (local testing)
 logger.info("Running in standalone mode (no Apify)")
 api_key = os.environ.get('OPENAI_API_KEY')

 if not api_key:
 logger.error(" OPENAI_API_KEY not set")
 return

 # Test with a simple URL
 scraper = UniversalScraper(
 api_key=api_key,
 use_camoufox=True,
 use_direct_llm=True
 )

 result = await scraper.scrape(
 url="https://news.ycombinator.com/",
 fields=["title", "url", "points"]
 )

 print(f"\n{'='*80}")
 print(f"Result: {result.get('success', False)}")
 print(f"Items: {len(result.get('data', []))}")
 print(f"Source: {result.get('source', 'unknown')}")
 print(f"{'='*80}\n")


if __name__ == "__main__":
 asyncio.run(main())
