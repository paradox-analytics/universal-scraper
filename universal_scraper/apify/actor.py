"""
Apify Actor for Universal Scraper
Enables deployment to Apify platform
"""

import asyncio
import os
import logging
from typing import Dict, Any, List

from apify import Actor

# Import scraper (handle both package and local imports)
try:
    from ..core.scraper import UniversalScraper
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.scraper import UniversalScraper

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main Actor execution function
    Reads input from Apify, runs scraper, pushes results to dataset
    """
    async with Actor:
        # Get input from Apify
        actor_input = await Actor.get_input() or {}
        
        Actor.log.info('🚀 Universal Scraper Actor Started')
        Actor.log.info(f'📋 Input: {actor_input}')
        
        # Extract configuration
        urls = actor_input.get('urls', [])
        fields = actor_input.get('fields', [])
        proxy_type = actor_input.get('proxyType', 'residential')
        ai_model = actor_input.get('aiModel', 'gpt-4o-mini')
        output_format = actor_input.get('outputFormat', 'json')
        api_keys = actor_input.get('apiKeys', {})
        
        # Validate inputs
        if not urls:
            raise ValueError("No URLs provided. Please provide 'urls' array in input.")
        
        if not fields:
            raise ValueError("No fields provided. Please provide 'fields' array in input.")
        
        if isinstance(urls, str):
            urls = [urls]
        
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(',')]
        
        Actor.log.info(f'🎯 Scraping {len(urls)} URL(s)')
        Actor.log.info(f'📋 Fields: {", ".join(fields)}')
        Actor.log.info(f'🤖 AI Model: {ai_model}')
        
        # Set up API keys from input
        api_key = None
        if api_keys.get('openai_api_key'):
            os.environ['OPENAI_API_KEY'] = api_keys['openai_api_key']
            api_key = api_keys['openai_api_key']
        elif api_keys.get('gemini_api_key'):
            os.environ['GEMINI_API_KEY'] = api_keys['gemini_api_key']
            api_key = api_keys['gemini_api_key']
        elif api_keys.get('anthropic_api_key'):
            os.environ['ANTHROPIC_API_KEY'] = api_keys['anthropic_api_key']
            api_key = api_keys['anthropic_api_key']
        
        # Set up Apify proxy
        proxy_configuration = None
        proxy_config = None
        
        if proxy_type and proxy_type != 'none':
            if proxy_type == 'residential':
                proxy_configuration = await Actor.create_proxy_configuration(
                    groups=['RESIDENTIAL'],
                    country_code='US'
                )
                Actor.log.info('🌍 Using Apify Residential Proxies (US)')
            else:
                proxy_configuration = await Actor.create_proxy_configuration()
                Actor.log.info('🖥️ Using Apify Datacenter Proxies')
            
            # Get proxy URL for scraper
            if proxy_configuration:
                proxy_info = await proxy_configuration.new_url()
                # Parse proxy info for scraper format
                # Apify provides proxy in format: http://username:password@host:port
                import re
                match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', proxy_info)
                if match:
                    username, password, host, port = match.groups()
                    proxy_config = {
                        'server': f'http://{host}:{port}',
                        'username': username,
                        'password': password
                    }
        else:
            Actor.log.warning('⚠️ No proxy selected')
        
        try:
            # Initialize scraper
            scraper = UniversalScraper(
                api_key=api_key,
                model_name=ai_model,
                proxy_config=proxy_config,
                enable_cache=True,
                enable_warming=True,
                log_level=logging.INFO
            )
            
            # Run scraping
            Actor.log.info('🕷️ Starting scraping...')
            
            total_items = 0
            for i, url in enumerate(urls, 1):
                Actor.log.info(f'📍 Scraping {i}/{len(urls)}: {url}')
                
                try:
                    result = scraper.scrape(url, fields)
                    
                    # Push data to dataset
                    for item in result['data']:
                        await Actor.push_data({
                            '__url': url,
                            '__timestamp': result['metadata']['timestamp'],
                            '__extraction_source': result['source'],
                            **item
                        })
                        total_items += 1
                    
                    Actor.log.info(f'  ✅ Extracted {len(result["data"])} items from {url}')
                    
                except Exception as e:
                    Actor.log.error(f'  ❌ Failed to scrape {url}: {str(e)}')
                    # Push error record
                    await Actor.push_data({
                        '__url': url,
                        '__error': str(e),
                        '__status': 'failed'
                    })
            
            # Save summary
            cache_stats = scraper.get_cache_stats()
            
            await Actor.set_value('summary', {
                'total_urls': len(urls),
                'total_items': total_items,
                'fields': fields,
                'ai_model': ai_model,
                'proxy_type': proxy_type,
                'cache_stats': cache_stats
            })
            
            Actor.log.info(f'✅ Scraping complete!')
            Actor.log.info(f'📊 Total items extracted: {total_items}')
            Actor.log.info(f'💾 Cache stats: {cache_stats}')
            
            # Close scraper
            scraper.close()
            
        except Exception as e:
            Actor.log.error(f'❌ Actor execution failed: {str(e)}')
            Actor.log.exception(e)
            raise


if __name__ == '__main__':
    asyncio.run(main())

