"""
Command Line Interface for Universal Scraper
"""

import argparse
import json
import csv
import logging
import sys
from pathlib import Path
from typing import List

from .core.scraper import UniversalScraper


def main():
    parser = argparse.ArgumentParser(
        description='Universal Web Scraper - AI-powered scraping for any website'
    )
    
    # Input
    parser.add_argument(
        '--url',
        type=str,
        help='Single URL to scrape'
    )
    parser.add_argument(
        '--urls',
        type=str,
        help='File containing URLs (one per line)'
    )
    
    # Fields
    parser.add_argument(
        '--fields',
        nargs='+',
        required=True,
        help='Fields to extract (space-separated)'
    )
    
    # AI Configuration
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for AI provider (or set via environment variable)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='AI model to use (default: gpt-4o-mini)'
    )
    
    # Proxy
    parser.add_argument(
        '--proxy-server',
        type=str,
        help='Proxy server URL'
    )
    parser.add_argument(
        '--proxy-username',
        type=str,
        help='Proxy username'
    )
    parser.add_argument(
        '--proxy-password',
        type=str,
        help='Proxy password'
    )
    
    # Cache
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Cache directory (default: ./cache)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON or CSV based on extension)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for multiple URLs'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    # Other
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input
    if not args.url and not args.urls:
        parser.error('Either --url or --urls is required')
    
    # Collect URLs
    urls = []
    if args.url:
        urls = [args.url]
    elif args.urls:
        with open(args.urls, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    # Setup proxy config
    proxy_config = None
    if args.proxy_server:
        proxy_config = {
            'server': args.proxy_server,
            'username': args.proxy_username or '',
            'password': args.proxy_password or ''
        }
    
    # Initialize scraper
    try:
        scraper = UniversalScraper(
            api_key=args.api_key,
            model_name=args.model,
            proxy_config=proxy_config,
            cache_dir=args.cache_dir,
            enable_cache=not args.no_cache,
            log_level=log_level
        )
    except Exception as e:
        print(f"❌ Failed to initialize scraper: {e}")
        sys.exit(1)
    
    # Scrape
    try:
        results = scraper.scrape_multiple(urls, args.fields)
        
        # Save results
        if args.output:
            save_results(results, args.output, args.format)
        elif args.output_dir:
            save_multiple_results(results, args.output_dir, args.format)
        else:
            # Print to stdout
            print(json.dumps(results, indent=2))
        
        # Print summary
        total_items = sum(len(r['data']) for r in results)
        print(f"\n✅ Scraping complete!")
        print(f"   URLs: {len(urls)}")
        print(f"   Items: {total_items}")
        
        scraper.close()
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        scraper.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        scraper.close()
        sys.exit(1)


def save_results(results: List, output_path: str, format: str):
    """Save results to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine all data
    all_data = []
    for result in results:
        all_data.extend(result['data'])
    
    if format == 'json' or output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"💾 Saved to {output_path} (JSON)")
    else:
        if all_data:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                writer.writerows(all_data)
            print(f"💾 Saved to {output_path} (CSV)")


def save_multiple_results(results: List, output_dir: str, format: str):
    """Save each URL's results to separate file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results, 1):
        filename = f"result_{i}.{format}"
        filepath = output_dir / filename
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(result['data'], f, indent=2)
        else:
            if result['data']:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result['data'][0].keys())
                    writer.writeheader()
                    writer.writerows(result['data'])
        
        print(f"💾 Saved {filepath}")


if __name__ == '__main__':
    main()

