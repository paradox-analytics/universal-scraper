#!/usr/bin/env python3
"""
Comprehensive Home Depot Validation Test

Tests:
1. HTML capture quality (full vs partial content)
2. GraphQL/JSON endpoint detection
3. Both proxy types (Residential + Web Unblocker)
4. Smart strategy caching

Goal: Determine the best scraping method and cache it for future use
"""
import asyncio
import sys
from pathlib import Path
import time
import os
import json
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from universal_scraper.core.json_detector import JSONDetector
from api.main import convert_proxy_config

load_dotenv()

# Home Depot test URL
TEST_URL = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"

async def test_proxy_configuration(name: str, proxy_config: dict, use_web_unblocker: bool = False, max_retries: int = 3):
    """Test a specific proxy configuration with retry logic"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {name}")
    print(f"{'='*80}")
    
    # Retry loop for intermittent Akamai blocks
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n🔄 Retry attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(3)  # Wait between retries
        
        start_time = time.time()
        
        # Create fetcher
        if use_web_unblocker:
            # Parse web unblocker credentials
            unlocker_url = os.getenv('BRIGHT_DATA_UNLOCKER_URL')
            import re
            match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', unlocker_url)
            if match:
                username, password, host, port = match.groups()
                web_unblocker_key = f"{host}:{port}:{username}:{password}"
            else:
                print("❌ Failed to parse Web Unblocker credentials")
                return None
                
            fetcher = CamoufoxFetcher(
                proxy_config=None,
                web_unblocker_api_key=web_unblocker_key,
                web_unblocker_zone='web_unlocker1',
                headless=True,
                timeout=120000,
                anti_detection_profile='random',
                humanize=True,
                stealth_mode=False,
                geoip=False
            )
        else:
            fetcher = CamoufoxFetcher(
                proxy_config=proxy_config,
                headless=True,
                timeout=120000,
                anti_detection_profile='random',
                humanize=True,
                stealth_mode=False,
                geoip=False
            )
        
        try:
            # Fetch the page
            result = await fetcher.fetch(
                url=TEST_URL,
                wait_for_selector="h1",
                wait_time=5000,
                scroll_to_bottom=False
            )
            
            elapsed = time.time() - start_time
            
            # Check if we should retry (403 Akamai block)
            status = result.get('status_code', 0)
            if status == 403 and attempt < max_retries - 1:
                print(f"   ⚠️  Got 403 Akamai block, will retry...")
                await fetcher.close()
                continue
            elif status == 429 and attempt < max_retries - 1:
                print(f"   ⚠️  Got 429 rate limit, will retry...")
                await fetcher.close()
                continue
            
            # Success or final attempt - break retry loop
            break
            
        except Exception as e:
            print(f"   ❌ Fetch error: {str(e)[:200]}")
            await fetcher.close()
            if attempt < max_retries - 1:
                continue
            return {
                'name': name,
                'success': False,
                'error': str(e),
                'elapsed': time.time() - start_time
            }
    
    # Process results (outside retry loop)
    try:
        elapsed = time.time() - start_time
        
        # Analyze results
        html = result.get('html', '')
        status = result.get('status_code', 0)
        api_calls = result.get('api_calls', [])
        json_data = result.get('json_data', [])
        
        print(f"\n📊 Results:")
        print(f"   Status: {status}")
        print(f"   HTML Size: {len(html):,} bytes")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   API Calls Captured: {len(api_calls)}")
        print(f"   JSON Responses: {len(json_data)}")
        
        # Validate HTML quality
        html_quality = "UNKNOWN"
        if status == 200:
            if len(html) > 100000:
                html_quality = "EXCELLENT"
            elif len(html) > 10000:
                html_quality = "GOOD"
            elif len(html) > 1000:
                html_quality = "PARTIAL"
            else:
                html_quality = "MINIMAL"
        elif status == 403:
            html_quality = "BLOCKED"
        elif status == 429:
            html_quality = "RATE_LIMITED"
        elif status == 502:
            html_quality = "PROXY_ERROR"
        
        print(f"   HTML Quality: {html_quality}")
        
        # Check for GraphQL endpoints
        graphql_endpoints = []
        if 'graphql' in html.lower():
            print(f"   ✅ GraphQL patterns detected in HTML")
            # Extract GraphQL endpoints from HTML
            import re
            patterns = [
                r'["\']([^"\']*?graphql[^"\']*?)["\']',
                r'["\']([^"\']*?/gql[^"\']*?)["\']',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if match.startswith('http') or match.startswith('/'):
                        graphql_endpoints.append(match)
            
            # Save HTML sample for offline analysis
            if status == 200 and len(html) > 10000:
                sample_dir = Path(__file__).parent / 'html_samples'
                sample_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time())
                safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                sample_file = sample_dir / f"homedepot_{safe_name}_{timestamp}.html"
                
                with open(sample_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                
                print(f"   💾 HTML sample saved: {sample_file.name}")
                
                # Also save extracted GraphQL endpoints
                if graphql_endpoints:
                    endpoints_file = sample_dir / f"homedepot_{safe_name}_{timestamp}_graphql.json"
                    with open(endpoints_file, 'w') as f:
                        json.dump({
                            'url': TEST_URL,
                            'timestamp': timestamp,
                            'endpoints': list(set(graphql_endpoints)),
                            'proxy': name
                        }, f, indent=2)
                    print(f"   💾 GraphQL endpoints saved: {endpoints_file.name}")
        
        # Check API calls for GraphQL
        graphql_calls = [call for call in api_calls if 'graphql' in call.lower() or '/gql' in call.lower()]
        if graphql_calls:
            print(f"   ✅ GraphQL API calls detected: {len(graphql_calls)}")
            for call in graphql_calls[:3]:
                print(f"      - {call[:80]}...")
        
        # Check JSON responses
        if json_data:
            print(f"   ✅ JSON data captured:")
            for idx, data in enumerate(json_data[:3]):
                url = data.get('url', 'unknown')
                size = len(str(data.get('data', '')))
                print(f"      {idx+1}. {url[:60]}... ({size:,} bytes)")
        
        # Use JSONDetector to find structured data
        if html and status == 200:
            detector = JSONDetector()
            json_results = detector.detect_and_extract(html, TEST_URL, json_data)
            
            if json_results.get('has_json'):
                print(f"\n   🎯 JSONDetector Results:")
                print(f"      Sources: {json_results.get('sources', [])}")
                print(f"      Data blocks: {len(json_results.get('data', []))}")
                
                # Check for product data
                for data_block in json_results.get('data', [])[:3]:
                    if isinstance(data_block, dict):
                        # Look for product-like fields
                        product_fields = ['name', 'title', 'price', 'sku', 'product', 'item']
                        found_fields = [f for f in product_fields if f in str(data_block).lower()]
                        if found_fields:
                            print(f"      ✅ Product data found with fields: {found_fields}")
        
        await fetcher.close()
        
        return {
            'name': name,
            'success': status == 200,
            'status': status,
            'html_size': len(html),
            'html_quality': html_quality,
            'elapsed': elapsed,
            'api_calls': len(api_calls),
            'json_responses': len(json_data),
            'graphql_endpoints': list(set(graphql_endpoints)),
            'graphql_calls': graphql_calls,
            'has_json': json_results.get('has_json', False) if status == 200 else False,
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:200]}")
        await fetcher.close()
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'elapsed': time.time() - start_time
        }

async def main():
    print("="*80)
    print("🏠 HOME DEPOT - COMPREHENSIVE VALIDATION TEST")
    print("="*80)
    print("\nGoal: Validate HTML capture, detect GraphQL endpoints, test all proxies")
    print("This will determine the best scraping strategy and cache it.\n")
    
    # Parse proxy configurations
    residential_url = os.getenv('BRIGHT_DATA_PROXY_URL')
    
    import re
    match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', residential_url)
    if not match:
        print("❌ Failed to parse residential proxy URL")
        return
    
    username, password, host, port = match.groups()
    
    residential_config = {
        'server': f"http://{host}:{port}",
        'username': username,
        'password': password
    }
    
    # Test configurations
    tests = [
        ("Residential Proxy (geoip=False)", residential_config, False),
        ("Web Unblocker (geoip=False)", None, True),
    ]
    
    results = []
    
    for name, proxy_config, use_unblocker in tests:
        result = await test_proxy_configuration(name, proxy_config, use_unblocker)
        if result:
            results.append(result)
        
        # Wait between tests to avoid rate limiting
        await asyncio.sleep(5)
    
    # Analyze results and determine best strategy
    print(f"\n{'='*80}")
    print("📊 SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r.get('success')]
    
    if successful:
        # Sort by HTML quality and speed
        quality_scores = {
            'EXCELLENT': 4,
            'GOOD': 3,
            'PARTIAL': 2,
            'MINIMAL': 1,
            'BLOCKED': 0,
            'RATE_LIMITED': 0,
            'PROXY_ERROR': 0
        }
        
        for r in successful:
            r['quality_score'] = quality_scores.get(r.get('html_quality', 'UNKNOWN'), 0)
        
        best = max(successful, key=lambda x: (x['quality_score'], -x['elapsed']))
        
        print(f"✅ Best Configuration: {best['name']}")
        print(f"   HTML Quality: {best['html_quality']}")
        print(f"   HTML Size: {best['html_size']:,} bytes")
        print(f"   Speed: {best['elapsed']:.1f}s")
        print(f"   GraphQL Endpoints: {len(best.get('graphql_endpoints', []))}")
        print(f"   JSON Data: {'Yes' if best.get('has_json') else 'No'}")
        
        # Save strategy to cache
        strategy = {
            'domain': 'www.homedepot.com',
            'timestamp': time.time(),
            'best_config': best['name'],
            'proxy_type': 'web_unblocker' if 'Unblocker' in best['name'] else 'residential',
            'html_quality': best['html_quality'],
            'graphql_endpoints': best.get('graphql_endpoints', []),
            'has_json': best.get('has_json', False),
            'recommended_approach': 'graphql' if best.get('graphql_calls') else 'html',
        }
        
        cache_file = Path(__file__).parent / '.scraping_strategies.json'
        
        # Load existing cache
        cache = {}
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        
        cache['www.homedepot.com'] = strategy
        
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"\n💾 Strategy cached to: {cache_file}")
        print(f"\n🎯 Recommended Approach: {strategy['recommended_approach'].upper()}")
        
        if strategy['recommended_approach'] == 'graphql':
            print(f"   Use GraphQL endpoints for direct data access")
            if best.get('graphql_endpoints'):
                print(f"   Endpoints found:")
                for ep in best.get('graphql_endpoints')[:5]:
                    print(f"      - {ep}")
        else:
            print(f"   Use HTML parsing with JSONDetector")
    else:
        print("❌ No successful configurations found")
        print("\nAll tests failed. Possible issues:")
        for r in results:
            if 'error' in r:
                print(f"   - {r['name']}: {r['error'][:100]}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())
