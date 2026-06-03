#!/usr/bin/env python3
"""
Real-World Test: Adaptive Anti-Blocking with Home Depot

Tests the adaptive system against Home Depot with production proxy settings.
This will test configuration learning, blocking detection, and adaptive strategies.
"""
import asyncio
import sys
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.adaptive_antiblocking_agent import AdaptiveAntiBlockingAgent
from universal_scraper.core.browser_config_generator import ConfigPreset
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from api.main import convert_proxy_config

# Load environment variables from .env
load_dotenv()

async def test_home_depot_adaptive():
    """Test Home Depot with adaptive anti-blocking"""
    
    print("=" * 80)
    print("🏠 HOME DEPOT - ADAPTIVE ANTI-BLOCKING TEST")
    print("=" * 80)
    
    # Use credentials from .env
    # IMPORTANT: Using RESIDENTIAL proxy instead of Web Unblocker
    # Web Unblocker has a global 5000 req/min rate limit for Home Depot
    proxy_url = os.getenv('BRIGHT_DATA_PROXY_URL')  # residential_proxy2
    if not proxy_url:
        print("❌ ERROR: BRIGHT_DATA_PROXY_URL not found in .env")
        return False
    
    # Parse the proxy URL: http://user:pass@host:port
    import re
    match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', proxy_url)
    if not match:
        print(f"❌ ERROR: Invalid proxy URL format: {proxy_url}")
        return False
    
    username, password, host, port = match.groups()

    # Production proxy configuration (using Residential Proxy)
    frontend_proxy_config = {
        'provider': 'brightdata',
        'externalProxy': {
            'server': f"{host}:{port}",
            'username': username,
            'password': password
        }
    }
    
    backend_proxy_config = convert_proxy_config(frontend_proxy_config)
    
    print(f"\n📋 Proxy Configuration:")
    print(f"   Provider: {frontend_proxy_config['provider']}")
    print(f"   Zone: {backend_proxy_config.get('web_unlocker_zone')}")
    print(f"   Server: {backend_proxy_config['server']}")
    
    # Home Depot URL
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    
    print(f"\n🎯 Target: {url[:80]}...")
    
    # Initialize adaptive agent
    agent = AdaptiveAntiBlockingAgent(
        max_parallel_tests=3,
        enable_llm_optimization=False
    )
    
    # Set initial timeout to 300s
    agent.learner.get_optimal_timeout = lambda domain, default_timeout=300000: 300000
    
    print(f"\n🤖 Adaptive Agent initialized")
    print(f"   Max parallel tests: 3")
    print(f"   Starting preset: BALANCED")
    
    # Progress tracking
    progress_updates = []
    
    async def progress_callback(update):
        """Track progress updates"""
        progress_updates.append(update)
        stage = update.get('stage', 'unknown')
        print(f"\n📊 Progress: {stage}")
        if 'attempt' in update:
            print(f"   Attempt {update['attempt']}/{update.get('max_attempts', '?')}")
    
    # Create fetcher function that uses CamoufoxFetcher
    async def fetch_with_camoufox(url, camoufox_config):
        """Wrapper to fetch with Camoufox using adaptive config"""
        print(f"\n🔧 Testing configuration:")
        print(f"   Humanize: {camoufox_config.get('humanize')}")
        print(f"   GeoIP: {camoufox_config.get('geoip')}")
        print(f"   Stealth: {camoufox_config.get('stealth')}")
        print(f"   Viewport: {camoufox_config.get('screen', {}).get('width')}x{camoufox_config.get('screen', {}).get('height')}")
        
        # Create Camoufox fetcher with residential proxy
        fetcher = CamoufoxFetcher(
            proxy_config=backend_proxy_config,
            headless=True,
            timeout=camoufox_config.get('timeout', 300000),
            anti_detection_profile='random',
            humanize=camoufox_config.get('humanize', True),
            stealth_mode=camoufox_config.get('stealth', False),
            geoip=camoufox_config.get('geoip', False) # Use config from agent
        )
        
        try:
            # Fetch using async Camoufox
            result = await fetcher.fetch(
                url=url,
                wait_for_selector="h1",
                wait_time=5000, # More wait for Home Depot
                scroll_to_bottom=False
            )
            
            return {
                'html': result.get('html', ''),
                'status_code': result.get('status_code', 0),
                'headers': result.get('headers', {}),
                'error': result.get('error', '')
            }
            
        except Exception as e:
            print(f"   ⚠️  Fetch error: {str(e)[:100]}")
            return {
                'html': '',
                'status_code': 0,
                'headers': {},
                'error': str(e)
            }
        finally:
            await fetcher.close()
    
    # Run adaptive fetch loop
    print(f"\n🚀 Starting adaptive fetch loop...")
    print(f"   This will test multiple configurations and learn what works")
    
    start_time = time.time()
    
    max_cycles = 2
    result = {}
    
    for cycle in range(1, max_cycles + 1):
        print(f"\n" + "=" * 40)
        print(f"🔄 ADAPTIVE CYCLE {cycle}/{max_cycles}")
        print(f"=" * 40)
        
        result = await agent.fetch_with_adaptation(
            url=url,
            fetcher_func=fetch_with_camoufox,
            preset=ConfigPreset.BALANCED,
            max_attempts=5,
            progress_callback=progress_callback
        )
        
        if result.get('success'):
            print(f"\n✅ Cycle {cycle} Successful!")
            break
        else:
            print(f"\n⚠️ Cycle {cycle} Failed. Agent should learn from this.")
            # Small pause
            await asyncio.sleep(2)
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    print(f"\n" + "=" * 80)
    print(f"📊 RESULTS")
    print(f"=" * 80)
    
    print(f"\nFetch Result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Status Code: {result.get('status_code', 0)}")
    print(f"   HTML Size: {len(result.get('html', '')):,} bytes")
    print(f"   Time Elapsed: {elapsed_time:.1f}s")
    
    # Blocking analysis
    if 'blocking_analysis' in result:
        analysis = result['blocking_analysis']
        print(f"\nBlocking Analysis:")
        print(f"   Type: {analysis.get('type_name', 'unknown')}")
        print(f"   Confidence: {analysis.get('confidence', 0):.2%}")
        print(f"   Is Blocked: {analysis.get('is_blocked', False)}")
        print(f"   Details: {analysis.get('details', 'N/A')}")
    
    # Configuration used
    if 'config_used' in result:
        config = result['config_used']
        print(f"\nConfiguration Used:")
        print(f"   Block Images: {config.get('blockImages')}")
        print(f"   Generate Canvas: {config.get('generateCanvasString')}")
        print(f"   Rotate Profile: {config.get('rotateProfile')}")
        print(f"   Proxy Rotation: {config.get('proxyRotationInterval')}")
    
    # Recommendations
    if 'recommendations' in result:
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Domain insights
    print(f"\n" + "=" * 80)
    print(f"🧠 LEARNING INSIGHTS")
    print(f"=" * 80)
    
    insights = agent.get_domain_insights('www.homedepot.com')
    stats = insights['stats']
    
    print(f"\nDomain: www.homedepot.com")
    print(f"   Total Attempts: {stats['total_attempts']}")
    print(f"   Successes: {stats.get('successes', 0)}")
    print(f"   Failures: {stats.get('failures', 0)}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    print(f"   Configs Tried: {stats['configs_tried']}")
    print(f"   Has Learned Config: {insights['has_learned_config']}")
    
    if stats.get('blocking_breakdown'):
        print(f"\nBlocking Breakdown:")
        for block_type, count in stats['blocking_breakdown'].items():
            print(f"   {block_type}: {count} times")
    
    # Progress summary
    print(f"\n" + "=" * 80)
    print(f"📈 PROGRESS SUMMARY")
    print(f"=" * 80)
    
    print(f"\nStages Completed: {len(progress_updates)}")
    for update in progress_updates:
        print(f"   - {update.get('stage')}")
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print(f"🎯 VERDICT")
    print(f"=" * 80)
    
    if result.get('success'):
        print(f"\n✅ SUCCESS! Adaptive system bypassed Home Depot's protection")
        print(f"   The system learned what works and can reuse this config")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS - System is learning")
        print(f"   Blocking Type: {result.get('blocking_analysis', {}).get('type_name', 'unknown')}")
        print(f"   Next Steps:")
        print(f"   1. The system has learned from these attempts")
        print(f"   2. Future requests will try different configurations")
        print(f"   3. Consider enabling Web Unblocker for Home Depot")
        return False

async def main():
    """Run the test"""
    try:
        success = await test_home_depot_adaptive()
        return success
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
