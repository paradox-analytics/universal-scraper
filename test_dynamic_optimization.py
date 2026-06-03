#!/usr/bin/env python3
"""
Test Dynamic Optimization with Home Depot

This test demonstrates the FULLY AUTONOMOUS adaptive system:
- Dynamically adjusts timeout based on previous attempts
- Selects optimal strategy (preset, Web Unblocker, etc.)
- Learns and adapts in real-time
- No user intervention required
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.adaptive_antiblocking_agent import AdaptiveAntiBlockingAgent
from universal_scraper.core.browser_config_generator import ConfigPreset
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from api.main import convert_proxy_config

async def test_dynamic_optimization():
    """Test dynamic optimization with Home Depot"""
    
    print("=" * 80)
    print("🤖 DYNAMIC OPTIMIZATION TEST - Home Depot")
    print("   Fully autonomous - no user intervention required")
    print("=" * 80)
    
    # Production proxy
    frontend_proxy_config = {
        'provider': 'brightdata',
        'externalProxy': {
            'server': 'brd.superproxy.io:33335',
            'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
            'password': 'REDACTED_PROXY_PASS'
        }
    }
    
    backend_proxy_config = convert_proxy_config(frontend_proxy_config)
    
    # Home Depot URL
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    
    # Initialize adaptive agent
    agent = AdaptiveAntiBlockingAgent(
        max_parallel_tests=2,  # Reduced for faster testing
        enable_llm_optimization=False
    )
    
    print(f"\n🎯 Target: {url[:80]}...")
    
    # Progress tracking
    async def progress_callback(update):
        """Show progress"""
        stage = update.get('stage', 'unknown')
        
        if stage == 'strategy_selected':
            strategy = update['strategy']
            timeout = update['timeout']
            print(f"\n🧠 INTELLIGENT STRATEGY SELECTED:")
            print(f"   Preset: {strategy['preset']}")
            print(f"   Timeout: {timeout}ms ({timeout/1000:.0f}s)")
            print(f"   Web Unblocker: {strategy['use_web_unblocker']}")
            print(f"   Proxy Rotation: {strategy['proxy_rotation']}")
            print(f"   Reason: {strategy['reason']}")
        elif stage == 'trying_web_unblocker':
            print(f"\n🛡️  Trying Web Unblocker (recommended by strategy)")
        elif stage == 'testing_variations':
            print(f"   Testing variation {update['attempt']}/{update['max_attempts']}")
    
    # Web Unblocker Credentials (for when system decides to switch)
    web_unblocker_config = {
        'server': 'https://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1',
        'password': 'REDACTED_PROXY_PASS'
    }

    # Create fetcher function
    async def fetch_with_camoufox(url, camoufox_config):
        """Fetch with Camoufox using adaptive config"""
        timeout = camoufox_config.get('timeout', 60000)
        
        # Determine which proxy to use
        current_proxy = backend_proxy_config
        
        # If system recommends Web Unblocker, switch credentials
        if camoufox_config.get('use_web_unblocker'):
            print(f"   🔄 Switching to Web Unblocker credentials as requested...")
            current_proxy = web_unblocker_config
            
        fetcher = CamoufoxFetcher(
            proxy_config=current_proxy,
            headless=True,
            timeout=timeout,  # Use dynamic timeout
            anti_detection_profile='random',
            humanize=True,
            stealth_mode=True
        )
        
        try:
            result = await fetcher.fetch(
                url=url,
                wait_for_selector=None,
                wait_time=2000,
                scroll_to_bottom=False
            )
            
            return {
                'html': result.get('html', ''),
                'status_code': result.get('status_code', 0),
                'headers': result.get('headers', {}),
                'error': result.get('error', '')
            }
            
        except Exception as e:
            return {
                'html': '',
                'status_code': 0,
                'headers': {},
                'error': str(e)
            }
        finally:
            await fetcher.close()
    
    # Run adaptive fetch - REQUEST 1 (Skipped for speed, simulating learning instead)
    print(f"\n🚀 REQUEST 1: Initial Attempt (Skipped, simulating learning)...")
    print(f"   Simulating 10 failed attempts to trigger adaptive logic...")
    
    # Simulate failed attempts to bump up the timeout count
    for _ in range(10):
        agent.learner.record_attempt(
            domain='www.homedepot.com', 
            config={'timeout': 60000}, 
            success=False,
            blocking_type='timeout',
            response_time=60.0
        )
    

        
    # Run adaptive fetch - REQUEST 2
    print(f"\n\n🚀 REQUEST 2: Adaptive Attempt (Applying Learned Strategy)...")
    print(f"   System should automatically use optimized settings (Web Unblocker + High Timeout)")
    
    start_time = time.time()
    
    result = await agent.fetch_with_adaptation(
        url=url,
        fetcher_func=fetch_with_camoufox,
        preset=ConfigPreset.BALANCED, # Pass same preset, agent should override it
        max_attempts=3,
        progress_callback=progress_callback
    )
    
    elapsed_time = time.time() - start_time
    

    
    # Results
    print(f"\n" + "=" * 80)
    print(f"📊 RESULTS")
    print(f"=" * 80)
    
    print(f"\nFetch Result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Status Code: {result.get('status_code', 0)}")
    print(f"   HTML Size: {len(result.get('html', '')):,} bytes")
    print(f"   Time Elapsed: {elapsed_time:.1f}s")
    print(f"   Strategy Used: {result.get('strategy_used', 'unknown')}")
    
    if 'timeout_used' in result:
        print(f"   Timeout Used: {result['timeout_used']}ms ({result['timeout_used']/1000:.0f}s)")
    
    # Blocking analysis
    if 'blocking_analysis' in result:
        analysis = result['blocking_analysis']
        print(f"\nBlocking Analysis:")
        print(f"   Type: {analysis.get('type_name', 'unknown')}")
        print(f"   Confidence: {analysis.get('confidence', 0):.2%}")
        print(f"   Details: {analysis.get('details', 'N/A')}")
    
    # Next strategy (if failed)
    if 'next_strategy' in result:
        next_strat = result['next_strategy']
        print(f"\nNext Strategy (for future attempts):")
        print(f"   Preset: {next_strat['preset']}")
        print(f"   Web Unblocker: {next_strat['use_web_unblocker']}")
        print(f"   Reason: {next_strat['reason']}")
    
    # Domain insights
    print(f"\n" + "=" * 80)
    print(f"🧠 LEARNING INSIGHTS")
    print(f"=" * 80)
    
    insights = agent.get_domain_insights('www.homedepot.com')
    stats = insights['stats']
    
    print(f"\nDomain: www.homedepot.com")
    print(f"   Total Attempts: {stats['total_attempts']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    print(f"   Configs Tried: {stats['configs_tried']}")
    print(f"   Has Learned Config: {insights['has_learned_config']}")
    
    # Verdict
    print(f"\n" + "=" * 80)
    print(f"🎯 VERDICT")
    print(f"=" * 80)
    
    if result.get('success'):
        print(f"\n✅ SUCCESS! System found working configuration")
        print(f"   Strategy: {result.get('strategy_used')}")
        print(f"   This config will be reused for future requests")
        return True
    else:
        print(f"\n🔄 LEARNING IN PROGRESS")
        print(f"   System is learning optimal settings for Home Depot")
        print(f"   Each attempt improves the strategy")
        print(f"   Next attempt will use: {result.get('next_strategy', {}).get('preset', 'unknown')}")
        
        if result.get('next_strategy', {}).get('use_web_unblocker'):
            print(f"   💡 System recommends Web Unblocker for this site")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(test_dynamic_optimization())
    sys.exit(0 if success else 1)
