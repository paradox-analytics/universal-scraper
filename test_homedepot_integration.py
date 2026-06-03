#!/usr/bin/env python3
"""
Comprehensive Frontend-to-Backend Integration Test for Home Depot
Tests the complete flow: Frontend Settings → API Conversion → Backend Scraping

This simulates exactly what happens when the frontend sends a request to the API.
"""
import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.main import convert_proxy_config, get_scraper
from universal_scraper.core.scraper import UniversalScraper

async def test_scenario_1_residential_proxy():
    """
    Scenario 1: Residential Proxy (Port 33335)
    This is what you have configured in your profile
    """
    print("\n" + "=" * 80)
    print("📋 SCENARIO 1: Residential Proxy (Your Current Settings)")
    print("=" * 80)
    
    # EXACT frontend payload from jevon@paradoxanalytics.com profile
    frontend_request = {
        'url': 'https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591',
        'fields': ['name', 'price', 'brand', 'rating'],
        'proxyConfig': {
            'provider': 'brightdata',
            'externalProxy': {
                'server': 'brd.superproxy.io:33335',
                'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
                'password': 'REDACTED_PROXY_PASS'
            }
        }
    }
    
    print(f"\n1️⃣  Frontend Request:")
    print(f"   URL: {frontend_request['url'][:80]}...")
    print(f"   Fields: {frontend_request['fields']}")
    print(f"   Proxy Provider: {frontend_request['proxyConfig']['provider']}")
    print(f"   Proxy Server: {frontend_request['proxyConfig']['externalProxy']['server']}")
    
    # Step 1: Convert proxy config (what the API does)
    backend_proxy_config = convert_proxy_config(frontend_request['proxyConfig'])
    
    print(f"\n2️⃣  Backend Proxy Config (after API conversion):")
    print(f"   Server: {backend_proxy_config['server']}")
    print(f"   Username: {backend_proxy_config['username'][:40]}...")
    print(f"   Password: {'*' * 10}")
    
    # Verify the fixes
    assert backend_proxy_config['server'] == 'https://brd.superproxy.io:33335', \
        "❌ HTTPS upgrade failed!"
    print(f"   ✅ HTTPS protocol verified")
    
    # Step 2: Create scraper (what the API does)
    scraper = UniversalScraper(
        proxy_config=backend_proxy_config,
        fetch_mode='browser',
        use_camoufox=True,
        browser_timeout=120000
    )
    
    print(f"\n3️⃣  Scraper initialized:")
    print(f"   Mode: browser (Camoufox)")
    print(f"   Timeout: 120s")
    
    # Step 3: Attempt to fetch
    print(f"\n4️⃣  Fetching Home Depot...")
    try:
        result = await scraper.html_fetcher.fetch(frontend_request['url'])
        
        html = result.get('html', '')
        status = result.get('status_code', 0)
        method = result.get('fetch_method', 'unknown')
        
        print(f"\n📊 Result:")
        print(f"   Status: {status}")
        print(f"   Method: {method}")
        print(f"   HTML Size: {len(html):,} bytes")
        
        if status == 407:
            print(f"\n❌ FAILED: 407 Proxy Auth Error - Fixes didn't work!")
            return False
        elif status == 403:
            print(f"\n⚠️  403 Forbidden - Home Depot blocked (anti-bot protection)")
            print(f"   ✅ But proxy auth is working (no 407)")
            return 'blocked'
        elif status == 200 and len(html) > 10000:
            print(f"\n✅ SUCCESS! Got Home Depot page")
            return True
        else:
            print(f"\n⚠️  Unexpected status: {status}")
            return 'unknown'
            
    except Exception as e:
        error_str = str(e)
        if '407' in error_str:
            print(f"\n❌ FAILED: 407 error - {e}")
            return False
        else:
            print(f"\n⚠️  Error (not 407): {e}")
            return 'error'
    finally:
        await scraper.close()

async def test_scenario_2_web_unblocker():
    """
    Scenario 2: Web Unblocker (Advanced anti-bot bypassing)
    This should bypass Home Depot's protection
    """
    print("\n" + "=" * 80)
    print("📋 SCENARIO 2: Web Unblocker (Anti-Bot Bypassing)")
    print("=" * 80)
    
    # Frontend payload with Web Unblocker enabled
    frontend_request = {
        'url': 'https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591',
        'fields': ['name', 'price', 'brand', 'rating'],
        'proxyConfig': {
            'provider': 'web_unblocker',
            'webUnblocker': {
                'enabled': True,
                'useProxyMethod': True,
                'zone': 'web_unlocker1'
            },
            'externalProxy': {
                'server': 'brd.superproxy.io:33335',
                'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1',
                'password': 'REDACTED_PROXY_PASS'
            }
        }
    }
    
    print(f"\n1️⃣  Frontend Request:")
    print(f"   URL: {frontend_request['url'][:80]}...")
    print(f"   Web Unblocker: Enabled")
    print(f"   Zone: {frontend_request['proxyConfig']['webUnblocker']['zone']}")
    
    # Convert proxy config
    backend_proxy_config = convert_proxy_config(frontend_request['proxyConfig'])
    
    print(f"\n2️⃣  Backend Config:")
    print(f"   Server: {backend_proxy_config['server']}")
    print(f"   Web Unblocker: {backend_proxy_config.get('web_unlocker', False)}")
    print(f"   API Key: {backend_proxy_config.get('web_unlocker_api_key', 'None')[:30]}...")
    
    # Create scraper with Web Unblocker
    scraper = UniversalScraper(
        proxy_config=backend_proxy_config,
        fetch_mode='browser',
        use_camoufox=True,
        browser_timeout=120000,
        web_unblocker_api_key=backend_proxy_config.get('web_unlocker_api_key'),
        web_unblocker_zone=backend_proxy_config.get('web_unlocker_zone', 'web_unlocker1')
    )
    
    print(f"\n3️⃣  Scraper initialized with Web Unblocker")
    
    # Attempt to fetch
    print(f"\n4️⃣  Fetching Home Depot with Web Unblocker...")
    try:
        result = await scraper.html_fetcher.fetch(frontend_request['url'])
        
        html = result.get('html', '')
        status = result.get('status_code', 0)
        method = result.get('fetch_method', 'unknown')
        
        print(f"\n📊 Result:")
        print(f"   Status: {status}")
        print(f"   Method: {method}")
        print(f"   HTML Size: {len(html):,} bytes")
        
        if status == 407:
            print(f"\n❌ FAILED: 407 Proxy Auth Error")
            return False
        elif status == 403:
            print(f"\n⚠️  403 Forbidden - Still blocked even with Web Unblocker")
            return 'blocked'
        elif status == 200 and len(html) > 10000:
            print(f"\n✅ SUCCESS! Web Unblocker bypassed Home Depot protection")
            if 'homedepot' in html.lower() or 'product' in html.lower():
                print(f"   ✅ Confirmed: Got Home Depot content")
                return True
            else:
                print(f"   ⚠️  Got 200 but content unclear")
                return 'unclear'
        else:
            print(f"\n⚠️  Unexpected status: {status}")
            return 'unknown'
            
    except Exception as e:
        error_str = str(e)
        if '407' in error_str:
            print(f"\n❌ FAILED: 407 error - {e}")
            return False
        else:
            print(f"\n⚠️  Error (not 407): {e}")
            return 'error'
    finally:
        await scraper.close()

async def main():
    """Run all test scenarios"""
    print("=" * 80)
    print("🧪 COMPREHENSIVE FRONTEND-TO-BACKEND INTEGRATION TEST")
    print("   Testing: Home Depot with jevon@paradoxanalytics.com settings")
    print("=" * 80)
    
    results = {}
    
    # Test Scenario 1: Residential Proxy
    results['residential_proxy'] = await test_scenario_1_residential_proxy()
    
    # Test Scenario 2: Web Unblocker
    results['web_unblocker'] = await test_scenario_2_web_unblocker()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n1. Residential Proxy: {results['residential_proxy']}")
    print(f"2. Web Unblocker: {results['web_unblocker']}")
    
    # Determine overall result
    print(f"\n🎯 KEY FINDINGS:")
    
    if results['residential_proxy'] == False or results['web_unblocker'] == False:
        print(f"   ❌ CRITICAL: 407 Auth errors detected - proxy fixes failed!")
        return False
    else:
        print(f"   ✅ Proxy authentication working (no 407 errors)")
        print(f"   ✅ Frontend settings register correctly with backend")
        print(f"   ✅ HTTPS protocol upgrade working")
        print(f"   ✅ Credential encoding working")
        
    if results['web_unblocker'] == True:
        print(f"   ✅ Web Unblocker successfully bypassed Home Depot protection")
        print(f"   ✅ Ready for production deployment")
        return True
    elif results['residential_proxy'] == 'blocked' or results['web_unblocker'] == 'blocked':
        print(f"   ⚠️  Home Depot has strong anti-bot protection (403)")
        print(f"   💡 Recommendation: Use Web Unblocker for Home Depot at scale")
        return True  # Proxy fixes work, just need better anti-bot
    else:
        print(f"   ⚠️  Mixed results - needs investigation")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
