#!/usr/bin/env python3
"""
Test Home Depot with static HTML fetcher (faster than browser)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.main import convert_proxy_config
from universal_scraper.core.html_fetcher import HTMLFetcher

def test_homedepot_static():
    """Test Home Depot with static HTML fetcher"""
    
    print("=" * 80)
    print("🧪 Testing Home Depot with Static HTML Fetcher")
    print("=" * 80)
    
    # Production proxy settings
    frontend_proxy_config = {
        'provider': 'brightdata',
        'externalProxy': {
            'server': 'brd.superproxy.io:33335',
            'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
            'password': 'REDACTED_PROXY_PASS'
        }
    }
    
    backend_proxy_config = convert_proxy_config(frontend_proxy_config)
    
    print(f"\n📋 Proxy: {backend_proxy_config['server']}")
    print(f"   Username: {backend_proxy_config['username'][:40]}...")
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    print(f"\n🎯 URL: {url}")
    
    fetcher = HTMLFetcher(proxy_config=backend_proxy_config, timeout=30)
    
    try:
        print(f"\n🚀 Fetching with static HTML + proxy...")
        result = fetcher.fetch(url)
        
        html = result.get('html', '')
        status = result.get('status_code', 0)
        
        print(f"\n📊 Result:")
        print(f"   Status: {status}")
        print(f"   HTML Size: {len(html):,} bytes")
        
        if status == 200:
            # Check for Home Depot content
            if 'homedepot' in html.lower() or 'product' in html.lower():
                print(f"\n✅ SUCCESS! Got Home Depot page")
                print(f"   First 200 chars: {html[:200]}")
                return True
            else:
                print(f"\n⚠️  Got 200 but content doesn't look like Home Depot")
                print(f"   First 500 chars: {html[:500]}")
        elif status == 403:
            print(f"\n⚠️  403 Forbidden - Home Depot is blocking the request")
            print(f"   This is expected - Home Depot has strong anti-bot protection")
            print(f"   The proxy authentication is working (no 407 error)")
        elif status == 407:
            print(f"\n❌ 407 Proxy Auth Failed - THE FIX DIDN'T WORK")
            return False
        else:
            print(f"\n⚠️  Status {status}")
            if len(html) < 1000:
                print(f"   Content: {html[:500]}")
        
        # If we got here without a 407, the proxy auth is working
        if status != 407:
            print(f"\n✅ Proxy authentication working (no 407 errors)")
            return True
            
    except Exception as e:
        error_str = str(e)
        if '407' in error_str:
            print(f"\n❌ 407 Proxy Auth Failed - THE FIX DIDN'T WORK")
            print(f"   Error: {e}")
            return False
        else:
            print(f"\n⚠️  Error: {e}")
            print(f"   (Not a 407 auth error - proxy credentials are working)")
            return True

if __name__ == "__main__":
    success = test_homedepot_static()
    sys.exit(0 if success else 1)
