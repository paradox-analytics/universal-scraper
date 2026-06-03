#!/usr/bin/env python3
"""
Quick proxy connection test - verify the fixes work with a simple URL first
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.main import convert_proxy_config
from universal_scraper.core.html_fetcher import HTMLFetcher

async def test_proxy_connection():
    """Test proxy connection with a simple, fast-loading URL"""
    
    print("=" * 80)
    print("🧪 Quick Proxy Connection Test")
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
    
    print(f"\n📋 Proxy Config:")
    print(f"   Server: {backend_proxy_config['server']}")
    print(f"   Username: {backend_proxy_config['username'][:40]}...")
    print(f"   Password: {'*' * 10}")
    
    # Verify HTTPS upgrade
    assert 'https://' in backend_proxy_config['server'], "Should use HTTPS for port 33335"
    print(f"\n✅ Protocol check passed: Using HTTPS")
    
    # Test with a simple, fast URL
    test_url = "https://httpbin.org/ip"
    print(f"\n🎯 Testing with: {test_url}")
    
    fetcher = HTMLFetcher(proxy_config=backend_proxy_config)
    
    try:
        result = fetcher.fetch(test_url)
        html = result.get('html', '')
        status = result.get('status_code', 0)
        
        print(f"\n📊 Result:")
        print(f"   Status: {status}")
        print(f"   HTML Size: {len(html)} bytes")
        print(f"   Content: {html[:200]}")
        
        if status == 200:
            print(f"\n✅ SUCCESS! Proxy connection working")
            print(f"   - No 407 Auth errors")
            print(f"   - No connection refused errors")
            print(f"   - Credentials are properly encoded")
            return True
        else:
            print(f"\n❌ FAILED: Status {status}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_proxy_connection())
    sys.exit(0 if success else 1)
