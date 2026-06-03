import sys
from pathlib import Path
from urllib.parse import quote

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.proxy_manager import ProxyManager
from universal_scraper.core.html_fetcher import HTMLFetcher
from api.main import convert_proxy_config

def test_proxy_manager_encoding():
    print("\n--- Testing ProxyManager Encoding ---")
    pm = ProxyManager()
    
    # Test with special characters in password
    server = "brd.superproxy.io:33335"
    user = "user@name"
    password = "pass:word"
    
    proxy_url = pm._build_proxy_url(server, user, password)
    print(f"Server: {server}")
    print(f"User: {user}")
    print(f"Pass: {password}")
    print(f"Resulting URL: {proxy_url}")
    
    assert "https://" in proxy_url
    assert quote(user) in proxy_url
    assert quote(password) in proxy_url
    print("✅ ProxyManager encoding and protocol test passed!")

def test_api_main_conversion():
    print("\n--- Testing api/main.py convert_proxy_config ---")
    
    # Test Bright Data default with port 33335
    frontend_config = {
        'provider': 'brightdata',
        'externalProxy': {
            'username': 'brd-customer-hl_123-zone-res',
            'password': 'password123'
        }
    }
    
    result = convert_proxy_config(frontend_config)
    print(f"Resulting server: {result['server']}")
    assert result['server'] == "https://brd.superproxy.io:33335"
    
    # Test explicit port 33335 upgrade
    frontend_config['externalProxy']['server'] = "brd.superproxy.io:33335"
    result = convert_proxy_config(frontend_config)
    print(f"Explicit server result: {result['server']}")
    assert result['server'] == "https://brd.superproxy.io:33335"
    
    # Test comma-separated parsing
    frontend_config['externalProxy']['server'] = "brd.superproxy.io,33335,user,pass"
    result = convert_proxy_config(frontend_config)
    print(f"Comma-separated result: {result['server']}")
    assert result['server'] == "https://brd.superproxy.io:33335"
    
    print("✅ api/main.py conversion test passed!")

def test_html_fetcher_session_proxies():
    print("\n--- Testing HTMLFetcher Session Proxies ---")
    
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'user@name',
        'password': 'pass:word'
    }
    
    fetcher = HTMLFetcher(proxy_config=proxy_config)
    proxies = fetcher.session.proxies
    print(f"Session proxies: {proxies}")
    
    proxy_url = proxies['http']
    assert "https://" in proxy_url
    assert quote(proxy_config['username']) in proxy_url
    assert quote(proxy_config['password']) in proxy_url
    print("✅ HTMLFetcher session proxies test passed!")

if __name__ == "__main__":
    try:
        test_proxy_manager_encoding()
        test_api_main_conversion()
        test_html_fetcher_session_proxies()
        print("\n✨ ALL PROXY FIX TESTS PASSED! ✨")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
