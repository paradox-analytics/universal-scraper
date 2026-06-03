"""
Test different Apify proxy configurations to find what works
"""

import os

def show_proxy_configs():
    """Show all possible Apify proxy configurations"""
    
    password = os.getenv('APIFY_PROXY_PASSWORD', 'NOT_SET')
    
    print("\n" + "="*80)
    print("🔍 APIFY PROXY CONFIGURATION GUIDE")
    print("="*80)
    
    print(f"\n📋 Current Settings:")
    print(f"   Password set: {'✅ Yes' if password != 'NOT_SET' else '❌ No'}")
    print(f"   Password: {password[:10]}..." if password != 'NOT_SET' else "   Password: NOT SET")
    
    print(f"\n🌐 Available Apify Proxy Services:")
    
    configs = [
        {
            "name": "Residential Proxies (Recommended)",
            "server": "http://proxy.apify.com:8000",
            "username": "groups-RESIDENTIAL",
            "description": "Best for anti-bot detection, rotates IPs"
        },
        {
            "name": "Residential Proxies (US only)",
            "server": "http://proxy.apify.com:8000",
            "username": "groups-RESIDENTIAL+country-US",
            "description": "US residential IPs only"
        },
        {
            "name": "Datacenter Proxies",
            "server": "http://proxy.apify.com:8000",
            "username": "groups-DATACENTER",
            "description": "Faster but easier to detect"
        },
        {
            "name": "Auto (Current - FAILING)",
            "server": "http://proxy.apify.com:8000",
            "username": "auto",
            "description": "Auto-select - but seems to fail"
        },
        {
            "name": "Account-specific (paradox-analytics)",
            "server": "http://proxy.apify.com:8000",
            "username": f"<your-apify-username>",
            "description": "Direct account username"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Server: {config['server']}")
        print(f"   Username: {config['username']}")
        print(f"   Password: <APIFY_PROXY_PASSWORD>")
        print(f"   📝 {config['description']}")
    
    print("\n" + "="*80)
    print("🔧 RECOMMENDED FIX")
    print("="*80)
    
    print(f"\n✅ Use RESIDENTIAL proxies for best anti-detection:")
    print(f"\n```python")
    print(f"proxy_config = {{")
    print(f"    'server': 'http://proxy.apify.com:8000',")
    print(f"    'username': 'groups-RESIDENTIAL',  # ← Change from 'auto'")
    print(f"    'password': '{password[:10]}...'")
    print(f"}}")
    print(f"```")
    
    print(f"\n📚 Apify Proxy Documentation:")
    print(f"   https://docs.apify.com/proxy")
    
    print(f"\n🔐 To check your Apify account settings:")
    print(f"   1. Go to: https://console.apify.com/proxy")
    print(f"   2. Find your username under 'paradox-analytics' account")
    print(f"   3. Verify residential proxies are enabled")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    show_proxy_configs()







