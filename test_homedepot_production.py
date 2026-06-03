#!/usr/bin/env python3
"""
Test Home Depot with Production Proxy Settings
Replicates the exact frontend configuration to verify proxy fixes work in production
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper
from dotenv import load_dotenv

load_dotenv()

async def test_homedepot_production():
    """Test Home Depot with exact production proxy settings"""
    
    print("=" * 80)
    print("🧪 Testing Home Depot with Production Proxy Settings")
    print("=" * 80)
    
    # EXACT PRODUCTION SETTINGS (from jevon@paradoxanalytics profile)
    # This replicates what the frontend sends to the API
    frontend_proxy_config = {
        'provider': 'brightdata',
        'externalProxy': {
            'server': 'brd.superproxy.io:33335',
            'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
            'password': 'rs2mvj79xi2t'
        }
    }
    
    # Convert using the same logic as api/main.py
    from api.main import convert_proxy_config
    backend_proxy_config = convert_proxy_config(frontend_proxy_config)
    
    print(f"\n📋 Frontend Config:")
    print(f"   Provider: {frontend_proxy_config['provider']}")
    print(f"   Server: {frontend_proxy_config['externalProxy']['server']}")
    print(f"   Username: {frontend_proxy_config['externalProxy']['username']}")
    print(f"   Password: {'*' * len(frontend_proxy_config['externalProxy']['password'])}")
    
    print(f"\n📋 Backend Config (after conversion):")
    print(f"   Server: {backend_proxy_config['server']}")
    print(f"   Username: {backend_proxy_config['username']}")
    print(f"   Password: {'*' * len(backend_proxy_config['password'])}")
    
    # Verify the fixes are applied
    assert backend_proxy_config['server'] == 'https://brd.superproxy.io:33335', \
        f"Expected https:// for port 33335, got {backend_proxy_config['server']}"
    print("\n✅ Protocol fix verified: Using HTTPS for port 33335")
    
    # Test Home Depot URL
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"
    fields = ["name", "price", "rating", "brand"]
    
    print(f"\n🎯 Target URL: {url}")
    print(f"📊 Fields: {', '.join(fields)}")
    
    # Initialize scraper with production settings
    scraper = UniversalScraper(
        api_key=None,  # No LLM needed for this test
        proxy_config=backend_proxy_config,
        fetch_mode='browser',  # Force browser mode (Home Depot requires it)
        use_camoufox=True,
        browser_timeout=120000
    )
    
    try:
        print(f"\n🚀 Fetching with Camoufox + Proxy...")
        print(f"   This will test:")
        print(f"   1. HTTPS protocol for port 33335")
        print(f"   2. URL-encoded credentials")
        print(f"   3. Camoufox anti-detection")
        
        # Just fetch the page (no extraction needed to test proxy)
        result = await scraper.html_fetcher.fetch(url)
        
        html = result.get('html', '')
        status = result.get('status_code', 0)
        fetch_method = result.get('fetch_method', 'unknown')
        
        print(f"\n📊 Fetch Result:")
        print(f"   Status: {status}")
        print(f"   Method: {fetch_method}")
        print(f"   HTML Size: {len(html):,} bytes")
        
        # Check for success indicators
        if status == 200 and len(html) > 10000:
            # Check for Home Depot content
            if 'homedepot' in html.lower() or 'product' in html.lower():
                print(f"\n✅ SUCCESS! Home Depot page fetched successfully")
                print(f"   - No 407 Auth errors")
                print(f"   - No connection refused errors")
                print(f"   - Proxy fixes are working!")
                return True
            else:
                print(f"\n⚠️  WARNING: Got HTML but doesn't look like Home Depot")
                print(f"   First 200 chars: {html[:200]}")
        else:
            print(f"\n❌ FAILED: Status {status}, HTML size {len(html)}")
            if result.get('error'):
                print(f"   Error: {result['error']}")
            if len(html) < 1000 and html:
                print(f"   HTML preview: {html[:500]}")
        
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await scraper.close()

if __name__ == "__main__":
    success = asyncio.run(test_homedepot_production())
    sys.exit(0 if success else 1)
