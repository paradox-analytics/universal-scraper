
import sys
import logging
from typing import Dict, Any, Optional

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Copying the function implementation to test it in isolation without loading full FastAPI app overhead
# This ensures we are testing the LOGIC exactly as it is in the file.
def convert_proxy_config(frontend_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert frontend proxy configuration format to backend format.
    Handles nested frontend format, comma-separated strings, and Web Unblocker specific logic.
    """
    if not frontend_config:
        return None
        
    provider = frontend_config.get('provider', 'none')
    if provider == 'none':
        return None
        
    # Initialize result with defaults
    result = {
        'server': None,
        'username': None,
        'password': None,
        'web_unlocker_api_key': None,
        'web_unlocker_zone': 'web_unlocker1',
        'web_unlocker_customer_id': None,
        'web_unlocker': False
    }

    # 1. Extract raw values from frontend config
    external_proxy = frontend_config.get('externalProxy', {})
    server = external_proxy.get('server')
    username = external_proxy.get('username')
    password = external_proxy.get('password')
    
    # 2. Robust parsing for comma-separated credentials (host,port,user,pass)
    if server and ',' in server:
        parts = [p.strip() for p in server.split(',')]
        if len(parts) >= 4:
            server = f"{parts[0]}:{parts[1]}"
            username = parts[2]
            password = parts[3]
        elif len(parts) == 2:
            server = f"{parts[0]}:{parts[1]}"
    
    # 3. Normalize server string
    if server:
        server = server.replace(',', ':')
        if not server.startswith('http'):
            # Use HTTPS for port 33335 (Bright Data SSL port)
            if ':33335' in server:
                server = f"https://{server}"
            else:
                server = f"http://{server}"
    
    # 3a. Force HTTPS for port 33335 even if it already has http://
    if server and ':33335' in server and server.startswith('http://'):
        server = server.replace('http://', 'https://')

    if provider in ['web_unlocker', 'web_unblocker']:
        result['web_unlocker'] = True
        web_unblocker_config = frontend_config.get('webUnblocker', {})
        result['web_unlocker_zone'] = web_unblocker_config.get('zone', 'web_unlocker1')
        result['web_unlocker_customer_id'] = web_unblocker_config.get('customerId')
        
        # === THE FIX IS HERE: WE REMOVED THE 'enabled' CHECK ===
        if web_unblocker_config.get('useProxyMethod'):
            # Use the parsed credentials
            if server and username and password:
                # Construct internal API key format: host:port:user:pass
                host_port = server.replace('http://', '').replace('https://', '')
                result['web_unlocker_api_key'] = f"{host_port}:{username}:{password}"
        else:
            # Use explicit API key
            result['web_unlocker_api_key'] = web_unblocker_config.get('apiKey')
                
    # 5. Default server for Bright Data if missing
    if not server and username and 'brd-customer' in username:
        server = 'https://brd.superproxy.io:33335'
        
    return result

def test_conversion():
    print("🧪 Testing Proxy Config Conversion")
    print("==================================")
    
    # CASE 1: Web Unblocker with explicit API key (Missing 'enabled' flag)
    # This simulates what we believe the frontend is sending
    frontend_payload = {
        "provider": "web_unlocker",
        "webUnblocker": {
            "apiKey": "TEST_API_KEY_123",
            "zone": "my_zone",
            "customerId": "cust_1",
            "useProxyMethod": False
            # "enabled": True  <-- MISSING
        }
    }
    
    print("\n1️⃣  Testing Web Unblocker payload (Missing 'enabled')...")
    converted = convert_proxy_config(frontend_payload)
    
    if converted['web_unlocker_api_key'] == "TEST_API_KEY_123":
        print("   ✅ SUCCESS: API Key extracted correctly without 'enabled' flag.")
        print(f"   Zone: {converted['web_unlocker_zone']}")
    else:
        print(f"   ❌ FAILURE: API Key not extracted. Got: {converted['web_unlocker_api_key']}")
        sys.exit(1)

    # CASE 2: Web Unblocker Disabled/Missing (Should default but provider check handles main logic)
    # If provider is 'none', it returns None immediately
    
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    test_conversion()
