import os
import logging
from typing import Optional, Dict, Any

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the function to test
import sys
sys.path.append('.')
from api.main import convert_proxy_config

def test_parsing():
    test_cases = [
        {
            "name": "Comma-separated full string in server field",
            "config": {
                "provider": "brightdata",
                "externalProxy": {
                    "server": "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS",
                    "username": "",
                    "password": ""
                }
            },
            "expected_server": "http://brd.superproxy.io:33335",
            "expected_user": "brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1",
            "expected_pass": "REDACTED_PROXY_PASS"
        },
        {
            "name": "Web Unblocker with comma-separated string",
            "config": {
                "provider": "web_unlocker",
                "webUnblocker": {"enabled": True, "useProxyMethod": True, "zone": "web_unlocker1"},
                "externalProxy": {
                    "server": "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS",
                    "username": "",
                    "password": ""
                }
            },
            "expected_server": "http://brd.superproxy.io:33335",
            "expected_user": "brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1",
            "expected_pass": "REDACTED_PROXY_PASS",
            "expected_api_key": "brd.superproxy.io:33335:brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS"
        },
        {
            "name": "Standard nested config",
            "config": {
                "provider": "custom",
                "externalProxy": {
                    "server": "myproxy.com:8080",
                    "username": "user1",
                    "password": "pass:with:colons"
                }
            },
            "expected_server": "http://myproxy.com:8080",
            "expected_user": "user1",
            "expected_pass": "pass:with:colons"
        }
    ]

    for case in test_cases:
        print(f"\n🧪 Testing: {case['name']}")
        result = convert_proxy_config(case['config'])
        
        if not result:
            print("❌ Failed: Result is None")
            continue
            
        print(f"Result: {result}")
        
        assert result['server'] == case['expected_server'], f"Expected server {case['expected_server']}, got {result['server']}"
        assert result['username'] == case['expected_user'], f"Expected user {case['expected_user']}, got {result['username']}"
        assert result['password'] == case['expected_pass'], f"Expected pass {case['expected_pass']}, got {result['password']}"
        
        if 'expected_api_key' in case:
            assert result['web_unlocker_api_key'] == case['expected_api_key'], f"Expected API key {case['expected_api_key']}, got {result['web_unlocker_api_key']}"
            
        print("✅ Passed!")

if __name__ == "__main__":
    test_parsing()
