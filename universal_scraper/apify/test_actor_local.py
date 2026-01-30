"""
Local testing script for Apify Actor
Allows testing the actor locally before deployment
"""

import asyncio
import json
import os
from pathlib import Path

# Mock Apify SDK for local testing
class MockActor:
 """Mock Apify Actor class for local testing"""

 def __init__(self):
 self.log = MockLogger()
 self.input_data = None
 self.output_data = []
 self.kv_store = {}

 async def __aenter__(self):
 return self

 async def __aexit__(self, exc_type, exc_val, exc_tb):
 pass

 async def get_input(self):
 """Load input from test_input.json"""
 input_file = Path(__file__).parent / 'test_input.json'
 if input_file.exists():
 with open(input_file, 'r') as f:
 self.input_data = json.load(f)
 return self.input_data
 return {}

 async def push_data(self, data):
 """Collect output data"""
 self.output_data.append(data)
 self.log.info(f" Output: {json.dumps(data, indent=2)}")

 async def set_value(self, key, value):
 """Store key-value"""
 self.kv_store[key] = value
 self.log.info(f" Stored '{key}': {json.dumps(value, indent=2)}")

 async def create_proxy_configuration(self, **kwargs):
 """Mock proxy configuration"""
 self.log.warning(' Mock proxy configuration (not using real proxies in local test)')
 return MockProxyConfig()


class MockProxyConfig:
 """Mock proxy configuration"""
 async def new_url(self):
 return 'http://user:pass@proxy.example.com:8080'


class MockLogger:
 """Mock logger"""
 def info(self, msg):
 print(f"[INFO] {msg}")

 def warning(self, msg):
 print(f"[WARN] {msg}")

 def error(self, msg):
 print(f"[ERROR] {msg}")

 def exception(self, exc):
 print(f"[EXCEPTION] {exc}")


async def test_actor():
 """Test the actor locally"""
 print(" Testing Apify Actor Locally")
 print("=" * 50)

 # Replace Actor with mock
 import sys
 from unittest.mock import MagicMock

 # Mock the apify module
 mock_apify = MagicMock()
 mock_actor = MockActor()
 mock_apify.Actor = mock_actor
 sys.modules['apify'] = mock_apify

 # Now import and run the actor
 from actor import main as actor_main

 # Patch the Actor reference in actor module
 import actor as actor_module
 actor_module.Actor = mock_actor

 # Run the actor
 try:
 await actor_main()

 print("\n" + "=" * 50)
 print(" Actor Test Complete!")
 print(f" Total items collected: {len(mock_actor.output_data)}")

 # Save output to file
 output_file = Path(__file__).parent / 'test_output.json'
 with open(output_file, 'w') as f:
 json.dump({
 'data': mock_actor.output_data,
 'kv_store': mock_actor.kv_store
 }, f, indent=2)
 print(f" Output saved to: {output_file}")

 except Exception as e:
 print(f"\n Actor test failed: {e}")
 import traceback
 traceback.print_exc()


if __name__ == '__main__':
 # Check for test input
 test_input_file = Path(__file__).parent / 'test_input.json'

 if not test_input_file.exists():
 print(" Creating sample test_input.json...")
 sample_input = {
 "urls": ["https://books.toscrape.com/"],
 "fields": ["title", "price", "rating"],
 "proxyType": "none",
 "aiModel": "gpt-4o-mini",
 "apiKeys": {
 "openai_api_key": os.environ.get('OPENAI_API_KEY', '')
 }
 }
 with open(test_input_file, 'w') as f:
 json.dump(sample_input, f, indent=2)
 print(f" Created {test_input_file}")
 print(f" Edit this file to customize your test input")
 print(f" Make sure to set OPENAI_API_KEY environment variable")
 print()

 # Run test
 asyncio.run(test_actor())


