import sys
from unittest.mock import MagicMock, patch
import asyncio

# Mock the dependencies that might fail to import or initialize
sys.modules['universal_scraper.core.hybrid_fetcher'] = MagicMock()
sys.modules['universal_scraper.core.web_unblocker_fetcher'] = MagicMock()

async def test_hybrid_init_logic():
    from api.main import suggest_fields_endpoint
    
    # Mock request object
    request = MagicMock()
    request.url = "https://www.homedepot.com/p/some-product"
    request.proxy_config = None
    request.browser_timeout = 90000
    
    # Patch HybridFetcher where it's defined
    with patch('universal_scraper.core.hybrid_fetcher.HybridFetcher') as MockHybridFetcher:
        # We need to mock the await hybrid_fetcher.fetch(request.url)
        mock_instance = MockHybridFetcher.return_value
        mock_instance.fetch = asyncio.Future()
        mock_instance.fetch.set_result({'html': '<html></html>', 'status_code': 200})
        
        try:
            await suggest_fields_endpoint(request)
        except Exception as e:
            # We might get errors later in the endpoint, but we only care about the init
            print(f"Caught expected error or finished: {e}")
            
        # Check how HybridFetcher was called
        # Note: Since it's imported locally, we check the call to the MockHybridFetcher
        args, kwargs = MockHybridFetcher.call_args
        print(f"HybridFetcher called with force_mode: {kwargs.get('force_mode')}")
        
        if kwargs.get('force_mode') == 'browser':
            print("✅ SUCCESS: force_mode='browser' for Home Depot")
        else:
            print("❌ FAILURE: force_mode was not 'browser' for Home Depot")

    # Test with non-Home Depot URL
    request.url = "https://example.com"
    with patch('universal_scraper.core.hybrid_fetcher.HybridFetcher') as MockHybridFetcher:
        mock_instance = MockHybridFetcher.return_value
        mock_instance.fetch = asyncio.Future()
        mock_instance.fetch.set_result({'html': '<html></html>', 'status_code': 200})
        
        try:
            await suggest_fields_endpoint(request)
        except:
            pass
            
        args, kwargs = MockHybridFetcher.call_args
        print(f"HybridFetcher called with force_mode for example.com: {kwargs.get('force_mode')}")
        
        if kwargs.get('force_mode') is None:
            print("✅ SUCCESS: force_mode=None for example.com")
        else:
            print("❌ FAILURE: force_mode was not None for example.com")

if __name__ == "__main__":
    asyncio.run(test_hybrid_init_logic())
