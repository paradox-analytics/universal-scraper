import asyncio
import logging
from unittest.mock import MagicMock, patch
from universal_scraper.core.hybrid_fetcher import HybridFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hybrid_fetcher_logic():
    # Test Case 1: Web Unblocker with Bearer Token
    proxy_config = {
        'web_unlocker_api_key': 'test-bearer-token',
        'web_unlocker_zone': 'web_unlocker1',
        'web_unlocker': True
    }
    
    url = "https://www.homedepot.com/test"
    
    # Mock WebUnblockerFetcher
    with patch('universal_scraper.core.hybrid_fetcher.WebUnblockerFetcher') as MockUnblocker:
        mock_instance = MockUnblocker.return_value
        mock_instance.fetch_async = MagicMock(side_effect=lambda *args, **kwargs: asyncio.Future())
        # Actually, AsyncMock is better
        from unittest.mock import AsyncMock
        mock_instance.fetch_async = AsyncMock(return_value={'html': '<html>Home Depot Content</html>', 'status': 200})
        
        # Initialize HybridFetcher WITHOUT force_mode="browser"
        fetcher = HybridFetcher(
            web_unblocker_api_key='test-bearer-token',
            web_unblocker_zone='web_unlocker1',
            force_mode=None
        )
        
        logger.info(f"Web Unblocker Fetcher initialized: {fetcher.web_unblocker_fetcher is not None}")
        
        logger.info("Testing HybridFetcher with Web Unblocker (Bearer Token)...")
        result = await fetcher.fetch(url)
        
        logger.info(f"Fetch method used: {result.get('fetch_method')}")
        assert result.get('fetch_method') == 'web_unblocker'
        mock_instance.fetch_async.assert_called_once()

    # Test Case 2: Forced Browser Mode (Current failing state)
    with patch('universal_scraper.core.camoufox_fetcher.CamoufoxFetcher') as MockCamoufox:
        mock_cf = MockCamoufox.return_value
        mock_cf.fetch.return_value = {'html': '<html>Browser Content</html>', 'status': 200}
        mock_cf._launch_browser.return_value = None
        
        fetcher = HybridFetcher(
            web_unblocker_api_key='test-bearer-token',
            web_unblocker_zone='web_unlocker1',
            force_mode='browser',
            use_camoufox=True
        )
        
        logger.info("Testing HybridFetcher with forced browser mode...")
        result = await fetcher.fetch(url)
        logger.info(f"Fetch method used: {result.get('fetch_method')}")
        assert result.get('fetch_method') == 'browser'
        # Verify it skipped WebUnblockerFetcher (static)
        # (In reality, it would try browser first and fail if proxy not configured correctly)

if __name__ == "__main__":
    asyncio.run(test_hybrid_fetcher_logic())
