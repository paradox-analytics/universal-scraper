import asyncio
import time
import logging
from universal_scraper.core.rate_limiter import AdaptiveRateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rate_limiter():
    limiter = AdaptiveRateLimiter()
    url = "https://example.com"
    domain = "example.com"
    
    print("\n--- Test 1: Default Delay ---")
    # First request should be instant
    start = time.time()
    await limiter.wait_for_token(url)
    print(f"Req 1 waited: {time.time() - start:.4f}s")
    
    # Second request immediately after should wait ~2s
    start = time.time()
    await limiter.wait_for_token(url)
    elapsed = time.time() - start
    print(f"Req 2 waited: {elapsed:.4f}s")
    assert elapsed >= 1.9, "Should wait at least ~2s"
    
    print("\n--- Test 2: Backoff on Block ---")
    # Report a block
    limiter.report_result(url, 429, is_blocked=True)
    current_delay = limiter._get_domain_delay(domain)
    print(f"Delay after block: {current_delay}s")
    assert current_delay > 2.0, "Delay should increase"
    
    print("\n--- Test 3: Recovery on Success ---")
    # Report success multiple times
    for i in range(5):
        limiter.report_result(url, 200)
        print(f"Delay after success {i+1}: {limiter._get_domain_delay(domain):.2f}s")
        
    assert limiter._get_domain_delay(domain) < current_delay, "Delay should decrease"
    
    print("\n✅ Rate Limiter Logic Verified!")

if __name__ == "__main__":
    asyncio.run(test_rate_limiter())
