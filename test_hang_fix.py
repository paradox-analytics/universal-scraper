
import asyncio
import logging
import sys
import time
from universal_scraper.core.hybrid_fetcher import HybridFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def test_hang_fix():
    print("\n🧪 TESTING HANG FIX FOR HOME DEPOT")
    print("====================================")
    print("Goal: Verify that HybridFetcher SKIPS the naive static fetch for Home Depot")
    print("      when the Web Unblocker Fast Path fails, avoiding the 300s tarpit hang.\n")

    url = "https://www.homedepot.com/p/Husky-2-Ton-Hydraulic-Trolley-Car-Jack-HPL4136-VT/311259745"
    
    # Initialize with a dummy API key to force Web Unblocker attempts
    # This simulates the production env where the key exists but might fail auth or be exhausted,
    # OR if we just want to verify the fallback logic.
    # Note: If we provide NO key, it might skip the fast path entirely? 
    # Let's check logic: "if self.web_unblocker_fetcher:" -> implies we need a key to trigger that path.
    
    fetcher = HybridFetcher(
        web_unblocker_api_key="dummy_key_to_trigger_init", # This ensures web_unblocker_fetcher is created
        use_camoufox=True,
        headless=True
    )
    
    try:
        # We need to simulate the Web Unblocker failing.
        # Since the key is dummy, it WILL fail authentication or connection.
        # Then we check if it falls back to 'static_fast_path' or SKIPS it.
        
        print(f"1. Fetching URL: {url}")
        print("   (Expectation: Web Unblocker fails -> Naive Static SKIPPED -> Browser Launched)")
        
        start_time = time.time()
        
        # We set a short timeout for the overall test to ensure we don't hang here
        result = await fetcher.fetch(url, browser_config={'timeout': 30000}) 
        
        duration = time.time() - start_time
        print(f"\n✅ Fetch completed in {duration:.2f} seconds")
        print(f"   Method used: {result.get('fetch_method')}")
        
        # Analyze logs to confirm the skip
        unblocker_logs = result.get('unblocker_log', [])
        
        print("\n🔍 LOG ANALYSIS:")
        found_skip_message = False
        found_static_attempt = False
        
        for entry in unblocker_logs:
            msg = entry['message']
            print(f"   - {msg}")
            if "Skipping naive static fetch" in msg:
                found_skip_message = True
            if "Using Static Fetcher for Fast Path" in msg:
                found_static_attempt = True
                
        print("\n📊 RESULTS:")
        if found_skip_message:
            print("   ✅ SUCCESS: 'Skipping naive static fetch' message found.")
        else:
            print("   ❌ FAILURE: Skip message NOT found.")
            
        if found_static_attempt:
            print("   ❌ FAILURE: 'Using Static Fetcher' message found (Should have been skipped).")
        else:
            print("   ✅ SUCCESS: Naive static fetch was correcty skipped.")
            
        if duration < 20: # Should be fast (browser launch time) vs 300s hang
            print("   ✅ SUCCESS: Operation completed quickly (no hang).")
        else:
             print("   ⚠️ WARNING: Operation took > 20s.")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(test_hang_fix())
