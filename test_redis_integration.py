import asyncio
import os
import time
from universal_scraper.core.redis_cache import RedisCache

async def test_redis_functionality():
    print("🚀 Starting Redis Integration Test...")
    
    # Use environment variable for Redis URL or default to localhost
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at: {redis_url}")
    
    cache = RedisCache(redis_url=redis_url)
    
    # Test 1: Basic Set/Get
    print("\nTest 1: Basic Set/Get")
    test_key = "test:basic_key"
    test_val = {"status": "ok", "timestamp": time.time()}
    
    await cache.set(test_key, test_val, ttl=60)
    retrieved_val = await cache.get(test_key)
    
    if retrieved_val == test_val:
        print("✅ Basic Set/Get passed")
    else:
        print(f"❌ Basic Set/Get failed. Expected {test_val}, got {retrieved_val}")

    # Test 2: TTL/Expiration
    print("\nTest 2: Expiration")
    expire_key = "test:expire_key"
    await cache.set(expire_key, "expiring", ttl=1)
    await asyncio.sleep(1.5)
    expired_val = await cache.get(expire_key)
    
    if expired_val is None:
        print("✅ Expiration passed")
    else:
        print(f"❌ Expiration failed. Value still exists: {expired_val}")

    # Test 3: Multiple Tenants/Namespaces
    print("\nTest 3: Namespaces/Multi-tenancy")
    # In our RedisCache, we can simulate namespaces by prefixing keys
    tenant1_key = "tenant1:key"
    tenant2_key = "tenant2:key"
    
    await cache.set(tenant1_key, "data1")
    await cache.set(tenant2_key, "data2")
    
    val1 = await cache.get(tenant1_key)
    val2 = await cache.get(tenant2_key)
    
    if val1 == "data1" and val2 == "data2":
        print("✅ Multi-tenancy simulation passed")
    else:
        print(f"❌ Multi-tenancy simulation failed")

    # Test 4: Pattern Matching/Deletion
    print("\nTest 4: Delete by Pattern")
    # Note: RedisCache might not have delete_by_pattern, let's check basic delete
    await cache.delete(tenant1_key)
    deleted_val = await cache.get(tenant1_key)
    if deleted_val is None:
        print("✅ Basic Deletion passed")
    else:
        print(f"❌ Basic Deletion failed")

    print("\n🚀 Redis Integration Test Completed")
    await cache.close()

if __name__ == "__main__":
    asyncio.run(test_redis_functionality())
