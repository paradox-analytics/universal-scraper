# 🏗️ SaaS Readiness Assessment & Multi-Tenancy Plan

## Current Architecture Analysis

### ✅ What Works for SaaS

1. **Cloud Run Auto-Scaling**
   - ✅ Auto-scales from 0 to 10 instances
   - ✅ Pay-per-request pricing
   - ✅ Handles concurrent requests
   - ⚠️ **Issue**: Max 10 instances may be limiting for high traffic

2. **Stateless API Design**
   - ✅ FastAPI is stateless
   - ✅ Can scale horizontally
   - ✅ No shared state between instances

### ❌ Critical Issues for Multi-Tenant SaaS

1. **Cache Isolation** 🚨 **CRITICAL**
   - ❌ Cache is **shared across all users** (no tenant isolation)
   - ❌ Uses local filesystem (`./cache`) - **won't work across Cloud Run instances**
   - ❌ Cache keys don't include tenant/user ID
   - **Impact**: Users can see each other's cached data, cache doesn't persist across instances

2. **No Tenant Identification** 🚨 **CRITICAL**
   - ❌ API keys are used for LLM calls, not tenant identification
   - ❌ No user/tenant ID in requests
   - ❌ No authentication/authorization system
   - **Impact**: Cannot track usage, enforce limits, or isolate data per tenant

3. **No Rate Limiting** ⚠️ **HIGH PRIORITY**
   - ❌ No per-tenant rate limits
   - ❌ No usage tracking
   - ❌ No quota enforcement
   - **Impact**: One tenant can consume all resources

4. **No Usage Tracking** ⚠️ **HIGH PRIORITY**
   - ❌ No metrics per tenant
   - ❌ No billing integration
   - ❌ No analytics dashboard
   - **Impact**: Cannot bill customers or track usage

5. **Scraper Pooling Issues** ⚠️ **MEDIUM**
   - ⚠️ Scraper instances pooled by API key (not tenant)
   - ⚠️ Multiple tenants with same LLM API key share scraper instance
   - **Impact**: Potential resource contention

## Required Changes for SaaS

### Phase 1: Multi-Tenancy Foundation (Critical)

#### 1.1 Tenant Identification System

```python
# api/middleware/auth.py
from fastapi import Header, HTTPException
from typing import Optional
import jwt
import os

async def get_tenant_id(
    authorization: Optional[str] = Header(None)
) -> str:
    """
    Extract tenant ID from JWT token or API key
    
    Options:
    1. JWT-based (recommended): Bearer token with tenant_id claim
    2. API Key-based: Map API key to tenant_id in database
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Option 1: JWT Token
    if authorization.startswith("Bearer "):
        token = authorization[7:]
        try:
            payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
            return payload["tenant_id"]
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    # Option 2: API Key mapping (temporary)
    # TODO: Replace with proper JWT auth
    tenant_id = await lookup_tenant_by_api_key(authorization)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return tenant_id
```

#### 1.2 Tenant-Isolated Cache

```python
# universal_scraper/core/tenant_cache.py
from typing import Optional, Dict, Any
import redis.asyncio as redis
import json
import os

class TenantCache:
    """
    Multi-tenant cache with isolation
    
    Cache Strategy:
    - Shared code cache (benefits all tenants)
    - Tenant-specific execution cache
    - Tenant-specific rate limits
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    
    async def get_code_cache(self, structure_hash: str) -> Optional[Dict]:
        """Get shared code cache (all tenants benefit)"""
        key = f"code:{structure_hash}"
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None
    
    async def set_code_cache(self, structure_hash: str, code: Dict, ttl: int = 86400):
        """Set shared code cache"""
        key = f"code:{structure_hash}"
        await self.redis_client.setex(key, ttl, json.dumps(code))
    
    async def get_execution_cache(self, url: str, fields: list) -> Optional[Dict]:
        """Get tenant-specific execution cache"""
        cache_key = self._generate_execution_key(url, fields)
        key = f"exec:{self.tenant_id}:{cache_key}"
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None
    
    async def set_execution_cache(self, url: str, fields: list, result: Dict, ttl: int = 3600):
        """Set tenant-specific execution cache"""
        cache_key = self._generate_execution_key(url, fields)
        key = f"exec:{self.tenant_id}:{cache_key}"
        await self.redis_client.setex(key, ttl, json.dumps(result))
    
    def _generate_execution_key(self, url: str, fields: list) -> str:
        """Generate cache key for execution results"""
        import hashlib
        fields_str = ','.join(sorted(fields))
        key_data = f"{url}:{fields_str}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
```

#### 1.3 Update API Endpoints

```python
# api/main.py
from api.middleware.auth import get_tenant_id

@app.post("/scrape")
async def scrape_endpoint(
    request: ScrapeRequest,
    tenant_id: str = Depends(get_tenant_id),  # Add tenant identification
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Scrape endpoint with tenant isolation"""
    
    # Initialize tenant-specific cache
    tenant_cache = TenantCache(tenant_id)
    
    # Check rate limits
    if not await check_rate_limit(tenant_id, request.url):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Check tenant-specific execution cache
    cached_result = await tenant_cache.get_execution_cache(request.url, request.fields)
    if cached_result:
        await track_usage(tenant_id, cache_hit=True)
        return cached_result
    
    # Proceed with scraping...
    scraper = get_scraper(x_api_key, request.mode, request.proxy_config)
    result = await scraper.scrape(...)
    
    # Cache result (tenant-specific)
    await tenant_cache.set_execution_cache(request.url, request.fields, result)
    
    # Track usage
    await track_usage(tenant_id, cache_hit=False, items_extracted=len(result.get('data', [])))
    
    return result
```

### Phase 2: Infrastructure Updates

#### 2.1 Add Redis for Shared Cache

```yaml
# infrastructure/redis.yaml (Cloud Memorystore)
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 16gb
    maxmemory-policy allkeys-lru
    save ""
    appendonly yes
```

**GCP Option**: Use Cloud Memorystore for Redis
```bash
gcloud redis instances create universal-scraper-cache \
  --size=16 \
  --region=us-central1 \
  --tier=standard \
  --redis-version=redis_7_0
```

#### 2.2 Update Cloud Run Configuration

```yaml
# infrastructure/cloudbuild/cloudbuild.yaml
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  args:
    - 'run'
    - 'deploy'
    - 'universal-scraper-api'
    - '--image'
    - 'gcr.io/$PROJECT_ID/universal-scraper-api:latest'
    - '--region'
    - 'us-central1'
    - '--platform'
    - 'managed'
    - '--memory'
    - '4Gi'  # Increased for browser automation
    - '--cpu'
    - '2'
    - '--timeout'
    - '300'
    - '--min-instances'
    - '1'  # Keep warm for faster cold starts
    - '--max-instances'
    - '100'  # Increased for SaaS scale
    - '--concurrency'
    - '10'  # Requests per instance
    - '--set-env-vars'
    - 'REDIS_URL=redis://[REDIS_IP]:6379'
```

#### 2.3 Add Database for Tenant Management

**Option 1: Cloud SQL (PostgreSQL)**
```sql
-- tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) NOT NULL, -- 'free', 'pro', 'enterprise'
    rate_limit_per_minute INTEGER DEFAULT 10,
    rate_limit_per_day INTEGER DEFAULT 1000,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- usage_logs table
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    endpoint VARCHAR(255),
    url TEXT,
    items_extracted INTEGER,
    cache_hit BOOLEAN,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_logs_tenant_date ON usage_logs(tenant_id, created_at);
```

**Option 2: Firestore (Serverless)**
```python
# Firestore structure
tenants/{tenant_id}
  - name: "Acme Corp"
  - plan: "pro"
  - rate_limit_per_minute: 100
  - rate_limit_per_day: 10000
  - created_at: timestamp

usage_logs/{log_id}
  - tenant_id: "tenant_123"
  - endpoint: "/scrape"
  - url: "https://example.com"
  - items_extracted: 50
  - cache_hit: true
  - created_at: timestamp
```

### Phase 3: Rate Limiting & Usage Tracking

#### 3.1 Rate Limiter

```python
# api/middleware/rate_limit.py
import redis.asyncio as redis
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(self, tenant_id: str, url: str) -> bool:
        """Check if tenant has exceeded rate limits"""
        tenant_config = await self.get_tenant_config(tenant_id)
        
        # Per-minute limit
        minute_key = f"rate:{tenant_id}:minute:{datetime.now().minute}"
        minute_count = await self.redis.incr(minute_key)
        if minute_count == 1:
            await self.redis.expire(minute_key, 60)
        
        if minute_count > tenant_config['rate_limit_per_minute']:
            return False
        
        # Per-day limit
        day_key = f"rate:{tenant_id}:day:{datetime.now().date()}"
        day_count = await self.redis.incr(day_key)
        if day_count == 1:
            await self.redis.expire(day_key, 86400)
        
        if day_count > tenant_config['rate_limit_per_day']:
            return False
        
        return True
```

#### 3.2 Usage Tracking

```python
# api/middleware/usage_tracking.py
from datetime import datetime
import asyncio

async def track_usage(
    tenant_id: str,
    endpoint: str,
    url: str,
    items_extracted: int,
    cache_hit: bool,
    execution_time_ms: int
):
    """Track usage for billing and analytics"""
    usage_log = {
        'tenant_id': tenant_id,
        'endpoint': endpoint,
        'url': url,
        'items_extracted': items_extracted,
        'cache_hit': cache_hit,
        'execution_time_ms': execution_time_ms,
        'created_at': datetime.utcnow().isoformat()
    }
    
    # Async write to database (don't block request)
    asyncio.create_task(write_usage_log(usage_log))
    
    # Update Redis counters for real-time metrics
    await update_usage_counters(tenant_id, items_extracted, cache_hit)
```

## Implementation Roadmap

### Week 1: Foundation
- [ ] Add Redis/Memorystore for shared cache
- [ ] Implement tenant identification middleware
- [ ] Update cache to be tenant-aware
- [ ] Add tenant database (Firestore or Cloud SQL)

### Week 2: Isolation & Limits
- [ ] Implement rate limiting per tenant
- [ ] Add usage tracking
- [ ] Update all endpoints to use tenant context
- [ ] Test multi-tenant isolation

### Week 3: Monitoring & Billing
- [ ] Add usage metrics dashboard
- [ ] Implement billing integration (Stripe/Paddle)
- [ ] Add tenant management API
- [ ] Set up alerts for rate limit violations

### Week 4: Scale Testing
- [ ] Load testing with multiple tenants
- [ ] Optimize cache hit rates
- [ ] Tune Cloud Run scaling parameters
- [ ] Document tenant onboarding process

## Cost Estimates (Multi-Tenant SaaS)

### Infrastructure Costs (100 tenants, 10K requests/day)

| Component | Cost/Month |
|-----------|------------|
| Cloud Run (100 instances max) | $200-500 |
| Cloud Memorystore Redis (16GB) | $600 |
| Cloud SQL PostgreSQL (db-f1-micro) | $25 |
| Cloud Storage (cache backup) | $5 |
| **Total Infrastructure** | **~$830/month** |

### Per-Tenant Costs
- **Free Tier**: 100 requests/day = $0.008/tenant/month
- **Pro Tier**: 10,000 requests/day = $0.83/tenant/month
- **Enterprise**: Custom pricing

## Security Considerations

1. **API Key Security**
   - Hash API keys in database
   - Rotate keys periodically
   - Use JWT tokens instead of API keys (Phase 2)

2. **Tenant Isolation**
   - Never expose tenant data to other tenants
   - Validate tenant_id on every request
   - Use Redis namespaces for cache isolation

3. **Rate Limiting**
   - Prevent DDoS attacks
   - Fair resource allocation
   - Protect against abuse

## Next Steps

1. **Immediate**: Add Redis and tenant identification
2. **Short-term**: Implement rate limiting and usage tracking
3. **Long-term**: Add billing, analytics dashboard, and advanced features




