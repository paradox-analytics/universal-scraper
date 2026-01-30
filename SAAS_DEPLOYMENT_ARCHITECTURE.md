# 🏗️ Universal Scraper - SaaS Deployment Architecture

**Target**: Production SaaS with multi-tenancy, high availability, and cost efficiency

---

## 📊 Caching Strategy (5-Layer Architecture)

### **Layer 1: In-Memory Cache (Redis/Valkey)** ⚡
**Purpose**: Hot path optimization  
**TTL**: 1-24 hours

```python
# Cache Structure
{
    # Execution Results (fastest)
    "exec:sha256(code):sha256(html[:1000])": {
        "data": [...],           # Extracted items
        "timestamp": 1699...,
        "execution_time": 2.3,
        "ttl": 3600             # 1 hour
    },
    
    # Generated Code (high value)
    "code:structure_hash": {
        "code": "def extract_data(soup)...",
        "fields": ["title", "price"],
        "confidence": 0.92,
        "generated_at": 1699...,
        "ttl": 86400            # 24 hours
    },
    
    # Field Mappings (domain-level)
    "fields:github.com:title,stars,author": {
        "mappings": {...},
        "domain_context": {...},
        "ttl": 604800           # 7 days
    },
    
    # Rate Limiting (per-tenant)
    "rate:tenant_123:github.com": {
        "count": 47,
        "window_start": 1699...,
        "ttl": 60               # 1 minute window
    }
}
```

**Why Redis/Valkey?**
- ✅ Sub-millisecond latency
- ✅ Atomic operations (rate limiting)
- ✅ TTL support (automatic expiration)
- ✅ Pub/Sub for cache invalidation
- ✅ Cluster mode for horizontal scaling

**Configuration**:
```yaml
redis:
  nodes: 3 (primary + 2 replicas)
  memory: 16GB per node
  eviction: allkeys-lru
  persistence: AOF (for code cache)
  cluster: true
```

---

### **Layer 2: Object Storage (S3/R2)** 💾
**Purpose**: Long-term code & field mapping storage  
**TTL**: Indefinite (until structure changes)

```python
# S3 Bucket Structure
s3://universal-scraper-cache/
├── code/
│   ├── by-hash/
│   │   ├── a3f2e1b9c4.json          # Code + metadata
│   │   └── 7d8e9f0a1b.json
│   └── by-domain/
│       ├── github.com/
│       │   ├── trending_v1.json     # Versioned
│       │   └── trending_v2.json
│       └── amazon.com/
│           └── products_v1.json
├── field-mappings/
│   ├── github.com.json              # Domain context
│   ├── github.com_title-stars.json  # Field semantics
│   └── amazon.com_title-price.json
└── html-samples/
    ├── github.com_sample_1.html     # For debugging
    └── amazon.com_sample_1.html
```

**Object Metadata**:
```json
{
    "structure_hash": "a3f2e1b9c4d5e6f7",
    "domain": "github.com",
    "url_pattern": "*/trending*",
    "fields": ["title", "stars", "author"],
    "generated_at": "2025-11-12T22:00:00Z",
    "last_validated": "2025-11-12T22:00:00Z",
    "validation_count": 147,
    "success_rate": 0.96,
    "avg_execution_time": 2.3,
    "version": 2
}
```

**Why S3/R2?**
- ✅ Infinite scalability
- ✅ 11 9's durability
- ✅ Low cost ($0.023/GB/month for S3)
- ✅ CDN integration (CloudFront)
- ✅ Versioning support

---

### **Layer 3: CDN Cache (CloudFront/Cloudflare)** 🌐
**Purpose**: Edge caching for public data  
**TTL**: 1-7 days

```yaml
# CloudFront Distribution
origins:
  - s3_code_cache:
      domain: universal-scraper-cache.s3.amazonaws.com
      path: /code/*
      ttl: 86400  # 1 day
  
  - s3_field_mappings:
      domain: universal-scraper-cache.s3.amazonaws.com
      path: /field-mappings/*
      ttl: 604800  # 7 days

behaviors:
  - pattern: /api/cache/code/*
    cache_policy: CacheOptimized
    compress: true
    viewer_protocol: https-only
```

**Cache Keys**:
```
GET /api/cache/code/a3f2e1b9c4d5e6f7
GET /api/cache/fields/github.com/title-stars-author
```

**Why CDN?**
- ✅ Global distribution
- ✅ Reduces S3 GET costs
- ✅ Sub-100ms latency worldwide
- ✅ DDoS protection

---

### **Layer 4: PostgreSQL (Persistent Metadata)** 🗄️
**Purpose**: Cache metadata, analytics, versioning

```sql
-- Code Cache Table
CREATE TABLE code_cache (
    id BIGSERIAL PRIMARY KEY,
    structure_hash VARCHAR(64) UNIQUE NOT NULL,
    domain VARCHAR(255) NOT NULL,
    url_pattern TEXT,
    fields JSONB NOT NULL,
    code TEXT NOT NULL,
    metadata JSONB,
    
    -- Performance Metrics
    success_rate DECIMAL(5,4) DEFAULT 1.0,
    avg_execution_time DECIMAL(8,3),
    total_executions INTEGER DEFAULT 0,
    
    -- Versioning
    version INTEGER DEFAULT 1,
    previous_version_id BIGINT REFERENCES code_cache(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP DEFAULT NOW(),
    last_validated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_domain (domain),
    INDEX idx_structure_hash (structure_hash),
    INDEX idx_last_used (last_used_at),
    INDEX idx_success_rate (success_rate)
);

-- Field Mappings Table
CREATE TABLE field_mappings (
    id BIGSERIAL PRIMARY KEY,
    domain VARCHAR(255) NOT NULL,
    fields_hash VARCHAR(64) NOT NULL,
    fields JSONB NOT NULL,
    mappings JSONB NOT NULL,
    domain_context JSONB,
    
    -- Metrics
    total_uses INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(domain, fields_hash)
);

-- Execution Log (for analytics)
CREATE TABLE execution_log (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    code_cache_id BIGINT REFERENCES code_cache(id),
    url TEXT NOT NULL,
    fields JSONB NOT NULL,
    
    -- Results
    items_extracted INTEGER,
    quality_score DECIMAL(5,4),
    execution_time DECIMAL(8,3),
    phase_used VARCHAR(20),  -- json, html, direct_llm
    cache_hit BOOLEAN,
    
    -- Costs
    llm_calls INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6),
    
    -- Timestamps
    executed_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_tenant_executed (tenant_id, executed_at),
    INDEX idx_cache_hit (cache_hit)
);
```

**Why PostgreSQL?**
- ✅ ACID compliance
- ✅ Complex queries (analytics)
- ✅ JSONB for flexible metadata
- ✅ Mature ecosystem

---

### **Layer 5: Distributed Cache Warming** 🔥
**Purpose**: Proactive cache population

```python
# Cache Warming Strategy
class CacheWarmer:
    def __init__(self):
        self.redis = Redis()
        self.s3 = S3Client()
        self.db = PostgreSQL()
    
    async def warm_popular_sites(self):
        """Warm cache for frequently accessed sites"""
        
        # Get top 100 most accessed site+field combos
        popular = await self.db.query("""
            SELECT domain, fields, COUNT(*) as hits
            FROM execution_log
            WHERE executed_at > NOW() - INTERVAL '7 days'
            GROUP BY domain, fields
            ORDER BY hits DESC
            LIMIT 100
        """)
        
        for item in popular:
            # Load from S3 → Redis
            code = await self.s3.get(f"code/{item.structure_hash}.json")
            fields = await self.s3.get(f"fields/{item.domain}_{item.fields_hash}.json")
            
            # Warm Redis
            await self.redis.setex(
                f"code:{item.structure_hash}",
                86400,  # 24h TTL
                code
            )
            await self.redis.setex(
                f"fields:{item.domain}:{item.fields}",
                604800,  # 7d TTL
                fields
            )
    
    async def warm_on_deploy(self):
        """Warm cache on new deployment"""
        # Load all code cache entries with success_rate > 0.9
        successful_code = await self.db.query("""
            SELECT structure_hash, code, metadata
            FROM code_cache
            WHERE success_rate > 0.9
            AND last_used_at > NOW() - INTERVAL '30 days'
        """)
        
        # Parallel load into Redis
        tasks = [
            self.redis.setex(f"code:{c.structure_hash}", 86400, c.code)
            for c in successful_code
        ]
        await asyncio.gather(*tasks)
```

---

## 🏗️ Infrastructure Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                      GLOBAL CDN (CloudFront)                 │
│  • Edge caching (code, field mappings)                      │
│  • DDoS protection                                           │
│  • SSL termination                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LOAD BALANCER (ALB/NLB)                        │
│  • Health checks                                             │
│  • SSL termination                                           │
│  • Tenant routing                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  API Server  │  │  API Server  │  │  API Server  │
│  (ECS/K8s)   │  │  (ECS/K8s)   │  │  (ECS/K8s)   │
│              │  │              │  │              │
│  • FastAPI   │  │  • FastAPI   │  │  • FastAPI   │
│  • 4 vCPU    │  │  • 4 vCPU    │  │  • 4 vCPU    │
│  • 8GB RAM   │  │  • 8GB RAM   │  │  • 8GB RAM   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
        ┌───────▼────────┐ ┌─────▼──────────┐
        │  Redis Cluster │ │  S3/R2 Buckets │
        │  (Cache)       │ │  (Long-term)   │
        └───────┬────────┘ └─────┬──────────┘
                │                 │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  PostgreSQL RDS │
                │  (Metadata)     │
                └─────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  Scraping Workers (Separate)   │
        │  • Camoufox pool                │
        │  • GPU instances (LLM)          │
        │  • Isolated network             │
        └────────────────────────────────┘
```

---

## 🚀 Component Specifications

### **1. API Servers (Auto-Scaling)**

```yaml
# ECS Task Definition (or K8s Deployment)
api_server:
  image: universal-scraper-api:latest
  cpu: 4 vCPU
  memory: 8 GB
  replicas:
    min: 3
    max: 50
    target_cpu: 70%
  
  environment:
    - REDIS_CLUSTER_ENDPOINT=redis-cluster.internal
    - S3_BUCKET=universal-scraper-cache
    - POSTGRES_HOST=postgres-primary.internal
    - LLM_API_KEY=secret:llm_api_key
    - CAMOUFOX_POOL_SIZE=0  # API servers don't run browsers
  
  health_check:
    path: /health
    interval: 30s
    timeout: 5s
  
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
```

**Scaling Logic**:
```python
# Auto-scale based on request queue depth
if queue_depth > 100:
    scale_out(replicas + 5)
elif queue_depth < 20 and replicas > min_replicas:
    scale_in(replicas - 2)
```

---

### **2. Redis Cluster (Cache Layer)**

```yaml
redis_cluster:
  mode: cluster
  nodes: 6 (3 primary + 3 replicas)
  
  primary_nodes:
    instance_type: cache.r7g.xlarge  # ARM-based, cost-efficient
    vcpu: 4
    memory: 26.32 GB
    network: 12.5 Gbps
  
  configuration:
    maxmemory-policy: allkeys-lru
    maxmemory: 20GB  # Per node
    tcp-keepalive: 300
    timeout: 0
    
    # Persistence (for code cache)
    save: "900 1 300 10 60 10000"  # RDB snapshots
    appendonly: yes
    appendfsync: everysec
  
  monitoring:
    - metric: cpu_utilization
      alarm_threshold: 75%
    - metric: memory_utilization
      alarm_threshold: 80%
    - metric: evictions
      alarm_threshold: 1000/min
```

**Cost**: ~$500-800/month for 3-node cluster

---

### **3. PostgreSQL RDS (Metadata)**

```yaml
postgres_rds:
  instance_class: db.r6g.xlarge  # ARM-based
  vcpu: 4
  memory: 32 GB
  storage: 500 GB (gp3)
  iops: 12000
  
  multi_az: true
  read_replicas: 2  # For analytics queries
  
  backup:
    retention: 7 days
    window: 03:00-04:00 UTC
  
  maintenance:
    window: Sun:04:00-Sun:05:00 UTC
  
  monitoring:
    enhanced: true
    interval: 60s
```

**Cost**: ~$400-600/month

---

### **4. Scraping Workers (Isolated)**

```yaml
# Separate worker pool for browser automation
scraping_workers:
  image: universal-scraper-worker:latest
  cpu: 8 vCPU
  memory: 16 GB
  gpu: 0  # Optional for LLM inference
  
  replicas:
    min: 2
    max: 20
    scale_metric: queue_depth
  
  environment:
    - CAMOUFOX_POOL_SIZE=4  # 4 browsers per worker
    - REDIS_ENDPOINT=redis-cluster.internal
    - S3_BUCKET=universal-scraper-cache
    - USE_LOCAL_LLM=false  # Use API for now
  
  network:
    # Isolated network for browsers
    subnet: private
    nat_gateway: true
    
    # Rotating proxy pool
    proxy_config:
      provider: brightdata/smartproxy
      pool_size: 100
      rotation: per-request
  
  volumes:
    - /tmp/camoufox-profiles:emptyDir  # Ephemeral
```

**Scaling Logic**:
```python
# Scale based on queue depth + browser availability
queue_depth = redis.llen('scraping_queue')
active_browsers = sum(worker.active_browsers for worker in workers)

if queue_depth > active_browsers * 2:
    scale_out()
elif queue_depth < active_browsers * 0.5:
    scale_in()
```

---

## 📊 Caching Logic (Detailed)

### **Request Flow with Caching**

```python
class UniversalScraperAPI:
    def __init__(self):
        self.redis = RedisCluster()
        self.s3 = S3Client()
        self.db = PostgreSQL()
        self.llm = LLMClient()
    
    async def scrape(self, url: str, fields: List[str], tenant_id: str):
        # Step 1: Rate limiting check
        if not await self.check_rate_limit(tenant_id, url):
            raise RateLimitExceeded()
        
        # Step 2: Check execution cache (fastest)
        cache_key = self.generate_cache_key(url, fields)
        cached_result = await self.redis.get(f"exec:{cache_key}")
        if cached_result:
            logger.info(f"Cache HIT (execution): {url}")
            await self.increment_usage_metrics(tenant_id, cache_hit=True)
            return cached_result
        
        # Step 3: Fetch HTML
        html = await self.fetch_html(url)
        structure_hash = self.generate_structure_hash(html)
        
        # Step 4: Check code cache (Redis)
        code = await self.redis.get(f"code:{structure_hash}")
        if code:
            logger.info(f"Cache HIT (code-redis): {structure_hash}")
            result = await self.execute_code(code, html)
            
            # Cache execution result
            await self.redis.setex(
                f"exec:{cache_key}",
                3600,  # 1 hour
                result
            )
            return result
        
        # Step 5: Check code cache (S3)
        code = await self.s3.get(f"code/{structure_hash}.json")
        if code:
            logger.info(f"Cache HIT (code-s3): {structure_hash}")
            
            # Warm Redis cache
            await self.redis.setex(f"code:{structure_hash}", 86400, code)
            
            result = await self.execute_code(code, html)
            await self.redis.setex(f"exec:{cache_key}", 3600, result)
            return result
        
        # Step 6: Check field mapping cache
        domain = self.extract_domain(url)
        fields_key = f"fields:{domain}:{':'.join(sorted(fields))}"
        
        field_hints = await self.redis.get(fields_key)
        if not field_hints:
            field_hints = await self.s3.get(f"field-mappings/{domain}_{fields_hash}.json")
            if field_hints:
                await self.redis.setex(fields_key, 604800, field_hints)
        
        # Step 7: Generate code (expensive - LLM calls)
        logger.info(f"Cache MISS - Generating code: {url}")
        
        # Queue for worker execution (don't block API)
        job_id = await self.queue_scraping_job({
            'url': url,
            'fields': fields,
            'structure_hash': structure_hash,
            'field_hints': field_hints,
            'tenant_id': tenant_id
        })
        
        # Wait for result (with timeout)
        result = await self.wait_for_job(job_id, timeout=60)
        
        # Step 8: Cache all the things
        if result['success']:
            # Cache code
            await self.redis.setex(f"code:{structure_hash}", 86400, result['code'])
            await self.s3.put(f"code/{structure_hash}.json", result['code_with_metadata'])
            
            # Cache execution
            await self.redis.setex(f"exec:{cache_key}", 3600, result['data'])
            
            # Cache field mappings (if new)
            if result.get('new_field_mappings'):
                await self.redis.setex(fields_key, 604800, result['field_mappings'])
                await self.s3.put(f"field-mappings/{domain}_{fields_hash}.json", result['field_mappings'])
            
            # Update PostgreSQL metadata
            await self.db.execute("""
                INSERT INTO code_cache (structure_hash, domain, fields, code, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (structure_hash) DO UPDATE
                SET last_used_at = NOW(),
                    total_executions = code_cache.total_executions + 1
            """, structure_hash, domain, fields, result['code'], result['metadata'])
        
        return result['data']
    
    def generate_cache_key(self, url: str, fields: List[str]) -> str:
        """Generate cache key for execution results"""
        # Include normalized URL + sorted fields
        normalized_url = self.normalize_url(url)
        fields_str = ':'.join(sorted(fields))
        return hashlib.sha256(f"{normalized_url}:{fields_str}".encode()).hexdigest()
```

---

## 💰 Cost Analysis (10,000 req/month)

### **Scenario 1: 95% Cache Hit Rate (Typical)**

| Component | Specification | Cost/Month |
|-----------|--------------|------------|
| **API Servers** | 3x ECS (4vCPU, 8GB) | $200 |
| **Redis Cluster** | 3-node (r7g.xlarge) | $600 |
| **PostgreSQL** | db.r6g.xlarge + 2 replicas | $500 |
| **S3 Storage** | 10GB code + 5GB fields | $0.35 |
| **S3 Requests** | 500 GET (cached by CDN) | $0.20 |
| **CloudFront** | 100GB transfer | $8.50 |
| **Scraping Workers** | 2x workers (8vCPU, 16GB) | $300 |
| **LLM API** | 500 calls @ $0.05 | $25 |
| **Proxy Pool** | Residential (optional) | $200 |
| **Total** | | **~$1,834/month** |

**Per-Request Cost**: $0.18

**Revenue at $1/request**: $10,000  
**Gross Margin**: **82%** 🎉

---

### **Scenario 2: 50% Cache Hit Rate (Worst Case)**

| Component | Cost/Month |
|-----------|------------|
| API Servers (same) | $200 |
| Redis Cluster (same) | $600 |
| PostgreSQL (same) | $500 |
| S3 (larger) | $2 |
| CloudFront (same) | $8.50 |
| **Scraping Workers** | **10x workers** | **$1,500** |
| **LLM API** | **5,000 calls @ $0.05** | **$250** |
| Proxy Pool | $200 |
| **Total** | **~$3,260/month** |

**Per-Request Cost**: $0.33  
**Gross Margin at $1/request**: **67%**

---

## 🔒 Security & Multi-Tenancy

### **Tenant Isolation**

```python
# Tenant-level caching with namespace isolation
class TenantCache:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.redis = Redis()
    
    def get_code(self, structure_hash: str):
        # Shared code cache (all tenants benefit)
        return self.redis.get(f"code:{structure_hash}")
    
    def get_execution_result(self, url: str, fields: List[str]):
        # Tenant-specific execution cache
        cache_key = self.generate_cache_key(url, fields)
        return self.redis.get(f"exec:{self.tenant_id}:{cache_key}")
    
    def set_execution_result(self, url: str, fields: List[str], data: dict):
        cache_key = self.generate_cache_key(url, fields)
        # Tenant-scoped TTL based on plan
        ttl = self.get_ttl_for_tenant()
        self.redis.setex(f"exec:{self.tenant_id}:{cache_key}", ttl, data)
```

### **Rate Limiting (Per Tenant)**

```python
class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
    
    async def check_limit(self, tenant_id: str, limit: int, window: int) -> bool:
        """
        Token bucket algorithm with Redis
        
        Args:
            tenant_id: Tenant identifier
            limit: Max requests per window
            window: Window in seconds
        """
        key = f"rate:{tenant_id}"
        
        # Lua script for atomic check-and-increment
        lua_script = """
        local current = redis.call('GET', KEYS[1])
        if current and tonumber(current) >= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('INCR', KEYS[1])
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return 1
        """
        
        allowed = await self.redis.eval(lua_script, keys=[key], args=[limit, window])
        return bool(allowed)
```

---

## 📈 Monitoring & Observability

### **Key Metrics**

```python
# Prometheus metrics
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Percentage of cache hits',
    ['layer', 'tenant']
)

scraping_duration = Histogram(
    'scraping_duration_seconds',
    'Time to complete scraping',
    ['phase', 'cached']
)

llm_cost = Counter(
    'llm_cost_usd',
    'Total LLM API cost',
    ['tenant', 'model']
)

code_cache_size = Gauge(
    'code_cache_entries',
    'Number of cached code entries',
    ['domain']
)

execution_quality = Histogram(
    'extraction_quality_score',
    'Quality score of extracted data',
    ['domain']
)
```

### **Dashboards**

```yaml
# Grafana Dashboard
panels:
  - title: Cache Hit Rates
    metrics:
      - redis_hit_rate
      - s3_hit_rate
      - cdn_hit_rate
    target: >95%
  
  - title: Response Times
    metrics:
      - p50_latency
      - p95_latency
      - p99_latency
    target: p95 < 5s
  
  - title: Cost per Request
    metrics:
      - llm_cost_per_request
      - infrastructure_cost_per_request
      - total_cost_per_request
    target: <$0.20
  
  - title: Quality Scores
    metrics:
      - avg_quality_score
      - quality_by_domain
    target: >90%
```

---

## 🚨 Failure Modes & Recovery

### **1. Redis Cluster Failure**

```python
# Fallback to S3 on Redis failure
try:
    code = await redis.get(f"code:{structure_hash}")
except RedisConnectionError:
    logger.error("Redis cluster unavailable - falling back to S3")
    code = await s3.get(f"code/{structure_hash}.json")
    
    # Alert ops team
    await alert_pagerduty("Redis cluster down")
```

### **2. S3 Unavailability**

```python
# Regenerate code on S3 failure
try:
    code = await s3.get(f"code/{structure_hash}.json")
except S3Error:
    logger.error("S3 unavailable - regenerating code")
    code = await llm.generate_code(html, fields)
    
    # Queue for S3 write when available
    await queue_s3_write(structure_hash, code)
```

### **3. LLM API Rate Limit**

```python
# Circuit breaker pattern
class LLMCircuitBreaker:
    def __init__(self):
        self.failure_count = 0
        self.last_failure = None
        self.state = 'closed'  # closed, open, half-open
    
    async def call_llm(self, prompt: str):
        if self.state == 'open':
            # Check if cooldown period passed
            if time.time() - self.last_failure > 60:
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpen("LLM circuit breaker open")
        
        try:
            result = await llm_client.complete(prompt)
            self.failure_count = 0
            self.state = 'closed'
            return result
        except RateLimitError:
            self.failure_count += 1
            self.last_failure = time.time()
            
            if self.failure_count >= 3:
                self.state = 'open'
            
            raise
```

---

## 🎯 Optimization Strategies

### **1. Prefetching (Predictive Caching)**

```python
# Predict what user will scrape next
class PredictiveCache:
    async def analyze_patterns(self, tenant_id: str):
        """Analyze tenant's scraping patterns"""
        recent_requests = await db.query("""
            SELECT url, fields, COUNT(*) as frequency
            FROM execution_log
            WHERE tenant_id = $1
            AND executed_at > NOW() - INTERVAL '7 days'
            GROUP BY url, fields
            ORDER BY frequency DESC
            LIMIT 20
        """, tenant_id)
        
        # Prefetch code for likely next requests
        for req in recent_requests:
            structure_hash = await self.get_structure_hash(req.url)
            code = await s3.get(f"code/{structure_hash}.json")
            
            # Warm Redis
            await redis.setex(f"code:{structure_hash}", 86400, code)
```

### **2. Batch Processing**

```python
# Batch similar requests
class BatchProcessor:
    async def process_batch(self, requests: List[ScrapeRequest]):
        """Process multiple requests for same site efficiently"""
        
        # Group by domain
        by_domain = defaultdict(list)
        for req in requests:
            by_domain[req.domain].append(req)
        
        results = []
        for domain, reqs in by_domain.items():
            # Single browser session for all
            async with camoufox_session() as session:
                for req in reqs:
                    result = await self.scrape_with_session(session, req)
                    results.append(result)
        
        return results
```

### **3. Smart Cache Invalidation**

```python
# Invalidate cache when structure changes
class CacheInvalidator:
    async def detect_structure_change(self, url: str, new_html: str):
        """Detect if website structure changed"""
        
        new_hash = generate_structure_hash(new_html)
        
        # Get previous hash
        domain = extract_domain(url)
        prev_hash = await redis.get(f"structure:{domain}")
        
        if prev_hash and prev_hash != new_hash:
            logger.info(f"Structure changed for {domain}: {prev_hash} → {new_hash}")
            
            # Invalidate all code cache for this domain
            pattern = f"code:*{domain}*"
            keys = await redis.keys(pattern)
            await redis.delete(*keys)
            
            # Trigger re-analysis
            await queue_analysis_job(url, new_html)
        
        # Update hash
        await redis.set(f"structure:{domain}", new_hash)
```

---

## 📚 Summary

### **Recommended Stack**

| Layer | Technology | Why |
|-------|-----------|-----|
| **API** | FastAPI + ECS/K8s | Async, auto-scaling, mature |
| **Cache (Hot)** | Redis Cluster | Sub-ms latency, atomic ops |
| **Cache (Cold)** | S3/R2 | Infinite scale, cheap |
| **CDN** | CloudFront | Global edge caching |
| **Database** | PostgreSQL RDS | ACID, analytics-ready |
| **Workers** | ECS/K8s (isolated) | Browser isolation, GPU option |
| **Monitoring** | Prometheus + Grafana | Industry standard |
| **Logging** | ELK Stack / DataDog | Centralized, searchable |

### **Cost Efficiency**

- **95% cache hit**: $0.18/request (82% margin at $1/request)
- **50% cache hit**: $0.33/request (67% margin)
- **Break-even**: ~1,800 requests/month at $1/request

### **Scalability**

- **Vertical**: Redis cluster (6-12 nodes), Postgres read replicas
- **Horizontal**: Auto-scaling API servers (3-50), Workers (2-20)
- **Global**: Multi-region deployment with CDN

### **Cache Hit Target**

- **Target**: 95%+ cache hit rate
- **Achieved through**:
  - Aggressive Redis caching (24h TTL)
  - S3 long-term storage (indefinite)
  - CDN edge caching (7d TTL)
  - Predictive prefetching
  - Cache warming on deploy

**Result**: Production-ready SaaS with excellent unit economics! 🚀

---

**Questions? Need help with implementation?**  
Contact: [Your SaaS deployment team]






