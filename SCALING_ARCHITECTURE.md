# 🚀 Scaling Architecture for Millions of Parallel Users

## Current Limitations

### Cloud Run Constraints
- **Max Instances**: 10 (current) → Need 1000+
- **Concurrency**: 10 requests/instance → Can optimize to 80+
- **Memory**: 2GB → May need 4-8GB for browser automation
- **Timeout**: 300s → May need longer for complex scrapes
- **Cold Starts**: ~1-3 seconds → Need warm instances

### Cache Limitations
- **Local Filesystem**: Won't work across instances
- **No Distributed Cache**: Can't share cache between instances
- **No Cache Persistence**: Lost on instance restart

## Target Architecture: Multi-Region, Auto-Scaling

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Load Balancer                       │
│              (Cloud Load Balancing - Anycast)                │
└──────────────┬────────────────────────────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────┐
    │   US-Central Region │          │   EU-West Region     │
    │                     │          │                      │
    │  ┌────────────────┐ │          │  ┌────────────────┐ │
    │  │  Cloud Run     │ │          │  │  Cloud Run     │ │
    │  │  (0-1000 inst) │ │          │  │  (0-1000 inst) │ │
    │  └────────┬───────┘ │          │  └────────┬───────┘ │
    │           │         │          │           │         │
    │  ┌────────▼───────┐ │          │  ┌────────▼───────┐ │
    │  │ Redis Cluster  │ │          │  │ Redis Cluster │ │
    │  │ (3 nodes)      │ │          │  │ (3 nodes)     │ │
    │  └────────────────┘ │          │  └────────────────┘ │
    │           │         │          │           │         │
    │  ┌────────▼───────┐ │          │  ┌────────▼───────┐ │
    │  │ Cloud Tasks    │ │          │  │ Cloud Tasks    │ │
    │  │ (Queue)        │ │          │  │ (Queue)        │ │
    │  └────────────────┘ │          │  └────────────────┘ │
    └──────────┬───────────┘          └──────────┬──────────┘
               │                                  │
    ┌──────────▼──────────────────────────────────▼──────────┐
    │         Global Redis Cluster (Cross-Region)             │
    │         (Cloud Memorystore Redis - HA)                  │
    └─────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Cloud SQL (HA)     │
    │   (PostgreSQL)       │
    │   - Tenant DB        │
    │   - Usage Logs       │
    └──────────────────────┘
```

## Scaling Strategy: Multi-Layer Approach

### Layer 1: Request Distribution (Global Load Balancer)

```yaml
# infrastructure/load_balancer.yaml
apiVersion: networking.gke.io/v1
kind: BackendConfig
metadata:
  name: universal-scraper-backend
spec:
  timeoutSec: 300
  connectionDraining:
    drainingTimeoutSec: 60
  healthCheck:
    checkIntervalSec: 10
    timeoutSec: 5
    healthyThreshold: 2
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health
  logConfig:
    enable: true
    sampleRate: 1.0
```

**Benefits**:
- Routes requests to nearest region (latency optimization)
- Health checks remove unhealthy instances
- Automatic failover between regions
- DDoS protection built-in

### Layer 2: Cloud Run Auto-Scaling (Per Region)

```yaml
# Updated cloudbuild.yaml
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
    - '4Gi'                    # Increased for browser automation
    - '--cpu'
    - '2'
    - '--timeout'
    - '300'
    - '--min-instances'
    - '10'                     # Keep warm pool (reduces cold starts)
    - '--max-instances'
    - '1000'                   # Scale to 1000 instances per region
    - '--concurrency'
    - '80'                     # Optimized for async I/O (browser automation)
    - '--cpu-throttling'
    - 'false'                  # Better performance for CPU-intensive tasks
    - '--execution-environment'
    - 'gen2'                   # Use 2nd gen for better performance
```

**Scaling Math**:
- **1000 instances × 80 concurrency = 80,000 concurrent requests per region**
- **2 regions × 80,000 = 160,000 concurrent requests globally**
- **At 2 requests/second average = 320,000 requests/second peak capacity**

### Layer 3: Distributed Redis Cache (Multi-Region)

```python
# universal_scraper/core/distributed_cache.py
import redis.asyncio as redis
from redis.cluster import RedisCluster
import json
import os

class DistributedCache:
    """
    Multi-region Redis cluster for shared cache
    
    Architecture:
    - Primary region: Write-through cache
    - Replica regions: Read-only replicas
    - Cross-region replication: <100ms latency
    """
    
    def __init__(self):
        # Primary Redis cluster (write)
        self.primary = RedisCluster(
            startup_nodes=[
                {"host": os.getenv("REDIS_PRIMARY_1"), "port": 6379},
                {"host": os.getenv("REDIS_PRIMARY_2"), "port": 6379},
                {"host": os.getenv("REDIS_PRIMARY_3"), "port": 6379},
            ],
            decode_responses=True,
            skip_full_coverage_check=True
        )
        
        # Replica Redis (read-only, lower latency)
        self.replica = RedisCluster(
            startup_nodes=[
                {"host": os.getenv("REDIS_REPLICA_1"), "port": 6379},
                {"host": os.getenv("REDIS_REPLICA_2"), "port": 6379},
            ],
            decode_responses=True,
            read_from_replicas=True,  # Read from replicas for better latency
            skip_full_coverage_check=True
        )
    
    async def get(self, key: str) -> Optional[Dict]:
        """Read from replica (faster, read-only)"""
        try:
            data = await self.replica.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Replica read failed: {e}, trying primary")
            # Fallback to primary
            data = await self.primary.get(key)
            return json.loads(data) if data else None
    
    async def set(self, key: str, value: Dict, ttl: int = 3600):
        """Write to primary (replicates to replicas)"""
        await self.primary.setex(key, ttl, json.dumps(value))
    
    async def get_multi(self, keys: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch read from replicas (optimized)"""
        pipeline = self.replica.pipeline()
        for key in keys:
            pipeline.get(key)
        results = await pipeline.execute()
        
        return {
            key: json.loads(result) if result else None
            for key, result in zip(keys, results)
        }
```

**Redis Cluster Configuration**:
```yaml
# Cloud Memorystore Redis (HA)
redis_version: redis_7_0
tier: standard
memory_size_gb: 64          # Per node
replica_count: 2            # 1 primary + 2 replicas per shard
node_count: 3               # 3 shards = 9 total nodes
transit_encryption: true
auth_enabled: true
```

**Capacity**:
- **64GB × 3 shards = 192GB total cache**
- **~100M cache entries (assuming 2KB avg per entry)**
- **Sub-millisecond latency (<1ms)**

### Layer 4: Queue-Based Processing (For Long-Running Tasks)

```python
# api/queue/processor.py
from google.cloud import tasks_v2
import asyncio

class ScrapingQueue:
    """
    Queue-based processing for long-running scrapes
    
    Benefits:
    - Prevents timeout issues
    - Better resource utilization
    - Can prioritize high-value tenants
    - Retry failed scrapes automatically
    """
    
    def __init__(self):
        self.client = tasks_v2.CloudTasksClient()
        self.queue_path = self.client.queue_path(
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_REGION"),
            queue="scraping-queue"
        )
    
    async def enqueue_scrape(
        self,
        tenant_id: str,
        url: str,
        fields: List[str],
        priority: int = 0  # Higher = more priority
    ) -> str:
        """Enqueue scraping task"""
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{os.getenv('API_URL')}/scrape/async",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Tenant-ID": tenant_id,
                },
                "body": json.dumps({
                    "url": url,
                    "fields": fields,
                }).encode(),
            },
            "schedule_time": None,  # Execute immediately
        }
        
        # Set priority (higher = processed first)
        if priority > 0:
            task["dispatch_deadline"] = {"seconds": 300}
        
        response = self.client.create_task(
            request={"parent": self.queue_path, "task": task}
        )
        return response.name
    
    async def get_queue_status(self) -> Dict:
        """Get queue metrics"""
        queue = self.client.get_queue(name=self.queue_path)
        return {
            "tasks_in_queue": queue.stats.tasks_count if queue.stats else 0,
            "oldest_task_age": queue.stats.oldest_estimated_arrival_time if queue.stats else None,
        }
```

**Cloud Tasks Configuration**:
```yaml
# infrastructure/cloud_tasks.yaml
queues:
  - name: scraping-queue
    rate_limits:
      max_dispatches_per_second: 1000
      max_concurrent_dispatches: 10000
    retry_config:
      max_attempts: 3
      max_retry_duration: 600s
      min_backoff: 1s
      max_backoff: 60s
    target:
      type: cloud-run
      service: universal-scraper-api
```

### Layer 5: Database Sharding (For Scale)

```python
# api/database/sharding.py
class TenantShardRouter:
    """
    Route tenants to database shards
    
    Strategy: Hash tenant_id to determine shard
    """
    
    def __init__(self):
        self.shards = [
            os.getenv("DB_SHARD_1"),  # 0-33% of tenants
            os.getenv("DB_SHARD_2"),  # 33-66% of tenants
            os.getenv("DB_SHARD_3"),  # 66-100% of tenants
        ]
    
    def get_shard(self, tenant_id: str) -> str:
        """Get database shard for tenant"""
        shard_index = hash(tenant_id) % len(self.shards)
        return self.shards[shard_index]
    
    async def get_tenant(self, tenant_id: str) -> Dict:
        """Get tenant from appropriate shard"""
        shard = self.get_shard(tenant_id)
        # Connect to shard and query
        return await query_shard(shard, tenant_id)
```

**Database Architecture**:
```sql
-- Shard 1: tenants_000000-333333
-- Shard 2: tenants_333334-666666
-- Shard 3: tenants_666667-999999

-- Each shard is a Cloud SQL instance
-- Read replicas per shard for read scaling
```

## Cost Optimization Strategies

### 1. Intelligent Caching (Reduce LLM Calls)

```python
# Cache hit rate optimization
class CacheOptimizer:
    """
    Maximize cache hits to reduce costs
    
    Strategy:
    - Domain-level pattern caching (reuse across URLs)
    - Field-level caching (reuse across different URLs on same domain)
    - Structure-based caching (reuse when HTML structure matches)
    """
    
    async def get_cached_pattern(self, domain: str, fields: List[str]) -> Optional[Dict]:
        """Check multiple cache levels"""
        # Level 1: Exact domain + fields match
        key1 = f"pattern:{domain}:{':'.join(sorted(fields))}"
        cached = await self.cache.get(key1)
        if cached:
            return cached
        
        # Level 2: Domain-level pattern (fields may differ)
        key2 = f"pattern:{domain}:*"
        cached = await self.cache.get(key2)
        if cached:
            # Adapt pattern to new fields
            return self.adapt_pattern(cached, fields)
        
        return None
```

**Expected Cache Hit Rates**:
- **First scrape**: 0% (must generate pattern)
- **Same domain, different URL**: 80-90% (domain pattern cache)
- **Same URL, different fields**: 60-70% (structure cache)
- **Same URL, same fields**: 95%+ (execution cache)

### 2. Spot Instances for Non-Critical Workloads

```yaml
# Use Cloud Run Jobs for batch processing (cheaper)
jobs:
  - name: batch-scraper
    schedule: "0 */6 * * *"  # Every 6 hours
    instance_type: spot       # 60-80% cheaper
    max_retries: 3
```

### 3. Regional Cost Optimization

```python
# Route to cheapest region when latency allows
REGION_COSTS = {
    "us-central1": 1.0,      # Baseline
    "us-east1": 0.95,         # 5% cheaper
    "asia-south1": 0.90,      # 10% cheaper
}

def select_region(tenant_region: str, latency_requirement: int) -> str:
    """Select cheapest region that meets latency requirement"""
    # Prefer tenant's region for low latency
    # Fallback to cheapest if latency requirement allows
    pass
```

## Monitoring & Auto-Scaling Policies

### Cloud Run Auto-Scaling Metrics

```yaml
# infrastructure/autoscaling.yaml
autoscaling:
  min_instances: 10           # Keep warm pool
  max_instances: 1000        # Scale to 1000 per region
  target_cpu_utilization: 70 # Scale when CPU > 70%
  target_concurrent_requests: 60  # Scale when > 60 concurrent
  target_request_count: 1000      # Scale when > 1000 req/min
```

### Custom Scaling Metrics

```python
# api/monitoring/scaling.py
class ScalingMetrics:
    """
    Custom metrics for intelligent scaling
    """
    
    async def get_scaling_signal(self) -> Dict:
        """Determine if scaling is needed"""
        metrics = {
            "queue_depth": await self.get_queue_depth(),
            "avg_response_time": await self.get_avg_response_time(),
            "error_rate": await self.get_error_rate(),
            "cache_hit_rate": await self.get_cache_hit_rate(),
        }
        
        # Scale up if:
        # - Queue depth > 1000 tasks
        # - Avg response time > 5s
        # - Error rate < 1% (healthy, can scale)
        
        # Scale down if:
        # - Queue depth < 100
        # - Avg response time < 1s
        # - Cache hit rate > 90% (less work needed)
        
        return self.calculate_scaling_decision(metrics)
```

## Capacity Planning

### Scenario: 1 Million Concurrent Users

**Assumptions**:
- Average request duration: 5 seconds
- Peak concurrent requests: 1,000,000
- Cache hit rate: 70% (30% need actual scraping)

**Required Resources**:

| Component | Configuration | Capacity |
|-----------|--------------|----------|
| **Cloud Run Instances** | 1000 instances × 2 regions | 160,000 concurrent |
| **Concurrency per Instance** | 80 requests | - |
| **Total Capacity** | 160,000 concurrent | **Need 6-7 regions** |
| **Redis Cluster** | 64GB × 3 shards × 2 regions | 384GB total |
| **Database** | 3 shards × 2 replicas | 6 instances |

**Cost Estimate** (1M concurrent users):
- **Cloud Run**: $5,000-10,000/month (pay per use)
- **Redis**: $1,200/month (2 regions)
- **Database**: $500/month (sharded)
- **Load Balancer**: $200/month
- **Total**: **~$7,000-12,000/month**

**Per-Request Cost**: $0.007-0.012 (at scale)

## Implementation Priority

### Phase 1: Foundation (Week 1-2)
1. ✅ Increase Cloud Run max instances to 1000
2. ✅ Add Redis cluster (Cloud Memorystore)
3. ✅ Implement distributed cache
4. ✅ Add tenant isolation

### Phase 2: Multi-Region (Week 3-4)
1. ✅ Deploy to 2-3 regions
2. ✅ Set up global load balancer
3. ✅ Configure Redis replication
4. ✅ Database read replicas

### Phase 3: Queue System (Week 5-6)
1. ✅ Cloud Tasks integration
2. ✅ Async scraping endpoints
3. ✅ Priority queues
4. ✅ Retry logic

### Phase 4: Optimization (Week 7-8)
1. ✅ Cache hit rate optimization
2. ✅ Auto-scaling policies
3. ✅ Cost monitoring
4. ✅ Performance tuning

## Next Steps

1. **Immediate**: Update Cloud Run config (max instances, concurrency)
2. **This Week**: Set up Redis cluster
3. **Next Week**: Multi-region deployment
4. **Month 2**: Queue system and optimization




