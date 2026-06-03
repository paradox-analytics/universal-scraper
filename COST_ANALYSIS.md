# 💰 Google Cloud Run Cost Analysis

## Current Configuration

- **Min Instances**: 10 (warm pool)
- **Max Instances**: 100
- **Memory**: 4Gi per instance
- **CPU**: 2 per instance
- **Concurrency**: 80 per instance
- **Region**: us-central1

## Cost Breakdown

### With Min Instances = 10 (Current)

**Always-On Costs** (24/7, even with zero traffic):
- 10 instances × 2 CPU = **20 CPU cores** always allocated
- 10 instances × 4Gi = **40Gi memory** always allocated

**Monthly Cost Estimate**:
- CPU: 20 cores × $0.00002400/second × 2,592,000 seconds/month = **~$1,244/month**
- Memory: 40Gi × $0.00000250/Gi-second × 2,592,000 seconds/month = **~$259/month**
- **Total Always-On Cost: ~$1,503/month** (even with zero requests)

**Additional Costs** (when traffic exists):
- Requests: $0.40 per million requests
- Egress: $0.12 per GB (first 10GB free)

### With Min Instances = 0 (Cost-Optimized)

**Always-On Costs**: **$0/month** (scales to zero when idle)

**Costs Only When Active**:
- CPU: $0.00002400/second per CPU
- Memory: $0.00000250/Gi-second
- Requests: $0.40 per million requests

**Example** (1000 requests/day, avg 5 seconds each):
- CPU time: 1000 × 5s × 2 CPU = 10,000 CPU-seconds/day = **~$0.24/day = ~$7/month**
- Memory: 1000 × 5s × 4Gi = 20,000 Gi-seconds/day = **~$0.05/day = ~$1.50/month**
- Requests: 30,000/month = **~$0.01/month**
- **Total: ~$8.50/month** (vs $1,503/month with min-instances=10)

**Trade-off**: Cold starts (~1-3 seconds) on first request after idle period

## Cost Comparison

| Configuration | Always-On Cost | With Traffic (1K req/day) | Best For |
|--------------|----------------|---------------------------|----------|
| **Min=10** | $1,503/month | $1,503/month | High-traffic, low-latency critical |
| **Min=0** | $0/month | ~$8.50/month | Cost-sensitive, can tolerate cold starts |
| **Min=1** | ~$150/month | ~$150/month | Balance between cost and latency |

## Recommendations

### For Development/Testing:
```yaml
min-instances: 0  # Scale to zero, pay only for usage
```

### For Production (Low-Medium Traffic):
```yaml
min-instances: 1  # Keep 1 warm instance (~$150/month)
max-instances: 100
```

### For Production (High Traffic):
```yaml
min-instances: 10  # Current setting (~$1,503/month)
max-instances: 100
```

### For Cost Optimization:
```yaml
min-instances: 0  # Scale to zero
max-instances: 100
concurrency: 80  # Higher concurrency = fewer instances needed
```

## How to Change Min Instances

### Option 1: Update cloudbuild.yaml
```yaml
- '--min-instances'
- '0'  # Change from 10 to 0
```

### Option 2: Update via gcloud CLI
```bash
gcloud run services update universal-scraper-api \
  --region=us-central1 \
  --min-instances=0
```

### Option 3: Update via Console
1. Go to Cloud Run → universal-scraper-api
2. Edit & Deploy New Revision
3. Set "Minimum number of instances" to 0
4. Deploy

## Cost Monitoring

### Check Current Costs:
```bash
# View Cloud Run costs
gcloud billing accounts list
gcloud billing projects describe soma-data-467016

# View Cloud Run metrics
gcloud run services describe universal-scraper-api \
  --region=us-central1 \
  --format="value(status.conditions)"
```

### Set Budget Alerts:
1. Go to Cloud Console → Billing → Budgets & Alerts
2. Create budget alert (e.g., $50/month)
3. Get notified when approaching limit

## Free Tier

Google Cloud Run Free Tier:
- **2 million requests/month** (free)
- **360,000 Gi-seconds memory** (free)
- **180,000 vCPU-seconds** (free)
- **1 GB egress** (free)

**With min-instances=0**: You can stay within free tier for low-medium traffic!

**With min-instances=10**: You'll exceed free tier immediately (always-on instances).

## Recommendation

**For your use case** (SaaS product, potentially low initial traffic):

1. **Start with min-instances=0**:
   - Pay only for actual usage
   - Stay within free tier for low traffic
   - Accept ~1-3 second cold starts

2. **Monitor traffic**:
   - If you see consistent traffic, increase to min-instances=1
   - If you see high traffic, increase to min-instances=5-10

3. **Set budget alerts**:
   - Alert at $50/month
   - Alert at $200/month
   - Alert at $500/month

4. **Optimize later**:
   - Once you have traffic data, optimize min-instances based on actual usage patterns
   - Consider regional deployment for lower latency

## Current Monthly Cost Estimate

**With min-instances=10** (current):
- Always-on: **~$1,503/month**
- Plus usage costs (requests, egress)

**With min-instances=0** (recommended):
- Always-on: **$0/month**
- Usage: **~$8-50/month** (depending on traffic)
- **Savings: ~$1,450-1,495/month**




