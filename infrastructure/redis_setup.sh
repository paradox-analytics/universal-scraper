#!/bin/bash
# Setup Redis Cluster for Multi-Tenant SaaS

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"soma-data-467016"}
REGION=${GCP_REGION:-"us-central1"}
REDIS_NAME="universal-scraper-cache"

echo "🚀 Setting up Redis cluster for multi-tenant SaaS..."

# Create Redis instance (Cloud Memorystore)
gcloud redis instances create $REDIS_NAME \
  --size=64 \
  --region=$REGION \
  --tier=standard \
  --redis-version=redis_7_0 \
  --network=default \
  --transit-encryption-mode=server-authentication \
  --auth-enabled \
  --display-name="Universal Scraper Cache" \
  --labels=environment=production,service=universal-scraper

# Get Redis IP
REDIS_IP=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(host)")

echo "✅ Redis cluster created!"
echo "📋 Redis IP: $REDIS_IP"
echo ""
echo "🔧 Next steps:"
echo "1. Update Cloud Run environment variables:"
echo "   REDIS_URL=redis://$REDIS_IP:6379"
echo ""
echo "2. Set Redis password (if auth enabled):"
echo "   gcloud redis instances get-auth-string $REDIS_NAME --region=$REGION"
echo ""
echo "3. Test connection:"
echo "   redis-cli -h $REDIS_IP -p 6379 ping"




