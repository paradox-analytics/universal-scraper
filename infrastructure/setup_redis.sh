#!/bin/bash
# Setup Cloud Memorystore (Redis) for ParaDocs
# This script creates a Redis instance and configures Cloud Run to connect to it

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-soma-data-467016}"
REGION="${REGION:-us-central1}"
REDIS_INSTANCE_NAME="${REDIS_INSTANCE_NAME:-paradocs-cache}"
REDIS_TIER="${REDIS_TIER:-BASIC}"  # BASIC (no HA) or STANDARD_HA (high availability)
REDIS_SIZE_GB="${REDIS_SIZE_GB:-1}"  # 1GB is minimum and sufficient for caching
NETWORK="${NETWORK:-default}"
SERVICE_NAME="${SERVICE_NAME:-universal-scraper-api}"

echo "🔧 Setting up Cloud Memorystore (Redis) for ParaDocs"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Redis Instance: $REDIS_INSTANCE_NAME"
echo "   Tier: $REDIS_TIER"
echo "   Size: ${REDIS_SIZE_GB}GB"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable redis.googleapis.com
gcloud services enable vpcaccess.googleapis.com

# Check if Redis instance already exists
EXISTING_REDIS=$(gcloud redis instances list --region=$REGION --filter="name:$REDIS_INSTANCE_NAME" --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_REDIS" ]; then
    echo "✅ Redis instance '$REDIS_INSTANCE_NAME' already exists"
else
    echo "🚀 Creating Redis instance..."
    gcloud redis instances create $REDIS_INSTANCE_NAME \
        --size=$REDIS_SIZE_GB \
        --region=$REGION \
        --tier=$REDIS_TIER \
        --network=$NETWORK \
        --redis-version=redis_7_0
    
    echo "⏳ Waiting for Redis instance to be ready..."
    gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION --format="value(state)"
fi

# Get Redis IP address
REDIS_HOST=$(gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION --format="value(port)")

echo "📍 Redis instance details:"
echo "   Host: $REDIS_HOST"
echo "   Port: $REDIS_PORT"

# Check if VPC connector exists
VPC_CONNECTOR_NAME="paradocs-connector"
EXISTING_CONNECTOR=$(gcloud compute networks vpc-access connectors list --region=$REGION --filter="name:$VPC_CONNECTOR_NAME" --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_CONNECTOR" ]; then
    echo "✅ VPC connector '$VPC_CONNECTOR_NAME' already exists"
else
    echo "🔗 Creating VPC Access Connector..."
    gcloud compute networks vpc-access connectors create $VPC_CONNECTOR_NAME \
        --region=$REGION \
        --network=$NETWORK \
        --range=10.8.0.0/28 \
        --min-instances=2 \
        --max-instances=3
fi

# Update Cloud Run service with Redis connection
echo "🔄 Updating Cloud Run service with Redis connection..."
REDIS_URL="redis://${REDIS_HOST}:${REDIS_PORT}"

gcloud run services update $SERVICE_NAME \
    --region=$REGION \
    --set-env-vars="REDIS_URL=$REDIS_URL" \
    --vpc-connector=$VPC_CONNECTOR_NAME \
    --vpc-egress=private-ranges-only

echo ""
echo "✅ Redis setup complete!"
echo ""
echo "📋 Configuration:"
echo "   REDIS_URL: $REDIS_URL"
echo "   VPC Connector: $VPC_CONNECTOR_NAME"
echo ""
echo "🧪 Test Redis connection by running a scrape and checking cache"
echo ""




