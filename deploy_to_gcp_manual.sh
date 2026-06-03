#!/bin/bash

# Universal Scraper - Manual Deploy to Google Cloud Run (without Cloud Build)
set -e

echo "🚀 Deploying Universal Scraper to Google Cloud Run (Manual)..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"soma-data-467016"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="universal-scraper-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"
echo "   Image: $IMAGE_NAME"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -t $IMAGE_NAME:latest .

# Push to Container Registry
echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars PORT=8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --format='value(status.url)')

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📋 Service Information:"
echo "   URL: $SERVICE_URL"
echo "   Health Check: $SERVICE_URL/health"
echo ""
echo "🧪 Test the API:"
echo "   curl -X POST $SERVICE_URL/scrape \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'X-API-Key: YOUR_API_KEY' \\"
echo "     -d '{\"url\": \"https://example.com\", \"fields\": [\"title\"]}'"
echo ""




