#!/bin/bash

# Universal Scraper - Deploy to Google Cloud Run
set -e

echo "🚀 Deploying Universal Scraper to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID from service account or prompt
PROJECT_ID=${GCP_PROJECT_ID:-"soma-data-467016"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="universal-scraper-api"

echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "🏗️  Building and deploying with Cloud Build..."
gcloud builds submit \
    --config=infrastructure/cloudbuild/cloudbuild.yaml

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

