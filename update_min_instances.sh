#!/bin/bash
# Update Cloud Run min-instances to reduce costs

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"soma-data-467016"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="universal-scraper-api"
MIN_INSTANCES=${1:-0}  # Default to 0 (scale to zero)

echo "💰 Updating Cloud Run min-instances to reduce costs..."
echo "   Service: $SERVICE_NAME"
echo "   Min Instances: $MIN_INSTANCES"
echo ""

gcloud config set project $PROJECT_ID

gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --min-instances=$MIN_INSTANCES

echo ""
echo "✅ Updated min-instances to $MIN_INSTANCES"
echo ""
echo "📊 Cost Impact:"
if [ "$MIN_INSTANCES" -eq "0" ]; then
  echo "   Always-on cost: \$0/month (scales to zero)"
  echo "   You'll only pay for actual usage"
elif [ "$MIN_INSTANCES" -eq "1" ]; then
  echo "   Always-on cost: ~\$150/month"
elif [ "$MIN_INSTANCES" -eq "10" ]; then
  echo "   Always-on cost: ~\$1,503/month"
fi
echo ""
echo "💡 Tip: Monitor your usage and adjust min-instances based on traffic patterns"




