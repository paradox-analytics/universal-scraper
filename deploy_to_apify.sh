#!/bin/bash
# Deploy Universal Scraper to Apify

set -e

echo "🚀 Deploying Universal Scraper to Apify..."

# Check if apify CLI is installed
if ! command -v apify &> /dev/null; then
    echo "❌ Apify CLI not found. Installing..."
    npm install -g apify-cli
fi

# Check if logged in
if ! apify info &> /dev/null; then
    echo "🔐 Please log in to Apify..."
    apify login
fi

# Navigate to project directory
cd "$(dirname "$0")"

echo "📦 Building Docker image..."

# Create .actor directory if it doesn't exist
mkdir -p .actor

# Copy actor config
cp universal_scraper/apify/.actor/actor.json .actor/
cp universal_scraper/apify/INPUT_SCHEMA.json .actor/

# Copy Dockerfile
cp universal_scraper/apify/Dockerfile .

# Deploy to Apify
echo "🚢 Deploying to Apify..."
apify push

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your actor is now available on Apify platform"
echo "📖 Visit https://console.apify.com to configure and run"

