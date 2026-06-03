#!/bin/bash

# Universal Scraper - Deploy to Apify with Camoufox Support
set -e

echo "🦊 Deploying Universal Scraper with Camoufox to Apify..."

# Navigate to Apify directory
cd "$(dirname "$0")/universal_scraper/apify"

# Check if Apify CLI is installed
if ! command -v apify &> /dev/null; then
    echo "❌ Apify CLI not found!"
    echo "Install with: npm install -g apify-cli"
    exit 1
fi

# Check if logged in
if ! apify info &> /dev/null; then
    echo "🔐 Please login to Apify:"
    apify login
fi

echo "✅ Apify CLI ready"

# Create .actor directory if it doesn't exist
mkdir -p .actor

# Create actor.json with metadata
cat > .actor/actor.json << 'EOF'
{
  "actorSpecification": 1,
  "name": "universal-scraper-camoufox",
  "title": "Universal Web Scraper (Camoufox + AI)",
  "description": "AI-powered universal scraper with Camoufox anti-detection. Works on any website.",
  "version": "2.0.0",
  "meta": {
    "templateId": "universal-scraper"
  },
  "input": "./INPUT_SCHEMA_V2.json",
  "dockerfile": "./Dockerfile"
}
EOF

echo "📝 Created actor.json"

# Copy the v2 actor as main
cp actor_v2.py actor.py

echo "✅ Updated actor.py"

# Push to Apify
echo "🚀 Pushing to Apify..."
apify push

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Go to: https://console.apify.com/"
echo "2. Find your 'universal-scraper-camoufox' actor"
echo "3. Click 'Run' and test with Reddit:"
echo ""
echo "   Test Input:"
echo "   {"
echo '     "mode": "scrape_only",'
echo '     "urls": [{"url": "https://www.reddit.com/r/webscraping/"}],'
echo '     "fields": ["title", "author", "upvotes", "comments"],'
echo '     "browserConfig": {"useCamoufox": true},'
echo '     "proxyConfiguration": {"useApifyProxy": true, "apifyProxyGroups": ["RESIDENTIAL"]},'
echo '     "apiKeys": {"openaiApiKey": "<YOUR_KEY>"},'
echo '     "crawlConfig": {"maxDepth": 0, "maxPages": 1, "handlePagination": false}'
echo "   }"
echo ""
echo "4. Check results - should extract 60+ Reddit posts!"







