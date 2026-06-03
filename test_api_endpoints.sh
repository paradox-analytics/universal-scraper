#!/bin/bash

# Test API Endpoints
API_URL="https://universal-scraper-api-968720932091.us-central1.run.app"
API_KEY="${1:-test-key}"

echo "Testing API Endpoints..."
echo "========================"
echo ""

# Test Health
echo "1. Testing /health endpoint..."
curl -s "$API_URL/health" | jq . || echo "Failed"
echo ""

# Test Scrape (will fail without valid API key, but tests endpoint)
echo "2. Testing /scrape endpoint..."
curl -s -X POST "$API_URL/scrape" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "url": "https://example.com",
    "fields": ["title"]
  }' | jq . || echo "Failed"
echo ""

# Test Crawl
echo "3. Testing /crawl endpoint..."
curl -s -X POST "$API_URL/crawl" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "start_urls": ["https://example.com"],
    "fields": ["title"]
  }' | jq . || echo "Failed"
echo ""

echo "Done!"




