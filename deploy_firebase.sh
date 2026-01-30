#!/bin/bash

# Deploy Frontend to Firebase Hosting
# This script will login, set project, and deploy

set -e

echo "🔥 Firebase Deployment Script"
echo "================================"
echo ""

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI not found. Installing..."
    npm install -g firebase-tools
fi

# Check if logged in
if ! firebase projects:list &> /dev/null; then
    echo "🔐 Please login to Firebase..."
    echo "   A browser window will open for authentication."
    firebase login
else
    echo "✅ Already logged in to Firebase"
fi

# Set Firebase project
echo ""
echo "📦 Setting Firebase project to: universal-scaper"
firebase use universal-scaper

# Verify build exists
if [ ! -d "frontend/dist" ]; then
    echo ""
    echo "🔨 Building frontend..."
    cd frontend
    npm run build
    cd ..
fi

# Deploy to Firebase Hosting
echo ""
echo "🚀 Deploying to Firebase Hosting..."
firebase deploy --only hosting

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app is live at:"
echo "   - https://universal-scaper.web.app"
echo "   - https://universal-scaper.firebaseapp.com"
echo ""




