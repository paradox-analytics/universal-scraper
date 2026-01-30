#!/bin/bash

# Firebase Authentication Setup Script
# This script helps you enable Firebase Authentication providers

echo "🔥 Firebase Authentication Setup"
echo "================================"
echo ""
echo "This script will help you enable Firebase Authentication providers."
echo ""

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI is not installed."
    echo "Install it with: npm install -g firebase-tools"
    exit 1
fi

# Get current project
PROJECT_ID=$(firebase use 2>&1 | grep -oP '(?<=\()[^)]+' | head -1)
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID="universal-scaper"
fi

echo "📋 Current Firebase Project: $PROJECT_ID"
echo ""
echo "To enable Firebase Authentication providers:"
echo ""
echo "1. Open Firebase Console:"
echo "   https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
echo ""
echo "2. Enable the following providers:"
echo ""
echo "   ✅ Email/Password:"
echo "      - Click 'Email/Password'"
echo "      - Toggle 'Enable' ON"
echo "      - Click 'Save'"
echo ""
echo "   ✅ Google Sign-In:"
echo "      - Click 'Google'"
echo "      - Toggle 'Enable' ON"
echo "      - Enter project support email"
echo "      - Click 'Save'"
echo ""
echo "   ✅ Apple Sign-In (Optional):"
echo "      - Click 'Apple'"
echo "      - Toggle 'Enable' ON"
echo "      - Enter Apple Services ID (if you have one)"
echo "      - Click 'Save'"
echo ""
echo "3. Verify Authorized Domains:"
echo "   - Go to Authentication → Settings"
echo "   - Ensure these domains are listed:"
echo "     • universal-scaper.web.app"
echo "     • universal-scaper.firebaseapp.com"
echo "     • localhost (for local development)"
echo ""
echo "Opening Firebase Console in your browser..."
echo ""

# Try to open the URL in the default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
else
    echo "Please manually open: https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
fi

echo ""
echo "✅ After enabling the providers, refresh your login page!"
echo ""




