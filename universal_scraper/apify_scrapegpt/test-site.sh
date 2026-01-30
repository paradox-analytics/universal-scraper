#!/bin/bash
# Quick site testing script
# Usage: ./test-site.sh [amazon|ticketmaster|leafly|custom]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY not set${NC}"
    echo "Export it first: export OPENAI_API_KEY=\"sk-your-key-here\""
    exit 1
fi

# Check for Apify token (needed for residential proxies)
if [ -z "$APIFY_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  WARNING: APIFY_TOKEN not set${NC}"
    echo "   Residential proxies will not work without it."
    echo "   Export it: export APIFY_TOKEN=\"apify_api_your-token-here\""
    echo "   Continuing anyway..."
    echo ""
fi

# Parse arguments
SITE=${1:-help}

if [ "$SITE" = "help" ] || [ "$SITE" = "-h" ] || [ "$SITE" = "--help" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  🧪 Universal Scraper - Quick Site Tester${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo
    echo "Usage: ./test-site.sh [site]"
    echo
    echo "Available test sites:"
    echo "  leafly        - Leafly menu (pagination test)"
    echo "  amazon        - Amazon Same-Day Store (auth required)"
    echo "  ticketmaster  - Ticketmaster (anti-bot, JavaScript-heavy)"
    echo "  custom        - Use test-input.json"
    echo
    echo "Examples:"
    echo "  ./test-site.sh leafly"
    echo "  ./test-site.sh amazon"
    echo "  ./test-site.sh ticketmaster"
    echo
    exit 0
fi

# Select input file based on site
case $SITE in
    leafly)
        INPUT_FILE="test-input.json"
        SITE_NAME="Leafly (Pagination Test)"
        ;;
    amazon)
        INPUT_FILE="test-amazon-ssd.json"
        SITE_NAME="Amazon Same-Day Store"
        ;;
    ticketmaster)
        INPUT_FILE="test-ticketmaster.json"
        SITE_NAME="Ticketmaster"
        ;;
    custom)
        INPUT_FILE="test-input.json"
        SITE_NAME="Custom Configuration"
        ;;
    *)
        echo -e "${RED}❌ Unknown site: $SITE${NC}"
        echo "Run './test-site.sh help' for available options"
        exit 1
        ;;
esac

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Set up environment
export APIFY_LOCAL_STORAGE_DIR="./apify_storage_local"
# Keep existing APIFY_TOKEN if set, otherwise empty for local-only mode
export APIFY_TOKEN="${APIFY_TOKEN:-}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🎯 Testing: $SITE_NAME${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo
echo -e "${GREEN}Configuration:${NC}"
echo "  Site: $SITE"
echo "  Input: $INPUT_FILE"
echo "  Storage: $APIFY_LOCAL_STORAGE_DIR"
echo "  OpenAI Key: ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
if [ -n "$APIFY_TOKEN" ]; then
    echo "  Apify Token: ${APIFY_TOKEN:0:15}...${APIFY_TOKEN: -4} ✅"
    echo "  Proxy: Residential IPs enabled 🌐"
else
    echo "  Apify Token: Not set"
    echo "  Proxy: Local only (no residential IPs)"
fi
echo

# Create storage structure
mkdir -p "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default"
mkdir -p "$APIFY_LOCAL_STORAGE_DIR/datasets/default"

# Clean previous results
rm -rf "$APIFY_LOCAL_STORAGE_DIR/datasets/default"/*

# Prepare input
echo -e "${BLUE}📋 Preparing input configuration...${NC}"
cat "$INPUT_FILE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
data['openaiApiKey'] = '$OPENAI_API_KEY'
print(json.dumps(data, indent=2))
" > "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json"

echo -e "${GREEN}✅ Ready to test${NC}"
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Running Apify Actor...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Run the actor (use Python directly for local testing)
START_TIME=$(date +%s)

echo -e "${YELLOW}💡 Running actor directly with Python (faster than Docker)...${NC}"
python3 actor.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Test Complete (${ELAPSED}s)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Show results
if [ -d "$APIFY_LOCAL_STORAGE_DIR/datasets/default" ]; then
    RESULT_FILES=$(ls -1 "$APIFY_LOCAL_STORAGE_DIR/datasets/default" 2>/dev/null | wc -l)
    
    if [ $RESULT_FILES -gt 0 ]; then
        echo -e "${GREEN}📊 Results Summary:${NC}"
        
        for file in "$APIFY_LOCAL_STORAGE_DIR/datasets/default"/*.json; do
            if [ -f "$file" ]; then
                if command -v jq &> /dev/null; then
                    ITEMS=$(cat "$file" | jq 'length' 2>/dev/null || echo "0")
                    echo "  Items extracted: $ITEMS"
                    
                    if [ "$ITEMS" != "0" ] && [ "$ITEMS" != "null" ]; then
                        echo "  Sample fields:"
                        cat "$file" | jq -r '.[0] | keys[]' 2>/dev/null | head -5 | sed 's/^/    - /' || echo "    (unable to parse)"
                        
                        echo
                        echo -e "${BLUE}📝 First item preview:${NC}"
                        cat "$file" | jq '.[0]' 2>/dev/null | head -20 || echo "  (unable to parse)"
                    fi
                else
                    echo "  (Install jq for better output: brew install jq)"
                    echo "  Raw file: $file"
                fi
                break
            fi
        done
        
        echo
        echo -e "${GREEN}💾 Full results:${NC}"
        echo "  $APIFY_LOCAL_STORAGE_DIR/datasets/default/"
    else
        echo -e "${YELLOW}⚠️  No results generated${NC}"
        echo "  Check logs above for errors"
    fi
else
    echo -e "${YELLOW}⚠️  No results directory${NC}"
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}💡 Next Steps:${NC}"
echo "  • Review logs above for warnings/errors"
echo "  • Check results in: $APIFY_LOCAL_STORAGE_DIR/datasets/default/"
echo "  • Edit $INPUT_FILE to adjust configuration"
echo "  • Test another site: ./test-site.sh [leafly|amazon|ticketmaster]"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

