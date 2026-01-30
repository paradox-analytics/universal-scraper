#!/bin/bash
# Local Apify Actor Testing Script
# Tests the actor in a local Apify environment without using credits

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🧪 Local Apify Actor Testing Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY environment variable not set${NC}"
    echo
    echo "Please set your OpenAI API key:"
    echo -e "  ${YELLOW}export OPENAI_API_KEY=\"sk-your-key-here\"${NC}"
    echo
    exit 1
fi

# Set up local storage directory
export APIFY_LOCAL_STORAGE_DIR="./apify_storage_local"
export APIFY_TOKEN=""  # Empty = local mode (no cloud connection)

echo -e "${GREEN}✅ Configuration:${NC}"
echo "   Local storage: $APIFY_LOCAL_STORAGE_DIR"
echo "   OpenAI API Key: ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
echo "   Mode: Local (no Apify credits used)"
echo

# Create local storage structure
mkdir -p "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default"
mkdir -p "$APIFY_LOCAL_STORAGE_DIR/datasets/default"

# Check if test-input.json exists
if [ ! -f "test-input.json" ]; then
    echo -e "${RED}❌ ERROR: test-input.json not found${NC}"
    echo "Please create test-input.json in the current directory"
    exit 1
fi

# Copy input file to local storage
echo -e "${BLUE}📋 Copying input configuration...${NC}"
cp test-input.json "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json"

# Update the input to use the environment variable for OpenAI key
echo -e "${BLUE}🔑 Injecting OpenAI API key from environment...${NC}"
cat "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
data['openaiApiKey'] = '$OPENAI_API_KEY'
print(json.dumps(data, indent=2))
" > "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json.tmp"
mv "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json.tmp" "$APIFY_LOCAL_STORAGE_DIR/key_value_stores/default/INPUT.json"

echo -e "${GREEN}✅ Local environment ready${NC}"
echo

# Clean previous results
if [ -d "$APIFY_LOCAL_STORAGE_DIR/datasets/default" ]; then
    echo -e "${YELLOW}🧹 Cleaning previous results...${NC}"
    rm -rf "$APIFY_LOCAL_STORAGE_DIR/datasets/default"/*
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Starting Apify Actor (Local Mode)...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Run the actor locally
apify run

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Actor execution complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Check if results were generated
if [ -d "$APIFY_LOCAL_STORAGE_DIR/datasets/default" ]; then
    RESULT_COUNT=$(ls -1 "$APIFY_LOCAL_STORAGE_DIR/datasets/default" 2>/dev/null | wc -l)
    if [ $RESULT_COUNT -gt 0 ]; then
        echo -e "${GREEN}📊 Results saved to:${NC}"
        echo "   $APIFY_LOCAL_STORAGE_DIR/datasets/default/"
        echo
        echo -e "${BLUE}📝 View results:${NC}"
        echo "   cat $APIFY_LOCAL_STORAGE_DIR/datasets/default/*.json | jq"
        echo
        
        # Show summary
        if command -v jq &> /dev/null; then
            echo -e "${GREEN}📈 Results Summary:${NC}"
            for file in "$APIFY_LOCAL_STORAGE_DIR/datasets/default"/*.json; do
                if [ -f "$file" ]; then
                    ITEMS=$(cat "$file" | jq 'length' 2>/dev/null || echo "0")
                    echo "   Items extracted: $ITEMS"
                    
                    # Show first item keys if available
                    if [ "$ITEMS" != "0" ]; then
                        echo "   Sample fields:"
                        cat "$file" | jq -r '.[0] | keys[]' 2>/dev/null | head -5 | sed 's/^/     - /'
                    fi
                    break
                fi
            done
        fi
    else
        echo -e "${YELLOW}⚠️  No results generated in dataset${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No dataset directory created${NC}"
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}💡 Tips:${NC}"
echo "   • Edit test-input.json to test different configurations"
echo "   • Check logs above for any errors or warnings"
echo "   • Results are in: $APIFY_LOCAL_STORAGE_DIR/datasets/default/"
echo "   • No Apify credits were used! ✨"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"








