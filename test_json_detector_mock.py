import asyncio
import json
import logging
import sys
from universal_scraper.core.json_detector import JSONDetector

# Enable logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('universal_scraper.core.json_detector')
logger.setLevel(logging.DEBUG)

async def test_mock_json():
    detector = JSONDetector()
    
    # Mock Home Depot-like JSON
    mock_json = {
        "props": {
            "pageProps": {
                "productData": {
                    "item": {
                        "identifiers": {"productName": "Mock Refrigerator", "modelNumber": "MOCK-123"},
                        "pricing": {"alternatePrice": 1299.0},
                        "description": "A mock description for testing.",
                        "brand": "MockBrand"
                    }
                }
            }
        }
    }
    
    html = f"""
    <html>
        <body>
            <script id="thd-pip-desktop-state" type="application/json">
                {json.dumps(mock_json)}
            </script>
        </body>
    </html>
    """
    
    fields = ["title", "brand", "model number", "price", "description"]
    
    print("🔍 Testing JSONDetector with mock HTML...")
    results = detector.detect_and_extract(html, "https://www.homedepot.com/p/mock/123")
    
    print(f"✅ Detection complete. Found: {results['json_found']}")
    if results['json_found']:
        print(f"✅ Data sources: {results['sources']}")
        # Pass the entire data list to extract_from_json
        items = detector.extract_from_json(results['data'], fields)
        print(f"✅ Extracted items: {items}")
    else:
        print("❌ No JSON found in mock HTML")

if __name__ == "__main__":
    asyncio.run(test_mock_json())
