
import asyncio
import json
import logging
import sys
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.json_detector import JSONDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def debug_price():
    print("🔎 Debugging Price Extraction Logic (Mock Data)")
    
    # Mock JSON representing likely Home Depot structure
    # They often have deeply nested props or split price fields
    mock_json = {
        "props": {
            "pageProps": {
                "productData": {
                    "item": {
                        "internet_number": "311259745",
                        "model_number": "HPL4136-VT",
                        "product_title": "2-Ton Hydraulic Trolley Car Jack",
                        "offers": {
                            "price": {
                                "currency": "$",     # <--- Suspect this is being grabbed
                                "value": 49.98,
                                "display": "$49.98"
                            },
                            "standardPrice": 49.98
                        },
                        "pricing": {
                            "currencySymbol": "$",
                            "value": 49.98
                        }
                    }
                }
            }
        }
    }
    
    detector = JSONDetector()
    
    # Manually trigger extraction on this mock data
    # We simulate what _extract_nextjs_data or similar would produce
    extracted_data = [mock_json]
    fields = ["price", "name", "model"]
    
    print(f"\n📋 Input Data: {json.dumps(mock_json, indent=2)}")
    
    print("\n🚀 Running Extraction...")
    results = detector.extract_from_json(extracted_data, fields)
    
    print("\n✅ Extraction Results:")
    print(json.dumps(results, indent=2))
    
    # Check if we reproduced the error
    if results and isinstance(results, list) and len(results) > 0:
        item = results[0]
        price = item.get('price')
        print(f"\n🔍 Extracted Price: '{price}'")
        
        if price == "$":
            print("🚨 REPRODUCED: Price is just '$'")
        elif price == 49.98 or price == "49.98" or price == "$49.98":
            print("✅ Price extracted correctly.")
        else:
            print(f"⚠️ Unexpected price value: {price}")

if __name__ == "__main__":
    asyncio.run(debug_price())
