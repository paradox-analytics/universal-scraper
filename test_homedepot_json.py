import asyncio
import json
import logging
import os
from universal_scraper.core.json_detector import JSONDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_homedepot_json():
    # Load the saved HTML
    if not os.path.exists("debug_homedepot.html"):
        print("❌ debug_homedepot.html not found")
        return
        
    with open("debug_homedepot.html", "r") as f:
        html = f.read()
        
    print(f"✅ Loaded {len(html)} chars")
    
    # Initialize detector
    detector = JSONDetector()
    
    # Run detection
    print("\n🔍 Running JSON Detection...")
    results = detector.detect_and_extract(html, "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591")
    
    print(f"\n📊 Results:")
    print(f"  JSON Found: {results['json_found']}")
    print(f"  Sources: {results['sources']}")
    
    if results['data']:
        print(f"  Data items: {len(results['data'])}")
        for i, item in enumerate(results['data']):
            print(f"\n  Item {i+1} ({item.get('_framework')}):")
            data = item.get('_data')
            if isinstance(data, list):
                print(f"    Type: List with {len(data)} items")
                if data:
                    print(f"    First item keys: {list(data[0].keys())[:10]}")
            elif isinstance(data, dict):
                print(f"    Type: Dict with keys {list(data.keys())[:10]}")
                if '@type' in data:
                    print(f"    @type: {data['@type']}")
                if 'name' in data:
                    print(f"    name: {data['name']}")

    # Test extraction
    fields = ['title', 'brand', 'model number', 'price', 'description', 'color', 'dimensions', 'weight', 'rating', 'warranty information', 'internal dispenser', 'fingerprint resistant finish', 'number of reviews', 'availability status']
    print(f"\n🎯 Testing Extraction for fields: {fields}")
    
    extracted = detector.extract_from_json(results['data'], fields)
    print(f"  Extracted {len(extracted)} items")
    if extracted:
        print(f"  First item: {json.dumps(extracted[0], indent=2)}")
        
    # Check sufficiency
    is_sufficient = detector.is_json_sufficient(results, fields)
    print(f"\nIs JSON sufficient? {is_sufficient}")

if __name__ == "__main__":
    asyncio.run(test_homedepot_json())
