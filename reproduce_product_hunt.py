
import sys
import os
import json
import logging

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.inline_json_extractor import InlineJSONExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_extraction():
    print("=" * 80)
    print("🧪 Testing InlineJSONExtractor on Product Hunt HTML")
    print("=" * 80)

    html_path = "product_hunt_raw_debug.html"
    if not os.path.exists(html_path):
        print(f"❌ Error: {html_path} not found.")
        return

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    print(f"📄 HTML size: {len(html):,} bytes")

    extractor = InlineJSONExtractor()
    results = extractor.extract(html)

    print(f"✅ Extracted {len(results)} items")

    if results:
        print("\n🔍 Sample Extracted Data:")
        for i, item in enumerate(results[:3]):
            print(f"\nItem {i+1}:")
            print(json.dumps(item, indent=2)[:500] + "...")
            
        # Check for Product Hunt specific data
        product_count = 0
        for item in results:
            data = item.get('data', {})
            # Check for products in various structures
            if isinstance(data, dict):
                # Check for Post type
                if data.get('__typename') == 'Post':
                    product_count += 1
                # Check for nested products
                # (This is a simplified check, real logic might be more complex)
            
        print(f"\n📦 Potential Products Found: {product_count}")

    else:
        print("❌ No data extracted.")

if __name__ == "__main__":
    test_extraction()
