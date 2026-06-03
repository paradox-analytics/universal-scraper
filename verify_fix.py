import logging
from universal_scraper.core.json_quality_validator import JSONQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_fix():
    print("🚀 Verifying JSONQualityValidator fix...")
    
    validator = JSONQualityValidator()
    
    # Test case 1: Clean data
    clean_data = {
        "name": "Product Hunt",
        "description": "The best new products in tech."
    }
    is_valid = validator.is_high_quality_value(clean_data)
    print(f"1. Clean data valid? {is_valid} (Expected: True)")
    if not is_valid:
        print("❌ FAILED: Clean data rejected")
        return

    # Test case 2: Garbage data (replacement characters)
    garbage_data = {
        "name": "Product Hunt",
        "description": "The best new products \ufffd\ufffd\ufffd in tech."
    }
    is_valid = validator.is_high_quality_value(garbage_data)
    print(f"2. Garbage data valid? {is_valid} (Expected: False)")
    if is_valid:
        print("❌ FAILED: Garbage data accepted")
        return
        
    # Test case 3: Nested garbage
    nested_garbage = {
        "items": [
            {"id": 1, "name": "Clean"},
            {"id": 2, "name": "Garbage \ufffd"}
        ]
    }
    is_valid = validator.is_high_quality_value(nested_garbage)
    print(f"3. Nested garbage valid? {is_valid} (Expected: False)")
    if is_valid:
        print("❌ FAILED: Nested garbage accepted")
        return

    print("\n✅ VERIFICATION SUCCESSFUL: Validator correctly rejects replacement characters.")

if __name__ == "__main__":
    verify_fix()
