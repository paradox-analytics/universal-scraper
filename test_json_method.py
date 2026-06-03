import sys
import os
sys.path.append(os.getcwd())

from universal_scraper.core.json_detector import JSONDetector

def test_method():
    detector = JSONDetector()
    print(f"Detector: {detector}")
    print(f"Has method: {hasattr(detector, '_extract_single_item_semantically')}")
    
    try:
        result = detector._extract_single_item_semantically({"name": "test"}, ["name"])
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_method()
