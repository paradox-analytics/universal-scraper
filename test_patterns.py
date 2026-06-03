import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "X-API-Key": "test-api-key",
    "X-Tenant-ID": "test-tenant"
}

def test_pattern_management():
    # 1. Save a pattern
    save_payload = {
        "url": "https://example.com/products",
        "fields": ["title", "price"],
        "pattern_data": {
            "title": "h1.title",
            "price": "span.price"
        },
        "visibility": "private"
    }
    
    print("Saving pattern...")
    response = requests.post(f"{BASE_URL}/save-pattern", json=save_payload, headers=HEADERS)
    print(f"Response: {response.status_code}")
    print(response.json())
    
    if response.status_code != 200:
        print("Failed to save pattern")
        return

    # 2. List patterns
    print("\nListing patterns...")
    response = requests.get(f"{BASE_URL}/list-patterns", headers=HEADERS)
    print(f"Response: {response.status_code}")
    patterns = response.json().get("patterns", [])
    print(f"Found {len(patterns)} patterns")
    for p in patterns:
        print(f" - {p['domain']}: {p['fields']}")

    # 3. Delete pattern
    if patterns:
        print("\nDeleting pattern...")
        delete_payload = {
            "domain": "example.com",
            "fields": ["title", "price"]
        }
        response = requests.post(f"{BASE_URL}/delete-pattern", json=delete_payload, headers=HEADERS)
        print(f"Response: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    # Ensure backend is running
    try:
        test_pattern_management()
    except requests.exceptions.ConnectionError:
        print("Error: Backend is not running on http://localhost:8000")
