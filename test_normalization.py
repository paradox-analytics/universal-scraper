"""
Test the normalization logic locally
"""
import json

# Simulate what the extraction returns
item = {
    "title": "medium-nylon-crescent-bag-navy-8-hokh",
    "price": 5200,
    "color": {
        "_id": "sanity-colorway-navy",
        "colorName": "Navy"
    },
    "product detail url": 7368519155809
}

url = "https://baggu.com/collections/crescent-bags"

# Apply normalization logic
normalized_item = {}
url_field_value = None

schema_fields = {
    'title': 'title',
    'name': 'name', 
    'price': 'price',
    'rating': 'rating',
    'review_count': 'review_count',
    'url': 'url',
    'description': 'description'
}

for key, value in item.items():
    if key.startswith('_'):
        continue
    
    normalized_key = key.lower().replace(' ', '_').replace('-', '_')
    
    # Handle URL fields
    if normalized_key in ['url', 'product_url', 'product_detail_url', 'producturl', 'link', 'href']:
        if isinstance(value, (int, float)):
            base_domain = '/'.join(url.split('/')[:3])
            title_slug = normalized_item.get('title', '').replace(' ', '-').lower()
            if title_slug:
                url_field_value = f"{base_domain}/products/{title_slug}"
            else:
                url_field_value = f"{base_domain}/products/{int(value)}"
        elif isinstance(value, str) and value.startswith('http'):
            url_field_value = value
        elif isinstance(value, str):
            from urllib.parse import urljoin
            url_field_value = urljoin(url, value)
        
        if url_field_value:
            normalized_item['url'] = url_field_value
        continue
    
    # Handle color field
    if normalized_key == 'color':
        if isinstance(value, dict):
            color_value = value.get('colorName') or value.get('name') or value.get('value')
            if color_value:
                normalized_item['color'] = str(color_value)
        elif value:
            normalized_item['color'] = str(value)
        continue
    
    # Map other fields
    if normalized_key in schema_fields:
        schema_field = schema_fields[normalized_key]
        if schema_field == 'price' and isinstance(value, (int, float)):
            if value > 1000:
                normalized_item[schema_field] = f"${value / 100:.2f}"
            else:
                normalized_item[schema_field] = f"${value:.2f}"
        else:
            normalized_item[schema_field] = str(value) if value is not None else None

# Add metadata
normalized_item['_url'] = url_field_value if url_field_value else url
if 'url' not in normalized_item:
    normalized_item['url'] = normalized_item.get('_url', url)

normalized_item['_metadata'] = {
    'fetch_method': 'json',
    'extraction_source': 'json',
    'execution_time': 0
}

print("Normalized item:")
print(json.dumps(normalized_item, indent=2))







