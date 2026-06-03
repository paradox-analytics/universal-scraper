import json
import re

# Read the HTML file
with open('debug_product_hunt.html', 'r') as f:
    html = f.read()

# Find the Apollo SSR Data Transport
search_str = 'window[Symbol.for("ApolloSSRDataTransport")]'
idx = html.find(search_str)

if idx != -1:
    push_idx = html.find('.push(', idx)
    if push_idx != -1:
        json_start = push_idx + 6
        
        # Extract balanced JSON
        start_char = html[json_start]
        if start_char == '{':
            end_char = '}'
        else:
            print("Unexpected start character")
            exit(1)
            
        stack = 1
        in_string = False
        escape = False
        
        for i in range(json_start + 1, len(html)):
            char = html[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"':
                in_string = not in_string
                continue
                
            if not in_string:
                if char == start_char:
                    stack += 1
                elif char == end_char:
                    stack -= 1
                    if stack == 0:
                        json_str = html[json_start:i+1]
                        break
        
        # Clean and parse
        json_str_clean = json_str.replace(':undefined', ':null').replace(',undefined', ',null')
        data = json.loads(json_str_clean)
        
        # Look for productCategory keys
        for key, value in data['rehydrate'].items():
            if isinstance(value, dict) and 'data' in value:
                data_content = value['data']
                if isinstance(data_content, dict) and 'productCategory' in data_content:
                    print(f"\n{key}: Has productCategory")
                    category = data_content['productCategory']
                    print(f"  Category keys: {list(category.keys())[:15]}")
                    
                    # Look for product arrays
                    for k, v in category.items():
                        if isinstance(v, (list, dict)):
                            print(f"  {k}: {type(v).__name__}")
                            if isinstance(v, list) and len(v) > 0:
                                print(f"    Length: {len(v)}")
                                if isinstance(v[0], dict):
                                    print(f"    First item keys: {list(v[0].keys())[:10]}")
                            elif isinstance(v, dict):
                                print(f"    Dict keys: {list(v.keys())[:10]}")
                                # Check for nested arrays
                                for k2, v2 in v.items():
                                    if isinstance(v2, list) and len(v2) > 5:
                                        print(f"      {k2}: list with {len(v2)} items")
                                        if isinstance(v2[0], dict):
                                            print(f"        First item: {json.dumps(v2[0], indent=2)[:300]}")
