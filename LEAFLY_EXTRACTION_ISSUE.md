# Leafly Extraction Issue Analysis

## Problem

Apify is extracting data from Leafly's `__NEXT_DATA__` but returning **raw minified field names** instead of mapping them to the user's requested fields.

### User Requested
```
"Extract the product name, price and description for all products"
→ Parsed to: ['product', 'name', 'price', 'description', 'products']
```

### What We Got
```json
{
  "E": "53558599-7455-4c3a-a404-90568c470481",
  "x": "449761",
  "t": 1,
  "n": {"id": 5151, "name": "Kaviar", "description": "..."},
  "m": "Kaviar",
  "u": 3.5,
  ...
}
```

### What We Should Get
```json
{
  "name": "KAVIAR BLUE UNICORN POOP INFUSED SHAKE 3.5G",
  "price": "$XX.XX",
  "description": "Cannabis flower is rich in trichomes...",
  ...
}
```

---

## Root Cause Analysis

The `JSONDetector.extract_from_json()` method has two modes:

1. **Auto-extraction mode** (line 392-393):
   ```python
   if not fields:
       return items  # Returns ALL fields as-is
   ```

2. **Field-filtering mode** (line 395-406):
   ```python
   for field in fields:
       value = self._find_field_in_json(item, field)
       if value is not None:
           extracted_item[field] = value
   ```

**The Apify run is hitting auto-extraction mode** (returning all 20+ minified fields) instead of field-filtering mode (returning only the 5 requested fields).

---

## Why Auto-Mode Is Activating

### Hypothesis 1: Fields List Is Empty
The `fields` parameter might be empty/None when `extract_from_json` is called, causing it to return all fields.

**Evidence**: The Apify output has 20+ fields per item, not the 5 requested.

### Hypothesis 2: Field Matching Is Failing
The `_find_field_in_json` method is supposed to do fuzzy matching, but might be:
- Matching too aggressively (grabbing everything)
- OR not matching at all (returning None for all fields)

If it returns None for all fields, the code at line 403 would skip the item:
```python
if extracted_item:  # Only add if we found at least one field
    extracted.append(extracted_item)
```

And we'd get 0 items back. But we're getting items, so matching IS happening.

### Hypothesis 3: Local vs. Apify Difference
Local test (terminal output we saw) showed actual product names like:
- "KAVIAR BLUE UNICORN POOP INFUSED SHAKE 3.5G"
- "CRESCO CHEM SCOUT FLOWER 3.5G"  

But Apify is showing minified field names.

This suggests:
- **Local**: Field filtering IS working properly
- **Apify**: Auto-extraction mode is running (all fields)

---

## Data Structure Analysis

Looking at the Apify output, the actual product data IS there:

```json
{
  "n": {
    "id": 5151,
    "slug": "kaviar",
    "name": "Kaviar",  // Brand name (not product name!)
    "description": "Experience the Trifecta..."  // Brand description
  },
  "m": "Kaviar",  // Brand name string
  "u": 3.5,  // Product size (grams)
  "f": "https://...",  // Image URL
  ...
}
```

**Missing**: The actual product name (e.g., "KAVIAR BLUE UNICORN POOP INFUSED SHAKE")

This means:
1. We're extracting from the wrong part of the JSON (brand info vs. product info)
2. OR the product names are in a different field that we're not accessing

---

## Local Test Output (What Worked)

From our terminal test, we saw:
```
Product 2:
  • name: KAVIAR BLUE UNICORN POOP INFUSED SHAKE 3.5G
  • description: Cannabis flower is rich in trichomes...

Product 5:
  • product: {'id': 479593, 'slug': 'cresco-chem-scout...'}
  • name: CRESCO CHEM SCOUT FLOWER 3.5G
  • description: Make room on your sash and in your stash...
```

This shows:
- ✅ Field filtering WAS working (only showing requested fields)
- ✅ Product names WERE extracted correctly
- ✅ Descriptions WERE extracted

But then it also showed:
- `product`: Full nested object (should just be name string)
- Suggests fuzzy matching was grabbing too much

---

## The Real Problem

**The Leafly JSON structure has TWO levels**:

1. **Menu Items Array**: Contains product IDs, sizes, images
   - Has minified field names: "x", "t", "n", "m", etc.
   - The "n" field contains brand info (not product name!)

2. **Product Details** (nested or separate): Contains actual product names
   - Needs to be looked up by ID or slug
   - OR is in a different part of the `__NEXT_DATA__`

**Our extraction is only getting level 1** (menu items with brand info), not level 2 (actual product details).

---

## Solution Options

### Option A: Improve Field Matching (Quick)
Make `_find_field_in_json` smarter:
- Look for product name in nested "strain" or "product" objects
- Prioritize fields at certain nesting levels
- Add domain-specific hints for cannabis dispensaries

### Option B: Better JSON Structure Analysis (Better)
Enhance `_find_item_arrays` to:
- Identify the BEST array (not just any array)
- Look for arrays with the most relevant field names
- Prefer arrays with actual product names vs. just brand names

### Option C: LLM-Powered Extraction (Fallback)
If JSON extraction doesn't find product names:
- Fall back to HTML semantic extraction
- Use LLM to generate proper CSS selectors
- This costs $0.02 but guarantees correct extraction

### Option D: Manual Leafly Pattern (Fastest for Demo)
Create a hardcoded pattern specifically for Leafly:
```python
LEAFLY_PATTERN = {
    "nextjs_path": "props.pageProps.menuData.menuItems",
    "product_name_field": "strain.name" or "product.name",
    "price_field": "variants[0].price",
    "description_field": "product.description"
}
```

---

## Recommended Action

**Test locally first** to confirm the extraction is actually working as we saw in the terminal output. Then:

1. If local IS working correctly:
   - The issue is in how Apify is calling/returning the data
   - Check the actor code for any mutations to `fields`
   - Check JSON serialization

2. If local is ALSO returning minified fields:
   - The terminal output we saw was misleading
   - Need to fix `_extract_fields_from_items` logic
   - Add better field mapping

3. Short-term fix:
   - Just use auto-extraction mode (`fields=None`)
   - Return ALL fields from JSON
   - Let the user filter client-side
   - Still better than navigation elements!

---

## Status

- ✅ JavaScript rendering works
- ✅ JSON detection works (`__NEXT_DATA__` found)
- ✅ JSON parsing works (items extracted)
- ❌ **Field mapping broken** (returning raw fields)

**Next Step**: Create a simple local test that prints the actual extracted data structure to see if it matches what Apify returned.




