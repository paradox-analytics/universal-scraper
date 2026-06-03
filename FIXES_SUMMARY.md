# Fixes Applied - Build 2.0.29

## Issues Fixed

### 1. Missing "title" Field
**Problem:** JSON extraction was finding "rating" and "review count" but missing "title" field.

**Solution:**
- Enhanced field synonym mapping in `json_detector.py` to include more variations for "title":
  - Added: `productname`, `product_name`, `displayname`, `display_name`, `headline`
  - Improved matching logic for title/name fields
- Added automatic Direct LLM fallback when JSON extraction misses fields
- Implemented field merging: When JSON extraction succeeds but misses fields, Direct LLM extraction supplements the missing fields

**Code Changes:**
- `json_detector.py`: Enhanced `_get_field_synonyms()` to include more title variations
- `scraper.py`: Added logic to detect missing fields and trigger Direct LLM supplementation
- `scraper.py`: Added merge logic to combine JSON + Direct LLM results

### 2. Same URL for All Items
**Problem:** All extracted items had the same page URL instead of product-specific URLs.

**Solution:**
- Updated `main.py` to extract product URLs from JSON data when available
- Uses product-specific URLs (`url`, `productUrl`, `product_url`, `link`, `href`) if present
- Falls back to page URL if no product URL found

**Code Changes:**
- `main.py`: Added logic to check for product URLs in extracted items
- `main.py`: Removes duplicate URL fields after setting `_url`

## How It Works Now

### Field Extraction Flow:
1. **JSON Extraction** (primary) - Extracts from captured JSON APIs
2. **Missing Field Detection** - Checks if all requested fields are present
3. **Direct LLM Supplementation** - If fields missing, uses Direct LLM to fill gaps
4. **Field Merging** - Combines JSON + Direct LLM results by position
5. **Final Result** - Complete items with all requested fields

### URL Extraction Flow:
1. Check for product-specific URL fields in extracted item
2. Use first available: `url`, `productUrl`, `product_url`, `link`, `href`
3. Fall back to page URL if none found
4. Remove duplicate URL fields to avoid redundancy

## Expected Results

After these fixes, you should see:
- ✅ **Title field** present in all items
- ✅ **Product-specific URLs** (if available in JSON) instead of same page URL
- ✅ **Better field coverage** through JSON + Direct LLM merging

## Testing

To verify the fixes work:
1. Run the Actor with Chewy.com URL
2. Check that items include "title" field
3. Verify URLs are product-specific (if available in source data)
4. Check logs for "Merging Direct LLM results" message when fields are missing

## Build Info

- **Build:** 2.0.29
- **Deployment Date:** 2025-11-30
- **Actor:** https://console.apify.com/organization/YwaXmjFoleWBUmZdU/actors/MSwDish8FXKQKiIyx







