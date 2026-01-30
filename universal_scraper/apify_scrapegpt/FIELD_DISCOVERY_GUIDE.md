# 🔍 Field Discovery Guide

**How to find and define fields for scraping new websites**

---

## 📋 Table of Contents

1. [Quick Start: Auto-Extraction Mode](#quick-start-auto-extraction-mode)
2. [When to Use Manual Field Specification](#when-to-use-manual-field-specification)
3. [Method 1: Browser Developer Tools](#method-1-browser-developer-tools)
4. [Method 2: Test & Refine Approach](#method-2-test--refine-approach)
5. [Method 3: JSON Inspection](#method-3-json-inspection)
6. [Common Field Patterns](#common-field-patterns)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Quick Start: Auto-Extraction Mode

**✅ Recommended for new websites you've never seen before**

Simply leave the `fields` array empty and the scraper will automatically extract all structured data:

```json
{
  "mode": "scrape_only",
  "startUrls": [
    {"url": "https://example.com/products/item-1"}
  ],
  "scrapeConfig": {
    "fetchMode": "browser"
    // No fields = auto-extraction!
  }
}
```

**Output:** Returns all discovered data
```json
{
  "product_name": "Blue Widget",
  "price": "$29.99",
  "rating": 4.5,
  "reviews": 128,
  "availability": "In Stock",
  "sku": "BW-001",
  "description": "High-quality blue widget..."
}
```

**After reviewing the output, you can refine with specific fields.**

---

## When to Use Manual Field Specification

Use specific fields when you want to:

1. ✅ **Control output structure** - Ensure consistent field names
2. ✅ **Reduce noise** - Only extract what you need
3. ✅ **Improve performance** - Faster extraction with targeted fields
4. ✅ **Standardize across sites** - Same field names for different websites
5. ✅ **Handle complex data** - Guide the AI for tricky extractions

---

## Method 1: Browser Developer Tools

**Best for: Understanding page structure**

### Step 1: Open Developer Tools

1. Visit the target website in Chrome/Firefox
2. Right-click → "Inspect" or press `F12`
3. Navigate to the page you want to scrape

### Step 2: Inspect Elements

**Find the data you want:**

```html
<!-- Example: Product page -->
<div class="product-container">
  <h1 class="product-title">Blue Widget</h1>
  <span class="price">$29.99</span>
  <div class="rating">
    <span class="stars">4.5</span>
    <span class="review-count">128 reviews</span>
  </div>
  <p class="description">High-quality blue widget...</p>
</div>
```

### Step 3: Identify Field Names

Look at:
- **Class names:** `product-title`, `price`, `rating`
- **Data attributes:** `data-product-name`, `data-price`
- **Text patterns:** "Price:", "$", "Rating:", etc.

### Step 4: Create Field List

Based on the structure, define fields:

```json
{
  "fields": [
    "product_title",    // or "name" (AI will match both)
    "price",
    "rating",
    "reviews",
    "description"
  ]
}
```

**💡 TIP:** The AI is smart - you don't need exact class names. Use semantic names like:
- `"name"`, `"title"`, `"product_name"` → AI will find the product name
- `"price"`, `"cost"`, `"amount"` → AI will find the price
- `"rating"`, `"score"`, `"stars"` → AI will find the rating

---

## Method 2: Test & Refine Approach

**Best for: Iterative discovery**

### Step 1: Run Auto-Extraction First

```json
{
  "mode": "scrape_only",
  "startUrls": [{"url": "https://example.com/product"}],
  "scrapeConfig": {
    "fetchMode": "browser"
    // No fields specified
  }
}
```

### Step 2: Review Output

Look at what the scraper found:

```json
{
  "title": "Blue Widget",
  "price_amount": "$29.99",
  "product_rating": 4.5,
  "total_reviews": 128,
  "in_stock": true,
  "product_description": "High-quality...",
  "manufacturer": "WidgetCo",
  "sku": "BW-001"
}
```

### Step 3: Select Fields You Need

Create a refined field list:

```json
{
  "fields": [
    "title",
    "price_amount",
    "product_rating",
    "in_stock",
    "sku"
  ]
}
```

### Step 4: Re-run with Specific Fields

Test again with your refined list to ensure consistency.

---

## Method 3: JSON Inspection

**Best for: Sites with embedded JSON**

Many modern websites embed data in JSON format. Look for:

### In HTML Source (View Page Source):

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org/",
  "@type": "Product",
  "name": "Blue Widget",
  "price": "29.99",
  "ratingValue": "4.5"
}
</script>
```

or

```html
<script id="__NEXT_DATA__">
{
  "props": {
    "product": {
      "name": "Blue Widget",
      "price": 29.99,
      "rating": 4.5
    }
  }
}
</script>
```

### Extract Field Names from JSON:

```json
{
  "fields": [
    "name",
    "price",
    "ratingValue"
  ]
}
```

**💡 TIP:** The scraper automatically detects and extracts from JSON-LD and embedded JSON!

---

## Common Field Patterns

### E-commerce Products

```json
{
  "fields": [
    "product_name",
    "brand",
    "price",
    "original_price",
    "discount",
    "rating",
    "review_count",
    "availability",
    "sku",
    "description",
    "images",
    "specifications"
  ]
}
```

### Listings (Real Estate, Jobs, etc.)

```json
{
  "fields": [
    "title",
    "location",
    "price",
    "date_posted",
    "description",
    "contact",
    "features"
  ]
}
```

### Articles/Blog Posts

```json
{
  "fields": [
    "title",
    "author",
    "publish_date",
    "category",
    "tags",
    "content",
    "excerpt"
  ]
}
```

### Profiles (LinkedIn, Social Media)

```json
{
  "fields": [
    "name",
    "title",
    "company",
    "location",
    "bio",
    "skills",
    "experience"
  ]
}
```

### Cannabis Products (Leafly Example)

```json
{
  "fields": [
    "product_name",
    "brand",
    "category",
    "thc_percentage",
    "cbd_percentage",
    "price",
    "weight",
    "strain_type",
    "effects",
    "description"
  ]
}
```

---

## Best Practices

### ✅ DO

1. **Use semantic names** - `"price"` not `"span-class-price-text"`
2. **Be descriptive** - `"thc_percentage"` not `"thc"`
3. **Use snake_case** - `"product_name"` not `"ProductName"`
4. **Start broad, then refine** - Use auto-extraction first
5. **Test on multiple pages** - Ensure consistency

### ❌ DON'T

1. **Don't use class names** - `"price"` not `"product-price-container"`
2. **Don't be too specific** - `"rating"` works better than `"five_star_rating_out_of_five"`
3. **Don't guess** - Use auto-extraction or inspect the page first
4. **Don't over-extract** - Only include fields you actually need

---

## Examples

### Example 1: New E-commerce Site

**Goal:** Scrape product listings

**Step 1:** Test with auto-extraction
```json
{
  "mode": "scrape_only",
  "startUrls": [{"url": "https://newshop.com/products/widget"}],
  "scrapeConfig": {}
}
```

**Step 2:** Review output
```json
{
  "item_name": "Blue Widget",
  "cost": "$29.99",
  "item_rating": "4.5 stars",
  "stock_status": "Available"
}
```

**Step 3:** Define specific fields
```json
{
  "fields": ["item_name", "cost", "item_rating", "stock_status"]
}
```

---

### Example 2: Cannabis Dispensary (Leafly)

**Goal:** Extract menu items with pricing and THC content

**Manual Discovery:**

1. Visit: `https://www.leafly.com/dispensary-info/example/menu`
2. Inspect product cards
3. Find: name, brand, price, THC%, CBD%

**Field Configuration:**
```json
{
  "mode": "scrape_only",
  "startUrls": [
    {"url": "https://www.leafly.com/dispensary-info/mammoth-holistics/menu"}
  ],
  "scrapeConfig": {
    "fetchMode": "browser",
    "fields": [
      "product_name",
      "brand",
      "category",
      "thc_percentage",
      "cbd_percentage",
      "price",
      "weight",
      "strain_type"
    ]
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"],
    "apifyProxyCountry": "US"
  }
}
```

---

### Example 3: Real Estate Listings

**Auto-discovery approach:**

```json
{
  "mode": "scrape_only",
  "startUrls": [
    {"url": "https://realty.com/property/123-main-st"}
  ],
  "scrapeConfig": {
    // Auto-extract first to see what's available
  }
}
```

**After review, refine:**
```json
{
  "fields": [
    "address",
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "lot_size",
    "year_built",
    "description",
    "agent_name",
    "agent_phone"
  ]
}
```

---

## 🛠️ Debugging Tips

### Problem: Empty Results

**Cause:** Field names don't match page content

**Solution:** 
1. Check if page requires JavaScript (`fetchMode: "browser"`)
2. Try auto-extraction to see what AI finds
3. Inspect HTML to verify data exists
4. Use more generic field names

### Problem: Inconsistent Data

**Cause:** Page structure varies

**Solution:**
1. Test on multiple page samples
2. Use schema validation (see `schema` in scrapeConfig)
3. Add fallback field names: `["name", "title", "product_name"]`

### Problem: Missing Fields

**Cause:** Data not visible or in dynamic content

**Solution:**
1. Use `fetchMode: "browser"` for JavaScript sites
2. Check Network tab for API calls (JSON data)
3. Increase `browserTimeout` if page loads slowly

---

## 🚀 Quick Reference

| I want to... | Method | Configuration |
|--------------|--------|---------------|
| **Try a new website** | Auto-extraction | `"fields": []` |
| **Know what's available** | Auto-extraction + review | `"fields": []` → review output |
| **Extract specific data** | Manual fields | `"fields": ["name", "price"]` |
| **Scrape JavaScript site** | Browser mode | `"fetchMode": "browser"` |
| **Standardize output** | Schema definition | Use `schema` in scrapeConfig |

---

## 📚 Additional Resources

- **README.md** - Overview and quick start
- **INPUT_SCHEMA.json** - Full configuration reference
- **Examples/** - Sample configurations for common use cases

---

## 💡 Pro Tips

1. **Start simple:** Use auto-extraction for discovery
2. **Iterate:** Refine fields based on actual output
3. **Test small:** Run on 1-2 pages first before scaling
4. **Use browser mode:** When in doubt, use `fetchMode: "browser"`
5. **Check the JSON:** Many sites embed data in JSON (faster extraction!)
6. **Semantic naming:** The AI understands context - use natural field names
7. **Document your fields:** Keep notes on what each field represents

---

**Need help?** 
- Check the logs for hints about what the scraper found
- Use auto-extraction to let the AI discover fields
- Review the examples for your type of website

Happy scraping! 🎉








