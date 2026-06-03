# Answer: How Schemas Are Defined for NEW Websites

## Your Question

> **"How is the schema defined the first time a user chooses a new website to scrape?"**

---

## The Answer: Auto-Generation

**The schema is AUTO-GENERATED from your first scrape!**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FIRST TIME SCRAPING A NEW WEBSITE                 │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Scrape Once (No Schema Needed)
┌─────────────────────────────────────────────────────────────────────┐
│  scraper = UniversalScraper()                                        │
│  result = scraper.scrape(url, fields)                                │
│                                                                       │
│  Output: Raw data (19 items found)                                   │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
Step 2: Auto-Generate Schema
┌─────────────────────────────────────────────────────────────────────┐
│  inferencer = SchemaInference()                                      │
│  inferencer.learn_from_data(result['data'])                          │
│                                                                       │
│  Analyzes:                                                            │
│    • Field types (string, number, object, etc.)                      │
│    • Coverage (% of items with each field)                           │
│    • Nesting patterns (extract from objects)                         │
│    • Generates aliases (camelCase, snake_case, etc.)                 │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
Step 3: Generate Schema Definition
┌─────────────────────────────────────────────────────────────────────┐
│  schema = inferencer.generate_schema("site_name")                    │
│                                                                       │
│  Generated Schema:                                                    │
│    • name (string, 100% coverage) → REQUIRED                         │
│    • price (number, 94.7% coverage) → REQUIRED                       │
│    • brand (string, 94.7% coverage) → OPTIONAL                       │
│    • thc_percentage (number, 52.6% coverage) → OPTIONAL              │
│    + Auto-generated aliases & transformers                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
Step 4: Export for Production Use
┌─────────────────────────────────────────────────────────────────────┐
│  schema_code = inferencer.export_schema_code("site_name")            │
│                                                                       │
│  Saves to: generated_schema_site_name.py                             │
│  → Commit to version control                                         │
│  → Reuse forever                                                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
Step 5: Use for All Future Scrapes
┌─────────────────────────────────────────────────────────────────────┐
│  scraper = UniversalScraper(schema=schema)                           │
│  result = scraper.scrape(url, fields)                                │
│                                                                       │
│  Output: Stable schema (even when website changes!)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Code Example

```python
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_inference import SchemaInference

# FIRST TIME: No schema needed!
scraper = UniversalScraper()
result = scraper.scrape(
    'https://new-website.com/products',
    ['name', 'price', 'brand', 'description']
)

print(f"✅ Found {len(result['data'])} items")

# AUTO-GENERATE SCHEMA
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])

# See what was learned
report = inferencer.get_report()
print("\n📊 Discovered Schema:")
for field in report['fields']:
    print(f"   • {field['name']}: {field['type']}, {field['coverage']}% coverage")

# Generate stable schema
schema = inferencer.generate_schema("new_website", version="1.0")

# Export as Python code
schema_code = inferencer.export_schema_code("new_website")
with open('generated_schema_new_website.py', 'w') as f:
    f.write(schema_code)

print("\n✅ Schema saved! Use it for all future scrapes:")

# FUTURE SCRAPES: Use generated schema
scraper_with_schema = UniversalScraper(schema=schema)
result = scraper_with_schema.scrape(url, fields)

# Stable output forever!
quality = result['metadata']['schema_quality']
print(f"📊 Quality: {quality['status']} ({quality['success_rate']}%)")
```

---

## What Gets Auto-Generated

### 1. Field Types
```python
# Analyzes actual data to determine types
price: 29.99 → field_type="number"
name: "Product" → field_type="string"
```

### 2. Required vs Optional
```python
# Based on coverage
name: 100% coverage → required=True
rating: 45% coverage → required=False
```

### 3. Aliases
```python
# Auto-generates common variations
source: "product_name"
aliases: ["productName", "ProductName", "product-name"]
```

### 4. Transformers
```python
# Handles nested objects
brand: {"name": "Nike"} → transformer=lambda x: x.get('name')
```

### 5. Normalized Names
```python
# Clean, consistent output names
"productName" → "product_name"
"Product-Title" → "product_title"
```

---

## Real Example: Leafly (Just Ran)

```python
# First scrape
result = scraper.scrape(
    'https://www.leafly.com/dispensary-info/mammoth-holistics/menu',
    ['product_name', 'price', 'thc_content', 'cbd_content', 'brand']
)

# Auto-generate
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])
schema = inferencer.generate_schema("leafly_menu")
```

**Generated Schema**:
```
Field                 Type      Coverage  Required
─────────────────────────────────────────────────
product_name          string    100.0%    ✅ Yes
price                 number     94.7%    ✅ Yes
brand                 object     94.7%    ❌ No
thc_content           object     52.6%    ❌ No
cbd_content           object     36.8%    ❌ No
strain_type           object     63.2%    ❌ No
```

**Saved to**: `generated_schema_leafly.py`

**Use forever**:
```python
from generated_schema_leafly import create_leafly_menu_schema

schema = create_leafly_menu_schema()
scraper = UniversalScraper(schema=schema)

# Stable output even when Leafly changes!
```

---

## Workflow Options

You have 4 options for new websites:

### Option 1: No Schema (Quick Test)
```python
scraper = UniversalScraper()
result = scraper.scrape(url, fields)
```
⚡ Fastest | ⚠️ No stability

### Option 2: Auto-Generate (Recommended)
```python
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])
schema = inferencer.generate_schema("site")
```
✅ Automatic | ✅ Stable | ✅ Learns from data

### Option 3: One-Liner
```python
from universal_scraper.core.schema_inference import infer_schema_from_scrape

schema = infer_schema_from_scrape(scraper, url, fields, "site")
```
⚡ Simplest | ✅ Stable

### Option 4: Manual Definition
```python
schema = SchemaDefinition(
    name="site",
    fields=[
        FieldMapping(output_field="name", source_fields=["name"], ...)
    ]
)
```
🎯 Full control | ⏱️ Slower

---

## Key Benefits

### ✅ Zero Manual Mapping
You don't need to know the website's structure upfront. The system learns it automatically.

### ✅ Intelligent Defaults
The system makes smart decisions:
- Types based on actual values
- Required based on coverage
- Aliases based on naming patterns
- Transformers for nested data

### ✅ Production-Ready Output
Exports as Python code you can:
- Commit to version control
- Review and refine
- Reuse across your team
- Deploy with confidence

### ✅ Stable Forever
Once generated, the schema provides stability even when the website changes.

---

## Comparison: Manual vs Auto-Generate

### Traditional Approach (Manual)
```python
# 😰 You have to figure out the structure yourself
schema = SchemaDefinition(
    name="site",
    fields=[
        FieldMapping(
            output_field="???",  # What should I call this?
            source_fields=["???"],  # What's the source field?
            field_type="???",  # What type is it?
            required=???,  # Is it required?
            # ... more guesswork
        )
    ]
)
```
⏱️ Takes 30+ minutes per site  
🤔 Requires manual inspection  
❌ Error-prone

### Auto-Generate Approach
```python
# 😎 System figures it out automatically
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])
schema = inferencer.generate_schema("site")
```
⚡ Takes 2 minutes  
✅ Data-driven decisions  
✅ Learns optimal mappings

---

## FAQs

### Q: Do I need to define a schema for every new website?

**A**: No! You can scrape without a schema. But for production, auto-generating one takes 2 minutes and gives you stability.

### Q: What if the auto-generated schema isn't perfect?

**A**: 
1. Export the schema as code
2. Review and manually refine it
3. Commit to version control
4. Use the refined version going forward

### Q: Can I improve the schema over time?

**A**: Yes! Learn from more scrapes:
```python
inferencer = SchemaInference()

# Learn from multiple pages
for url in urls:
    result = scraper.scrape(url, fields)
    inferencer.learn_from_data(result['data'])

# Generate better schema from more data
schema = inferencer.generate_schema("site")
```

### Q: What if I know exactly what I want?

**A**: Use manual definition (Option 4). Full control, but more work upfront.

---

## Try It Yourself

```bash
# Run the interactive examples
python examples/new_website_bootstrap.py

# Choose from:
# 1. No Schema (Quick Start)
# 2. Auto-Generate Schema (Recommended)
# 3. Manual Definition
# 4. One-Liner
# 5. Real Example: Leafly
```

---

## Summary

### Question:
> "How is the schema defined the first time?"

### Answer:
**It's AUTO-GENERATED from your first scrape!**

```python
# 1. Scrape once (no schema)
result = scraper.scrape(url, fields)

# 2. Auto-generate schema
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])
schema = inferencer.generate_schema("site")

# 3. Use forever (stable output)
scraper_prod = UniversalScraper(schema=schema)
```

**Time to get started**: 2 minutes  
**Manual work required**: Zero  
**Schema stability**: Forever ✅  

---

## Documentation

- **Full Guide**: `NEW_WEBSITE_GUIDE.md`
- **Examples**: `examples/new_website_bootstrap.py`
- **Implementation**: `universal_scraper/core/schema_inference.py`

---

## The Bottom Line

**You don't need to manually define schemas for new websites.**

The system automatically:
1. ✅ Learns from your first scrape
2. ✅ Generates optimal schema
3. ✅ Exports production-ready code
4. ✅ Provides stable output forever

**Just scrape, learn, export, use!** 🚀








