# Scraping a NEW Website: Bootstrap Guide

## The Question

> **"How is the schema defined the first time a user chooses a new website to scrape?"**

## Quick Answer

You have **4 options** when scraping a new website:

1. **No Schema** - Fast, but output may change ⚡
2. **Auto-Generate** - Recommended for new sites ✅  
3. **Manual Definition** - Full control 🎯
4. **One-Liner** - Simplest workflow 🚀

---

## Option 1: No Schema (Quick Start)

**When to use**: Testing, exploration, prototyping

```python
from universal_scraper.core.scraper import UniversalScraper

# Just scrape - no schema needed
scraper = UniversalScraper()

result = scraper.scrape(
    'https://new-website.com/products',
    fields=['name', 'price', 'description']
)

# Works immediately!
print(result['data'])
```

**Pros**: 
- ✅ Fastest to get started
- ✅ No setup required

**Cons**:
- ❌ Output may change when website structure changes
- ❌ No schema stability

---

## Option 2: Auto-Generate Schema (Recommended)

**When to use**: First time scraping a new production website

```python
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_inference import SchemaInference

# Step 1: Scrape once without schema
scraper = UniversalScraper()
result = scraper.scrape(
    'https://new-website.com/products',
    ['name', 'price', 'brand', 'description']
)

# Step 2: Auto-generate schema from the data
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])

# Step 3: Generate schema
schema = inferencer.generate_schema(
    name="new_website_products",
    version="1.0"
)

# Step 4: Use schema for all future scrapes
scraper_with_schema = UniversalScraper(schema=schema)

# Now you have stable output!
result = scraper_with_schema.scrape(url, fields)
```

**Pros**:
- ✅ Learns optimal field mappings from actual data
- ✅ Automatic alias generation
- ✅ Provides stable output going forward
- ✅ Can be exported as code

**Cons**:
- ❌ Requires initial scrape to generate

---

## Option 3: One-Liner (Simplest)

**When to use**: Rapid prototyping

```python
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_inference import infer_schema_from_scrape

scraper = UniversalScraper()

# Auto-generate schema from first scrape (one line!)
schema = infer_schema_from_scrape(
    scraper=scraper,
    url='https://new-website.com/products',
    fields=['name', 'price', 'description'],
    schema_name="new_site"
)

# Use it immediately
scraper_with_schema = UniversalScraper(schema=schema)
result = scraper_with_schema.scrape(url, fields)
```

**Pros**:
- ✅ Simplest workflow
- ✅ One function call

**Cons**:
- ❌ Less control over schema details

---

## Option 4: Manual Definition (Full Control)

**When to use**: You know exactly what you need

```python
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_manager import SchemaDefinition, FieldMapping

# Define schema manually
schema = SchemaDefinition(
    name="custom_products",
    version="1.0",
    fields=[
        FieldMapping(
            output_field="product_name",
            source_fields=["name", "title", "product_name"],
            field_type="string",
            required=True
        ),
        FieldMapping(
            output_field="price_usd",
            source_fields=["price", "cost"],
            field_type="number",
            required=True,
            transformer=lambda x: float(str(x).replace('$', ''))
        ),
        # ... more fields
    ]
)

# Use it
scraper = UniversalScraper(schema=schema)
result = scraper.scrape(url, fields)
```

**Pros**:
- ✅ Full control over every aspect
- ✅ Custom transformers
- ✅ Optimal for specific needs

**Cons**:
- ❌ Requires upfront knowledge of website structure

---

## Complete Workflow: New Website to Production

### Phase 1: Discovery (No Schema)

```python
# Quick test to see if it works
scraper = UniversalScraper()
result = scraper.scrape(url, ['name', 'price', 'brand'])

print(f"Found {len(result['data'])} items")
print(f"Sample: {result['data'][0]}")
```

### Phase 2: Schema Generation

```python
# Auto-generate schema
from universal_scraper.core.schema_inference import SchemaInference

inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])

# Review what was learned
report = inferencer.get_report()
print(f"Discovered fields: {report['fields_discovered']}")

for field in report['fields']:
    print(f"  • {field['name']}: {field['type']}, {field['coverage']}% coverage")
```

### Phase 3: Export Schema

```python
# Generate schema
schema = inferencer.generate_schema("my_website", version="1.0")

# Export as Python code
schema_code = inferencer.export_schema_code("my_website")

# Save to file
with open('schemas/my_website_schema.py', 'w') as f:
    f.write(schema_code)

# ✅ Now you can commit this to version control!
```

### Phase 4: Production Use

```python
# In production code
from schemas.my_website_schema import create_my_website_schema

schema = create_my_website_schema()

scraper = UniversalScraper(
    schema=schema,
    strict_schema=False
)

# Stable output forever!
result = scraper.scrape(url, fields)
```

---

## Real Example: Leafly

Here's how it works with a real website (Leafly):

```python
import os
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_inference import SchemaInference

# First scrape
scraper = UniversalScraper(model_name='gpt-4o-mini', fetch_mode='hybrid')
result = scraper.scrape(
    'https://www.leafly.com/dispensary-info/mammoth-holistics/menu',
    ['product_name', 'price', 'thc_content', 'cbd_content', 'brand']
)

print(f"✅ Scraped {len(result['data'])} items")

# Auto-generate schema
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])

# Review
report = inferencer.get_report()
print("\n📊 Discovered Schema:")
for field in report['fields'][:5]:
    print(f"   • {field['name']} ({field['type']}, {field['coverage']}% coverage)")

# Generate
schema = inferencer.generate_schema("leafly_menu", version="1.0")

# Export
schema_code = inferencer.export_schema_code("leafly_menu")
with open('generated_schema_leafly.py', 'w') as f:
    f.write(schema_code)

print("\n✅ Schema saved to: generated_schema_leafly.py")

# Use in production
scraper_prod = UniversalScraper(schema=schema, strict_schema=False)
result = scraper_prod.scrape(url, fields)

quality = result['metadata']['schema_quality']
print(f"\n📊 Quality: {quality['status']} ({quality['success_rate']}%)")
```

**Output**:
```
✅ Scraped 19 items

📊 Discovered Schema:
   • product_name (string, 100.0% coverage)
   • price (number, 94.7% coverage)
   • thc_content (object, 52.6% coverage)
   • cbd_content (object, 36.8% coverage)
   • brand (object, 94.7% coverage)

✅ Schema saved to: generated_schema_leafly.py

📊 Quality: healthy (94.74%)
```

---

## What Gets Auto-Generated

When you use schema inference, it automatically:

### 1. Analyzes Field Types

```python
# Detects that 'price' is a number
FieldMapping(
    output_field="price",
    field_type="number",  # ← Detected automatically
    ...
)
```

### 2. Generates Aliases

```python
# If source field is 'product_name', generates:
FieldMapping(
    output_field="product_name",
    source_fields=["product_name"],
    aliases=["productName", "ProductName", "product-name"]  # ← Auto-generated
)
```

### 3. Determines Required vs Optional

```python
# If field present in 90%+ of items = required
FieldMapping(
    output_field="name",
    required=True,  # ← Based on coverage
)

# If field present in <90% = optional
FieldMapping(
    output_field="rating",
    required=False,  # ← Based on coverage
)
```

### 4. Creates Transformers for Complex Fields

```python
# If field is sometimes an object:
FieldMapping(
    output_field="brand",
    transformer=lambda x: x.get('name') if isinstance(x, dict) else x  # ← Auto-generated
)
```

### 5. Normalizes Field Names

```python
# Source: "productName" or "product_name" or "Product-Name"
# Output: "product_name" (normalized to snake_case)
```

---

## Exporting Schemas

The `export_schema_code()` function generates production-ready Python code:

```python
inferencer = SchemaInference()
inferencer.learn_from_data(data)
schema_code = inferencer.export_schema_code("my_site")

# Generates:
"""
'''
Auto-generated schema for my_site
Generated from 100 observations
'''

from universal_scraper.core.schema_manager import SchemaDefinition, FieldMapping


def create_my_site_schema() -> SchemaDefinition:
    '''Create schema for my_site'''
    return SchemaDefinition(
        name="my_site",
        version="1.0",
        fields=[
            FieldMapping(
                output_field="name",
                source_fields=['name', 'title'],
                field_type="string",
                required=True,
                aliases=['productName', 'product_name']
            ),
            # ... more fields
        ]
    )
"""

# Save to file and commit to version control!
```

---

## Best Practices

### 1. Start with Auto-Generation

```python
# Don't manually define schemas for new sites
# Let the system learn from actual data

inferencer = SchemaInference()
inferencer.learn_from_data(initial_scrape['data'])
schema = inferencer.generate_schema("site_name")
```

### 2. Learn from Multiple Scrapes

```python
# Better: Learn from multiple pages
inferencer = SchemaInference()

for url in sample_urls:
    result = scraper.scrape(url, fields)
    inferencer.learn_from_data(result['data'])

# More robust schema based on more data
schema = inferencer.generate_schema("site_name")
```

### 3. Review Before Production

```python
# Always review the generated schema
report = inferencer.get_report()

for field in report['fields']:
    print(f"{field['name']}: {field['coverage']}% coverage")
    if field['coverage'] < 50:
        print(f"  ⚠️  Warning: Low coverage!")
```

### 4. Refine as Needed

```python
# Generate initial schema
schema = inferencer.generate_schema("site")

# Review and manually adjust if needed
# Then export for production use
schema_code = inferencer.export_schema_code("site")
```

### 5. Version Control Your Schemas

```
project/
  schemas/
    amazon_products.py      ← Schema v1.0
    ebay_listings.py        ← Schema v1.0
    leafly_menu.py          ← Schema v1.0
```

---

## Comparison Table

| Approach | Setup Time | Stability | Control | Best For |
|----------|-----------|-----------|---------|----------|
| **No Schema** | 1 min | ⚠️ Low | High | Quick tests |
| **Auto-Generate** | 5 min | ✅ High | Medium | New production sites |
| **One-Liner** | 2 min | ✅ Medium | Low | Prototypes |
| **Manual** | 30 min | ✅ Highest | Full | Specific requirements |

---

## Summary

### The Answer to Your Question:

> **"How is the schema defined the first time?"**

**Recommended Workflow**:

1. **First scrape**: Run without schema to see what data is available
2. **Auto-generate**: Use `SchemaInference` to learn optimal schema
3. **Review**: Check the generated schema and field coverage
4. **Export**: Save as Python code for version control
5. **Use**: Deploy with stable schema for all future scrapes

**Code**:
```python
# 1. Scrape
scraper = UniversalScraper()
result = scraper.scrape(url, fields)

# 2. Auto-generate
inferencer = SchemaInference()
inferencer.learn_from_data(result['data'])
schema = inferencer.generate_schema("site_name")

# 3. Export
with open('schema_site_name.py', 'w') as f:
    f.write(inferencer.export_schema_code("site_name"))

# 4. Use forever
scraper_prod = UniversalScraper(schema=schema)
```

**Result**: 
- ✅ Stable output schema
- ✅ Auto-adapts to website changes
- ✅ Zero manual field mapping
- ✅ Production-ready

**Try it**: `python examples/new_website_bootstrap.py`








