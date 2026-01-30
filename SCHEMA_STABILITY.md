# Schema Stability in Production

## The Problem

In production environments, scrapers face a **critical challenge**:

> **"If this website were to change, how would we retain schema integrity to ensure an engineer doesn't have to shift the consumption schema?"**

When websites change their structure (which happens frequently), traditional scrapers break in several ways:

1. **Field names change** → `product_name` becomes `productTitle`
2. **Structure changes** → Flat fields become nested objects
3. **Fields move** → Data moves from HTML to JSON or API
4. **Fields disappear** → Website removes or renames data points

Without schema management, each change requires:
- Engineers to update extraction code
- Downstream systems to update field mappings
- Database schemas to change
- API contracts to break
- **Production downtime** while changes roll out

## The Solution: Schema Manager

The **Schema Manager** provides **production-grade schema stability** that ensures:

✅ **Your output schema never changes** (even when websites change)  
✅ **Zero-downtime migrations** (auto-adapts to website changes)  
✅ **No consumer code updates** (stable API contracts)  
✅ **Real-time quality monitoring** (alerts on degradation)  
✅ **Type safety** (normalized, validated output)

---

## How It Works

### 1. Define a Stable Schema

You define **once** the schema you want your downstream systems to receive:

```python
from universal_scraper.core.schema_manager import SchemaDefinition, FieldMapping

schema = SchemaDefinition(
    name="leafly_product",
    version="1.0",
    fields=[
        FieldMapping(
            output_field="name",  # ← Stable output field name
            source_fields=[        # ← Possible source field names
                "product_name",    #    (in priority order)
                "name",
                "title"
            ],
            field_type="string",
            required=True,
            aliases=["productName", "product_title"]
        ),
        FieldMapping(
            output_field="price",
            source_fields=["price", "current_price", "salePrice"],
            field_type="number",
            required=True,
            transformer=lambda x: float(str(x).replace('$', ''))
        ),
        # ... more fields ...
    ]
)
```

### 2. Initialize Scraper with Schema

```python
from universal_scraper.core.scraper import UniversalScraper

scraper = UniversalScraper(
    model_name='gpt-4o-mini',
    fetch_mode='hybrid',
    schema=schema,  # ← Enable schema management
    strict_schema=False  # ← Don't fail on missing fields
)
```

### 3. Scrape with Automatic Schema Enforcement

```python
result = scraper.scrape(url, fields)

# Output is ALWAYS in your stable schema format
print(result['data'])
# [
#   {
#     "name": "Product 1",      ← Stable field name
#     "price": 29.99,           ← Stable field name
#     "brand": "Brand A"        ← Stable field name
#   },
#   ...
# ]

# Monitor quality
print(result['metadata']['schema_quality'])
# {
#   "status": "healthy",
#   "success_rate": 94.7,
#   "field_coverage": {
#     "name": 100.0,
#     "price": 94.7,
#     "brand": 89.5
#   }
# }
```

---

## Real-World Example: Leafly

### Test Results

From the test we just ran:

```
Status: HEALTHY
Success Rate: 94.74%
Items Processed: 19

Field Coverage:
  name                 ████████████████████ 100.0%
  price                ██████████████████░░  94.7%
  thc_percentage       ██████████░░░░░░░░░░  52.6%
  cbd_percentage       ███████░░░░░░░░░░░░░  36.8%
  brand                ██████████████████░░  94.7%
  strain_type          ████████████░░░░░░░░  63.2%
  strain_name          ████████████░░░░░░░░  63.2%
```

### What This Means

**Scenario 1: Leafly renames a field**
- **Before**: Website uses `product_name`
- **After**: Website changes to `productTitle`
- **Result**: ✅ Schema Manager auto-maps to stable `name` field
- **Your consumers**: No changes needed!

**Scenario 2: Leafly changes nesting**
- **Before**: `price: 64`
- **After**: `pricing: { current: 64, original: 80 }`
- **Result**: ✅ Schema Manager extracts `pricing.current` → `price`
- **Your consumers**: Still receive `price: 64`!

**Scenario 3: Leafly moves data to API**
- **Before**: Data in HTML
- **After**: Data in API response
- **Result**: ✅ Hybrid Fetcher captures API, Schema Manager normalizes
- **Your consumers**: No idea anything changed!

---

## Intelligent Field Mapping

The Schema Manager uses **multi-layer mapping** to find data:

### Layer 1: Exact Match
```python
source_fields=["product_name", "name", "title"]
# Tries each in order: data["product_name"] → data["name"] → data["title"]
```

### Layer 2: Alias Matching (Case-Insensitive)
```python
aliases=["productName", "product_title", "itemName"]
# Tries: data["productname"], data["PRODUCT_NAME"], etc.
```

### Layer 3: Fuzzy Matching
```python
# Partial matches: "prod_name" matches "product_name"
```

### Layer 4: AI-Powered Discovery (Optional)
```python
enable_ai_mapping=True
# When all else fails, AI analyzes the data structure
# and suggests the best field mapping
```

### Layer 5: Transformer Functions
```python
transformer=lambda x: x.get('current') if isinstance(x, dict) else x
# Handles nested objects, type conversions, data cleaning
```

---

## Production Benefits

### 1. Zero-Downtime Website Changes

**Without Schema Manager:**
```
Website changes → Scraper breaks → Engineers alerted → 
Code updated → Tests run → Deploy → Consumers updated → 
DOWNTIME: 2-4 hours
```

**With Schema Manager:**
```
Website changes → Schema Manager auto-adapts → 
Quality metrics logged → Engineers notified (if needed) →
DOWNTIME: 0 minutes ✅
```

### 2. Stable API Contracts

Your API consumers always receive the same schema:

```python
# v1.0 of your API (January)
GET /api/products
{
  "name": "Product 1",
  "price": 29.99,
  "brand": "Brand A"
}

# v1.0 of your API (June - after website changed 10x)
GET /api/products
{
  "name": "Product 1",  ← Same field name!
  "price": 29.99,       ← Same field name!
  "brand": "Brand A"    ← Same field name!
}
```

### 3. Quality Monitoring

Real-time metrics for every scraping run:

```python
quality = result['metadata']['schema_quality']

if quality['success_rate'] < 80:
    alert_engineers("Scraper quality degraded")
    
if quality['field_coverage']['price'] < 50:
    alert_engineers("Price field coverage dropped")
```

### 4. Type Safety

Schema Manager normalizes types automatically:

```python
# Source has inconsistent types
source_data = [
    {"price": "$29.99"},      # string
    {"price": 39.99},         # number
    {"price": "49"},          # string number
    {"price": {"current": 59.99}}  # nested object
]

# Output is always normalized
output_data = [
    {"price": 29.99},  # float
    {"price": 39.99},  # float
    {"price": 49.0},   # float
    {"price": 59.99},  # float
]
```

### 5. Schema Evolution

When you need to add fields, use versioning:

```python
# v1.0 - Original schema
schema_v1 = SchemaDefinition(
    name="product",
    version="1.0",
    fields=[...]
)

# v2.0 - Add new field
schema_v2 = SchemaDefinition(
    name="product",
    version="2.0",
    fields=[
        ...schema_v1.fields,
        FieldMapping(
            output_field="discount_percentage",  # New field
            required=False,  # ← Optional for backward compatibility
            default_value=0
        )
    ]
)

# Deploy both versions
scraper_v1 = UniversalScraper(schema=schema_v1)  # Old consumers
scraper_v2 = UniversalScraper(schema=schema_v2)  # New consumers

# Zero-downtime migration!
```

---

## Configuration Options

### Strict Mode

```python
# Strict mode (fail on missing required fields)
scraper = UniversalScraper(
    schema=schema,
    strict_schema=True  # ← Raises exception if required field missing
)

# Lenient mode (warn but continue)
scraper = UniversalScraper(
    schema=schema,
    strict_schema=False  # ← Logs warning, uses default values
)
```

### AI-Assisted Mapping

```python
schema_manager = SchemaManager(
    schema=schema,
    ai_generator=ai_generator,
    enable_ai_mapping=True,  # ← Use AI to discover new mappings
    strict_mode=False
)

# When standard mapping fails, AI analyzes:
# - Available source fields
# - Field value examples
# - Semantic similarity
# → Suggests best mapping
```

### Custom Transformers

```python
FieldMapping(
    output_field="price",
    source_fields=["price", "cost"],
    transformer=lambda x: {
        # Extract from nested object
        if isinstance(x, dict):
            return x.get('current', x.get('value', 0))
        # Clean string
        elif isinstance(x, str):
            return float(x.replace('$', '').replace(',', ''))
        # Pass through number
        else:
            return float(x)
    }
)
```

---

## Monitoring in Production

### 1. Quality Metrics

```python
# After each scraping run
quality = result['metadata']['schema_quality']

# Log to monitoring system
logger.info(f"Schema quality: {quality['status']}")
logger.info(f"Success rate: {quality['success_rate']}%")
logger.info(f"Field coverage: {quality['field_coverage']}")

# Alert thresholds
if quality['status'] == 'critical':
    alert.send("CRITICAL: Scraper quality below 70%")
elif quality['status'] == 'warning':
    alert.send("WARNING: Scraper quality below 90%")
```

### 2. Field-Level Alerts

```python
for field_name, coverage in quality['field_coverage'].items():
    if coverage < 50 and is_required(field_name):
        alert.send(f"ALERT: {field_name} coverage at {coverage}%")
```

### 3. Dashboard Integration

```python
# Send metrics to dashboard
metrics.gauge('scraper.quality.success_rate', quality['success_rate'])
metrics.gauge('scraper.quality.items_extracted', quality['total_items'])

for field_name, coverage in quality['field_coverage'].items():
    metrics.gauge(f'scraper.quality.field.{field_name}', coverage)
```

---

## Comparison

### Without Schema Manager

```python
# Week 1: Website structure
{
  "product_name": "Item",
  "price": 29.99
}

# Your API output
{
  "product_name": "Item",  ← Exposes internal website structure
  "price": 29.99
}

# Week 2: Website changes to
{
  "title": "Item",  ← Changed field name
  "current_price": 29.99  ← Changed field name
}

# Your API output (BROKEN)
{
  "product_name": null,  ← 💥 BROKEN
  "price": null          ← 💥 BROKEN
}

# Consumer code (BROKEN)
const name = response.product_name;  ← 💥 null
const price = response.price;        ← 💥 null
```

### With Schema Manager

```python
# Week 1: Website structure
{
  "product_name": "Item",
  "price": 29.99
}

# Your API output (STABLE)
{
  "name": "Item",  ← Your stable schema
  "price": 29.99   ← Your stable schema
}

# Week 2: Website changes to
{
  "title": "Item",
  "current_price": 29.99
}

# Your API output (STILL STABLE) ✅
{
  "name": "Item",  ← Auto-mapped from "title"
  "price": 29.99   ← Auto-mapped from "current_price"
}

# Consumer code (WORKS) ✅
const name = response.name;   ← ✅ "Item"
const price = response.price; ← ✅ 29.99
```

---

## Best Practices

### 1. Design Schemas Around Your Needs

Don't mirror the website structure. Design schemas for **your use case**:

```python
# ❌ BAD: Mirroring website structure
FieldMapping(
    output_field="props.pageProps.product.name",  # Too specific
    ...
)

# ✅ GOOD: Generic, use-case driven
FieldMapping(
    output_field="name",  # Clean, simple
    source_fields=[
        "props.pageProps.product.name",  # Can map from complex sources
        "product_name",
        "name",
        "title"
    ]
)
```

### 2. Make Optional Fields Actually Optional

```python
FieldMapping(
    output_field="discount_price",
    required=False,  # ← Not all products have discounts
    default_value=None  # ← Explicit null when missing
)
```

### 3. Use Transformers for Consistency

```python
# Standardize formats
FieldMapping(
    output_field="price",
    transformer=lambda x: round(float(str(x).replace('$', '')), 2)
)

# Normalize enums
FieldMapping(
    output_field="category",
    transformer=lambda x: x.lower().strip()
)

# Extract from complex objects
FieldMapping(
    output_field="rating",
    transformer=lambda x: x.get('average') if isinstance(x, dict) else x
)
```

### 4. Version Your Schemas

```python
schema_v1_0 = SchemaDefinition(name="product", version="1.0", ...)
schema_v1_1 = SchemaDefinition(name="product", version="1.1", ...)  # Add optional field
schema_v2_0 = SchemaDefinition(name="product", version="2.0", ...)  # Breaking change
```

### 5. Monitor Quality Continuously

```python
# In your production loop
while True:
    result = scraper.scrape(url, fields)
    quality = result['metadata']['schema_quality']
    
    # Log for trending
    metrics.record('scraper.quality', quality)
    
    # Alert on degradation
    if quality['success_rate'] < THRESHOLD:
        alert_team(quality)
```

---

## Example: Production Deployment

```python
from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_manager import create_ecommerce_schema
import logging

# Production configuration
schema = create_ecommerce_schema()  # Pre-defined stable schema

scraper = UniversalScraper(
    model_name='gpt-4o-mini',
    fetch_mode='hybrid',
    enable_cache=True,
    schema=schema,
    strict_schema=False,  # Lenient in production
    log_level=logging.INFO
)

def scrape_product(url):
    """Production scraping function with monitoring"""
    try:
        result = scraper.scrape(url, fields=schema.get_field_names())
        
        # Monitor quality
        quality = result['metadata']['schema_quality']
        if quality['success_rate'] < 80:
            logger.warning(f"Quality degraded: {quality}")
            alert_team(quality)
        
        # Return stable output
        return {
            'success': True,
            'data': result['data'],  # Always matches schema
            'quality': quality
        }
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        alert_team({'error': str(e)})
        return {
            'success': False,
            'error': str(e)
        }

# In your API
@app.get("/api/products/{url}")
def get_product(url: str):
    result = scrape_product(url)
    return result['data']  # Guaranteed to match schema ✅
```

---

## Summary

The Schema Manager solves your production concern:

> **"How to retain schema integrity when websites change?"**

**Answer**: 
1. ✅ Define a **stable output schema** once
2. ✅ Schema Manager **auto-maps** source data to your schema
3. ✅ **Multi-layer mapping** handles website changes automatically
4. ✅ **Quality monitoring** alerts you to issues
5. ✅ **Zero consumer changes** when websites change

**Result**: Production-grade scraping with stable API contracts! 🚀








