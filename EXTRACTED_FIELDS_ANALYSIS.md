# Extracted Fields Analysis - ScrapeGraphAI Test Results

**Date:** November 19, 2025  
**Source:** scrapegraphai_test_results.log

## Amazon Laptop Search

### Requested Fields (via prompt)
```
"Extract all laptop product listings with product title, price, and rating"
```

### Actual Field Names Returned
```json
{
  "laptops": [
    {
      "title": "...",    // ← Returned as "title" (not "product_title")
      "price": "...",    // ← Returned as "price" ✓
      "rating": "..."    // ← Returned as "rating" ✓
    }
  ]
}
```

### Field Analysis

| Field | Requested Name | Returned Name | Data Type | Format | Sample Values |
|-------|---------------|---------------|-----------|--------|---------------|
| **Title** | "product title" | `title` | String | Full text | "HP 15.6 inch Laptop, HD Touchscreen Display, AMD Ryzen 3 7320U, 8 GB RAM..." |
| **Price** | "price" | `price` | String | $XXX.XX | "$356.28", "$749.00", "$169.99" |
| **Rating** | "rating" | `rating` | String (should be number) | X.X | "4.4", "4.8", "4.2" |

### Data Quality Metrics

**Total Items:** 13 laptops

**Field Completeness:**
- `title`: 13/13 (100%) ✅
- `price`: 13/13 (100%) ✅
- `rating`: 13/13 (100%) ✅

**Data Type Consistency:**
- Prices: All formatted as "$XXX.XX" ✅
- Ratings: All in "X.X" format (should be numeric) ⚠️

**Value Range:**
- Prices: $119.00 - $749.00
- Ratings: 3.9 - 5.0
- Title lengths: 60-200 characters

### Sample Items

**Item 1: Budget Laptop**
```json
{
  "title": "HP 14 Laptop, Intel Celeron N4020, 4 GB RAM, 64 GB Storage, 14-inch Micro-edge HD Display, Windows 11 Home, Thin & Portable, 4K Graphics, One Year of Microsoft 365 (14-dq0040nr, Snowflake White)",
  "price": "$170.23",
  "rating": "4.1"
}
```

**Item 2: Premium Laptop**
```json
{
  "title": "Apple 2025 MacBook Air 13-inch Laptop with M4 chip: Built for Apple Intelligence, 13.6-inch Liquid Retina Display, 16GB Unified Memory, 256GB SSD Storage, 12MP Center Stage Camera, Touch ID; Midnight",
  "price": "$749.00",
  "rating": "4.8"
}
```

**Item 3: Mid-range Laptop**
```json
{
  "title": "HP 15.6 inch Laptop, HD Touchscreen Display, AMD Ryzen 3 7320U, 8 GB RAM, 128 GB SSD, AMD Radeon Graphics, Windows 11 Home in S Mode, Natural Silver, 15-fc0099nr",
  "price": "$356.28",
  "rating": "4.4"
}
```

---

## Hacker News Front Page

### Requested Fields (via prompt)
```
"Extract all article listings with title, points, and comments count"
```

### Actual Field Names Returned
```json
{
  "articles": [
    {
      "title": "...",     // ← Returned as "title" ✓
      "points": 292,      // ← Returned as "points" ✓ (numeric!)
      "comments": 153     // ← Returned as "comments" (not "comments_count")
    }
  ]
}
```

### Field Analysis

| Field | Requested Name | Returned Name | Data Type | Format | Sample Values |
|-------|---------------|---------------|-----------|--------|---------------|
| **Title** | "title" | `title` | String | Full text | "The Death of Arduino?", "Building more with GPT-5.1-Codex-Max" |
| **Points** | "points" | `points` | Integer ✅ | Numeric | 292, 264, 380, 132 |
| **Comments** | "comments count" | `comments` | Integer ✅ | Numeric | 153, 156, 419, 29 |

### Data Quality Metrics

**Total Items:** 30 articles

**Field Completeness:**
- `title`: 30/30 (100%) ✅
- `points`: 30/30 (100%) ✅
- `comments`: 30/30 (100%) ✅

**Data Type Consistency:**
- All points are integers ✅
- All comments are integers ✅
- All titles are non-empty strings ✅

**Value Range:**
- Points: 3 - 380
- Comments: 0 - 419
- Title lengths: 20-90 characters

### Sample Items

**High-engagement Article**
```json
{
  "title": "Europe is scaling back GDPR and relaxing AI laws",
  "points": 380,
  "comments": 419
}
```

**Medium-engagement Article**
```json
{
  "title": "The Death of Arduino?",
  "points": 292,
  "comments": 153
}
```

**Low-engagement Article**
```json
{
  "title": "Racing karts on a Rust GPU kernel driver",
  "points": 9,
  "comments": 1
}
```

**Zero-comment Article**
```json
{
  "title": "Sam 3D: Powerful 3D Reconstruction for Physical World Images",
  "points": 19,
  "comments": 0
}
```

---

## Reddit /r/programming

### Requested Fields (via prompt)
```
"Extract all post listings with post title, author username, and upvotes"
```

### Actual Result
```json
{
  "post_listings": [
    {
      "post_title": "You've been blocked by network security.",
      "author_username": "NA",
      "upvotes": "NA"
    }
  ]
}
```

### Status: BLOCKED ❌

**Issue:** Reddit detected bot traffic and blocked the request

**Field Names Used:**
- `post_title` ✓ (matched request)
- `author_username` ✓ (matched request)
- `upvotes` ✓ (matched request)

**Note:** ScrapeGraphAI correctly understood the field names from the prompt, but was blocked before extraction.

---

## Key Insights from Field Extraction

### 1. Field Name Intelligence

**ScrapeGraphAI's LLM is smart about field names:**

| Prompt Says | LLM Returns | Analysis |
|-------------|-------------|----------|
| "product title" | `title` | ✅ Simplified to common name |
| "comments count" | `comments` | ✅ Simplified to common name |
| "author username" | `author_username` | ✅ Kept full name (more specific) |
| "post title" | `post_title` | ✅ Kept full name (disambiguation) |

**Insight:** The LLM understands semantic equivalence and chooses sensible field names.

### 2. Data Type Handling

**Mixed quality on data types:**

| Field | Expected Type | Actual Type | Status |
|-------|--------------|-------------|--------|
| Amazon `price` | String | String | ✅ Correct (includes $) |
| Amazon `rating` | Number | String | ⚠️ Should be number |
| HN `points` | Number | Integer | ✅ Perfect |
| HN `comments` | Number | Integer | ✅ Perfect |

**Insight:** Data types are inconsistent - sometimes strings, sometimes numbers.

### 3. Wrapper Key Names

**ScrapeGraphAI adds semantic wrapper keys:**

| Source | Wrapper Key | Items Inside |
|--------|-------------|--------------|
| Amazon | `laptops` | 13 items |
| Hacker News | `articles` | 30 items |
| Reddit | `post_listings` | 1 item |

**Insight:** LLM infers semantic container names from the prompt/content.

### 4. Field Value Quality

**All extracted values are semantically correct:**

✅ **Amazon:**
- Titles are actual product names (not "Featured" or "Best Seller")
- Prices are actual prices (not "Free shipping" or "Sale!")
- Ratings are actual ratings (not "4.5 out of 5 stars" or "View reviews")

✅ **Hacker News:**
- Titles are actual article titles (not "More", "Past", "Comments")
- Points are actual vote counts (not rank numbers like 1, 2, 3)
- Comments are actual comment counts (not "discuss" or "hide")

**Insight:** LLM has excellent semantic understanding of what each field means.

---

## Comparison: Prompt → Fields Mapping

### Amazon Test

**Prompt:**
> "Extract all laptop product listings with product title, price, and rating"

**Expected Fields:** `product_title`, `price`, `rating`

**Actual Fields:** `title`, `price`, `rating`

**Match Quality:** 95% (title simplified but correct)

### Hacker News Test

**Prompt:**
> "Extract all article listings with title, points, and comments count"

**Expected Fields:** `title`, `points`, `comments_count`

**Actual Fields:** `title`, `points`, `comments`

**Match Quality:** 95% (comments_count → comments)

### Reddit Test

**Prompt:**
> "Extract all post listings with post title, author username, and upvotes"

**Expected Fields:** `post_title`, `author_username`, `upvotes`

**Actual Fields:** `post_title`, `author_username`, `upvotes` (but blocked)

**Match Quality:** 100% (perfect match)

---

## Recommendations for Our DirectLLMExtractor

### 1. Improve Data Type Consistency

Currently our system returns strings for everything. We should:

```python
# Add type inference
def infer_and_convert_types(items: List[Dict]) -> List[Dict]:
    """Convert string values to appropriate types"""
    for item in items:
        for key, value in item.items():
            if value is None or value == "":
                continue
            
            # Try to convert to number
            if key in ['price', 'rating', 'points', 'comments', 'upvotes', 'score']:
                # Remove currency symbols
                cleaned = re.sub(r'[$€£,]', '', str(value))
                try:
                    # Try integer first
                    if '.' not in cleaned:
                        item[key] = int(cleaned)
                    else:
                        item[key] = float(cleaned)
                except ValueError:
                    pass  # Keep as string
    
    return items
```

### 2. Field Name Normalization

Our system should understand semantic equivalence:

```python
FIELD_SYNONYMS = {
    'product_title': ['title', 'name', 'product_name'],
    'price': ['cost', 'amount', 'product_price'],
    'rating': ['score', 'stars', 'review_score'],
    'comments_count': ['comments', 'comment_count', 'num_comments'],
    'author': ['author_username', 'username', 'user', 'by'],
}

def normalize_field_names(items: List[Dict], requested_fields: List[str]) -> List[Dict]:
    """Normalize field names to match user's request"""
    # Implementation details...
```

### 3. Response Structure

Our system should add semantic wrapper keys like ScrapeGraphAI:

```python
def wrap_response(items: List[Dict], url: str, context: str) -> Dict:
    """Wrap items in semantic container"""
    # Infer container name from context
    if 'product' in context.lower() or 'amazon' in url:
        wrapper_key = 'products'
    elif 'article' in context.lower() or 'news' in url:
        wrapper_key = 'articles'
    elif 'post' in context.lower() or 'reddit' in url:
        wrapper_key = 'posts'
    else:
        wrapper_key = 'items'
    
    return {
        wrapper_key: items,
        'metadata': {
            'count': len(items),
            'source': url,
            'extracted_at': datetime.now().isoformat()
        }
    }
```

---

## Summary Statistics

### Total Fields Extracted

| Source | Fields Requested | Fields Returned | Field Name Match | Data Completeness |
|--------|-----------------|-----------------|------------------|-------------------|
| **Amazon** | 3 | 3 | 95% | 100% |
| **Hacker News** | 3 | 3 | 95% | 100% |
| **Reddit** | 3 | 3 (blocked) | 100% | N/A |

### Data Type Distribution

**ScrapeGraphAI's data types:**
- Strings: 6 fields (60%)
- Integers: 2 fields (20%)
- N/A: 2 fields (20%, due to block)

**Ideal distribution:**
- Strings: 4 fields (40%) - titles, text
- Numbers: 4 fields (40%) - prices, ratings, counts
- N/A: 2 fields (20%) - blocked

**Gap:** Prices and ratings should be numbers, not strings

---

## Conclusion

### What ScrapeGraphAI Does Well

1. ✅ **Perfect field completeness** - 100% of items have all fields filled
2. ✅ **Semantic understanding** - Correct values for each field type
3. ✅ **Smart field naming** - Simplifies names appropriately
4. ✅ **Structured output** - Clean JSON with semantic wrappers

### What We Should Improve

1. 🔧 **Type consistency** - Convert numeric fields to actual numbers
2. 🔧 **Field name normalization** - Handle synonyms (title vs product_title)
3. 🔧 **Wrapper keys** - Add semantic containers like "products", "articles"
4. 🔧 **Quality filtering** - More aggressive filtering to match their 100% completeness

### Implementation Priority

1. **HIGH:** Add quality mode toggle (conservative/balanced/aggressive)
2. **HIGH:** Type inference for numeric fields
3. **MEDIUM:** Field name normalization
4. **LOW:** Semantic wrapper keys (nice to have)




