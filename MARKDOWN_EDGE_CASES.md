# Markdown Conversion: Edge Cases & Limitations

**Analysis Date:** November 25, 2025  
**Purpose:** Identify potential issues before implementing HTML→Markdown conversion

---

## Executive Summary

While markdown conversion is used successfully by ScrapeGraphAI, there are **8 significant edge cases** where data can be lost or corrupted. Understanding these helps us build a hybrid approach.

---

## Critical Edge Cases

### 1. **Data Attributes (HIGH RISK)**

**Problem:** HTML data attributes contain structured data that markdown conversion strips entirely.

```html
<!-- Original HTML -->
<div class="product" 
     data-price="19.99" 
     data-stock="142" 
     data-sku="ABC123"
     data-rating="4.5">
  <span class="display-price">$19.99</span>
</div>
```

```markdown
<!-- After markdown conversion -->
$19.99
```

**Lost Data:**
- `data-price="19.99"` → Gone (but duplicated in display)
- `data-stock="142"` → **Completely lost** ❌
- `data-sku="ABC123"` → **Completely lost** ❌
- `data-rating="4.5"` → **Completely lost** ❌

**Impact:** E-commerce sites (Amazon, eBay, Product Hunt) often store accurate numeric values in data attributes while displaying formatted strings.

**Mitigation:**
```python
def extract_data_attributes(html: str) -> dict:
    """Extract data-* attributes before markdown conversion"""
    soup = BeautifulSoup(html, 'html.parser')
    data_elements = {}
    
    for elem in soup.find_all(attrs=lambda x: x and any(k.startswith('data-') for k in x.keys())):
        for attr, value in elem.attrs.items():
            if attr.startswith('data-'):
                key = attr.replace('data-', '')
                data_elements[key] = value
    
    return data_elements
```

---

### 2. **Complex Tables (MEDIUM RISK)**

**Problem:** html2text converts tables to plain text, but complex tables with colspan/rowspan lose structure.

```html
<table>
  <tr>
    <th colspan="2">Product Info</th>
    <th>Price</th>
  </tr>
  <tr>
    <td>Name</td>
    <td>Description</td>
    <td>$19.99</td>
  </tr>
</table>
```

```markdown
<!-- html2text output -->
Product Info | Price
---|---
Name | Description | $19.99
```

**Issues:**
- Column alignment can be wrong
- Nested tables become unreadable
- Headers may not align with data

**Impact:** Financial data, comparison tables, spec sheets.

**Mitigation:** Extract tables separately before markdown conversion:
```python
def extract_tables_as_json(html: str) -> List[dict]:
    """Convert tables to structured JSON before markdown"""
    import pandas as pd
    tables = pd.read_html(html)
    return [t.to_dict(orient='records') for t in tables]
```

---

### 3. **Form Data & Select Options (HIGH RISK)**

**Problem:** Form elements contain valuable data (dropdown options, input values) that markdown ignores.

```html
<select name="size" id="product-size">
  <option value="S">Small</option>
  <option value="M" selected>Medium</option>
  <option value="L">Large</option>
  <option value="XL">X-Large</option>
</select>

<input type="hidden" name="product_id" value="12345">
```

```markdown
<!-- After markdown conversion -->
(completely empty - forms are removed)
```

**Lost Data:**
- All dropdown options → **Gone** ❌
- Hidden input values → **Gone** ❌
- Selected default value → **Gone** ❌

**Impact:** E-commerce (size/color variants), filters, search parameters.

**Mitigation:**
```python
def extract_form_data(html: str) -> dict:
    """Extract form field data before markdown conversion"""
    soup = BeautifulSoup(html, 'html.parser')
    
    forms = {}
    for select in soup.find_all('select'):
        name = select.get('name', 'unnamed')
        options = [opt.text for opt in select.find_all('option')]
        forms[name] = options
    
    for input_elem in soup.find_all('input', type='hidden'):
        name = input_elem.get('name')
        value = input_elem.get('value')
        if name and value:
            forms[name] = value
    
    return forms
```

---

### 4. **JSON-LD and Script Data (CRITICAL)**

**Problem:** Markdown conversion strips all `<script>` tags, including JSON-LD structured data.

```html
<script type="application/ld+json">
{
  "@type": "Product",
  "name": "iPhone 15",
  "offers": {
    "price": "999.00",
    "availability": "InStock"
  }
}
</script>
```

```markdown
<!-- After markdown conversion -->
(completely empty - script tags removed)
```

**Impact:** This is **catastrophic** for our JSON-first strategy!

**Mitigation:** **Always extract JSON-LD BEFORE markdown conversion**
```python
def safe_markdown_conversion(html: str) -> tuple[str, dict]:
    """Extract JSON-LD first, then convert to markdown"""
    # Step 1: Extract all JSON-LD
    json_ld = extract_json_ld(html)
    
    # Step 2: Extract data attributes
    data_attrs = extract_data_attributes(html)
    
    # Step 3: Convert remaining HTML to markdown
    markdown = html2text.html2text(html)
    
    return markdown, {'json_ld': json_ld, 'data_attrs': data_attrs}
```

---

### 5. **CSS-Generated Content (LOW RISK)**

**Problem:** Content added via CSS `::before` / `::after` pseudo-elements is invisible to HTML parsing.

```css
.price::before { content: "$"; }
.stock-status.in-stock::after { content: " ✓ In Stock"; }
```

```html
<span class="price">19.99</span>
<span class="stock-status in-stock"></span>
```

**Visual display:** "$19.99 ✓ In Stock"  
**Markdown output:** "19.99" (missing $ and status)

**Impact:** Price formatting, status indicators, icons.

**Mitigation:** Usually not critical since the actual data is present. Can add post-processing rules:
```python
# Add common price formatting
if field == 'price' and not value.startswith('$'):
    value = f"${value}"
```

---

### 6. **Image Alt Text vs Image Content (MEDIUM RISK)**

**Problem:** Images with embedded text (infographics, charts, promotional banners) are lost.

```html
<img src="sale-banner.jpg" alt="Summer Sale - 50% Off All Items">
```

```markdown
![Summer Sale - 50% Off All Items](sale-banner.jpg)
```

**What's preserved:** Alt text ✓  
**What's lost:** Actual image content (if alt is inaccurate)

**Impact:** Product images, promotional content, charts.

**Mitigation:** For critical use cases, consider image-to-text OCR or vision models.

---

### 7. **Nested/Accordion Content (MEDIUM RISK)**

**Problem:** Collapsed or tabbed content appears but loses its hierarchical context.

```html
<div class="accordion">
  <button class="accordion-header">Specifications</button>
  <div class="accordion-content" style="display:none">
    <p>Weight: 150g</p>
    <p>Dimensions: 10x5x2cm</p>
  </div>
</div>
```

```markdown
<!-- Markdown output -->
Specifications

Weight: 150g

Dimensions: 10x5x2cm
```

**Issue:** The relationship between "Specifications" and the content below is lost. LLM might not understand this is a grouped section.

**Mitigation:** Use semantic chunking that preserves header-content relationships.

---

### 8. **Unicode/Encoding Edge Cases (LOW RISK)**

**Problem:** Special characters may be mangled during conversion.

```html
<span class="price">€19,99</span>  <!-- European format -->
<span class="rating">★★★★☆</span>  <!-- Unicode stars -->
<span class="brand">Pokémon™</span>
```

**Potential issues:**
- Currency symbols: €, £, ¥
- Special characters: ™, ®, ©
- Emojis: 🔥, ⭐, ✓
- Non-ASCII names: Café, naïve, Zürich

**Mitigation:** Ensure UTF-8 encoding throughout:
```python
h = HTML2Text()
h.unicode_snob = True  # Preserve unicode characters
```

---

## Risk Matrix

| Edge Case | Risk Level | Frequency | Data Loss | Mitigation Difficulty |
|-----------|------------|-----------|-----------|----------------------|
| Data Attributes | 🔴 HIGH | Very Common | Critical | Medium |
| JSON-LD Scripts | 🔴 CRITICAL | Common | Critical | Easy (extract first) |
| Form/Select Data | 🔴 HIGH | Common | Significant | Medium |
| Complex Tables | 🟡 MEDIUM | Occasional | Moderate | Hard |
| Nested Content | 🟡 MEDIUM | Common | Context loss | Medium |
| CSS Content | 🟢 LOW | Rare | Minor | Easy |
| Image Content | 🟡 MEDIUM | Occasional | Variable | Hard (OCR) |
| Unicode | 🟢 LOW | Rare | Minor | Easy |

---

## Recommended Hybrid Approach

Instead of pure markdown conversion, implement a **staged extraction pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                    HTML Input                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: JSON-First Extraction                             │
│  - Extract JSON-LD                                          │
│  - Extract JSON from script tags                            │
│  - Extract API response data                                │
│  IF complete data found → RETURN (skip markdown)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Structured Data Extraction                        │
│  - Extract data-* attributes                                │
│  - Extract form/select options                              │
│  - Extract tables as JSON                                   │
│  - Store in metadata dict                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Markdown Conversion                               │
│  - Convert cleaned HTML to markdown                         │
│  - Preserve unicode                                         │
│  - Maintain semantic structure                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: LLM Extraction                                    │
│  - Send markdown + metadata to LLM                          │
│  - Prompt includes: "Also check metadata for: {fields}"     │
│  - Merge LLM results with structured data                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Code

```python
class HybridMarkdownExtractor:
    """
    Hybrid extraction that captures structured data before markdown conversion
    """
    
    def __init__(self):
        self.h = HTML2Text()
        self.h.unicode_snob = True
        self.h.ignore_images = False
        self.h.ignore_links = False
    
    def extract(self, html: str, url: str) -> ExtractedContent:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Stage 1: JSON-LD (highest priority)
        json_ld = self._extract_json_ld(soup)
        
        # Stage 2: Structured data
        structured = {
            'data_attributes': self._extract_data_attrs(soup),
            'form_data': self._extract_forms(soup),
            'tables': self._extract_tables(soup),
            'meta_tags': self._extract_meta(soup),
        }
        
        # Stage 3: Markdown conversion
        markdown = self.h.handle(str(soup))
        
        return ExtractedContent(
            markdown=markdown,
            json_ld=json_ld,
            structured_data=structured,
            source_url=url
        )
    
    def _extract_json_ld(self, soup) -> List[dict]:
        scripts = soup.find_all('script', type='application/ld+json')
        return [json.loads(s.string) for s in scripts if s.string]
    
    def _extract_data_attrs(self, soup) -> dict:
        data = {}
        for elem in soup.find_all(attrs=True):
            for attr, value in elem.attrs.items():
                if attr.startswith('data-') and value:
                    key = attr.replace('data-', '')
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
        return data
    
    def _extract_forms(self, soup) -> dict:
        forms = {}
        for select in soup.find_all('select'):
            name = select.get('name', select.get('id', 'unnamed'))
            forms[name] = [opt.text.strip() for opt in select.find_all('option')]
        return forms
    
    def _extract_tables(self, soup) -> List[List[List[str]]]:
        tables = []
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables
    
    def _extract_meta(self, soup) -> dict:
        meta = {}
        for tag in soup.find_all('meta'):
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                meta[name] = content
        return meta
```

---

## Conclusion

**Don't use pure markdown conversion.** Instead:

1. ✅ **Extract JSON-LD first** (handles 50%+ of sites with structured data)
2. ✅ **Extract data attributes** (captures hidden values)
3. ✅ **Extract form/select data** (captures variants/options)
4. ✅ **Then convert to markdown** (for LLM processing)
5. ✅ **Merge all sources** (best of both worlds)

This hybrid approach gives us:
- **ScrapeGraphAI's advantages** (clean markdown for LLM)
- **Oxylabs' advantages** (JSON-first extraction)
- **Our unique advantage** (data attribute capture)

**Expected improvement over pure markdown:**
- Quality: 90% → 95%+ (fewer edge case failures)
- Completeness: Data attributes no longer lost
- Reliability: JSON-LD bypasses markdown entirely when available


