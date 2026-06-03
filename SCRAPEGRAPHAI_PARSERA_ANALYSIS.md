# ScrapeGraphAI & Parsera Implementation Analysis

## Source Code Review

After analyzing the actual GitHub repositories for both libraries, here are the key techniques implemented:

### From ScrapeGraphAI (`github.com/ScrapeGraphAI/Scrapegraph-ai`)

**1. Script Tag JSON Extraction** (`utils/cleanup_html.py`)
```python
# Extract JSON from window/document variables
json_pattern = r"(?:const|let|var)?\s*\w+\s*=\s*({[\s\S]*?});?$"
data_pattern = r"(?:window|document)\.(\w+)\s*=\s*([^;]+);"
```

**Key Features:**
- Extracts JSON from `const/let/var` assignments
- Captures `window.*` and `document.*` variable data
- Preserves short script content for context

**2. HTML Cleanup Strategy** (`utils/cleanup_html.py`)
```python
# Preserve key attributes during cleanup
attrs_to_keep = ["class", "id", "href", "src", "type"]
```

**3. Markdown Conversion** (`utils/convert_to_md.py`)
- Uses `html2text` library
- Sets `ignore_links = False` 
- Sets `body_width = 0` (no wrapping)

### From Parsera (`github.com/raznem/parsera`)

**1. Overlapping Chunks with Context Passing** (`engine/chunks_extractor.py`)
```python
overlap_factor = 3  # 33% overlap
chunk_overlap = chunk_size // overlap_factor

# Pass previous data to each chunk for continuity
previous_tail = json.dumps(previous_data[cutoff:])
```

**Key Features:**
- 33% chunk overlap to catch truncated items
- Sequential context passing: each chunk receives previous chunk's items
- LLM-based intelligent merging of overlapping data

**2. Chunk Merging Strategy**
```python
# Prefer data "further from the border between files"
# When merging conflicts, take values from the middle of chunks
```

**3. Uses `markdownify` Library** (not html2text)
- Different markdown conversion approach
- Better table handling in some cases

---

## Implemented Improvements

### 1. Parsera-Style Context Passing
**File:** `universal_scraper/core/direct_llm_extractor.py`

- **Overlapping chunks (33%)** - Uses `_chunk_html_with_overlap()` 
- **Previous items context** - Each chunk receives last items from previous chunks
- **Sequential start + parallel finish** - First 2 chunks sequential to establish patterns, rest in parallel

### 2. ScrapeGraphAI Script Extraction
**File:** `universal_scraper/core/hybrid_extractor.py`

- **`_extract_script_data()` method** - Extracts:
  - JSON from variable assignments (`const data = {...}`)
  - `window.*` and `document.*` variables
  - `__INITIAL_STATE__`, `__NEXT_DATA__`, `__PRELOADED_STATE__`
  - Short inline scripts (<500 chars)

### 3. Combined Approach

Our hybrid extractor now captures:
1. JSON-LD (structured data)
2. Script data (ScrapeGraphAI style)
3. Data attributes
4. Form data
5. Hidden inputs
6. Tables
7. Meta tags
8. CSS class data (ratings, votes)
9. Labeled numbers (universal numeric extraction)

---

## Key Differences: Our Approach vs Competitors

| Feature | ScrapeGraphAI | Parsera | Universal Scraper |
|---------|--------------|---------|-------------------|
| Markdown Library | html2text | markdownify | html2text |
| Chunk Overlap | None | 33% | 33% |
| Context Passing | None | Full | Last 3-5 items |
| Script Extraction | Yes | No | Yes |
| Data Attributes | No | No | Yes |
| CSS Class Data | No | No | Yes |
| Labeled Numbers | No | No | Yes |
| Parallel Processing | No | No | Yes |

---

## Expected Quality Improvements

After implementing these techniques:

1. **Better continuity** - Overlapping chunks catch truncated items
2. **More data sources** - Script extraction finds embedded JSON
3. **Faster processing** - Parallel chunks after pattern establishment
4. **Better numeric handling** - CSS class data + labeled numbers

---

## Test Results

Run `python3 test_simple_competitive.py` to verify improvements.

Target metrics:
- Stack Overflow: 65% → 85%+ (improved answers extraction)
- Product Hunt: 48% → 70%+ (improved votes extraction)
- Overall Average: 79% → 85%+


