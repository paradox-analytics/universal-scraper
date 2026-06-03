# Cache System Explanation

## What We're Caching

The universal scraper uses a **multi-layer cache system** to avoid LLM calls and speed up extraction:

### 1. **DOM Digest Cache** (Layer 1)
- **What**: Page structure fingerprints (hash of cleaned HTML structure)
- **Key**: `domain + path_pattern + dom_fingerprint`
- **Purpose**: Fast detection of layout changes
- **When Used**: Before checking template cache - if structure matches, we know the page type
- **Benefit**: <10ms check to see if we've seen this page structure before

### 2. **Template Spec Cache** (Layer 2) 
- **What**: Deterministic extraction templates (JSON spec with CSS selectors, field mappings, normalizers)
- **Key**: `site + page_type + dom_fingerprint`
- **Purpose**: Execute extraction without LLM calls
- **When Used**: After DOM digest match - if template exists, execute deterministically
- **Benefit**: <50ms extraction (vs 5-15s with LLM)

### 3. **Direct LLM Cache** (Layer 3)
- **What**: LLM extraction results (structured data)
- **Key**: `domain + fields + structure_hash`
- **Purpose**: Return cached results if same page + fields requested
- **When Used**: If no template spec exists, check if we've extracted this before
- **Benefit**: Instant return (0ms) if exact match

### 4. **Pattern Cache** (Layer 4)
- **What**: Learned selector patterns from successful extractions
- **Key**: `domain + field_name`
- **Purpose**: Bootstrap template generation with known-good selectors
- **When Used**: When generating new template specs, use learned patterns as examples
- **Benefit**: Better template quality, faster generation

## Cache Flow

```
Request (URL + Fields)
    ↓
1. Generate DOM Digest (structure fingerprint)
    ↓
2. Check DOM Digest Cache
    ├─ HIT → Get page_type, check Template Spec Cache
    │          ├─ HIT → Execute template (<50ms, no LLM)
    │          └─ MISS → Check Direct LLM Cache
    │                     ├─ HIT → Return cached results (0ms)
    │                     └─ MISS → Generate template with LLM
    │                                ├─ Use Pattern Cache for examples
    │                                ├─ Generate Template Spec
    │                                ├─ Cache Template Spec
    │                                └─ Execute & return results
    └─ MISS → Check Direct LLM Cache
               ├─ HIT → Return cached results
               └─ MISS → Direct LLM extraction
                          ├─ Learn patterns → Pattern Cache
                          ├─ Cache results → Direct LLM Cache
                          └─ Return results
```

## What Gets Repurposed

When you hit the **same page + fields** again:

1. **First Request**: 
   - DOM Digest generated → cached
   - Template Spec generated → cached
   - Patterns learned → cached
   - Results cached

2. **Second Request** (same URL + fields):
   - DOM Digest Cache HIT → instant page type detection
   - Template Spec Cache HIT → deterministic extraction (<50ms)
   - **No LLM calls** → $0.00 cost, instant results

3. **Similar Page** (same domain, different path, same fields):
   - DOM Digest might match → Template Spec reused
   - Or new Template Spec generated → cached for this page type
   - Patterns from Pattern Cache help bootstrap new template

## Cache Storage

- **Local Development**: File-based cache (`./cache/`)
- **Production (Cloud Run)**: Redis cache (if configured)
- **Apify**: KV Store (persists across runs)

## Cache Invalidation

Caches are invalidated when:
- Page structure changes significantly (new DOM digest)
- Fields change (new cache key)
- Manual cache clear

## Performance Impact

- **First extraction**: 5-15s, $0.02-0.05 (LLM cost)
- **Cached extraction**: <50ms, $0.00 (deterministic)
- **Speed improvement**: 100-300x faster
- **Cost savings**: 100% (no LLM calls)



