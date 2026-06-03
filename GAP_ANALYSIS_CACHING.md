# Gap Analysis: Caching & Repurposing Architecture

## Current Flow

1. **Probabilistic (First Run)**
   - User navigates to URL → Field discovery → Extract
   - Uses Direct LLM extraction
   - Generates extraction code/pattern
   - Stores in cache (code cache or Direct LLM cache)

2. **Deterministic (Cached Pattern Found)**
   - Checks cache by URL + fields
   - If pattern exists, reuses it
   - No LLM call needed

3. **Repurposing (Same Domain, Different URL)**
   - **GAP**: Currently only checks exact URL match
   - **NEED**: Check domain-level patterns
   - **NEED**: Structural similarity matching

## Gaps Identified

### 1. **Domain-Level Pattern Matching**
- **Current**: Cache key is `{url}_{fields_hash}`
- **Needed**: Cache key should include `{domain}_{structure_hash}_{fields_hash}`
- **Benefit**: Same domain, similar structure = reuse pattern

### 2. **Structural Similarity Detection**
- **Current**: Exact match only
- **Needed**: HTML structure hash comparison
- **Implementation**: Use DOM structure hash (already exists in `hash_generator`)
- **Threshold**: 0.85 similarity = reuse pattern

### 3. **Pattern Storage Format**
- **Current**: Stores code/LLM results
- **Needed**: Store pattern metadata:
  - Domain
  - Structure hash
  - Field mappings
  - Extraction strategy (CSS selectors, JSON paths, etc.)
  - Confidence score

### 4. **Frontend Pattern Discovery**
- **Current**: Backend-only pattern detection
- **Needed**: Frontend can suggest patterns from:
  - Detected elements
  - Field selectors
  - User-defined mappings
- **Benefit**: User validates pattern before caching

### 5. **Pattern Reuse Logic**
- **Current**: Only exact URL match
- **Needed**: 
  ```
  IF domain matches AND structure_hash similar (>=0.85):
    Reuse pattern (deterministic)
  ELSE IF domain matches:
    Try pattern with adaptation (semi-deterministic)
  ELSE:
    Use Direct LLM (probabilistic) → Cache new pattern
  ```

## Proposed Architecture

### Pattern Cache Structure
```json
{
  "pattern_id": "uuid",
  "domain": "producthunt.com",
  "structure_hash": "abc123...",
  "fields": ["title", "description", "url"],
  "extraction_strategy": {
    "type": "css_selectors",
    "container": ".product-item",
    "fields": {
      "title": ".title",
      "description": ".description"
    }
  },
  "confidence": 0.95,
  "created_at": "2025-01-01T00:00:00Z",
  "usage_count": 42,
  "last_used": "2025-01-15T00:00:00Z"
}
```

### Flow Enhancement

1. **Pattern Discovery (Frontend)**
   - User navigates → Fields suggested
   - User validates/edits fields
   - Frontend generates pattern metadata
   - Sends to backend for caching

2. **Pattern Matching (Backend)**
   - Extract domain from URL
   - Generate structure hash from HTML
   - Search cache: `domain + structure_hash + fields`
   - If match found → Use cached pattern (deterministic)
   - If domain match but structure different → Try adaptation
   - If no match → Direct LLM → Cache new pattern

3. **Pattern Adaptation**
   - If structure similar but not exact:
     - Use cached selectors as hints
     - Run lightweight LLM to adapt selectors
     - Cache adapted pattern

4. **Pattern Validation**
   - After extraction, validate results
   - If quality low → Mark pattern as "needs_update"
   - User can update pattern from frontend

## Implementation Plan

### Phase 1: Fix Current Issues
1. ✅ Fix extracted fields display
2. ✅ Fix proxy/web unblocker inline settings
3. ✅ Fix agent log labels
4. ✅ Fix direct LLM extraction

### Phase 2: Domain-Level Caching
1. Update cache key structure
2. Add structure hash to cache
3. Implement domain-level pattern search

### Phase 3: Pattern Repurposing
1. Add structural similarity matching
2. Implement pattern adaptation
3. Add pattern validation

### Phase 4: Frontend Pattern Management
1. Pattern discovery UI
2. Pattern validation UI
3. Pattern sharing (public/private)




