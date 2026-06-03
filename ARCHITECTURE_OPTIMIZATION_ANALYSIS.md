# Architecture Optimization Analysis
## Multi-Tenant, Deterministic, Universal Scraper at Subsecond Scale

**Date**: December 26, 2025  
**Goal**: Evaluate and implement optimizations for subsecond product-scale scraping with deterministic caching

---

## Current Architecture Assessment

### ✅ What We Already Have

1. **Multi-layer Caching** (Partial)
   - ✅ Direct LLM result cache (domain + fields)
   - ✅ Code cache (extraction code)
   - ✅ Pattern cache (Smart Pattern Cache)
   - ✅ JSON structure analysis cache
   - ⚠️ Missing: Raw fetch cache, DOM digest cache

2. **Template System** (Partial)
   - ✅ Pattern learning from successful extractions
   - ✅ Deterministic pattern execution
   - ⚠️ Missing: Strict template spec JSON format
   - ⚠️ Missing: Deterministic runtime extractor

3. **Model Selection** (Single Model)
   - ⚠️ Currently: Single model (`gpt-4o-mini`) for all tasks
   - ❌ Missing: 3-tier model selection

4. **Speed Optimizations** (Partial)
   - ✅ HTML cleaning (82-87% reduction)
   - ✅ Chunking optimization
   - ⚠️ Missing: Compressed HTML representation
   - ⚠️ Missing: Heuristic prefiltering

5. **Bootstrapping** (Partial)
   - ✅ Pattern learning from extractions
   - ⚠️ Missing: Site-specific selector library
   - ⚠️ Missing: Training examples for template generation

---

## Optimization Roadmap

### 1. 3-Tier Model Selection

**Current State**: Single model (`gpt-4o-mini`) for all tasks

**Target Architecture**:
```
A. Router/Classification (fastest, most frequent)
   - Model: gpt-4o-mini or claude-haiku
   - Tasks:
     * "Is this page type X?"
     * "Do we already have a working template?"
     * "Has layout changed enough to re-learn selectors?"
   - Frequency: Every request
   - Latency Target: <100ms

B. Template Generation (occasional, higher value)
   - Model: gpt-4o-mini or claude-sonnet-3.5
   - Tasks:
     * Generate extraction plan (selectors + field map)
     * Propose fallback rules
     * Create template JSON spec
   - Frequency: ~5-10% of requests (cache miss)
   - Latency Target: <2s

C. Recovery Mode (rare, hardest)
   - Model: gpt-4o or claude-opus
   - Tasks:
     * Re-derive template when validation fails
     * Handle odd layouts
     * Multi-step flows
   - Frequency: <1% of requests
   - Latency Target: <10s (acceptable for rare cases)
```

**Implementation Plan**:
1. Create `ModelRouter` class
2. Add model selection logic to:
   - `SmartPatternCache` (router/classification)
   - `AICodeGenerator` (template generation)
   - `DirectLLMExtractor` (recovery mode)
3. Add configuration for model tiers
4. Track model usage and costs

**Files to Modify**:
- `universal_scraper/core/model_router.py` (NEW)
- `universal_scraper/core/smart_pattern_cache.py`
- `universal_scraper/core/ai_generator.py`
- `universal_scraper/core/direct_llm_extractor.py`
- `universal_scraper/core/scraper.py`

---

### 2. Deterministic Template Spec System

**Current State**: LLM generates code directly, some non-determinism

**Target Architecture**:
```
LLM Output: Template Spec JSON (strict schema)
{
  "page_fingerprint_features": {...},
  "selectors": {
    "field_name": {
      "primary": "css_selector",
      "fallbacks": ["alt_selector1", "alt_selector2"],
      "priority": 1
    }
  },
  "normalizers": {
    "price": "parse_currency",
    "date": "parse_date"
  },
  "validators": {
    "title": {
      "required": false,
      "min_length": 3,
      "regex": null
    }
  },
  "fallbacks": {
    "pagination": {
      "type": "url_param",
      "param": "page",
      "next_selector": ".next-button"
    }
  },
  "confidence": 0.95,
  "why_these_selectors": "Found repeating article elements with consistent structure"
}

Runtime Extractor:
- Runs selectors deterministically (temperature=0, strict JSON)
- Normalizes deterministically
- Validates deterministically
- If fails → re-call LLM (recovery mode)
```

**Implementation Plan**:
1. Create `TemplateSpec` dataclass
2. Modify `AICodeGenerator` to output template spec JSON
3. Create `DeterministicExtractor` class
4. Add validation layer
5. Set temperature=0 for template generation

**Files to Create/Modify**:
- `universal_scraper/core/template_spec.py` (NEW)
- `universal_scraper/core/deterministic_extractor.py` (NEW)
- `universal_scraper/core/ai_generator.py`
- `universal_scraper/core/scraper.py`

---

### 3. 3-Layer Cache Strategy

**Current State**: Partial caching (Direct LLM, code, patterns)

**Target Architecture**:
```
Layer 1: Raw Fetch Cache
- Key: URL + headers + geo + device + cookie context + timestamp bucket
- Store: HTML, final URL, response headers, screenshots (optional)
- Purpose: Repeatable debugging + avoid re-fetching
- TTL: 1 hour (configurable)

Layer 2: DOM Digest Cache (Fingerprinting)
- Key: domain + path_pattern + dom_fingerprint
- Fingerprint Generation:
  * Strip scripts/styles
  * Normalize whitespace
  * Drop dynamic IDs/classes (or hash class tokens)
  * Hash "tag path histogram" or "DOM shape signature"
- Store: "page type" + version
- Purpose: Detect "same layout as before" quickly (no LLM)
- TTL: 24 hours

Layer 3: Template Cache (The Gold)
- Key: site + page_type + dom_fingerprint (or cluster_id)
- Store:
  * Template JSON spec
  * Last-success timestamp
  * Success rate counters
  * Sample extracted outputs
- Purpose: Most pages run with zero LLM involvement
- TTL: 7 days (or until validation fails)
- Clustering: LSH/simhash for minor layout variations
```

**Implementation Plan**:
1. Create `RawFetchCache` class
2. Create `DOMDigestGenerator` class
3. Enhance `SmartPatternCache` with DOM fingerprinting
4. Add clustering (LSH/simhash) for layout variations
5. Integrate all layers into `HybridFetcher`

**Files to Create/Modify**:
- `universal_scraper/core/raw_fetch_cache.py` (NEW)
- `universal_scraper/core/dom_digest.py` (NEW)
- `universal_scraper/core/smart_pattern_cache.py`
- `universal_scraper/core/hybrid_fetcher.py`

---

### 4. Bootstrapping System

**Current State**: Pattern learning exists but not systematic

**Target Architecture**:
```
When user scrapes first few pages:
1. Save successful field-to-selector mappings as training examples
2. Build site-specific selector library:
   - Canonical selectors + alternates
   - Field patterns (e.g., "product title tends to be h1 inside main")
3. Feed next template-generation with:
   - "Here are selectors that worked on 12 similar pages"
   - "Here's the validation profile of this site"
4. Makes mid-tier model more consistent and faster to converge
```

**Implementation Plan**:
1. Create `SelectorLibrary` class
2. Enhance pattern learning to save training examples
3. Add site-specific selector patterns
4. Feed examples to template generation
5. Track success rates per selector pattern

**Files to Create/Modify**:
- `universal_scraper/core/selector_library.py` (NEW)
- `universal_scraper/core/extraction_pattern.py`
- `universal_scraper/core/ai_generator.py`

---

### 5. Speed Optimizations

**Current State**: HTML cleaning, chunking optimization

**Target Architecture**:
```
1. Compressed HTML Representation
   - Don't send full HTML to LLM
   - Send:
     * Top-N candidate nodes per field (from heuristics)
     * Simplified DOM (only tags + attributes you care about)
     * Rendered text blocks with their DOM paths

2. Heuristic Prefiltering Before LLM
   - Probable price nodes contain currency
   - Titles are largest headings near top
   - Availability has known phrases
   - Run before LLM call (reduces context size)

3. Parallelization
   - Parallelize fetch + parse
   - Keep LLM calls serial and rare
   - Use streaming only if interactive
```

**Implementation Plan**:
1. Create `HTMLCompressor` class
2. Create `HeuristicPrefilter` class
3. Enhance `HTMLStructureAnalyzer` with prefiltering
4. Add parallel fetch/parse where possible
5. Optimize chunking strategy

**Files to Create/Modify**:
- `universal_scraper/core/html_compressor.py` (NEW)
- `universal_scraper/core/heuristic_prefilter.py` (NEW)
- `universal_scraper/core/html_structure_analyzer.py`
- `universal_scraper/core/hybrid_fetcher.py`

---

### 6. Recommended "Speed-First" Model Policy

**Implementation Flow**:
```
1. Try existing template by fingerprint match
   → If match: Use template (0 LLM calls, <50ms)

2. If no match: Run tiny model (or pure heuristic) to classify page type
   → Confirm novelty
   → Latency: <100ms

3. Generate template with mid model
   → Create template spec JSON
   → Latency: <2s

4. Validate on 3-5 pages
   → If pass: Store template
   → If fail: Escalate to large model recovery

5. If extraction fails in production:
   → Escalate to large model recovery
   → Store new template version if it improves success rate
```

**Implementation Plan**:
1. Integrate all optimizations above
2. Add validation pipeline
3. Add success rate tracking
4. Add automatic template versioning

---

## Priority Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. ✅ Make all fields optional by default (DONE)
2. Implement DOM digest cache (Layer 2)
3. Add heuristic prefiltering
4. Compress HTML before LLM

### Phase 2: Core Architecture (3-5 days)
1. Implement 3-tier model selection
2. Create deterministic template spec system
3. Implement raw fetch cache (Layer 1)
4. Enhance template cache with clustering

### Phase 3: Advanced Features (5-7 days)
1. Implement bootstrapping system
2. Add selector library
3. Add validation pipeline
4. Add automatic template versioning

---

## Expected Performance Improvements

### Current Performance (Baseline)
- Cache hit: ~500ms (Direct LLM cache)
- Cache miss: ~5-10s (LLM extraction)
- Quality: 81-99% (varies by site)

### Target Performance (After Optimizations)
- Cache hit (template): <50ms (deterministic extraction)
- Cache miss (template generation): <2s (mid model)
- Recovery mode: <10s (large model, rare)
- Quality: >90% consistently
- LLM calls: <5% of requests (vs ~50% currently)

### Cost Reduction
- Current: ~$0.001-0.01 per request
- Target: ~$0.0001-0.001 per request (90% reduction)

---

## Next Steps

1. ✅ Make fields optional (DONE)
2. Implement Phase 1 quick wins
3. Test with 3 URLs
4. Measure performance improvements
5. Iterate on Phase 2 and 3

---

## Notes

- All optimizations maintain backward compatibility
- Deterministic extraction ensures reproducibility
- Multi-tenant support via Redis cache backend
- Subsecond scale requires aggressive caching (target: >95% cache hit rate)



