# Extraction Consistency: HTML vs JSON vs CSS

This document outlines the consistent approach used across all data acquisition types (HTML, JSON, CSS, etc.), following ScrapeGraphAI's guardrails and patterns.

## Architecture Overview

All extraction types follow the same **two-phase approach**:

1. **Phase 1: Fast Pattern Detection** (no LLM)
   - Uses heuristics/pattern matching
   - High confidence threshold (≥0.85) → skip LLM
   - Fast, cost-effective, cached

2. **Phase 2: LLM Analysis** (fallback)
   - Only called if Phase 1 confidence < 0.85
   - Uses LLM to understand structure
   - Cached for reuse

## HTML Extraction

### HTMLStructureAnalyzer

**Phase 1: DOM Pattern Detection**
- Uses `DOMPatternDetector` (fast, no LLM)
- Detects repeating elements, custom components, data attributes
- Confidence ≥ 0.85 → skip LLM, return immediately
- Source: `dom_pattern_detection`

**Phase 2: LLM Analysis**
- Only if DOM confidence < 0.85
- Uses `SmartHTMLSampler` for intelligent sampling
- Analyzes HTML structure to guide code generation
- Source: `llm_analysis`

**Usage:**
- Called before code generation (Step 5.5)
- Cached by domain + structure hash
- Used in all HTML extraction paths

## JSON Extraction

### JSONStructureAnalyzer

**Phase 1: Fast JSON Pattern Detection** (NEW)
- Uses heuristics to detect field mappings
- Exact matches, case-insensitive, synonyms
- Confidence ≥ 0.85 → skip LLM, return immediately
- Source: `pattern_detection`

**Phase 2: LLM Analysis**
- Only if pattern confidence < 0.85
- Analyzes JSON schema and field mappings
- Uses pattern detection hints in LLM prompt
- Source: `llm_analysis`

**Usage:**
- Context-driven JSON extraction path ✅
- Traditional JSON extraction path ✅
- Pagination extraction path ✅ (NEW)
- Cached by domain + structure hash + fields

## Consistency Guarantees

### 1. Guardrails
- ✅ Both use fast pattern detection first
- ✅ Both skip LLM if confidence ≥ 0.85
- ✅ Both cache results for reuse
- ✅ Both use LLM only when needed

### 2. Caching Strategy
- ✅ Both cache by domain + structure hash
- ✅ Both check cache before analysis
- ✅ Both mark source (pattern_detection vs llm_analysis)

### 3. Confidence Scoring
- ✅ Both return confidence 0.0-1.0
- ✅ Both use 0.85 threshold for LLM skip
- ✅ Both merge pattern + LLM results

### 4. Error Handling
- ✅ Both have fallback analysis
- ✅ Both handle LLM failures gracefully
- ✅ Both return structured analysis dict

## Usage Across Extraction Paths

### HTML Extraction Paths
- ✅ Main extraction (Step 5.5)
- ✅ Code generation (before generation)
- ✅ Pattern-based extraction

### JSON Extraction Paths
- ✅ Context-driven extraction (line ~750)
- ✅ Traditional extraction (line ~860)
- ✅ Pagination extraction (line ~1480) (NEW)

## Future: CSS/Other Extraction Types

When adding new extraction types (CSS, XML, etc.), follow the same pattern:

1. Create `{Type}StructureAnalyzer` class
2. Implement `_detect_{type}_patterns()` (Phase 1)
3. Implement LLM analysis (Phase 2)
4. Use same caching strategy
5. Use same confidence thresholds
6. Integrate into all extraction paths

## Benefits

1. **Cost Efficiency**: Skip LLM when pattern detection is confident
2. **Speed**: Fast pattern detection is instant vs seconds for LLM
3. **Consistency**: Same approach across all data types
4. **Reliability**: Pattern detection provides fallback if LLM fails
5. **Caching**: Results cached for repeated requests

## Example Flow

### HTML Extraction
```
1. HTMLStructureAnalyzer.analyze(html, url)
   → Phase 1: DOM pattern detection (confidence: 0.92)
   → ✅ Skip LLM (confidence ≥ 0.85)
   → Return cached analysis
```

### JSON Extraction
```
1. JSONStructureAnalyzer.analyze(json_data, url, fields)
   → Phase 1: JSON pattern detection (confidence: 0.78)
   → Phase 2: LLM analysis (confidence < 0.85)
   → Merge pattern + LLM results
   → Cache and return
```

## Guardrails Summary

| Aspect | HTML | JSON | Status |
|--------|------|------|--------|
| Fast pattern detection | ✅ DOMPatternDetector | ✅ JSON pattern detection | ✅ Consistent |
| LLM fallback | ✅ If confidence < 0.85 | ✅ If confidence < 0.85 | ✅ Consistent |
| Caching | ✅ Domain + structure | ✅ Domain + structure + fields | ✅ Consistent |
| Confidence threshold | ✅ 0.85 | ✅ 0.85 | ✅ Consistent |
| Source marking | ✅ pattern_detection/llm | ✅ pattern_detection/llm | ✅ Consistent |
| All paths covered | ✅ Yes | ✅ Yes (including pagination) | ✅ Consistent |







