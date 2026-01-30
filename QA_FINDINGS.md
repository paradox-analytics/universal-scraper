# Comprehensive QA Findings - Universal Hybrid Scraper

## Test Date
November 18, 2025

## Executive Summary

Tested 6 diverse data sources. **Only 1/6 passed completely**.

### ✅ What's Working Perfectly:
1. **JSON Quality Validation** - Content quality validator correctly rejects analytics/tracking data
2. **JSON Extraction (when quality is high)** - Leafly extracted 18 clean products
3. **Universal Fetching** - Camoufox/Playwright/static selection working

### ❌ Critical Issues Blocking Universal Capability:

## Issue #1: HTML Extraction API Inconsistency (CRITICAL)

**Problem:** Multiple components have mismatched APIs, breaking the HTML fallback workflow.

### Symptoms:
- Amazon: 2,178,914 → 4 bytes (HTML destroyed)
- eBay: 3,128,229 → 4 bytes (HTML destroyed)
- Reddit: 312,154 → 4 bytes (HTML destroyed)

### Root Causes:

1. **HTML Cleaner Returns Dict, Not String**
   ```python
   # WRONG (current test code):
   cleaned = html_cleaner.clean(html)  # Returns Dict!
   dom_detector.detect_patterns(cleaned)  # Expects String!
   
   # CORRECT:
   cleaned_result = html_cleaner.clean(html)
   cleaned_html = cleaned_result['html']  # Extract string
   dom_detector.detect_patterns(cleaned_html)
   ```

2. **DOM Detector Output Structure Mismatch**
   ```python
   # WRONG (actor.py line 248):
   repeating_containers = dom_patterns.get('repeating_containers', [])
   # ↑ This key doesn't exist!
   
   # CORRECT:
   best_pattern = dom_patterns.get('best_pattern')
   selector = best_pattern.get('selector')
   containers = soup.select(selector)
   ```

3. **Method Name Error**
   ```python
   # WRONG (test code):
   dom_detector.detect_repeating_patterns(html)  # Method doesn't exist!
   
   # CORRECT:
   dom_detector.detect_patterns(html)
   ```

### Impact:
- **100% failure rate** on HTML extraction
- Amazon, eBay, Reddit, Hacker News, Product Hunt all fail
- Only Leafly succeeds (JSON-only)

### Required Fixes:

1. **Fix actor.py (lines 245-250)**:
   ```python
   # Current (BROKEN):
   cleaned_html = self.html_cleaner.clean(html)
   dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
   repeating_containers = dom_patterns.get('repeating_containers', [])
   
   # Fixed:
   cleaned_result = self.html_cleaner.clean(html)
   cleaned_html = cleaned_result['html']
   dom_patterns = self.dom_detector.detect_patterns(cleaned_html)
   best_pattern = dom_patterns.get('best_pattern')
   
   if best_pattern:
       from bs4 import BeautifulSoup
       soup = BeautifulSoup(cleaned_html, 'html.parser')
       selector = best_pattern.get('selector', '')
       containers = soup.select(selector)
   else:
       containers = []
   ```

2. **Fix pattern serialization**:
   ```python
   # Current (BROKEN):
   repeating_containers=[str(c) for c in containers[:3]]  # Wrong!
   
   # Fixed:
   container_samples = []
   for container in containers[:3]:
       container_samples.append({
           'tag': container.name,
           'classes': ' '.join(container.get('class', [])),
           'text_preview': container.get_text()[:100]
       })
   repeating_containers=container_samples
   ```

---

## Issue #2: JSON Quality Validation Working TOO Well

**Problem:** Product Hunt rejected at 50% quality (extracted "ProductCategoryEdge" GraphQL types).

### Analysis:
- This is a **borderline case**
- GraphQL schema types (`ProductCategoryEdge`) scored 50% vs 60% threshold
- Need to investigate if real product data exists alongside schema types

### Potential Solutions:
1. Lower quality threshold to 50% (risky - might accept garbage)
2. Better minified key inference for GraphQL responses
3. Investigate if Product Hunt has better JSON endpoints

---

## Test Results Summary

| Source | Type | Expected | Items | Method | Status | Quality Score |
|--------|------|----------|-------|--------|--------|---------------|
| **Leafly** | JS-heavy | 10+ | 18 | json | ✅ PASS | Excellent |
| **Amazon** | Mixed | 10+ | 0 | none | ❌ FAIL | JSON: 46.2% (rejected) |
| **eBay** | Mixed | 10+ | 0 | none | ❌ FAIL | JSON: 27.5% (rejected) |
| **Reddit** | Static | 15+ | 0 | none | ❌ FAIL | No JSON |
| **Hacker News** | Static | 20+ | 0 | none | ❌ FAIL | No JSON |
| **Product Hunt** | Modern JS | 5+ | 0 | none | ❌ FAIL | JSON: 50.0% (rejected) |

---

## Quality Validation Performance

### ✅ Correctly Rejected (Analytics Garbage):
1. **Amazon** (46.2%): `byg_desktop_optimistic_qs_t1` - Internal metrics
2. **eBay** (27.5%): `si=977af6df19a0ab3b2235adaefffb2e8a,c=1,serviceCorrelationId` - Tracking IDs

### ✅ Correctly Accepted (Real Products):
1. **Leafly** (>60%): Clean product names, prices, descriptions

### ⚠️ Borderline Case:
1. **Product Hunt** (50.0%): GraphQL schema types (`ProductCategoryEdge`)
   - Needs investigation: Are real products in same JSON?

---

## Recommended Action Plan

### Priority 1: Fix HTML Extraction (CRITICAL)
**Estimated Time:** 1-2 hours
**Impact:** Enables 5/6 sources to work

1. Fix actor.py API calls:
   - Update lines 245-250 (HTML cleaner dict handling)
   - Update lines 248-250 (DOM detector output handling)
   - Update lines 257-260 (Container serialization)

2. Test HTML extraction on:
   - Amazon (analytics rejected → HTML fallback)
   - eBay (analytics rejected → HTML fallback)
   - Reddit (no JSON → HTML)
   - Hacker News (no JSON → HTML)

### Priority 2: Validate Product Hunt
**Estimated Time:** 30 minutes
**Impact:** Clarify if 50% threshold is appropriate

1. Manual inspection of Product Hunt JSON
2. Determine if real product data exists
3. Adjust strategy (lower threshold vs better extraction)

### Priority 3: End-to-End QA Retest
**Estimated Time:** 1 hour
**Impact:** Validate true universal capability

1. Run comprehensive QA on all 6 sources
2. Verify 80%+ success rate (5/6 or 6/6)
3. Validate data quality for each source

---

## Success Criteria for Production Deployment

- [x] JSON quality validation working (detecting analytics garbage)
- [ ] HTML fallback extraction working
- [ ] 80%+ success rate across diverse sources (5/6 minimum)
- [ ] High-quality structured data output
- [ ] No API inconsistencies or crashes

**Current Status:** 1/6 passing (16.7%)
**Target Status:** 5/6+ passing (83.3%+)

---

## Conclusion

The **core architecture is sound**:
- ✅ Universal fetching (static/JS/JSON)
- ✅ JSON quality validation (analytics detection)
- ✅ Intelligent field matching (minified keys, synonyms)

But **implementation has critical bugs**:
- ❌ API inconsistencies between components
- ❌ HTML fallback completely broken
- ❌ Wrong method names and return types

**Recommendation:** Fix Priority 1 issues, then re-run comprehensive QA. System should achieve 80%+ success rate and be production-ready.




