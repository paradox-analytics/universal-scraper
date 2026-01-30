# ✅ **CONTEXT-DRIVEN UNIVERSAL SYSTEM - IMPLEMENTED**

**Date**: November 8, 2025  
**Status**: Core modules complete - Ready for integration

---

## 🎉 **WHAT'S BEEN BUILT**

### **1. Context Manager** (`context_manager.py`) ✅
**Purpose**: Parse user intent and guide all scraping decisions

**Features**:
- ✅ **LLM-driven inference** - NO hardcoded patterns
- ✅ **Flexible input** - Accepts string or structured dict
- ✅ **Smart caching** - One LLM call per unique context
- ✅ **Confidence scoring** - Know how reliable the inference is
- ✅ **URL-aware** - Can use URL for additional context

**Example Usage**:
```python
from universal_scraper.core import ContextManager

# Initialize
context_mgr = ContextManager(
    api_key="sk-...",
    model="gpt-4o-mini"
)

# Parse context
context = context_mgr.parse_context(
    "Extract concert events with artist name, venue, and ticket prices",
    url="https://www.ticketmaster.com/discover/concerts"
)

# Result:
# Context(type=events, 4 fields, confidence=0.95)
# Fields: ['artist_name', 'venue', 'date', 'ticket_price']
```

---

### **2. Data Validator** (`data_validator.py`) ✅
**Purpose**: Validate extracted data matches user's goal

**Features**:
- ✅ **Context-aware validation** - Checks against user's goal
- ✅ **Prevents false positives** - Rejects cart config, footer data, etc.
- ✅ **Confidence scoring** - How sure is the validation?
- ✅ **Actionable suggestions** - "Try next JSON source" or "Use HTML extraction"
- ✅ **Smart caching** - Caches by item structure

**Example Usage**:
```python
from universal_scraper.core import LLMDataValidator

validator = LLMDataValidator(api_key="sk-...")

# Validate extraction
result = validator.validate_extraction(
    items=[{"title": "American Express", "image": "..."}],  # Footer data
    url="https://www.ticketmaster.com/discover/concerts",
    context=context  # User wants "concert events"
)

# Result:
# {
#     "is_target_data": False,
#     "confidence": 0.9,
#     "reasoning": "These are footer partner links, not concert events",
#     "detected_type": "footer_links",
#     "suggestion": "try_next_json_source"
# }
```

---

### **3. JSON Analyzer** (`json_analyzer.py`) ✅
**Purpose**: Rank multiple JSON sources by relevance

**Features**:
- ✅ **Context-driven ranking** - Picks best JSON for user's goal
- ✅ **Handles multiple sources** - __NEXT_DATA__, APIs, JSON-LD, etc.
- ✅ **Structural analysis** - Looks for item arrays, field names, patterns
- ✅ **Confidence scoring** - How likely is each source?
- ✅ **Smart caching** - Caches by source structure

**Example Usage**:
```python
from universal_scraper.core import LLMJsonAnalyzer

analyzer = LLMJsonAnalyzer(api_key="sk-...")

# Rank sources
rankings = analyzer.rank_sources(
    json_sources={
        '__NEXT_DATA__': {...},  # Footer/navigation data
        'api_response_1': {...},  # Concert listings API
        'json_ld': {...}  # Schema.org metadata
    },
    url="https://www.ticketmaster.com/discover/concerts",
    context=context
)

# Result:
# [
#     {
#         "source": "api_response_1",
#         "confidence": 0.95,
#         "reasoning": "Has events array with concert data",
#         "estimated_items": 50
#     },
#     {
#         "source": "__NEXT_DATA__",
#         "confidence": 0.40,
#         "reasoning": "Contains pageProps but mostly navigation",
#         "estimated_items": 10
#     },
#     ...
# ]
```

---

## 🔄 **THE COMPLETE FLOW**

```
┌─────────────────────────────────────────┐
│  USER INPUT                             │
│  "Extract concert events with dates"   │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │   CONTEXT MANAGER         │
        │   LLM infers:             │
        │   - type: "events"        │
        │   - fields: [artist, ...]│
        └───────────────────────────┘
                    ↓
            EXTRACTION CONTEXT
                    ↓
        ┌───────────────────────────┐
        │   FETCH PAGE              │
        │   - HTML + captured JSON  │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │   JSON ANALYZER           │
        │   Ranks 3 sources:        │
        │   1. API (0.95)          │
        │   2. __NEXT_DATA__ (0.40)│
        │   3. JSON-LD (0.10)      │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │   TRY SOURCE #1 (API)     │
        │   Extract items           │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │   DATA VALIDATOR          │
        │   ✅ These ARE events!    │
        │   Confidence: 0.95        │
        └───────────────────────────┘
                    ↓
            ✅ RETURN DATA
```

---

## 📂 **FILES CREATED**

| File | Lines | Purpose |
|------|-------|---------|
| `context_manager.py` | 267 | Parse & enrich user intent |
| `data_validator.py` | 235 | Validate extracted data |
| `json_analyzer.py` | 314 | Rank JSON sources |
| **Total** | **816** | **Core intelligence layer** |

---

## 🔧 **INTEGRATION STATUS**

### ✅ **Completed**
- [x] Create `ContextManager` module
- [x] Create `LLMDataValidator` module
- [x] Create `LLMJsonAnalyzer` module
- [x] Add to `core/__init__.py` exports
- [x] Zero hardcoded patterns (fully LLM-driven)
- [x] Smart caching for performance
- [x] Confidence scoring throughout

### ⏳ **Next Steps**
- [ ] Update `scraper.py` to use context system
- [ ] Update `UniversalScraper.__init__` to accept `extraction_context` parameter
- [ ] Integrate JSON analyzer into JSON detection flow
- [ ] Integrate data validator after extraction
- [ ] Update `actor.py` to pass context from input
- [ ] Update `INPUT_SCHEMA.json` with `extractionContext` field
- [ ] Test on Amazon, Ticketmaster, Leafly
- [ ] Documentation & examples

---

## 🎯 **EXPECTED IMPACT**

### **Before (Current State)**
```python
# Amazon SSD Store
→ Finds cart config JSON
→ len(items) == 1
→ ✅ "Sufficient!"
→ ❌ Returns cart config
```

### **After (With Context System)**
```python
# Amazon SSD Store
context = "Extract product listings with name, price, rating"

→ Finds 3 JSON sources
→ LLM ranks: products_api=0.95, cart=0.20, footer=0.10
→ Extracts from products_api
→ LLM validates: ✅ "These are products!"
→ ✅ Returns actual products
```

---

## 💰 **LLM COSTS**

| Operation | Calls Per URL | Cacheable | Cost (gpt-4o-mini) |
|-----------|---------------|-----------|---------------------|
| Context inference | 1 | ✅ Yes (by prompt) | $0.0001 |
| JSON source ranking | 1 | ✅ Yes (by structure) | $0.0002 |
| Data validation | 1-3 | ✅ Yes (by structure) | $0.0001 each |
| **Total per unique page** | **3-5** | | **~$0.0005** |

**With caching**:
- Same goal used 100 times: 1 context inference call only
- Same JSON structure: Ranking cached
- Result: **$0.0005/unique page, $0.0001/repeated page**

---

## 🧪 **TESTING PLAN**

### **Phase 1: Unit Tests**
```python
# Test context inference
def test_context_inference():
    mgr = ContextManager(api_key="...")
    ctx = mgr.parse_context("Extract concert events")
    assert ctx.data_type == "events"
    assert "date" in ctx.fields

# Test validation
def test_validation_rejects_wrong_data():
    validator = LLMDataValidator(api_key="...")
    result = validator.validate_extraction(
        items=[{"title": "Footer Link"}],
        context=event_context
    )
    assert result['is_target_data'] == False
```

### **Phase 2: Integration Tests**
```python
# Test Amazon (should work with context)
scraper = UniversalScraper(
    api_key="...",
    extraction_context="Extract product listings"
)
result = scraper.scrape("https://www.amazon.com/...")
assert len(result['data']) > 10
assert 'name' in result['data'][0]

# Test Ticketmaster (should work with context)
scraper = UniversalScraper(
    extraction_context="Extract concert events"
)
result = scraper.scrape("https://www.ticketmaster.com/...")
assert 'artist' in result['data'][0] or 'event' in result['data'][0]
```

### **Phase 3: Edge Cases**
- Multiple JSON sources (all irrelevant)
- No JSON sources (HTML only)
- Ambiguous data (products + reviews mixed)
- LLM failure scenarios

---

## 📊 **SUCCESS METRICS**

| Metric | Current | Target |
|--------|---------|--------|
| **Leafly** | ✅ Works | ✅ Still works |
| **Ticketmaster** | ❌ Footer (11 items) | ✅ Events (50+ items) |
| **Amazon** | ❌ Cart (1 item) | ✅ Products (50+ items) |
| **False positive rate** | ~66% (2/3 failed) | <10% |
| **LLM calls per page** | 1-2 | 3-5 (cached) |

---

## 🚀 **DEPLOYMENT CHECKLIST**

1. **Integration** (Next step)
   - [ ] Update `scraper.py` with context-driven logic
   - [ ] Update `actor.py` to accept `extractionContext`
   - [ ] Update `INPUT_SCHEMA.json`

2. **Testing**
   - [ ] Unit tests for new modules
   - [ ] Integration tests with scraper
   - [ ] Test on 3 edge case sites

3. **Documentation**
   - [ ] Update README with context examples
   - [ ] Update APIFY_DEPLOYMENT.md
   - [ ] Create CONTEXT_USAGE_GUIDE.md

4. **Deployment**
   - [ ] Deploy to Apify
   - [ ] Test in production
   - [ ] Monitor LLM costs

---

## 💡 **KEY DESIGN DECISIONS**

### **1. Why LLM for Everything?**
**Static patterns fail fast:**
```python
# ❌ Breaks on:
if 'product' in text:  # "production", "byproduct", "reproduce"
if url.contains('/shop/'):  # /shopping-cart/, /workshop/
```

**LLM understands semantics:**
```python
# ✅ Works for:
"Extract brewery info" → businesses
"Get tour dates" → events (concerts)
"Scrape NFT listings" → digital_assets
```

### **2. Why Context-Driven?**
**User knows best** - Don't guess what they want, let them tell us.

### **3. Why Three Separate Modules?**
**Single Responsibility** - Each module does one thing well:
- Context Manager: Understand intent
- JSON Analyzer: Pick best source
- Data Validator: Verify correctness

---

## 🎯 **NEXT IMMEDIATE ACTION**

**Update `scraper.py`** to integrate these modules (ETA: 2-3 hours)

Key changes:
1. Add `extraction_context` parameter to `__init__`
2. Initialize context manager, analyzer, validator
3. Use analyzer to rank JSON sources
4. Use validator to verify extracted data
5. Fall back to HTML if validation fails

**Ready to proceed?**








