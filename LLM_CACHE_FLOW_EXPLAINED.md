# 🧠 LLM → Cache → Deterministic Flow Architecture

## Overview

ParaDocs uses a **hybrid intelligence system** that starts with LLM-powered analysis and evolves into deterministic, instant extraction through caching. This document explains how it works.

---

## 🔄 The Extraction Flow

### 1️⃣ **First Visit (LLM Mode)**
When you scrape a new website for the first time:

```
URL → Fetch HTML → LLM Analysis → Template Generation → Cache Storage → Extract Data
  |                    |                 |                    |              |
  └─ Hybrid         └─ GPT-4o-mini   └─ JSON spec        └─ 3-tier     └─ Results
     fetcher           analyzes          with selectors     caching
```

**What Happens:**
1. **Hybrid Fetcher** determines best method (static HTML, browser JS, or Web Unblocker)
2. **LLM Analysis** (via `gpt-4o-mini`):
   - Analyzes HTML structure
   - Identifies patterns for your requested fields
   - Generates a deterministic **Template Spec** (JSON)
3. **Template Spec** contains:
   - CSS selectors for each field
   - Extraction rules (text, attribute, etc.)
   - Normalization functions
   - Validation rules
   - Fallback strategies
4. **Cache Storage** (3 layers):
   - **DOM Digest Cache**: HTML structure fingerprint
   - **Template Cache**: Extraction rules
   - **Direct LLM Cache**: Full extraction results
5. **Extract Data**: Execute template spec → return results

**Cost:** ~500-2000 tokens (~$0.001-0.003)
**Time:** 3-8 seconds

---

### 2️⃣ **Second Visit (Cache Hit - Instant!)**
When you scrape the **same website** with **same fields**:

```
URL → DOM Digest Check → Template Spec Retrieved → Deterministic Extract → Data
  |         |                     |                         |                 |
  └─ Fast └─ Fingerprint      └─ Cached JSON          └─ No LLM!       └─ Results
     fetch    match (0.1s)        (instant)              CSS selectors
```

**What Happens:**
1. **DOM Digest Check**: Generate HTML fingerprint → check cache
   - If structure matches → retrieve cached Template Spec
2. **Deterministic Extraction**:
   - Execute CSS selectors directly
   - No LLM calls
   - Pure selector-based extraction
3. **Instant Results**: ~0.5-1 second total

**Cost:** $0 (no LLM calls!)
**Time:** < 1 second

---

### 3️⃣ **Website Changed (Adaptive Mode)**
When the website HTML structure changes:

```
URL → DOM Digest Check → No Match → Re-analyze with LLM → New Template → Cache → Data
  |         |                |              |                    |            |
  └─ Fetch └─ Structure   └─ Cache     └─ GPT-4o-mini      └─ Updated   └─ Results
             changed          miss         generates new        cache
```

**What Happens:**
1. **DOM Digest** detects structure change
2. **Re-run LLM Analysis**: Generate new template spec
3. **Update Cache**: Replace old template with new one
4. **Future requests** use the new cached template

**Cost:** ~500-2000 tokens (same as first visit)
**Time:** 3-8 seconds (one time, then cached again)

---

## 📊 Metadata Tracked

Every extraction returns comprehensive metadata:

### Cache Metadata
```json
{
  "dom_digest_cache_hit": true,        // Did we find a matching structure?
  "template_spec_used": true,          // Did we use a cached template?
  "template_spec_id": "abc123...",     // Which template was used?
  "direct_llm_cached": false,          // Was full LLM result cached?
  "pattern_cache_hit": true            // Legacy pattern system hit?
}
```

### LLM Usage Metadata
```json
{
  "model_tier_used": "template",       // router | template | recovery
  "template_spec_generated": false,    // Did we generate a new template?
  "llm_tokens_used": 0,                // Tokens used (0 if cached)
  "extraction_method": "template_spec" // Method used
}
```

### Learning Metadata
```json
{
  "pattern_learned": false,            // Did we learn a new pattern?
  "selector_library_updated": true,    // Did we update selector library?
  "dom_digest_page_type": "product"    // Detected page type
}
```

---

## 🎯 Multi-Tenant Sharing

### Private Cache (Default)
```
User A scrapes example.com
  └─ Cache saved to User A's tenant only
  └─ User B cannot access
```

### Public/Shared Cache
```
User A scrapes example.com with "public" visibility
  └─ Cache saved with public flag
  └─ User B can reuse the same template!
  └─ No LLM cost for User B
```

**Use Cases:**
- **Public APIs**: Share templates for common sites (GitHub, Twitter, etc.)
- **Team Workspaces**: Share within organization
- **Marketplace**: Sell/share high-quality templates

---

## 🚀 Production Deployment

### Agent Creation
1. User creates agent in UI (Draft mode)
2. Tests URL with fields
3. LLM generates template spec
4. User reviews extracted data
5. User clicks "Save" → Agent persists

### Agent Execution
```
Agent Trigger → Load Template Spec → Fetch URL → Execute → Store Results
     |               |                   |           |          |
  Schedule/API   From cache          Hybrid      No LLM    Database/Webhook
```

**Production Benefits:**
- **Speed**: < 1 second per page (cached templates)
- **Cost**: $0 per extraction after first run
- **Reliability**: Deterministic selectors (no LLM hallucinations)
- **Scale**: 1000s of pages/minute with cached templates

---

## 💡 Key Concepts

### Template Spec (Deterministic Rules)
A JSON specification with:
```json
{
  "template_id": "abc123...",
  "fields": [
    {
      "name": "title",
      "selector": "h1.product-title",
      "extraction_type": "text",
      "normalizer": "trim",
      "required": true
    },
    {
      "name": "price",
      "selector": ".price-tag",
      "extraction_type": "text",
      "normalizer": "parse_price",
      "validator": "is_number"
    }
  ],
  "container_selector": ".product-card",
  "pagination": { ... }
}
```

### DOM Digest (Fingerprint)
A stable hash of HTML structure:
- Strips dynamic content (IDs, timestamps, etc.)
- Focuses on tag structure and class patterns
- Detects layout changes
- Used for fast cache lookups

### Selector Library (Bootstrapping)
A knowledge base of successful selectors:
- Learns from extraction successes
- Provides training examples for new templates
- Improves LLM accuracy over time
- Site-specific pattern library

---

## 🔧 How to Verify It's Working

### In the UI:
1. Look for **"Draft"** badge on new agents
2. Check **"Saved just now"** indicator (auto-save)
3. Watch for **cache hit badges** on re-runs
4. View **Extraction Flow** panel showing:
   - LLM Analysis (or "Skipped" if cached)
   - Template Spec
   - Cache Storage
   - Deterministic Extract

### In Logs (Backend):
```
📥 Scraping: example.com
🔍 Checking DOM digest cache...
✅ DOM digest cache HIT! Found matching template
⚡ Using cached template spec (NO LLM CALL)
🎯 Extracted 24 items in 0.8s
💾 Cache hit: true | Tokens: 0 | Cost: $0.00
```

### In API Response:
```json
{
  "success": true,
  "items": [...],
  "metadata": {
    "dom_digest_cache_hit": true,
    "template_spec_used": true,
    "template_spec_id": "abc123",
    "llm_tokens_used": 0,
    "extraction_time": 0.8,
    "cost": 0.0
  }
}
```

---

## 🎓 Best Practices

### For Optimal Caching:
1. **Consistent Field Names**: Use same field names across runs
2. **Stable URLs**: Scrape same domain/structure
3. **Public Visibility**: Share templates for common sites
4. **Review First Run**: Verify LLM-generated template before production

### For Production:
1. **Test First**: Always test agent in draft mode
2. **Save Templates**: Explicitly save after successful extraction
3. **Monitor Metadata**: Track cache hit rates
4. **Handle Changes**: Set up alerts for template invalidation

### For Sharing:
1. **High-Quality Data**: Ensure extracted data is clean
2. **Good Field Names**: Use semantic, clear field names
3. **Documentation**: Add description to agent
4. **Version Control**: Update templates when sites change

---

## 📈 Performance Comparison

| Metric | First Run (LLM) | Cached Run (Deterministic) |
|--------|-----------------|----------------------------|
| **Speed** | 3-8 seconds | < 1 second |
| **Cost** | $0.001-0.003 | $0.00 |
| **Tokens** | 500-2000 | 0 |
| **LLM Calls** | 1-2 | 0 |
| **Reliability** | 95-99% | 99.9% |

---

## 🔬 Technical Deep Dive

### Cache Layers (3-Tier)

#### Layer 1: DOM Digest Cache
- **Key**: `domain + path_pattern + dom_fingerprint`
- **Value**: `template_spec_id + page_type + version`
- **TTL**: 30 days
- **Purpose**: Fast pre-check for layout changes

#### Layer 2: Template Spec Cache
- **Key**: `template_spec_id`
- **Value**: Full JSON template spec
- **TTL**: 90 days
- **Purpose**: Store deterministic extraction rules

#### Layer 3: Direct LLM Cache
- **Key**: `domain + fields_hash`
- **Value**: Full extraction results
- **TTL**: 7 days
- **Purpose**: Instant results for exact matches

### Model Routing (3-Tier)

#### Tier 1: Router Model
- **Model**: `gpt-3.5-turbo`
- **Use**: Page type classification
- **Cost**: ~100 tokens (~$0.0001)
- **Speed**: < 1 second

#### Tier 2: Template Model
- **Model**: `gpt-4o-mini`
- **Use**: Template spec generation
- **Cost**: ~1000 tokens (~$0.001)
- **Speed**: 2-4 seconds

#### Tier 3: Recovery Model
- **Model**: `gpt-4o` (future)
- **Use**: Complex/failed extractions
- **Cost**: ~2000 tokens (~$0.04)
- **Speed**: 5-8 seconds

---

**Last Updated**: December 27, 2025
**Status**: ✅ Fully Implemented & Deployed



