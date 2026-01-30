# LLM Usage Architecture - When LLMs Are Called

## ❌ **NO - LLM is NOT used for every request**

The Universal Scraper uses a **multi-layer optimization strategy** to minimize LLM calls and costs.

---

## 🎯 **Request Flow (LLM Call Optimization)**

### **Layer 1: JSON-First Extraction** ⚡ **$0.00 (No LLM)**

**When:** ~30-50% of sites (Next.js, React, Vue, etc.)

```python
1. Fetch HTML
2. Detect embedded JSON (__NEXT_DATA__, JSON-LD, etc.)
3. Extract directly from JSON
4. ✅ Return data - NO LLM CALL
```

**Cost:** $0.00  
**Speed:** Instant  
**Example:** Chewy.com, most e-commerce sites

---

### **Layer 2: Code Cache** ⚡ **$0.00 (No LLM)**

**When:** Same page structure as previously scraped

```python
1. Generate structural hash from HTML
2. Check code cache by hash + fields
3. If cache hit: Execute cached extraction code
4. ✅ Return data - NO LLM CALL
```

**Cost:** $0.00  
**Speed:** Instant  
**Cache Key:** `structure_hash + fields`

**Example:** 
- First scrape of `amazon.com/products`: LLM call (~$0.01)
- Second scrape of `amazon.com/products`: Cache hit ($0.00)
- Scrape `amazon.com/products?page=2`: Cache hit ($0.00) - same structure!

---

### **Layer 3: Direct LLM Extraction** 💰 **~$0.01-0.03 per request**

**When:** JSON extraction fails AND code cache miss

```python
1. JSON extraction failed
2. Code cache miss
3. Use Direct LLM extraction (like ScrapeGraphAI)
4. ✅ Return data - 1 LLM CALL
```

**Cost:** ~$0.01-0.03 per request  
**Speed:** 2-5 seconds  
**Note:** This is the fallback, not the primary method

---

### **Layer 4: Pattern-Based Code Generation** 💰 **~$0.01 per structure (cached)**

**When:** Direct LLM fails AND code cache miss

```python
1. Direct LLM failed
2. Code cache miss
3. Generate extraction code with LLM (1 call)
4. Cache code by structure hash
5. Execute code
6. ✅ Return data - 1 LLM CALL (cached for future)
```

**Cost:** ~$0.01 per unique structure (then $0.00 for similar pages)  
**Speed:** 3-10 seconds first time, instant after caching

---

## 📊 **Cost Breakdown by Scenario**

### **Scenario 1: JSON-Heavy Site (e.g., Chewy.com)**

```
Request 1: JSON extraction → $0.00 ✅
Request 2: JSON extraction → $0.00 ✅
Request 3: JSON extraction → $0.00 ✅
...
Total: $0.00 for all requests
```

### **Scenario 2: HTML Site with Caching (e.g., Hacker News)**

```
Request 1: Code cache miss → Generate code → $0.01
Request 2: Code cache hit → $0.00 ✅
Request 3: Code cache hit → $0.00 ✅
Request 4: Code cache hit → $0.00 ✅
...
Total: $0.01 for first request, $0.00 for all subsequent
```

### **Scenario 3: New Site Every Time**

```
Request 1 (site A): Direct LLM → $0.02
Request 2 (site B): Direct LLM → $0.02
Request 3 (site C): Direct LLM → $0.02
...
Total: ~$0.02 per unique site
```

---

## 🎯 **LLM Call Frequency**

### **When LLM IS Called:**

1. **JSON Source Ranking** (optional, context-driven)
   - Only if multiple JSON sources found
   - Cached per domain
   - ~$0.005 per domain

2. **Data Validation** (optional, context-driven)
   - Only if context validation enabled
   - Validates extracted data quality
   - ~$0.005 per validation

3. **Direct LLM Extraction** (fallback)
   - Only if JSON extraction fails
   - ~$0.01-0.03 per request
   - No caching (each page is unique)

4. **Code Generation** (fallback)
   - Only if code cache miss
   - Cached by structure hash
   - ~$0.01 per unique structure

5. **Pagination Analysis** (optional)
   - Only if LLM pagination enabled
   - Cached per domain
   - ~$0.01 per domain

6. **Field Mapping** (optional)
   - Only if semantic field mapping enabled
   - Cached per domain
   - ~$0.01 per domain

7. **HTML Structure Analysis** (optional)
   - Only if structure analyzer enabled
   - Cached per domain
   - ~$0.01 per domain

### **When LLM is NOT Called:**

1. ✅ **JSON extraction** (most sites)
2. ✅ **Code cache hits** (same structure)
3. ✅ **Pattern cache hits** (learned patterns)
4. ✅ **Fast pagination detection** (pattern-based)
5. ✅ **Traditional JSON detection** (no context)

---

## 💰 **Cost Comparison**

### **Universal Scraper (This Architecture):**

**1000 pages, 10 unique structures:**
- JSON extraction: 300 pages × $0.00 = **$0.00**
- Code cache hits: 600 pages × $0.00 = **$0.00**
- Code generation: 10 structures × $0.01 = **$0.10**
- Direct LLM fallback: 100 pages × $0.02 = **$2.00**
- **Total: $2.10**

### **ScrapeGraphAI (LLM per request):**

**1000 pages:**
- Direct LLM: 1000 pages × $0.02 = **$20.00**
- **Total: $20.00**

### **Savings: 10x cheaper** 🎉

---

## 🔑 **Key Optimization Features**

### **1. JSON-First Architecture**
- **Priority:** Extract from embedded JSON first
- **LLM Calls:** 0
- **Success Rate:** ~30-50% of sites

### **2. Code Caching**
- **Cache Key:** Structural hash + fields
- **LLM Calls:** 1 per unique structure (then 0)
- **Hit Rate:** ~90%+ for similar pages

### **3. Pattern Learning**
- **Caches:** Learned extraction patterns
- **LLM Calls:** 1 per pattern (then 0)
- **Reuse:** Across similar sites

### **4. Context-Driven Validation**
- **Optional:** Only validates when needed
- **Cached:** Per domain
- **Cost:** ~$0.005 per domain

---

## 📈 **Real-World Example: Chewy.com**

### **First Request:**
```
1. Fetch HTML → JSON detected (Next.js)
2. Extract from JSON → ✅ Success
3. LLM Calls: 0
4. Cost: $0.00
```

### **Subsequent Requests:**
```
1. Fetch HTML → JSON detected
2. Extract from JSON → ✅ Success
3. LLM Calls: 0
4. Cost: $0.00
```

**Result:** $0.00 for all requests! 🎉

---

## 🎯 **Summary**

| Method | LLM Calls | Cost per Request | Cached? |
|--------|-----------|------------------|---------|
| **JSON Extraction** | 0 | $0.00 | N/A |
| **Code Cache Hit** | 0 | $0.00 | ✅ Yes |
| **Pattern Cache Hit** | 0 | $0.00 | ✅ Yes |
| **Direct LLM** | 1 | $0.01-0.03 | ❌ No |
| **Code Generation** | 1 | $0.01 | ✅ Yes (by structure) |
| **JSON Ranking** | 1 | $0.005 | ✅ Yes (per domain) |
| **Data Validation** | 1 | $0.005 | ✅ Yes (per domain) |

---

## ✅ **Answer: NO**

The architecture **does NOT use an LLM for every request**. It uses:

1. **JSON-first** extraction (no LLM)
2. **Code caching** (no LLM after first)
3. **Pattern caching** (no LLM after learning)
4. **LLM only as fallback** when all else fails

**Typical LLM call rate:** 10-30% of requests (mostly first-time structure discovery)







