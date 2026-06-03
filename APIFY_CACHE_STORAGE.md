# Apify Cache Storage - Where Cache Lives

## 📍 **Cache Storage Location**

### **In Apify (Production):**

**Storage Backend:** Apify Key-Value Store (KV Store)

**Location:** Cloud storage managed by Apify

**Persistence:** ✅ **Persists across runs** - cache is shared between all runs of the same actor!

**Access:**
- **Read**: `await Actor.get_value(key)`
- **Write**: `await Actor.set_value(key, value)`
- **Delete**: `await Actor.set_value(key, None)`

---

## 🔑 **Cache Key Format**

### **Direct LLM Cache Keys:**
```
direct_llm:{url_hash}:{structure_hash}:{fields_hash}
```

**Example:**
```
direct_llm:a1b2c3d4:e5f6g7h8i9j0k1l2m3n4o5p6:i7j8k9l0
```

**Components:**
- `url_hash`: MD5 hash of URL (8 chars) - optional
- `structure_hash`: SHA256 hash of HTML structure (16 chars)
- `fields_hash`: MD5 hash of sorted fields (8 chars)

---

## 💾 **What Gets Cached**

### **Cache Entry Structure:**
```json
{
  "items": [
    {"title": "...", "description": "..."},
    ...
  ],
  "structure_hash": "e5f6g7h8i9j0k1l2",
  "fields": ["title", "description"],
  "timestamp": 1732700000.0,
  "url": "https://news.ycombinator.com",
  "item_count": 30
}
```

---

## 🔄 **Cache Lifecycle**

### **First Run:**
```
1. Actor starts
2. Scrapes URL → Cache miss
3. Direct LLM → $0.02
4. Caches result in KV Store
5. Actor ends
```

### **Second Run (Same Actor):**
```
1. Actor starts
2. Scrapes same URL → Cache hit ✅
3. Validates cache → Passes ✅
4. Returns cached data → $0.00
5. Actor ends
```

**Key Point:** Cache persists between runs because KV Store is persistent!

---

## 📊 **Cache Sharing**

### **Within Same Actor:**
- ✅ All runs share the same KV Store
- ✅ Cache from Run 1 is available in Run 2
- ✅ Cache from Run 2 is available in Run 3
- ✅ **Persists forever** (until manually deleted or TTL expires)

### **Between Different Actors:**
- ❌ Each actor has its own KV Store
- ❌ Cache is NOT shared between actors
- ✅ Each actor builds its own cache

---

## 🗑️ **Cache Expiration**

### **TTL-Based Expiration:**
- Default: 1 hour (`cache_ttl=3600`)
- Configurable via `DirectLLMExtractor(cache_ttl=...)`
- Checked during validation

### **Manual Deletion:**
- Via Apify dashboard: Key-Value Store → Delete key
- Via code: `await Actor.set_value(key, None)`

---

## 📈 **Cache Statistics**

### **Viewing Cache in Apify:**

1. Go to Actor dashboard
2. Navigate to **Storage** → **Key-Value Store**
3. Search for keys starting with `direct_llm:`
4. View cache entries, timestamps, sizes

### **Cache Size:**
- Each entry: ~1-10KB (depends on item count)
- KV Store limit: 10MB per key
- Total KV Store: Unlimited (within Apify limits)

---

## ✅ **Benefits of Apify KV Store**

1. **Persistence**: Cache survives between runs ✅
2. **Sharing**: All runs of same actor share cache ✅
3. **Cloud Storage**: No local disk needed ✅
4. **Scalability**: Handles millions of keys ✅
5. **Reliability**: Managed by Apify infrastructure ✅

---

## 🎯 **Example: Real-World Usage**

### **Day 1 - First Run:**
```
Run 1: Scrapes Hacker News
- Cache miss
- Direct LLM → $0.02
- Caches result in KV Store
- Key: direct_llm:abc123:def456:ghi789
```

### **Day 1 - Second Run (Same Day):**
```
Run 2: Scrapes Hacker News again
- Cache hit ✅
- Validates → Passes ✅
- Returns cached data → $0.00
```

### **Day 2 - Third Run:**
```
Run 3: Scrapes Hacker News
- Cache hit ✅ (still in KV Store!)
- Validates → Passes ✅
- Returns cached data → $0.00
```

**Total Cost:** $0.02 for 3 runs (vs $0.06 without caching)

---

## 🔧 **Configuration**

### **Enable/Disable Caching:**
```python
# In scraper initialization
scraper = UniversalScraper(
    api_key=api_key,
    use_direct_llm=True,
    # Caching enabled by default
)
```

### **Cache TTL:**
```python
# In DirectLLMExtractor initialization
direct_llm_extractor = DirectLLMExtractor(
    api_key=api_key,
    enable_cache=True,
    cache_ttl=3600  # 1 hour (default)
)
```

---

## 📝 **Summary**

**Cache Storage in Apify:**
- **Backend**: Apify Key-Value Store (KV Store)
- **Location**: Cloud storage (managed by Apify)
- **Persistence**: ✅ **Persists across runs**
- **Sharing**: ✅ Shared between all runs of the same actor
- **Access**: Via `Actor.get_value()` / `Actor.set_value()`

**Cache Keys:**
- Format: `direct_llm:{structure_hash}:{fields_hash}`
- Stored in KV Store
- Accessible from any run of the same actor

**Result:** Cache works seamlessly in Apify, providing massive cost savings for repeated requests! 🎉







