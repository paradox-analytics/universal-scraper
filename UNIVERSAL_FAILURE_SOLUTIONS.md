# 🔧 Universal Solutions for Test Failures

## 📊 Failure Analysis

### **Failure 1: Reddit (47% quality - null fields)**
**Problem**: Items extracted but `author`, `upvotes`, `comments` are null  
**Root Cause**: Custom `<shreddit-post>` elements store data in HTML attributes, not nested text  
**Universal Solution**: Enhanced attribute extraction guidance in AI prompts

### **Failure 2: Craigslist (0% quality - date field null)**
**Problem**: 340 items extracted but `date` field consistently null  
**Root Cause**: Temporal fields (dates, times) need better semantic mapping  
**Universal Solution**: Enhanced temporal field detection in Field Mapper

### **Failure 3: Product Hunt & TechCrunch (0 items)**
**Problem**: No items extracted at all  
**Root Cause**: Heavy JS rendering requires adaptive wait times  
**Universal Solution**: Smart wait strategy with DOM mutation detection

---

## 🎯 Universal Solutions (No Site-Specific Code)

### **Solution 1: Enhanced Attribute Extraction**
**Applies to**: Any site with custom components (Reddit, modern SPAs)

**Implementation**:
1. When null ratio is high (>50%), provide specific attribute extraction guidance
2. Tell LLM to check: `data-*`, `aria-*`, `itemprop`, custom attributes
3. Show HTML sample with attributes visible
4. Add retry with "attributes first" strategy

**Benefit**: Works for ANY custom component architecture

---

### **Solution 2: Temporal Field Detection**
**Applies to**: Any site with dates, times, timestamps (Craigslist, news sites, forums)

**Implementation**:
1. In Field Mapper, detect temporal field names: `date`, `time`, `posted`, `published`, `updated`, `created`
2. Add specific guidance for temporal extraction:
   - Check for `<time>` tags
   - Check for `datetime` attributes
   - Check for relative dates ("2 hours ago")
   - Check for formatted dates ("Nov 12, 2024")
3. Provide temporal-specific code examples

**Benefit**: Works for ANY date/time format across all sites

---

### **Solution 3: Smart Wait Strategy**
**Applies to**: Any JS-heavy site (Product Hunt, TechCrunch, modern SPAs)

**Implementation**:
1. Detect page rendering state (DOM mutations)
2. Adaptive wait based on:
   - Network activity (API calls)
   - DOM stability (no changes for 500ms)
   - Known patterns (skeleton loaders, spinners)
3. Maximum wait: 10s (prevent hanging)

**Benefit**: Works for ANY async rendering pattern

---

### **Solution 4: Zero-Item Diagnostic**
**Applies to**: Any site returning 0 items

**Implementation**:
1. When 0 items extracted, run diagnostics:
   - Is HTML size reasonable? (>1KB)
   - Are there repeating patterns? (check DOM)
   - Is content in shadow DOM? (check for custom elements)
   - Is content loaded via JS? (check for loading indicators)
2. Provide diagnostic feedback to LLM for retry
3. Try alternative patterns (tables, divs, lists, custom elements)

**Benefit**: Self-diagnosing system for ANY extraction failure

---

## 🔧 Implementation Priority

1. **Enhanced Attribute Extraction** (HIGH) - Fixes Reddit immediately
2. **Temporal Field Detection** (HIGH) - Fixes Craigslist immediately  
3. **Smart Wait Strategy** (MEDIUM) - Fixes Product Hunt/TechCrunch
4. **Zero-Item Diagnostic** (MEDIUM) - Prevents future 0-item failures

All solutions are **100% universal** and require **no site-specific code**.







