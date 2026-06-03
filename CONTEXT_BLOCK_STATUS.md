# Context-Block Extraction - Implementation Status

**Date**: November 14, 2024  
**Session**: Fixing sibling element blindness

---

## 🎯 The Problem

The system assumes all data is nested inside repeating containers, but many websites use **sibling-based layouts** where related data is in adjacent elements.

**Example (Stack Overflow)**:
```html
<div class="s-post-summary">           ← System finds this
    <h3>Question Title</h3>            ✅ Extracted
</div>
<div class="s-post-summary--stats">   ← SIBLING (never checked)
    <span class="vote-count">42</span> ❌ MISSING
</div>
```

---

## ✅ What's Completed

### **Phase 1: DOM Pattern Detector** ✅

**Status**: COMPLETE & WORKING

**Changes**:
- Added `_analyze_sibling_patterns()` method
- Detects consistent sibling elements across repeating containers
- Returns `sibling_analysis` with 3 types:
  - `parent_context_block`: Parent contains container + siblings
  - `sibling_group`: Container + immediate siblings
  - `container_only`: No siblings (fallback)

**Test Results**:
```
📦 Context block: parent_context_block
📦 Found 1 consistent siblings
```

✅ **Sibling detection is working!**

---

### **Phase 2: HTML Sampling** ✅

**Status**: COMPLETE & WORKING

**Changes**:
- Added `_extract_context_blocks()` to `SmartHTMLSampler`
- Extracts full parent elements (includes container + siblings)
- Modified `_extract_smart_html_sample()` in `scraper.py` to pass `sibling_analysis`

**What It Does**:
- For `parent_context_block`: Extracts full parent elements
- For `sibling_group`: Extracts container + next 3 siblings
- Limits to 30KB max

✅ **Context blocks are being extracted!**

---

### **Phase 3: LLM Prompts** ✅

**Status**: COMPLETE

**Changes**:
- Updated `adaptive_dom_detector.py` Pass 2 prompt
- Added explicit sibling awareness section
- Provides Stack Overflow example

**Prompt Addition**:
```
**CRITICAL: DATA MAY BE IN SIBLING ELEMENTS, NOT JUST CHILDREN!**

Many websites (Stack Overflow, GitHub, Indeed) use sibling-based layouts...

Example (Stack Overflow):
<div class="s-post-summary">  ← Main container
    <h3>Question Title</h3>   ✅ Inside container
</div>
<div class="s-post-summary--stats">  ← SIBLING (not child!)
    <span class="vote-count">42</span>  ← votes field is HERE
</div>

**WHERE TO LOOK FOR NULL FIELDS:**
1. ✅ Inside the container (children, grandchildren)
2. ✅ **SIBLING elements** (next/previous siblings of the container)
3. ✅ Parent element (shared across all items)
```

✅ **LLM knows about siblings!**

---

### **Phase 4: Code Generation** ⚠️

**Status**: INCOMPLETE - THIS IS THE ISSUE

**Problem**: Generated code still only looks inside the main container

**Generated Code** (Stack Overflow):
```python
containers = soup.find_all('div', class_='s-post-summary js-post-summary')

for elem in containers:
    item = {}
    
    # ✅ Title extraction works (inside container)
    title_elem = elem.select_one('h3 a')
    item['title'] = title_elem.text.strip() if title_elem else None
    
    # ❌ Votes extraction fails (in sibling, not in elem!)
    votes_elem = elem.select_one('span.vote-count-post')  
    item['votes'] = votes_elem.text.strip() if votes_elem else None
```

**What It Should Be**:
```python
# Option 1: Iterate over parent elements (recommended)
parents = soup.find_all('div', class_='parent-container')

for parent in parents:
    item = {}
    
    # Extract from main container
    container = parent.select_one('.s-post-summary')
    if container:
        title_elem = container.select_one('h3 a')
        item['title'] = title_elem.text.strip() if title_elem else None
    
    # Extract from sibling
    sibling = parent.select_one('.s-post-summary--stats')
    if sibling:
        votes_elem = sibling.select_one('span.vote-count-post')
        item['votes'] = votes_elem.text.strip() if votes_elem else None
```

---

## ❌ What's Missing

### **The Core Issue**

Even though:
- ✅ Context blocks include siblings
- ✅ LLM prompt mentions siblings
- ❌ Generated code ignores siblings

**Root Cause**: The AI generator prompt needs explicit instructions on HOW to write code that accesses siblings.

---

## 🔧 The Fix Needed

### **Update `ai_generator.py` Prompt**

Add to the code generation prompt:

```
**IMPORTANT: IF THE HTML SAMPLE INCLUDES PARENT ELEMENTS WITH SIBLINGS:**

The HTML sample may contain "context blocks" (parent > container + siblings).
If you see parent elements wrapping multiple related elements, you MUST:

1. Iterate over the PARENT elements, not just the main containers
2. Extract data from BOTH the main container AND its siblings

Example Structure:
<parent>
    <div class="main-container">
        <h3>Title</h3>  ← Field 1
    </div>
    <div class="metadata">
        <span>votes</span>  ← Field 2 (SIBLING!)
    </div>
</parent>

Code Pattern:
parents = soup.find_all('parent-selector')
for parent in parents:
    item = {}
    
    # Extract from main container
    container = parent.select_one('.main-container')
    item['title'] = container.select_one('h3').text
    
    # Extract from sibling
    sibling = parent.select_one('.metadata')
    item['votes'] = sibling.select_one('span').text
```

---

## 📊 Test Results

### **Before Context-Block Implementation**

| Site | Quality | Issue |
|------|---------|-------|
| Stack Overflow | 50% | votes always None (in sibling) |
| GitHub | 33% | stars always None (in sibling) |
| Indeed | 25% | salary always None (in sibling) |

### **After Phases 1-3** (Current State)

| Site | Quality | Status |
|------|---------|--------|
| Stack Overflow | 50% | ✅ Siblings detected, ❌ not extracted |
| GitHub | ? | Not tested yet |
| Indeed | ? | Not tested yet |

### **Expected After Phase 4 Fix**

| Site | Quality | Expected |
|------|---------|----------|
| Stack Overflow | 90%+ | ✅ votes extracted from siblings |
| GitHub | 85%+ | ✅ stars extracted from siblings |
| Indeed | 80%+ | ✅ salary extracted from siblings |

---

## 🚀 Next Steps

### **Immediate (Complete Phase 4)**

1. Update `ai_generator.py` prompt with sibling code generation instructions
2. Add example code showing parent iteration pattern
3. Test on Stack Overflow to validate

### **Then Test**

Test on all 3 sibling-based layouts:
- Stack Overflow (`votes` in sibling)
- GitHub Trending (`stars` in sibling)
- Indeed (`salary` in sibling)

---

## 💰 Cost Impact

**No significant cost increase**:
- Sibling detection: 0 LLM calls (heuristic only)
- Context block extraction: Same sample size
- Code generation: Same number of LLM calls

**Estimated**: ~$0.005/scrape (unchanged)

---

## ✅ Summary

**What's Working**:
- ✅ Sibling detection (Phase 1)
- ✅ Context block extraction (Phase 2)
- ✅ Sibling-aware prompts (Phase 3)

**What's Not Working**:
- ❌ Generated code doesn't iterate over parents
- ❌ Generated code doesn't access sibling elements

**The Fix**: Update AI generator prompt with explicit parent iteration instructions.

**Time to Complete**: ~30 minutes
- Update prompt: 10 min
- Test & validate: 20 min

---

**Status**: 75% COMPLETE - Just need Phase 4 prompt update! 🎯






