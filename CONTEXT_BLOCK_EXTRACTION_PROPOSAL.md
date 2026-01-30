# 🎯 Context-Block Extraction - Universal Solution

**Problem**: Current system assumes all data is inside the repeating container, but many sites use sibling/adjacent layouts.

---

## 🔍 Current Architecture (Container-Based)

```
┌─────────────────────────────────┐
│ <div class="item"> ← Container │
│   <h3>Title</h3>     ✅         │
│   <span>Price</span> ✅         │
│ </div>                          │
└─────────────────────────────────┘

Works great when all data is nested inside!
```

---

## ❌ Where It Fails (Sibling-Based Layouts)

### **Stack Overflow**
```html
<div class="s-post-summary">           ← Found by DOM detector
    <h3>Title</h3>                     ✅ Extracted
</div>
<div class="s-post-summary--stats">   ← SIBLING (never checked)
    <span class="vote-count">42</span> ❌ Missing
</div>
```

### **GitHub Trending**
```html
<article class="Box-row">              ← Found by DOM detector
    <h2>Repository Name</h2>           ✅ Extracted
</article>
<div class="f6 color-fg-muted mt-2">  ← SIBLING (never checked)
    <span>★ 1,234</span>               ❌ Missing
</div>
```

### **Indeed**
```html
<div class="job-card">                 ← Found by DOM detector
    <h2>Job Title</h2>                 ✅ Extracted
</div>
<div class="metadata">                 ← SIBLING (never checked)
    <span>Salary</span>                ❌ Missing
    <span>Location</span>              ❌ Missing
</div>
```

---

## ✅ Proposed Solution: Context-Block Extraction

### **Concept**: Extract a "context block" that includes the container + siblings + parent context

```
┌─────────────────────────────────────────┐
│ <div class="list">  ← Parent context   │
│                                         │
│   <div class="item"> ← Main container  │
│     <h3>Title</h3>   ✅               │
│   </div>                                │
│                                         │
│   <div class="metadata"> ← Sibling     │
│     <span>Price</span>   ✅ Now found! │
│   </div>                                │
│                                         │
│   <div class="actions">  ← Sibling     │
│     <button>Buy</button> ✅ Now found! │
│   </div>                                │
│                                         │
└─────────────────────────────────────────┘

Extract the entire "context block" per item!
```

---

## 🔧 Implementation Strategy

### **Phase 1: Detect Context Block Boundaries**

Instead of just finding the repeating element, find the **repeating context block**.

```python
def detect_context_block(soup, main_container_selector):
    """
    Find the context block that includes:
    - Main container
    - Sibling elements (before and after)
    - Parent wrapper (if exists)
    """
    
    containers = soup.select(main_container_selector)
    
    # Analyze the parent structure
    first_container = containers[0]
    parent = first_container.parent
    
    # Check if siblings are consistent across all containers
    sibling_patterns = analyze_sibling_patterns(containers)
    
    # Determine context block selector
    if sibling_patterns['consistent']:
        # e.g., "div.list > *" (all children of parent)
        return {
            'type': 'parent_children',
            'selector': f'{parent.name}.{parent.get("class")[0]} > *',
            'group_by': main_container_selector,
            'include_siblings': True
        }
    else:
        # Fallback to original container
        return {
            'type': 'container_only',
            'selector': main_container_selector,
            'group_by': None,
            'include_siblings': False
        }
```

---

### **Phase 2: Enhanced HTML Sampling**

Send the LLM a sample that includes siblings, not just the container.

```python
def extract_context_sample(soup, context_block):
    """
    Extract HTML sample that includes full context.
    """
    
    if context_block['type'] == 'parent_children':
        # Get parent element (includes all children + siblings)
        containers = soup.select(context_block['group_by'])
        parent = containers[0].parent
        
        # Sample: 2-3 complete parent blocks
        sample = []
        for i, container in enumerate(containers[:3]):
            parent_block = container.parent
            sample.append(str(parent_block))
        
        return '\n'.join(sample)
    else:
        # Original approach
        containers = soup.select(context_block['selector'])
        return '\n'.join(str(c) for c in containers[:3])
```

---

### **Phase 3: Update LLM Prompt**

Guide the LLM to look at siblings, not just children.

```python
prompt = f"""
**CRITICAL: Data may be in SIBLINGS, not just children!**

You are extracting data from this HTML structure:

```html
{context_sample}
```

**Fields to extract**: {fields}

**WHERE TO LOOK:**
1. ✅ Inside the main container (children)
2. ✅ SIBLING elements (before or after the container)
3. ✅ Parent element (shared across items)
4. ✅ Adjacent <div>, <span>, <section> tags

**COMMON PATTERNS:**
- Stack Overflow: Votes in sibling <div class="stats">
- GitHub: Stars in sibling <div class="metadata">
- Indeed: Salary in sibling <div class="job-info">

**INSTRUCTIONS:**
For each field, provide:
1. CSS selector (relative to PARENT, not container)
2. Whether it's in container or sibling
3. Specific extraction method

**EXAMPLE OUTPUT:**
{{
    "title": {{
        "location": "container",  // Inside main container
        "selector": "h3.title"
    }},
    "votes": {{
        "location": "sibling",  // In adjacent sibling element
        "selector": ".stats .vote-count"
    }}
}}
"""
```

---

### **Phase 4: Modify Code Generation**

Generate code that groups siblings together.

```python
def generate_extraction_code_with_siblings(context_block, fields, field_locations):
    """
    Generate code that handles sibling-based layouts.
    """
    
    if context_block['type'] == 'parent_children':
        # Group siblings by parent
        code = f"""
from bs4 import BeautifulSoup

def extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    
    # Find all parent blocks
    parents = soup.select('{context_block['selector']}')
    
    for parent in parents:
        item = {{}}
        
        # Extract from main container
        container = parent.select_one('{context_block['group_by']}')
        if container:
            item['title'] = container.select_one('h3').get_text(strip=True)
        
        # Extract from sibling elements
        sibling = parent.select_one('.stats')
        if sibling:
            item['votes'] = sibling.select_one('.vote-count').get_text(strip=True)
        
        if item:
            results.append(item)
    
    return results
"""
    else:
        # Original approach
        code = generate_original_code(context_block, fields)
    
    return code
```

---

## 📊 Expected Impact

### **Before (Container-Only)**

| Site | Items | Quality | Issue |
|------|-------|---------|-------|
| Stack Overflow | 15 | 50% | Votes in sibling ❌ |
| GitHub | 11 | 33% | Stars in sibling ❌ |
| Indeed | 16 | 25% | Salary in sibling ❌ |

### **After (Context-Block)**

| Site | Items | Quality | Improvement |
|------|-------|---------|-------------|
| Stack Overflow | 15 | **90%+** | Sibling detected ✅ |
| GitHub | 25 | **85%+** | Sibling detected ✅ |
| Indeed | 20 | **80%+** | Sibling detected ✅ |

---

## 🎯 Why This Is Universal

### **Handles All Layout Patterns**

1. **Nested (current system works)**
   ```html
   <div class="item">
       <h3>Title</h3>
       <span>Price</span>
   </div>
   ```

2. **Sibling (new system handles)**
   ```html
   <div class="item">
       <h3>Title</h3>
   </div>
   <div class="meta">
       <span>Price</span>
   </div>
   ```

3. **Mixed (new system handles)**
   ```html
   <article class="item">
       <h3>Title</h3>  <!-- Inside -->
   </article>
   <div class="meta">
       <span>Price</span>  <!-- Sibling -->
   </div>
   <div class="actions">
       <button>Buy</button>  <!-- Sibling -->
   </div>
   ```

---

## 💰 Cost Impact

**No significant cost increase**:
- Same number of LLM calls
- Slightly larger HTML samples (2-3 parent blocks instead of 2-3 containers)
- More accurate extraction = fewer retries = **potential cost savings**

**Estimated**:
- Current: $0.005/scrape (with retries for failures)
- With context blocks: $0.006/scrape (larger samples, but fewer failures)
- **Net**: ~20% cost increase, but 2-3x higher success rate

---

## 🚀 Implementation Plan

### **Step 1: Add Sibling Detection** (2 hours)
- Modify `dom_pattern_detector.py` to analyze sibling patterns
- Detect if siblings are consistent across repeating elements

### **Step 2: Update HTML Sampling** (1 hour)
- Modify `smart_sampler.py` to include parent context
- Extract full "context blocks" instead of just containers

### **Step 3: Enhance LLM Prompts** (1 hour)
- Update `adaptive_dom_detector.py` prompts to mention siblings
- Add specific guidance for sibling-based layouts

### **Step 4: Update Code Generation** (2 hours)
- Modify `ai_generator.py` to handle sibling selectors
- Generate code that groups siblings by parent

### **Step 5: Test & Validate** (2 hours)
- Test on Stack Overflow, GitHub, Indeed
- Validate quality improvements (50% → 90%+)

**Total**: ~8 hours of implementation

---

## ✅ Success Criteria

After implementation, we should achieve:

- ✅ **Stack Overflow**: 90%+ quality (votes extracted)
- ✅ **GitHub Trending**: 85%+ quality (stars extracted)
- ✅ **Indeed**: 80%+ quality (salary extracted)
- ✅ **Universal**: Works on any site (nested, sibling, or mixed)
- ✅ **No regressions**: Sites that already work remain at 90%+

---

## 📋 Next Steps

**Option 1**: Implement context-block extraction (fixes Stack Overflow, GitHub, Indeed)  
**Option 2**: Test embedding cache first (demonstrates learning capability)  
**Option 3**: Move to proxy integration (fixes blocked sites)

**Recommendation**: Implement context-block extraction first - it's the **fundamental architectural fix** that will unlock the remaining 50% of sites.






