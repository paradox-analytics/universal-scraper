# Universal Scraper - Architecture Principles

## Core Insight

> "The main learning from the last 3 years of LLM research is that **next token prediction works surprisingly well for agentic tasks**. That's the actual breakthrough result. The exact way it's done is secondary."

This means:
- **Autoregressive/iterative generation** is the key mechanism, not any specific architecture
- Transformers are just one implementation - the principle is more general
- We should design for **progressive refinement**, not single-shot extraction

---

## Implications for Web Scraping

### 1. Iterative Extraction > Single-Shot

Instead of trying to extract everything perfectly in one pass:

```
❌ Old approach: HTML → LLM → Complete JSON (one shot)

✅ New approach: HTML → LLM → Partial JSON → LLM → Refined JSON → LLM → Complete JSON
```

Each "step" builds on the previous, like next-token prediction builds on previous tokens.

### 2. Self-Correction Through Context

The model should see its own previous outputs:

```python
# Parsera-style: pass previous items as context
previous_items = []
for chunk in chunks:
    items = extract(chunk, context=previous_items[-5:])
    previous_items.extend(items)
```

This is autoregressive extraction - each output becomes input for the next step.

### 3. Progressive Refinement

Start coarse, then refine:

```
Pass 1: Extract titles (easy, high confidence)
Pass 2: Extract prices (add to existing items)
Pass 3: Extract ratings (fill remaining gaps)
Pass 4: Validate and fix inconsistencies
```

Each pass has full context of previous passes.

### 4. Quality Through Iteration, Not Architecture

ScrapeGraphAI's graph architecture isn't the secret - it's that they:
- Make multiple passes
- Let the model see previous outputs
- Allow self-correction

We can achieve this simpler:

```python
# Simple iterative refinement (no complex graphs needed)
items = initial_extract(html, fields)

for i in range(max_iterations):
    quality = calculate_quality(items)
    if quality >= threshold:
        break
    
    # Give model its own output + guidance
    items = refine_extract(html, fields, 
        previous_attempt=items,
        feedback=f"Quality was {quality}%, improve these fields: {low_quality_fields}"
    )
```

---

## Architectural Changes

### Current State

```
HTML → Clean → Chunk → Parallel LLM calls → Deduplicate → Result
```

### Proposed: Iterative Refinement

```
HTML → Clean → Initial Extract (single pass) →
  ↓
Quality Check ← (iterate if < threshold)
  ↓
Refinement Pass (with previous output as context) →
  ↓
Quality Check ← (iterate if < threshold)  
  ↓
Final Result
```

### Key Principles

1. **Context accumulation**: Each step sees all previous outputs
2. **Self-correction**: Model can fix its own mistakes when shown them
3. **Progressive difficulty**: Extract easy fields first, hard fields later
4. **Iteration > parallelism**: For quality, sequential refinement beats parallel extraction

---

## Implementation Priorities

### High Impact (Aligned with Core Insight)

1. **Iterative refinement loop** - Let model self-correct
2. **Full context passing** - Previous outputs inform next extraction
3. **Quality-driven iteration** - Stop when good enough, not after fixed steps

### Lower Priority (Architecture Details)

- Graph-based workflows (complex, secondary)
- Specific chunking strategies (implementation detail)
- Model-specific optimizations (secondary)

---

## Code Changes Needed

### 1. Add Refinement Loop to DirectLLMExtractor

```python
async def extract_with_refinement(self, html, fields, max_iterations=3):
    items = await self._extract_single_pass(html, fields)
    
    for i in range(max_iterations):
        quality = self._calculate_quality(items, fields)
        if quality >= 0.8:  # Good enough
            break
        
        # Self-correct with context
        items = await self._refine_extraction(
            html, fields, 
            previous_items=items,
            iteration=i+1
        )
    
    return items
```

### 2. Add Refinement Prompt

```python
def _build_refinement_prompt(self, html, fields, previous_items):
    """Let model see and fix its previous output"""
    
    # Calculate what's missing
    missing = self._find_missing_fields(previous_items, fields)
    
    return f"""
    You previously extracted this data:
    {json.dumps(previous_items, indent=2)}
    
    The following fields are incomplete or missing:
    {missing}
    
    Re-examine the content and provide improved extraction:
    {html}
    """
```

---

## Implementation Status ✅

### Implemented: `_extract_with_refinement()`

```python
async def _extract_with_refinement(self, html, fields, context, quality_mode, 
                                    max_iterations=3, quality_threshold=0.7):
    """
    Iterative refinement - mimics next-token prediction for extraction.
    Each output informs the next step until quality threshold met.
    """
    items = await self._extract_single_pass(html, fields, context, quality_mode)
    quality = self._calculate_quality(items, fields)
    
    for iteration in range(2, max_iterations + 1):
        if quality >= quality_threshold:
            break
        
        missing_fields = self._find_incomplete_fields(items, fields)
        refined_items = await self._refine_extraction(
            html, fields, items, missing_fields, context, quality_mode
        )
        
        new_quality = self._calculate_quality(refined_items, fields)
        if new_quality > quality:
            items = refined_items
            quality = new_quality
        else:
            break  # No improvement, stop
    
    return items
```

### Key Files Modified

- `universal_scraper/core/direct_llm_extractor.py`
  - Added `_extract_with_refinement()` - iterative refinement loop
  - Added `_refine_extraction()` - shows model its own output
  - Added `_find_incomplete_fields()` - identifies what to refine
  - Added `_build_refinement_prompt()` - context-aware prompt
  - Added `_get_refinement_system_prompt()` - refinement guidance

---

## Summary

The breakthrough isn't transformers or graphs - it's that **autoregressive, iterative refinement** works for complex tasks. 

Our scraper now:
1. ✅ **Iterates** rather than single-shot
2. ✅ **Passes context** from previous extractions  
3. ✅ **Self-corrects** by showing the model its own output
4. ✅ **Stops when quality is met**, not after fixed steps

This is simpler than ScrapeGraphAI's graph architecture but captures the essential insight about why LLMs work for agentic tasks.

