# Direct LLM Extraction Optimization Analysis

## Problem Identified

**Current Issue:** 101 minutes for a single page (Chewy.com)
- Web Unblocker returns 6.8MB of HTML
- HTML is passed to Direct LLM **WITHOUT cleaning**
- Creates 646 chunks (way too many!)
- Even with parallel processing (BATCH_SIZE=4), that's 162 batches
- Sequential bottlenecks + rate limits = extremely slow

## Root Cause

Looking at `scraper.py` line 777-801:
```python
# IMPORTANT: Pass RAW HTML to direct LLM extractor
# The HybridMarkdownExtractor inside does its own cleaning
clean_result = self.html_cleaner.clean(html)
logger.info(f"   HTML reduced: {clean_result['reduction_percent']:.1f}%")

# But then passes RAW HTML:
direct_llm_items = await self.direct_llm_extractor.extract(
    html=html,  # RAW HTML - ❌ NOT CLEANED!
    ...
)
```

**The HTML cleaner is called but its output is NOT used!**

## How ScrapeGraphAI/Parsera Handle This

Based on research:
1. **Pre-process HTML aggressively** - Remove scripts, styles, nav, footer, ads BEFORE chunking
2. **Smart chunking** - Focus on content areas, not entire page
3. **Larger batch sizes** - Process 10-20 chunks in parallel (not 4)
4. **Content-aware chunking** - Split at product/item boundaries, not arbitrary HTML tags

## Optimizations Needed

### 1. Use Cleaned HTML (CRITICAL)
- Pass `clean_result['html']` instead of raw `html`
- Should reduce 6.8MB → ~500KB-1MB (85-90% reduction)
- Reduces chunks from 646 → ~50-100 chunks

### 2. Increase Batch Size
- Current: BATCH_SIZE = 4
- Recommended: BATCH_SIZE = 10-20 (for better parallelism)
- With 50-100 chunks: 5-10 batches instead of 162

### 3. Smarter Chunking
- Current: Splits at arbitrary HTML tags (`</div>`, `</article>`)
- Better: Focus on product/item containers
- Use semantic boundaries (product cards, list items)

### 4. Content-Aware Processing
- Detect product listing areas first
- Extract only relevant sections
- Skip navigation, footer, ads entirely

## Expected Impact

**Before:**
- 6.8MB HTML → 646 chunks → 162 batches → 101 minutes

**After (with cleaned HTML):**
- 6.8MB → cleaned → ~800KB → ~80 chunks → 8 batches → **~10-15 minutes**

**After (with cleaned HTML + larger batches):**
- 6.8MB → cleaned → ~800KB → ~80 chunks → 4 batches (BATCH_SIZE=20) → **~5-8 minutes**

## Implementation Priority

1. **HIGH:** Use cleaned HTML (immediate 10x speedup)
2. **MEDIUM:** Increase BATCH_SIZE to 10-20
3. **LOW:** Smarter chunking (content-aware)







