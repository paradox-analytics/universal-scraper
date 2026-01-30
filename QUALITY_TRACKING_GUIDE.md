# Quality Tracking & Cache Reuse Guide

## Overview

The test script (`test_three_urls_local.py`) now includes comprehensive quality tracking and cache reuse monitoring to ensure data extraction meets the 90% quality threshold consistently.

## Quality Threshold: 90%

All extractions must meet a **90% quality score** to be considered successful. The quality score is calculated from:
1. **Scraper's internal quality metric** (from metadata) - preferred
2. **Field coverage average** - fallback if scraper metric unavailable

### Quality Calculation

- **Field Coverage**: Percentage of items that have each expected field populated
- **Quality Score**: Average of all field coverage percentages
- **Threshold Check**: `quality_score >= 0.90`

## Issue Categories

### 1. Quality Below 90% Threshold
- **Gap Analysis**: Shows how far below 90% the extraction is
- **Example**: `Quality: 81.0% (gap: 9.0%)`

### 2. Missing Fields (0% Coverage)
- Fields that are completely absent from all extracted items
- **Action Required**: Field may not exist on page or extraction logic needs improvement

### 3. Low Coverage Fields (<50% Coverage)
- Fields that are present in less than half of items
- **Action Required**: Extraction logic may be inconsistent or field is optional

## Cache Reuse Tracking

The system tracks cache usage to identify when patterns/code are being repurposed:

### Cache Types

1. **Code Cache**: Cached extraction code for faster subsequent runs
2. **Pattern Cache**: Cached extraction patterns learned from previous extractions
3. **Direct LLM Cache**: Cached LLM extraction results (domain + fields match)

### Cache Indicators

- `code_cached`: Boolean - whether extraction code was reused
- `pattern_cached`: Boolean - whether extraction pattern was reused
- `direct_llm_cached`: Boolean - whether Direct LLM result was cached
- `extraction_source`: String - 'json', 'html', or 'auto_pagination'
- `early_exit`: Boolean - whether optimization skipped HTML extraction

### Expected Behavior

- **First Run**: Cache misses expected, full extraction
- **Subsequent Runs**: Cache hits expected, faster extraction
- **Pattern Learning**: Patterns should be learned and reused for similar pages

## Test Configuration

### Pagination
- **Disabled**: `enable_auto_pagination=False`
- Only single pages are tested (no pagination)

### Quality Mode
- **Mode**: `balanced` (good compromise between quality and speed)
- **Alternatives**: `conservative` (≥70% fields, highest quality) or `aggressive` (≥30% fields, most items)

### Extraction Method
- **Primary**: Direct LLM extraction (like ScrapeGraphAI)
- **Fallback**: HTML extraction if JSON fails

## Logging Output

### Per-Site Results
```
Results for ProductHunt:
  Success: True
  Items Extracted: 30
  Quality Score: 81.4%
  ❌ QUALITY: Below 90% threshold (gap: 8.6%)
  
  Cache Usage:
    Extraction Source: html
    Code Cached: False
    Pattern Cached: False
    Direct LLM Cached: True
    Early Exit: True
  
  Missing Fields: maker, image
  Low Coverage Fields: comments
```

### Summary Section
```
FUNDAMENTAL ISSUES TRACKING
================================================================================

❌ QUALITY BELOW 90% THRESHOLD (2 sites):
  • ProductHunt: 81.4% (gap: 8.6%)
    Missing: maker, image
    Low coverage: comments
  • Metacritic: 81.0% (gap: 9.0%)
    Low coverage: platform

📚 CACHE REUSE PATTERNS (2 sites):
  • ProductHunt:
    - Source: html
    - Direct LLM cached: True
  • Metacritic:
    - Source: html
    - Direct LLM cached: True
```

## Troubleshooting

### Quality Below 90%

1. **Check Missing Fields**
   - Verify fields exist on the page
   - Check if field names match page structure
   - Review extraction context description

2. **Check Low Coverage Fields**
   - Some fields may be optional (e.g., "platform" for movies)
   - Consider making optional fields truly optional
   - Review extraction logic for consistency

3. **Improve Extraction Context**
   - More specific context descriptions help LLM extraction
   - Include examples of expected data format

### No Cache Reuse

1. **First Run**: Expected - no cache exists yet
2. **Subsequent Runs**: Should see cache hits
   - Check cache directory exists and is writable
   - Verify same domain + fields are being used
   - Check cache TTL hasn't expired

### Pattern Not Learning

1. **Check Pattern Cache**
   - Patterns are learned after successful extractions
   - Requires multiple items with consistent structure
   - May not generate if structure is too variable

## Best Practices

1. **Run Tests Twice**: First run establishes cache, second run shows reuse
2. **Review Missing Fields**: May indicate page structure changes
3. **Monitor Quality Gaps**: Track improvements over time
4. **Cache Analysis**: Identify which sites benefit most from caching

## Next Steps

1. Run test script: `python3 test_three_urls_local.py`
2. Review quality scores for all 3 sites
3. Identify fields causing quality gaps
4. Improve extraction context or field definitions
5. Verify cache reuse on second run



