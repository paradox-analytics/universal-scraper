# Phase 2 Implementation Status

**Date**: December 26, 2025  
**Goal**: 3-Tier Model Selection + Deterministic Template Spec System

---

## ✅ Completed

### 1. Model Router (3-Tier Model Selection)
- **File**: `universal_scraper/core/model_router.py`
- **Features**:
  - Router tier: Fast classification/routing (gpt-4o-mini)
  - Template tier: Template generation (gpt-4o-mini, upgradeable)
  - Recovery tier: Complex cases (gpt-4o)
  - Cost estimation per tier
  - Automatic tier selection based on task type
- **Status**: ✅ Core implementation complete

### 2. Template Spec System
- **File**: `universal_scraper/core/template_spec.py`
- **Features**:
  - `TemplateSpec` dataclass with JSON serialization
  - `FieldSelector` with primary/fallback selectors
  - `PaginationConfig` for pagination rules
  - Normalizers and validators
  - Template validation
- **Status**: ✅ Core implementation complete

### 3. Deterministic Extractor
- **File**: `universal_scraper/core/deterministic_extractor.py`
- **Features**:
  - Executes template specs deterministically
  - CSS/XPath selector support
  - Normalizer application (currency, date, number parsing)
  - Validator execution
  - Fallback selector support
  - No LLM calls during extraction
- **Status**: ✅ Core implementation complete

---

## 🚧 Integration Required

### 4. AICodeGenerator Enhancement
- **Current**: Generates Python code (BeautifulSoup)
- **Needed**: Add option to generate template spec JSON
- **File**: `universal_scraper/core/ai_generator.py`
- **Changes Required**:
  1. Add `generate_template_spec()` method
  2. Use ModelRouter to select template tier model
  3. Set temperature=0 for deterministic output
  4. Require strict JSON schema
  5. Output TemplateSpec JSON instead of Python code

### 5. Scraper Integration
- **File**: `universal_scraper/core/scraper.py`
- **Changes Required**:
  1. Initialize ModelRouter
  2. Use template spec path when available
  3. Fall back to code generation if template spec fails
  4. Store template specs in cache
  5. Use DeterministicExtractor for template spec execution

---

## 📋 Implementation Plan

### Step 1: Enhance AICodeGenerator
```python
def generate_template_spec(
    self,
    cleaned_html: str,
    fields: List[str],
    url: Optional[str] = None,
    extraction_context: Optional[str] = None,
    structure_analysis: Optional[Dict[str, Any]] = None,
    model_router: Optional[ModelRouter] = None
) -> TemplateSpec:
    """
    Generate template spec JSON (deterministic)
    
    Uses template tier model with temperature=0
    Requires strict JSON schema
    """
    # Use template tier model
    model = model_router.get_model(ModelTier.TEMPLATE) if model_router else self.model_name
    
    # Generate template spec JSON
    # ... LLM call with strict JSON schema ...
    
    # Parse and validate
    template_spec = TemplateSpec.from_json(llm_response)
    
    return template_spec
```

### Step 2: Integrate into Scraper
```python
# In scraper.py __init__
self.model_router = ModelRouter(
    router_model="gpt-4o-mini",
    template_model="gpt-4o-mini",
    recovery_model="gpt-4o",
    api_key=api_key
)

# In scrape() method
# Check for template spec cache first
template_spec = await self._get_cached_template_spec(url, fields, structure_hash)

if template_spec:
    # Use deterministic extractor
    extractor = DeterministicExtractor()
    items = extractor.extract(cleaned_html, template_spec)
else:
    # Generate template spec
    template_spec = await self.ai_generator.generate_template_spec(
        cleaned_html, fields, url, context, structure_analysis, self.model_router
    )
    # Cache and extract
    await self._cache_template_spec(template_spec)
    items = extractor.extract(cleaned_html, template_spec)
```

---

## 🎯 Expected Benefits

### Performance
- **Template Spec Execution**: <50ms (deterministic, no LLM)
- **Template Generation**: <2s (template tier model)
- **Recovery Mode**: <10s (recovery tier model, rare)

### Cost Reduction
- **Router Tier**: ~$0.0001 per call (frequent, cheap)
- **Template Tier**: ~$0.001 per call (occasional, balanced)
- **Recovery Tier**: ~$0.01 per call (rare, expensive)
- **Overall**: 70-80% cost reduction vs single model

### Determinism
- Template specs are deterministic (temperature=0)
- Runtime extraction is deterministic (no LLM)
- Reproducible results across runs

---

## 📝 Next Steps

1. ✅ Model Router - DONE
2. ✅ Template Spec - DONE
3. ✅ Deterministic Extractor - DONE
4. 🔄 Enhance AICodeGenerator (in progress)
5. 🔄 Integrate into Scraper (pending)
6. 🔄 Test with 3 URLs (pending)

---

## 🔧 Technical Notes

### Template Spec Format
```json
{
  "template_id": "producthunt_com_products_v1",
  "page_fingerprint_features": {
    "repeating_element": ".product-item",
    "tag_paths": "div>div>article>h2",
    "class_patterns": "product-item=20"
  },
  "selectors": [
    {
      "field_name": "name",
      "primary": "h2.product-title",
      "fallbacks": ["h2", ".title"],
      "selector_type": "css",
      "normalizer": "extract_text",
      "required": true
    },
    {
      "field_name": "price",
      "primary": ".price",
      "fallbacks": [".cost", "[data-price]"],
      "selector_type": "css",
      "normalizer": "parse_currency",
      "required": false
    }
  ],
  "confidence": 0.95,
  "why_these_selectors": "Found repeating product-item containers with consistent structure"
}
```

### Model Tier Usage
- **Router**: "Is this a product listing page?" (<100ms)
- **Template**: "Generate selectors for name, price, image" (<2s)
- **Recovery**: "Previous selectors failed, find alternatives" (<10s, rare)

---

## ✅ Phase 2 Foundation Complete

The core components are implemented and ready for integration.
Next: Enhance AICodeGenerator and integrate into scraper flow.



