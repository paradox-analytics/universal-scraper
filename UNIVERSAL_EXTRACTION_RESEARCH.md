# Universal HTML Extraction - Deep Dive Research

## Objective
Understand how other successful universal scraping solutions handle HTML extraction and apply those learnings to fix our implementation.

---

## Research Summary: Competing Solutions

### 1. **ScrapeGraphAI** (Most Similar to Our Approach)

**Architecture:**
- **Graph-Based Nodes**: Modular pipeline (FetchNode → ParseNode → RAGNode → GenerateAnswerNode)
- **LLM-Driven**: Uses GPT/Claude to interpret HTML and extract data
- **RAG Approach**: Chunks HTML into smaller pieces, embeds them, then LLM extracts from relevant chunks

**Key Innovation:**
- **Direct LLM Extraction**: Instead of generating "patterns", they pass HTML directly to LLM with extraction prompt
- **Chunking Strategy**: Splits large HTML into manageable pieces (not just cleaning)
- **Graph Pipelines**: Different pipelines for different use cases (SmartScraperGraph, SearchGraph, etc.)

**How It Works:**
1. Fetch HTML (with JS rendering if needed)
2. Clean HTML (remove scripts/styles)
3. **Chunk HTML** into smaller pieces (4000-8000 tokens each)
4. Embed chunks (optional, for RAG)
5. Pass chunks + user prompt directly to LLM: "Extract {fields} from this HTML"
6. LLM returns structured JSON

**Pros:**
- Simple: No pattern generation, direct extraction
- Flexible: LLM adapts to any structure
- High quality: LLM understands context

**Cons:**
- Expensive: LLM call per page ($0.01-0.05 each)
- Slow: GPT-4 takes 5-15 seconds
- No caching: Each request is fresh LLM call

---

### 2. **Diffbot** (Commercial, API-Based)

**Architecture:**
- **Machine Learning Models**: Trained on millions of pages
- **Computer Vision**: Analyzes page visual structure, not just HTML
- **Pattern Library**: Maintains extraction patterns for 50+ content types
- **Automatic Type Detection**: Classifies page as article, product, discussion, etc.

**How It Works:**
1. Fetch page
2. Render visual representation
3. Classify page type (article, product, etc.)
4. Apply pre-trained extraction model for that type
5. Return structured data

**Pros:**
- Very fast (< 1 second)
- High accuracy (trained models)
- Universal (50+ page types)

**Cons:**
- Expensive ($299-999/month)
- Closed source (can't replicate)
- API dependency

---

### 3. **Parsera** (LLM-Based Extractor)

**Architecture:**
- **Pure LLM**: Relies entirely on LLM to extract from HTML
- **Minimal Preprocessing**: Just cleans HTML
- **Token Optimization**: Strips HTML to minimal structure

**How It Works:**
1. Fetch HTML
2. Strip to bare minimum (remove attributes, classes, keep structure)
3. Pass to GPT-4: "Extract {fields} from this simplified HTML"
4. Parse LLM output as JSON

**Pros:**
- Extremely simple architecture
- High quality extraction
- Flexible (works on anything)

**Cons:**
- Very expensive (full GPT-4 call per page)
- Slow
- No caching or optimization

---

### 4. **Firecrawl** (Scraping + LLM)

**Architecture:**
- **Hybrid**: Combines traditional scraping with LLM
- **Markdown Conversion**: Converts HTML to Markdown for LLM
- **Schema Enforcement**: Uses JSON schema to guide extraction

**How It Works:**
1. Fetch with anti-bot evasion
2. Convert HTML to clean Markdown
3. Pass Markdown + JSON schema to LLM
4. LLM fills in schema with extracted data

**Pros:**
- Markdown is much smaller than HTML (lower costs)
- Schema ensures consistent output
- Good for LLM consumption

**Cons:**
- Markdown conversion loses some structure
- Still expensive (LLM per page)
- Limited to text-heavy content

---

### 5. **AutoScraper** (Research Paper, arXiv:2404.12753)

**Architecture:**
- **Two-Stage Process**:
  1. Code generation (LLM generates BeautifulSoup code)
  2. Code execution (Runs generated code on HTML)
- **Iterative Refinement**: If extraction fails, regenerates code

**How It Works:**
1. User provides example/prompt
2. LLM generates Python extraction code
3. Execute code on page
4. If fails, show error to LLM, regenerate code
5. Repeat until successful

**Pros:**
- Generates reusable code
- Can cache/reuse code for similar pages
- Adaptive (fixes itself)

**Cons:**
- Code generation is slow
- Security risk (executing generated code)
- Iterative process can be expensive

---

## Key Insights from Research

### **Common Pattern: Direct LLM Extraction**

**What successful solutions do:**
```
HTML → Clean/Simplify → Pass to LLM → Get Structured Data
```

**What our system tried:**
```
HTML → Clean → Detect Patterns → Generate Pattern → Execute Pattern → Data
```

**The Problem with Our Approach:**
- Too many steps where things can go wrong
- Pattern generation is itself an extraction problem
- Semantic strategies are brittle
- No LLM sees the actual content during extraction

**Why Direct LLM Works Better:**
1. **LLM understands context**: Can distinguish between product listings and sidebar filters
2. **No intermediate abstraction**: Pattern generation is error-prone
3. **Adaptive**: LLM figures out structure on the fly
4. **Semantic understanding**: Knows "author" should be a username, not a timestamp

---

### **Critical Insight: HTML Chunking**

Most successful solutions **chunk HTML** instead of trying to extract from entire page:

**ScrapeGraphAI approach:**
```python
# 1. Split HTML into chunks (4000-8000 tokens each)
chunks = split_html_into_chunks(cleaned_html, max_tokens=6000)

# 2. For each chunk, ask LLM
for chunk in chunks:
    prompt = f"""
    Extract the following fields from this HTML:
    Fields: {fields}
    
    HTML:
    {chunk}
    
    Return only JSON with extracted data.
    """
    
    items = llm.extract(prompt)
    all_items.extend(items)
```

**Benefits:**
- Works with large pages (doesn't hit token limits)
- LLM can focus on smaller sections
- Parallelizable (process chunks in parallel)
- More accurate (less noise per chunk)

---

### **Critical Insight: Markdown Conversion**

**Firecrawl approach:**
```
HTML → Markdown → LLM
```

**Benefits:**
- Markdown is 10x smaller than HTML
- LLM trained on Markdown (better performance)
- Easier for LLM to parse
- Preserves semantic structure (headers, lists, etc.)

**Our current approach:**
```
HTML → Clean HTML → LLM pattern generation
```

**Problem:**
- Still sending messy HTML to LLM
- LLM has to parse HTML tags, classes, IDs
- Harder for LLM to understand content

---

### **Critical Insight: Schema-Guided Extraction**

**Firecrawl/Parsera approach:**
```python
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "price": {"type": "number"},
            "rating": {"type": "number"}
        }
    }
}

prompt = f"""
Extract data matching this schema:
{json.dumps(schema)}

From this HTML:
{html}
"""
```

**Benefits:**
- LLM knows exact expected structure
- Type validation built-in
- Consistent output format
- Can specify required vs optional fields

**Our current approach:**
- Just field names, no types or structure
- No validation on output
- LLM doesn't know if field should be string, number, array, etc.

---

## Recommended Architecture Changes

### **Option A: Direct LLM Extraction (ScrapeGraphAI-style)**

Replace our broken pattern generation with direct extraction:

```python
class DirectLLMExtractor:
    def extract(self, html: str, fields: List[str]) -> List[Dict]:
        # 1. Clean HTML
        cleaned = clean_html(html)
        
        # 2. Chunk into manageable pieces
        chunks = chunk_html(cleaned, max_tokens=6000)
        
        # 3. Extract from each chunk
        all_items = []
        for chunk in chunks:
            prompt = f"""
            Extract items with these fields: {fields}
            
            HTML:
            {chunk}
            
            Return JSON array of extracted items.
            """
            
            items = await llm.extract(prompt)
            all_items.extend(items)
        
        return all_items
```

**Pros:**
- Simple, proven approach
- High quality (LLM understands context)
- No pattern generation failures

**Cons:**
- $0.01-0.05 per page
- 5-15 seconds per page
- No caching (unless we cache responses)

---

### **Option B: HTML → Markdown → LLM (Firecrawl-style)**

Convert HTML to Markdown first:

```python
class MarkdownExtractor:
    def extract(self, html: str, fields: List[str]) -> List[Dict]:
        # 1. Convert to Markdown
        markdown = html_to_markdown(html)
        
        # 2. Create schema
        schema = self._create_schema(fields)
        
        # 3. Extract with schema
        prompt = f"""
        Extract data matching this schema:
        {json.dumps(schema)}
        
        From this Markdown:
        {markdown}
        
        Return JSON array.
        """
        
        return await llm.extract(prompt)
```

**Pros:**
- Smaller input (10x token reduction)
- Better LLM performance (trained on Markdown)
- Cheaper ($0.001-0.01 per page)

**Cons:**
- Markdown conversion loses some structure
- Not as accurate for complex layouts

---

### **Option C: Hybrid - Pattern Cache + Direct LLM Fallback**

Keep our pattern cache but use direct LLM as fallback:

```python
class HybridExtractor:
    def extract(self, html: str, fields: List[str]) -> List[Dict]:
        # 1. Try cached pattern (if available)
        embedding = generate_embedding(html)
        cached_pattern = pattern_cache.find_similar(embedding, fields)
        
        if cached_pattern:
            # Use fast pattern-based extraction
            return semantic_extractor.extract(html, cached_pattern)
        
        # 2. Fallback to direct LLM extraction
        items = await direct_llm_extract(html, fields)
        
        # 3. Optionally: Try to generate reusable pattern from results
        pattern = infer_pattern_from_results(html, items, fields)
        if pattern:
            pattern_cache.save(embedding, pattern, fields)
        
        return items
```

**Pros:**
- Fast for cached domains (0.1s, $0.0001)
- High quality for new domains (LLM extraction)
- Cost-effective (only expensive on first request)

**Cons:**
- Complex architecture
- Pattern inference is hard

---

## Immediate Action Plan

### **Phase 1: Test ScrapeGraphAI Locally (2 hours)**

Install and test their approach on our failing sources:

```bash
pip install scrapegraphai-py
```

Test on Amazon, eBay, Reddit, Hacker News with actual code to understand their extraction mechanism.

### **Phase 2: Implement Direct LLM Extraction (4 hours)**

Create new `DirectLLMExtractor` class:
- HTML chunking
- Schema-guided prompts
- Direct extraction (no patterns)

Test on our 6 sources to measure:
- Quality (data inspection)
- Cost (token usage)
- Speed (extraction time)

### **Phase 3: Compare Approaches (2 hours)**

| Approach | Quality | Cost/Page | Speed | Cacheability |
|----------|---------|-----------|-------|--------------|
| Current (Pattern) | 16% | $0.02 | 40s | ✅ Yes |
| Direct LLM | ? | $0.01-0.05 | 10-15s | ❌ No |
| Markdown + LLM | ? | $0.001-0.01 | 5-10s | ❌ No |
| Hybrid (Cache + LLM) | ? | $0.0001-0.05 | 0.1-15s | ✅ Yes |

### **Phase 4: Decide & Implement (8-16 hours)**

Based on Phase 2 results, choose best approach and fully implement.

---

## Next Steps

1. **Install ScrapeGraphAI** and run comparative tests
2. **Analyze their code** to understand implementation details
3. **Build DirectLLMExtractor** proof of concept
4. **Test on our 6 sources** with quality inspection
5. **Make data-driven decision** on architecture

---

## Open Questions

1. How does ScrapeGraphAI handle HTML chunking exactly?
2. What prompts do they use for extraction?
3. How do they handle token limits?
4. Do they have any caching/optimization?
5. How does Markdown conversion affect accuracy?
6. Can we combine caching with direct LLM extraction?

Let's find out by testing their actual implementation!




