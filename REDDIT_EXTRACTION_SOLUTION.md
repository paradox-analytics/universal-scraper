# Reddit Extraction Solution

## Problem Summary

Reddit uses custom HTML elements (`<shreddit-post>`) with data stored in **HTML attributes**, not nested elements:

```html
<shreddit-post 
    post-title="Scraping Walmart store specific aisle data for a product"
    author="jpcoder"
    score="2"
    comment-count="4"
    created-timestamp="2025-11-10T16:39:28.815000+0000">
</shreddit-post>
```

## Why Both Approaches Fail

### 1. **Our Code Generation Approach** ❌
- AI generates code like: `post.select_one('.title')` 
- Looks for nested elements, not attributes
- Returns `None` for all fields

### 2. **ScrapeGraphAI's Direct LLM Approach** ❌  
- Converts HTML → Markdown (loses attributes)
- Markdown shows: " votes • comments" (no numbers)
- LLM can't extract what isn't there

## The Solution

**For attribute-based sites, skip AI and parse directly:**

```python
from bs4 import BeautifulSoup

def extract_reddit_posts(html):
    soup = BeautifulSoup(html, 'html.parser')
    posts = soup.find_all('shreddit-post')
    
    items = []
    for post in posts:
        item = {
            'title': post.get('post-title'),
            'author': post.get('author'),
            'upvotes': post.get('score'),
            'comments_count': post.get('comment-count'),
            'permalink': post.get('permalink'),
            'created': post.get('created-timestamp'),
            'subreddit': post.get('subreddit-name')
        }
        items.append(item)
    
    return items
```

## Test Results

### ✅ Direct Attribute Extraction (Our Solution)
- Extracted: **13 posts**
- All fields populated: ✅ title, ✅ author, ✅ upvotes, ✅ comments

### ⚠️ ScrapeGraphAI Approach  
- Extracted: **10 posts**
- Partial data: ✅ title, ✅ author, ❌ upvotes (null), ❌ comments (0)

### ❌ Code Generation (Before Fix)
- Extracted: **28 posts** (found elements)
- No data: ❌ All fields null (wrong selectors)

## Recommendations

### 1. **Pattern Detection System**
Add a pre-check before AI generation:

```python
def detect_extraction_pattern(html):
    """Detect if site uses attribute-based data storage"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Check for custom elements with data attributes
    custom_elements = soup.find_all(lambda tag: '-' in tag.name)
    
    if custom_elements:
        # Check if they have data in attributes
        sample = custom_elements[0]
        attrs = sample.attrs
        
        # If many attributes, likely attribute-based
        if len(attrs) > 5:
            return 'attribute_based'
    
    return 'nested_elements'
```

### 2. **Hybrid Extraction Strategy**

```python
def smart_extract(html, url, fields):
    pattern = detect_extraction_pattern(html)
    
    if pattern == 'attribute_based':
        # Use direct attribute extraction
        return extract_from_attributes(html, fields)
    else:
        # Use AI code generation
        return generate_and_execute_code(html, fields)
```

### 3. **Improve AI Prompt**
Add explicit instructions about attributes:

```
IMPORTANT: Check if data is stored in HTML ATTRIBUTES first!

Example for Reddit:
- ❌ DON'T: post.select_one('h3').text
- ✅ DO: post.get('post-title')

Many modern sites use:
- Custom elements: <shreddit-post>, <product-card>
- Data attributes: data-title, data-price, data-id
- Element attributes: score, author, permalink
```

## Root Cause Analysis

### Why AI Missed Attributes

1. **Sample didn't include posts** - Posts were at position 45,000+, sample showed first 8,000 chars
2. **AI looks for visible patterns** - Trained on nested HTML, not attribute-based storage
3. **Markdown conversion loses attributes** - html2text strips custom elements entirely

### Fixes Applied

1. ✅ **Smart content sampling** - Find content markers (posts/products) and sample from there
2. ✅ **Added attribute extraction example** - Prompt now includes custom element example
3. ✅ **Increased sample size** - From 8K to 15K chars
4. ⚠️ **Still needs pattern detection** - Should skip AI for obvious attribute-based sites

## When to Use Each Approach

### Use Direct Attribute Extraction:
- Custom web components (`<custom-element>`)
- Heavy use of data attributes (`data-*`)
- Modern frameworks (Lit, Stencil, Web Components)
- Sites like: Reddit, some SPAs

### Use AI Code Generation:
- Traditional nested HTML
- Class/ID-based selectors
- Standard semantic HTML
- Sites like: Most e-commerce, blogs, news

### Use Direct LLM (ScrapeGraphAI style):
- Complex layouts hard to describe
- Mixed content types
- When AI can "see" all data in text
- Emergency fallback

## Conclusion

Neither approach (code generation nor direct LLM) handles attribute-based extraction well without explicit support. The solution is:

1. **Detect the pattern** (attributes vs. nested)
2. **Route accordingly** (direct parsing vs. AI)
3. **Improve prompts** to teach AI about attributes
4. **Smart sampling** to ensure AI sees actual content

For Reddit specifically: Parse `<shreddit-post>` attributes directly. It's faster, cheaper, and more reliable than AI extraction.







