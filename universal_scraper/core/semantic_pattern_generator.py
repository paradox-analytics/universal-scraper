"""
Semantic Pattern Generator - Creates semantic extraction patterns using LLM

Instead of generating brittle CSS selector code, this generates resilient
semantic patterns that describe WHAT to extract, not HOW.

Key Difference:
- Old: Generate Python code with CSS selectors → breaks on layout changes
- New: Generate semantic extraction patterns → resilient to layout changes
"""

import json
import logging
from typing import List, Dict, Any, Optional
import litellm

logger = logging.getLogger(__name__)


class SemanticPatternGenerator:
    """
    Generate semantic extraction patterns using LLM.
    
    These patterns are:
    1. More resilient than CSS selectors (semantic, not structural)
    2. Cacheable (reusable across similar websites)
    3. Human-readable (easy to debug/modify)
    4. LLM-free to execute (only generation needs LLM)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize semantic pattern generator.
        
        Args:
            api_key: API key for LLM provider (defaults to OPENAI_API_KEY env var)
            model_name: Model to use for pattern generation
        """
        import os
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model_name = model_name
        logger.info(f" Semantic Pattern Generator initialized (model={model_name})")
    
    async def generate_pattern(
        self,
        html_sample: str,
        fields,  # Can be List[str] or str (natural language)
        context: Optional[str] = None,
        repeating_containers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate semantic extraction pattern from HTML.
        
        Args:
            html_sample: Cleaned/sampled HTML
            fields: Either:
                - List of field names: ["title", "price", "rating"]
                - Natural language string: "Extract product names, prices, and ratings"
            context: Optional context about what data to extract
            repeating_containers: Optional list of detected container signatures
            
        Returns:
            Semantic pattern dict ready for SemanticExtractor
            
        Example return:
        {
            "title": {
                "primary": {"type": "heading", "position": "first"},
                "fallbacks": [
                    {"type": "bold_text", "min_length": 20},
                    {"type": "link_text"},
                    {"type": "attribute", "name": "data-title"}
                ],
                "validation": {"not_empty": true, "min_length": 3}
            },
            "price": {
                "primary": {"type": "currency"},
                "fallbacks": [
                    {"type": "attribute", "name": "data-price"},
                    {"type": "text_contains", "pattern": "\\$\\d+"}
                ]
            }
        }
        """
        # Handle natural language field specification
        if isinstance(fields, str):
            logger.info(f"  Natural language input detected: '{fields[:100]}...'")
            fields = await self._parse_natural_language_fields(fields, html_sample)
            logger.info(f"    Parsed to fields: {fields}")
        elif isinstance(fields, list) and len(fields) == 1 and len(fields[0]) > 50:
            # User might have put a long description as a single array element
            logger.info(f"  Natural language detected in array: '{fields[0][:100]}...'")
            fields = await self._parse_natural_language_fields(fields[0], html_sample)
            logger.info(f"    Parsed to fields: {fields}")
        
        logger.info(f" Generating semantic pattern for {len(fields)} fields")
        
        # Build prompt
        prompt = self._build_pattern_prompt(
            html_sample=html_sample,
            fields=fields,
            context=context,
            repeating_containers=repeating_containers
        )
        
        try:
            # Call LLM to generate semantic pattern
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing HTML structure and creating resilient data extraction patterns."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent patterns
                response_format={"type": "json_object"}
            )
            
            pattern_json = response.choices[0].message.content
            pattern = json.loads(pattern_json)
            
            logger.info(f" Generated semantic pattern with {len(pattern)} fields")
            logger.debug(f"Pattern: {json.dumps(pattern, indent=2)}")
            
            return pattern
            
        except Exception as e:
            logger.error(f" Failed to generate semantic pattern: {e}")
            # Return fallback pattern
            return self._generate_fallback_pattern(fields)
    
    def _build_pattern_prompt(
        self,
        html_sample: str,
        fields: List[str],
        context: Optional[str],
        repeating_containers: Optional[List[str]]
    ) -> str:
        """
        Build LLM prompt for semantic pattern generation.
        """
        prompt = f"""Analyze this HTML and generate a SEMANTIC EXTRACTION PATTERN for the following fields: {', '.join(fields)}

**CRITICAL: OUTPUT SEMANTIC PATTERNS, NOT CSS SELECTORS**

Instead of brittle CSS selectors like 'div.product > h2.title', output semantic strategies that describe WHAT to extract based on meaning, not structure.

HTML Sample:
```html
{html_sample[:15000]}  
```

"""
        
        if context:
            prompt += f"""
Extraction Context:
{context}

"""
        
        if repeating_containers:
            # Ensure repeating_containers is serializable (remove slice objects)
            serializable_containers = []
            for container in repeating_containers[:5]:
                if isinstance(container, dict):
                    # Filter out non-serializable values like slice objects
                    clean_container = {}
                    for key, value in container.items():
                        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            clean_container[key] = value
                        elif isinstance(value, slice):
                            # Convert slice to string representation
                            clean_container[key] = f"[{value.start}:{value.stop}]"
                        else:
                            clean_container[key] = str(value)
                    serializable_containers.append(clean_container)
                else:
                    serializable_containers.append(str(container))
            
            prompt += f"""
Detected Repeating Containers:
{json.dumps(serializable_containers, indent=2)}

"""
        
        prompt += """
For each field, provide:

1. **Primary Strategy**: The most reliable way to extract this field semantically
   - Use semantic extraction types (heading, bold_text, link_text, currency, image, date, etc.)
   - Do NOT use CSS selectors in primary strategy
   
2. **Fallback Strategies**: Alternative approaches if primary fails
   - List 2-3 fallbacks in order of preference
   - Only use CSS selectors as last resort
   
3. **Validation Rules** (optional): Constraints to validate extracted data
   - not_empty, min_length, max_length, pattern, type

Available Strategy Types:
- **heading**: Extract from h1-h6 tags (params: position="first"|"last")
- **bold_text**: Extract from <strong>/<b> tags (params: min_length=N)
- **link_text**: Extract from <a> tags (params: return="text"|"href", position="first"|"last")
- **attribute**: Extract from data-* or aria-* attributes (params: name="attr-name")
- **currency**: Extract currency values with $ € £ (params: symbols=["$", "€"])
- **number**: Extract numeric values (params: pattern=regex)
- **date**: Extract dates from <time> or text (auto-detects common formats)
- **image**: Extract from <img> tags (params: return="src"|"alt")
- **text_contains**: Extract text matching pattern (params: pattern=regex)
- **first_text**: Extract first non-empty text (params: min_length=N)
- **semantic_element**: Extract from semantic tags (params: tag="article"|"section"|etc)
- **css_selector**: LAST RESORT - Use CSS selector (params: selector=".class")

Output Format (JSON):
{
  "field_name": {
    "primary": {
      "type": "strategy_type",
      "param1": "value1",
      "param2": "value2"
    },
    "fallbacks": [
      {"type": "strategy_type", "param": "value"},
      {"type": "strategy_type", "param": "value"}
    ],
    "validation": {
      "not_empty": true,
      "min_length": 3
    }
  }
}

Examples:

**Good Semantic Pattern for "title":**
{
  "title": {
    "primary": {"type": "heading", "position": "first"},
    "fallbacks": [
      {"type": "bold_text", "min_length": 20},
      {"type": "link_text"},
      {"type": "attribute", "name": "data-title"}
    ],
    "validation": {"not_empty": true, "min_length": 3}
  }
}

**Good Semantic Pattern for "price":**
{
  "price": {
    "primary": {"type": "currency"},
    "fallbacks": [
      {"type": "attribute", "name": "data-price"},
      {"type": "text_contains", "pattern": "\\\\$\\\\d+\\\\.\\\\d{2}"}
    ]
  }
}

**Bad Pattern (don't do this):**
{
  "title": {
    "primary": {"type": "css_selector", "selector": "div.product-card > h2.title"}
  }
}

Now analyze the HTML and generate semantic patterns for: """ + ', '.join(fields) + """

Return ONLY valid JSON, no explanations."""

        return prompt
    
    def _generate_fallback_pattern(self, fields: List[str]) -> Dict[str, Any]:
        """
        Generate simple fallback pattern if LLM fails.
        
        This provides basic semantic strategies for common fields.
        """
        logger.warning(" Using fallback pattern (LLM generation failed)")
        
        pattern = {}
        
        for field in fields:
            field_lower = field.lower()
            
            # Title-like fields
            if any(keyword in field_lower for keyword in ['title', 'name', 'heading', 'product']):
                pattern[field] = {
                    "primary": {"type": "heading", "position": "first"},
                    "fallbacks": [
                        {"type": "bold_text", "min_length": 10},
                        {"type": "link_text"},
                        {"type": "first_text", "min_length": 5}
                    ],
                    "validation": {"not_empty": True, "min_length": 3}
                }
            
            # Price-like fields
            elif any(keyword in field_lower for keyword in ['price', 'cost', 'amount']):
                pattern[field] = {
                    "primary": {"type": "currency"},
                    "fallbacks": [
                        {"type": "attribute", "name": "data-price"},
                        {"type": "number", "pattern": r"\d+\.?\d*"}
                    ]
                }
            
            # Date-like fields
            elif any(keyword in field_lower for keyword in ['date', 'time', 'posted', 'published']):
                pattern[field] = {
                    "primary": {"type": "date"},
                    "fallbacks": [
                        {"type": "attribute", "name": "datetime"},
                        {"type": "first_text", "min_length": 5}
                    ]
                }
            
            # Image-like fields
            elif any(keyword in field_lower for keyword in ['image', 'img', 'photo', 'picture']):
                pattern[field] = {
                    "primary": {"type": "image", "return": "src"},
                    "fallbacks": [
                        {"type": "attribute", "name": "data-src"},
                        {"type": "attribute", "name": "src"}
                    ]
                }
            
            # Link-like fields
            elif any(keyword in field_lower for keyword in ['url', 'link', 'href']):
                pattern[field] = {
                    "primary": {"type": "link_text", "return": "href"},
                    "fallbacks": [
                        {"type": "attribute", "name": "data-url"},
                        {"type": "attribute", "name": "href"}
                    ]
                }
            
            # Description-like fields
            elif any(keyword in field_lower for keyword in ['description', 'desc', 'text', 'content', 'body']):
                pattern[field] = {
                    "primary": {"type": "first_text", "min_length": 20},
                    "fallbacks": [
                        {"type": "semantic_element", "tag": "article"},
                        {"type": "first_text", "min_length": 10}
                    ]
                }
            
            # Rating-like fields
            elif any(keyword in field_lower for keyword in ['rating', 'score', 'stars']):
                pattern[field] = {
                    "primary": {"type": "attribute", "name": "data-rating"},
                    "fallbacks": [
                        {"type": "number", "pattern": r"\d+\.?\d*"},
                        {"type": "first_text", "min_length": 1}
                    ]
                }
            
            # Generic field
            else:
                pattern[field] = {
                    "primary": {"type": "first_text", "min_length": 1},
                    "fallbacks": [
                        {"type": "heading"},
                        {"type": "link_text"}
                    ]
                }
        
        return pattern
    
    def validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validate that a semantic pattern is well-formed.
        
        Args:
            pattern: Pattern dict to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(pattern, dict):
            logger.error("Pattern must be a dictionary")
            return False
        
        valid_strategy_types = {
            'heading', 'bold_text', 'link_text', 'attribute', 'currency',
            'number', 'date', 'image', 'text_contains', 'first_text',
            'semantic_element', 'css_selector', 'xpath'
        }
        
        for field_name, field_pattern in pattern.items():
            # Check field pattern structure
            if not isinstance(field_pattern, dict):
                logger.error(f"Field '{field_name}' pattern must be a dict")
                return False
            
            # Check primary strategy
            primary = field_pattern.get('primary')
            if not primary or not isinstance(primary, dict):
                logger.error(f"Field '{field_name}' missing primary strategy")
                return False
            
            if primary.get('type') not in valid_strategy_types:
                logger.error(f"Field '{field_name}' has invalid primary strategy type: {primary.get('type')}")
                return False
            
            # Check fallbacks (optional)
            fallbacks = field_pattern.get('fallbacks', [])
            if not isinstance(fallbacks, list):
                logger.error(f"Field '{field_name}' fallbacks must be a list")
                return False
            
            for fallback in fallbacks:
                if not isinstance(fallback, dict):
                    logger.error(f"Field '{field_name}' fallback must be a dict")
                    return False
                
                if fallback.get('type') not in valid_strategy_types:
                    logger.error(f"Field '{field_name}' has invalid fallback strategy type")
                    return False
        
        logger.info(f" Pattern validated successfully ({len(pattern)} fields)")
        return True
    
    async def _parse_natural_language_fields(
        self,
        natural_language: str,
        html_sample: str
    ) -> List[str]:
        """
        Parse natural language field specification into structured field names.
        
        Examples:
        - "Extract product name, price and description" → ["product_name", "price", "description"]
        - "Get all job titles, companies, and locations" → ["job_title", "company", "location"]
        - "Scrape article headlines, authors, and publish dates" → ["headline", "author", "publish_date"]
        
        Args:
            natural_language: User's natural language description
            html_sample: HTML sample to provide context
            
        Returns:
            List of standardized field names
        """
        logger.info(" Parsing natural language field specification...")
        
        prompt = f"""The user wants to extract data from a website. They described what they want in natural language.
Your job is to parse their description and output a JSON array of standardized field names.

User's Request:
"{natural_language}"

HTML Context (first 3000 chars to understand what's available):
```html
{html_sample[:3000]}
```

Rules for field names:
1. Use lowercase_with_underscores format (e.g., "product_name", not "Product Name")
2. Be specific and clear (e.g., "publish_date" not just "date")
3. Common field names: title, name, product_name, price, description, author, date, url, image, rating, etc.
4. Analyze the HTML to understand what data is actually available
5. Output 3-8 fields typically (unless user is very specific)

Example outputs:
- User: "product info" → ["product_name", "price", "description", "image"]
- User: "all articles with title and author" → ["title", "author", "publish_date", "url"]
- User: "job listings" → ["job_title", "company", "location", "salary", "description"]

Output JSON format:
{{
  "fields": ["field1", "field2", "field3"]
}}"""

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,  # CRITICAL: Pass API key!
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction expert who converts natural language requests into structured field names."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_json = response.choices[0].message.content
            result = json.loads(result_json)
            fields = result.get("fields", [])
            
            if not fields or not isinstance(fields, list):
                logger.warning(f"  LLM returned invalid fields: {result}")
                # Fallback: extract words from natural language
                return self._extract_fields_from_text(natural_language)
            
            logger.info(f" Parsed {len(fields)} fields from natural language")
            return fields
            
        except Exception as e:
            logger.error(f" Failed to parse natural language fields: {e}")
            # Fallback: extract words from text
            return self._extract_fields_from_text(natural_language)
    
    def _extract_fields_from_text(self, text: str) -> List[str]:
        """
        Fallback method to extract field names from natural language text.
        Simple regex-based extraction.
        """
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 
                     'extract', 'get', 'scrape', 'fetch', 'find', 'all', 'every', 'each'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        fields = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Take first 5 unique words as fields
        fields = list(dict.fromkeys(fields))[:5]
        
        # Ensure we have at least some fields
        if not fields:
            fields = ['title', 'description', 'url']
        
        logger.info(f" Fallback extraction found: {fields}")
        return fields

