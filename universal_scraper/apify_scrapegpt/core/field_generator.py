"""
Natural Language Field Generator

Converts natural language prompts to structured field definitions.
Inspired by Oxylabs AI Scraper's schema generation.

Universal approach: Works for any website/domain.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class NaturalLanguageFieldGenerator:
    """
    Generate structured fields from natural language prompts.
    
    Example:
        generator = NaturalLanguageFieldGenerator(api_key="...")
        fields = await generator.generate_fields(
            prompt="I want product names, prices in USD, and star ratings",
            url="https://example.com/products"
        )
        # Returns: ['product_name', 'price', 'rating']
    """
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        """
        Initialize field generator.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o-mini is sufficient)
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize LLM client
        try:
            from litellm import acompletion
            self.llm_client = acompletion
        except ImportError:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        logger.info(f" Natural Language Field Generator initialized (model={model})")
    
    async def generate_fields(
        self,
        prompt: str,
        url: Optional[str] = None,
        return_descriptions: bool = False
    ) -> Union[List[str], Dict[str, str]]:
        """
        Convert natural language prompt to structured fields.
        
        Universal approach: Works for any domain/website by analyzing
        the natural language description and optional URL context.
        
        Args:
            prompt: Natural language description (e.g., "I want product names and prices")
            url: Optional URL for domain context
            return_descriptions: If True, return Dict[field_name, description]
            
        Returns:
            List of field names or Dict of {field_name: description}
            
        Example:
            Input: "I want game titles, developers, prices in USD, and genres as arrays"
            Output: ['title', 'developer', 'price', 'genre']
            
            Or with return_descriptions=True:
            Output: {
                'title': 'Game title or name',
                'developer': 'Game developer/publisher',
                'price': 'Price in USD currency',
                'genre': 'Array of game genres'
            }
        """
        logger.info(f" Generating fields from prompt: {prompt[:80]}...")
        
        # Extract domain context if URL provided
        domain_context = ""
        if url:
            domain = urlparse(url).netloc
            domain_context = f"\nDomain context: {domain} ({self._infer_domain_type(url)})"
        
        # Build LLM prompt
        llm_prompt = f"""Convert this natural language request into structured field definitions.

USER REQUEST: "{prompt}"{domain_context}

INSTRUCTIONS:
1. Extract field names from the request
2. Use snake_case, lowercase field names
3. Keep field names concise (1-2 words max)
4. Infer data types from context:
   - "prices" → number
   - "ratings" → number
   - "genres", "tags", "categories" → array
   - "name", "title", "description" → string
   - "date", "time" → string (ISO format)
5. Handle implicit requirements:
   - "prices in USD" → field is "price", description mentions currency
   - "star ratings" → field is "rating", description mentions scale
   - "as array" or "as list" → note in description

OUTPUT FORMAT (valid JSON):
{{
    "fields": [
        {{
            "name": "field_name",
            "description": "Clear description of what this field contains",
            "type": "string|number|array",
            "examples": ["example1", "example2"]
        }}
    ]
}}

EXAMPLES:

Input: "I want product titles and prices"
Output:
{{
    "fields": [
        {{"name": "title", "description": "Product title or name", "type": "string", "examples": ["iPhone 15", "Nike Air Max"]}},
        {{"name": "price", "description": "Product price", "type": "number", "examples": ["999.99", "89.95"]}}
    ]
}}

Input: "Get job titles, companies, locations, and salaries"
Output:
{{
    "fields": [
        {{"name": "job_title", "description": "Job position or role", "type": "string", "examples": ["Software Engineer", "Product Manager"]}},
        {{"name": "company", "description": "Company or employer name", "type": "string", "examples": ["Google", "Meta"]}},
        {{"name": "location", "description": "Job location (city, state, or remote)", "type": "string", "examples": ["San Francisco, CA", "Remote"]}},
        {{"name": "salary", "description": "Salary or compensation", "type": "string", "examples": ["$120k-150k", "$80,000/year"]}}
    ]
}}

Now generate the output for the user's request:
"""
        
        try:
            # Call LLM
            response = await self.llm_client(
                model=self.model,
                messages=[{"role": "user", "content": llm_prompt}],
                api_key=self.api_key,
                temperature=0.3,  # Low temperature for consistency
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed = json.loads(content)
            fields_data = parsed.get('fields', [])
            
            logger.info(f"    Generated {len(fields_data)} fields")
            
            # Return format based on user preference
            if return_descriptions:
                result = {
                    field['name']: field.get('description', '')
                    for field in fields_data
                }
                for field in fields_data:
                    logger.info(f"      • {field['name']}: {field.get('description', '')[:60]}...")
                return result
            else:
                result = [field['name'] for field in fields_data]
                logger.info(f"      Fields: {', '.join(result)}")
                return result
                
        except Exception as e:
            logger.error(f"    Field generation failed: {e}")
            raise RuntimeError(f"Failed to generate fields from prompt: {e}")
    
    def _infer_domain_type(self, url: str) -> str:
        """
        Infer domain type from URL for better context.
        
        Universal heuristics based on common patterns.
        """
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
        
        # E-commerce
        if any(x in domain for x in ['shop', 'store', 'amazon', 'ebay', 'etsy', 'walmart']):
            return 'e-commerce'
        
        # Jobs
        if any(x in domain for x in ['job', 'career', 'indeed', 'linkedin']) or 'jobs' in path:
            return 'job board'
        
        # News
        if any(x in domain for x in ['news', 'cnn', 'bbc', 'techcrunch', 'nytimes']):
            return 'news/media'
        
        # Social
        if any(x in domain for x in ['reddit', 'twitter', 'facebook', 'instagram']):
            return 'social media'
        
        # Real estate
        if any(x in domain for x in ['zillow', 'realtor', 'redfin', 'trulia']):
            return 'real estate'
        
        # Forums
        if any(x in domain for x in ['stackoverflow', 'forum', 'discourse']):
            return 'forum/community'
        
        # Default
        return 'general website'

