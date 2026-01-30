"""
LLM Data Validator (Context-Aware)

Validates that extracted data matches the user's extraction goal.
Prevents false positives (e.g., cart config when user wants products).
"""

import json
import logging
from typing import List, Dict, Any, Optional
import litellm
from .context_manager import ExtractionContext

logger = logging.getLogger(__name__)


class LLMDataValidator:
    """
    Uses LLM to validate extracted data matches user's intent
    Critical for universal scraping - prevents extracting wrong data
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        enable_cache: bool = True
    ):
        """
        Initialize data validator
        
        Args:
            api_key: OpenAI API key (or any LiteLLM-supported provider)
            model: Model to use for validation
            enable_cache: Cache validation results
        """
        self.api_key = api_key
        self.model = model
        self.enable_cache = enable_cache
        self._cache = {}
        
        logger.info(f" Data Validator initialized with {model}")
    
    def validate_extraction(
        self,
        items: List[Dict[str, Any]],
        url: str,
        context: ExtractionContext
    ) -> Dict[str, Any]:
        """
        Validate that extracted items match user's extraction goal
        
        Args:
            items: Extracted data items
            url: Source URL
            context: User's extraction context
        
        Returns:
            {
                "is_target_data": bool,
                "confidence": float,
                "reasoning": str,
                "detected_type": str,
                "suggestion": str
            }
        """
        
        if not items:
            return {
                'is_target_data': False,
                'confidence': 1.0,
                'reasoning': 'No items extracted',
                'detected_type': 'empty',
                'suggestion': 'Try different extraction method'
            }
        
        # Check cache
        cache_key = self._get_cache_key(items, url, context)
        if self.enable_cache and cache_key in self._cache:
            logger.info(" Using cached validation result")
            return self._cache[cache_key]
        
        # Prepare sample for LLM (limit size)
        sample_items = items[:3]  # Show up to 3 items
        
        # Truncate large values
        truncated_sample = []
        for item in sample_items:
            truncated = {}
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 200:
                    truncated[key] = value[:200] + "..."
                elif isinstance(value, (list, dict)) and len(str(value)) > 200:
                    truncated[key] = str(value)[:200] + "..."
                else:
                    truncated[key] = value
            truncated_sample.append(truncated)
        
        # Calculate field match rate (heuristic)
        field_match_rate = self._calculate_field_match_rate(items, context)
        
        prompt = f"""You are an expert at validating web scraping results.

{context.to_llm_prompt_section()}

URL: {url}

EXTRACTED ITEMS (sample of {len(items)} total):
{json.dumps(truncated_sample, indent=2)}

Field match analysis: {field_match_rate:.1%} of requested fields present

VALIDATION TASK:
Does this extracted data match the user's goal: "{context.goal}"?

Is this the PRIMARY {context.data_type} data that the user wants?
Or is it something else (metadata, config, navigation, analytics, etc.)?

IMPORTANT - Use these thresholds:
 ACCEPT if:
   - Data type is correct ({context.data_type}) AND 60%+ of requested fields are present
   - OR data type is correct AND there are 10+ substantial items
   - OR this is clearly the main page content

 REJECT if:
   - Obviously wrong type (footer when user wants events, cart config when user wants products)
   - Analytics/tracking data only
   - Navigation/menu items only
   - Cookie/consent banners
   - Empty or trivial data

Think:
1. Is the data TYPE correct ({context.data_type})?
2. Are 60%+ of the requested fields present (current: {field_match_rate:.1%})?
3. Is this substantive content (not just metadata)?

Respond in JSON:
{{
    "is_target_data": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation focusing on type match and field coverage",
    "detected_type": "products|events|footer_links|cart_config|navigation|metadata|etc",
    "suggestion": "html_extraction|try_next_json_source|looks_good|need_different_url"
}}

Be pragmatic - accept partial matches if data type is correct and most fields are present.
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a web scraping validation expert. Be critical and precise. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent validation
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content
            
            # Validate response structure
            if not isinstance(result, dict):
                raise ValueError("LLM returned non-dict response")
            
            # Ensure all required fields exist
            result.setdefault('is_target_data', False)
            result.setdefault('confidence', 0.5)
            result.setdefault('reasoning', 'No reasoning provided')
            result.setdefault('detected_type', 'unknown')
            result.setdefault('suggestion', 'try_different_method')
            
            # Log validation result
            status_icon = "" if result['is_target_data'] else ""
            logger.info(f"{status_icon} Validation: {result['is_target_data']} (confidence: {result['confidence']:.2f})")
            logger.info(f"   Detected: {result['detected_type']}")
            logger.info(f"   Reasoning: {result['reasoning']}")
            
            # Cache result
            if self.enable_cache:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f" LLM validation failed: {e}")
            # Default to accepting data if validation fails (fail-open)
            return {
                'is_target_data': True,
                'confidence': 0.3,
                'reasoning': f'Validation failed: {str(e)} - Defaulting to accepting data',
                'detected_type': 'unknown',
                'suggestion': 'validation_error'
            }
    
    def _calculate_field_match_rate(
        self,
        items: List[Dict],
        context: ExtractionContext
    ) -> float:
        """
        Calculate what % of requested fields are present in extracted items
        Returns 0.0-1.0
        """
        if not items or not context.fields:
            return 0.5  # Neutral if no fields specified
        
        # Get all fields from first item (representative)
        extracted_fields = set(items[0].keys()) if items else set()
        
        # Normalize field names for comparison
        extracted_lower = {f.lower() for f in extracted_fields}
        requested_lower = {f.lower() for f in context.fields}
        
        # Count exact matches
        exact_matches = len(extracted_lower & requested_lower)
        
        # Count fuzzy matches (field name contains requested field or vice versa)
        fuzzy_matches = 0
        for req in requested_lower:
            for ext in extracted_lower:
                if req in ext or ext in req:
                    fuzzy_matches += 1
                    break
        
        # Use best match count
        match_count = max(exact_matches, fuzzy_matches)
        
        # Calculate rate
        if len(context.fields) == 0:
            return 0.5  # Neutral if no fields requested
        
        rate = match_count / len(context.fields)
        return min(rate, 1.0)  # Cap at 100%
    
    def _get_cache_key(
        self,
        items: List[Dict],
        url: str,
        context: ExtractionContext
    ) -> str:
        """Generate cache key from items structure and context"""
        import hashlib
        
        # Create signature from item structure (fields only, not values)
        if items:
            item_signature = json.dumps(sorted(items[0].keys()))
        else:
            item_signature = "empty"
        
        cache_input = f"{url}|{context.goal}|{item_signature}|{len(items)}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def quick_validate(
        self,
        items: List[Dict],
        expected_type: str
    ) -> bool:
        """
        Quick heuristic validation without LLM (for performance)
        Returns True if data looks plausible
        """
        if not items:
            return False
        
        # Check if we have multiple items (good sign)
        if len(items) < 1:
            return False
        
        # Check if items have reasonable field count
        if items:
            field_count = len(items[0])
            if field_count < 2:  # Too few fields
                return False
            if field_count > 100:  # Suspiciously many fields
                return False
        
        # Passed basic checks
        return True

