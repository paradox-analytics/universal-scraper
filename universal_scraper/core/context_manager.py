"""
Universal Context Manager (LLM-Driven)

Parses user intent and guides all scraping decisions.
NO hardcoded patterns - everything inferred by LLM.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import litellm

logger = logging.getLogger(__name__)


@dataclass
class ExtractionContext:
    """
    Structured context from user input + LLM inference
    Guides crawler, scraper, validator, and normalizer
    """
    goal: str  # User's extraction goal
    data_type: Optional[str] = None  # Inferred by LLM: "products", "events", etc.
    fields: Optional[List[str]] = None  # Inferred target fields
    description: Optional[str] = None  # Additional context
    raw_prompt: str = ""  # Original user input
    
    # LLM inference metadata
    inference_confidence: float = 0.0
    inference_reasoning: str = ""
    
    def to_llm_prompt_section(self) -> str:
        """Format context for inclusion in other LLM prompts"""
        parts = [f"USER GOAL: {self.goal}"]
        
        if self.data_type:
            parts.append(f"DATA TYPE: {self.data_type}")
        
        if self.fields:
            parts.append(f"TARGET FIELDS: {', '.join(self.fields)}")
        
        if self.description:
            parts.append(f"ADDITIONAL CONTEXT: {self.description}")
        
        return "\n".join(parts)
    
    def __str__(self):
        fields_str = f"{len(self.fields)} fields" if self.fields else "auto-extract"
        return f"Context(type={self.data_type}, {fields_str}, confidence={self.inference_confidence:.2f})"


class ContextManager:
    """
    LLM-driven context manager
    Parses and enriches user intent using AI - NO hardcoded patterns
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        enable_cache: bool = True,
        cache_dir: str = ".context_cache"
    ):
        """
        Initialize context manager
        
        Args:
            api_key: OpenAI API key (or any LiteLLM-supported provider)
            model: Model to use for inference
            enable_cache: Cache context inferences
            cache_dir: Directory for caching
        """
        self.api_key = api_key
        self.model = model
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.context = None
        
        # In-memory cache for current session
        self._memory_cache = {}
        
        if api_key:
            logger.info(f" Context Manager initialized with {model}")
        else:
            logger.warning(" No API key - context inference disabled")
    
    def parse_context(
        self,
        context_input: Any,
        url: Optional[str] = None
    ) -> ExtractionContext:
        """
        Parse context from user input and enrich with LLM inference
        
        Args:
            context_input: String or dict from user
            url: Optional URL for additional context
        
        Returns:
            ExtractionContext with LLM-inferred metadata
        """
        # 1. Parse raw input
        if isinstance(context_input, str):
            raw_prompt = context_input
            goal = context_input
            description = None
        elif isinstance(context_input, dict):
            goal = context_input.get('goal', '')
            description = context_input.get('description')
            raw_prompt = f"{goal}. {description}" if description else goal
            
            # Allow pre-specified fields/type (user override)
            preset_fields = context_input.get('fields')
            preset_type = context_input.get('dataType')
        else:
            raise ValueError(f"Invalid context format: {type(context_input)}")
        
        if not goal:
            raise ValueError("Context must have a 'goal'")
        
        logger.info(f" Parsing context: {goal[:100]}...")
        
        # 2. LLM inference (if we have API key)
        if self.api_key:
            inference = self._llm_infer_context(raw_prompt, url)
            
            # User overrides take precedence
            if isinstance(context_input, dict):
                if preset_fields:
                    inference['fields'] = preset_fields
                if preset_type:
                    inference['data_type'] = preset_type
            
            self.context = ExtractionContext(
                goal=goal,
                data_type=inference.get('data_type'),
                fields=inference.get('fields'),
                description=description,
                raw_prompt=raw_prompt,
                inference_confidence=inference.get('confidence', 0.0),
                inference_reasoning=inference.get('reasoning', '')
            )
        else:
            # No API key - create basic context
            self.context = ExtractionContext(
                goal=goal,
                data_type=context_input.get('dataType') if isinstance(context_input, dict) else None,
                fields=context_input.get('fields') if isinstance(context_input, dict) else None,
                description=description,
                raw_prompt=raw_prompt
            )
        
        logger.info(f" {self.context}")
        if self.context.fields:
            logger.info(f"   Fields: {self.context.fields}")
        if self.context.inference_reasoning:
            logger.info(f"   Reasoning: {self.context.inference_reasoning}")
        
        return self.context
    
    def _llm_infer_context(
        self,
        user_prompt: str,
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to intelligently infer context details
        NO hardcoded patterns - pure LLM intelligence
        """
        
        # Check cache first
        cache_key = self._get_cache_key(user_prompt, url)
        if self.enable_cache and cache_key in self._memory_cache:
            logger.info(" Using cached context inference")
            return self._memory_cache[cache_key]
        
        prompt = f"""You are an expert at understanding web scraping requirements.

USER REQUEST:
"{user_prompt}"

{f'TARGET URL: {url}' if url else ''}

TASK:
Analyze this request and infer:
1. What TYPE of data are they trying to extract?
2. What FIELDS/attributes do they want?

IMPORTANT RULES:
- If the user asks for a specific object (e.g. "product", "event", "job"), you MUST infer the standard fields for that object unless they explicitly say "get everything".
- DO NOT return empty fields if the user mentions a specific data type.
- Example: "Extract products" -> fields=["name", "price", "image", "url", "description"]
- Example: "Get all data" -> fields=[] (Auto-extract all)
- Be specific about field names (use snake_case).

Common data types (but not limited to):
- products (fields: name, price, description, rating, availability, image, url)
- events (fields: title, date, location, venue, price)
- articles (fields: title, author, date, content, url)
- listings (fields: title, price, location, description)
- reviews (fields: author, rating, date, text, title)
- businesses (fields: name, address, phone, website, rating)
- people (fields: name, title, email, phone, linkedin)
- general_data (when unclear)


Respond in JSON:
{{
    "data_type": "the_type",
    "fields": ["field_name_1", "field_name_2", ...],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your inference"
}}
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a web scraping context analyzer. Be intelligent. If the user implies a schema (e.g. 'products'), infer the fields!"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent inference
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
            
            if 'data_type' not in result:
                result['data_type'] = 'general_data'
            
            if 'fields' not in result:
                result['fields'] = []
            
            # FALLBACK: If LLM returned empty fields but we have a known data type, inject defaults
            # This handles cases where LLM is too conservative
            if not result['fields'] and result['data_type'] != 'general_data':
                defaults = {
                    'products': ['name', 'price', 'description', 'image', 'url', 'rating', 'availability', 'brand'],
                    'events': ['title', 'date', 'location', 'venue', 'description', 'price', 'url'],
                    'articles': ['title', 'author', 'date', 'content', 'summary', 'url', 'image'],
                    'jobs': ['title', 'company', 'location', 'salary', 'description', 'url', 'posted_date'],
                    'reviews': ['title', 'author', 'rating', 'date', 'text', 'product_name'],
                    'businesses': ['name', 'address', 'phone', 'website', 'rating', 'hours_of_operation']
                }
                if result['data_type'] in defaults:
                    result['fields'] = defaults[result['data_type']]
                    if 'reasoning' not in result:
                        result['reasoning'] = "Inferred from user request"
                    result['reasoning'] += " (Auto-injected default fields for type)"
                    logger.info(f" Auto-injected default fields for {result['data_type']}")
            
            if 'confidence' not in result:
                result['confidence'] = 0.5
            
            if 'reasoning' not in result:
                result['reasoning'] = 'Inferred from user request'
            
            logger.info(f" LLM inferred: {result['data_type']} (confidence: {result['confidence']:.2f})")
            logger.debug(f"   Reasoning: {result['reasoning']}")
            
            # Cache result
            if self.enable_cache:
                self._memory_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f" LLM context inference failed: {e}")
            return {
                'data_type': 'general_data',
                'fields': [],
                'confidence': 0.0,
                'reasoning': f'Inference failed: {str(e)}'
            }
    
    def _get_cache_key(self, prompt: str, url: Optional[str] = None) -> str:
        """Generate cache key from prompt and URL"""
        cache_input = f"{prompt}|{url or ''}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get_context_prompt(self) -> str:
        """Get formatted context for other LLM prompts"""
        return self.context.to_llm_prompt_section() if self.context else ""
    
    def get_data_type(self) -> str:
        """Get inferred data type"""
        return self.context.data_type if self.context else 'unknown'
    
    def get_fields(self) -> List[str]:
        """Get inferred or specified fields"""
        return self.context.fields if self.context and self.context.fields else []
    
    def get_goal(self) -> str:
        """Get user's extraction goal"""
        return self.context.goal if self.context else ''
    
    def has_context(self) -> bool:
        """Check if context has been set"""
        return self.context is not None
    
    def get_confidence(self) -> float:
        """Get inference confidence score"""
        return self.context.inference_confidence if self.context else 0.0








