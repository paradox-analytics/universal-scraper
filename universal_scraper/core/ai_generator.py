"""
AI Code Generator
Generates BeautifulSoup extraction code using multiple AI providers
"""

import os
import logging
from typing import Dict, Any, List, Optional
import litellm
from litellm import completion

logger = logging.getLogger(__name__)


class AICodeGenerator:
    """Generates extraction code using AI (OpenAI, Gemini, Claude, etc.)"""
    
    # Default model preferences
    DEFAULT_MODELS = {
        'openai': 'gpt-4o-mini',
        'gemini': 'gemini-2.0-flash-exp',
        'claude': 'claude-3-haiku-20240307'
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Initialize AI Code Generator
        
        Args:
            api_key: API key for AI provider (auto-detects from env if None)
            model_name: Model to use (auto-selects if None)
            temperature: Generation temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or self._detect_api_key()
        self.model_name = model_name or self._detect_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set API key in environment for litellm
        if self.api_key:
            self._set_api_key_env()
        
        logger.info(f"🤖 AI Generator initialized: {self.model_name}")
    
    def generate_extraction_code(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate BeautifulSoup extraction code
        
        Args:
            cleaned_html: Cleaned HTML to analyze
            fields: Fields to extract
            url: Source URL for context
            
        Returns:
            Dict with 'code', 'explanation', 'model_used' keys
        """
        logger.info(f"🤖 Generating extraction code for {len(fields)} fields...")
        
        # Build prompt
        prompt = self._build_prompt(cleaned_html, fields, url)
        
        try:
            # Generate code using litellm
            response = completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert web scraping developer. Generate clean, efficient BeautifulSoup extraction code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            # Extract code from response
            code = self._extract_code_from_response(generated_text)
            
            logger.info(f"✅ Generated {len(code)} characters of code")
            
            return {
                'code': code,
                'explanation': generated_text,
                'model_used': self.model_name,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {str(e)}")
            raise
    
    def _build_prompt(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str] = None
    ) -> str:
        """Build prompt for AI code generation"""
        
        # Limit HTML size for prompt (take first 5000 chars)
        html_sample = cleaned_html[:5000]
        if len(cleaned_html) > 5000:
            html_sample += "\n... (truncated)"
        
        prompt = f"""Generate Python BeautifulSoup code to extract the following fields from this HTML:

Fields to extract: {', '.join(fields)}

HTML structure:
```html
{html_sample}
```

Requirements:
1. Return a function called `extract_data(soup)` that takes a BeautifulSoup object
2. The function should return a list of dictionaries, where each dict has the requested fields
3. Handle cases where fields might be missing (use None or empty string)
4. Use efficient selectors (CSS selectors preferred)
5. Include error handling for robust extraction
6. If the HTML contains repeating structures (like product cards), extract ALL of them
7. Add brief comments explaining the extraction logic

Example format:
```python
def extract_data(soup):
    items = []
    
    # Find all container elements
    containers = soup.select('.product-card')
    
    for container in containers:
        item = {{}}
        
        # Extract field1
        field1_elem = container.select_one('.field1')
        item['field1'] = field1_elem.text.strip() if field1_elem else None
        
        items.append(item)
    
    return items
```

Generate the extraction code now. Only output the Python code, nothing else."""
        
        if url:
            prompt = f"URL: {url}\n\n" + prompt
        
        return prompt
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from AI response"""
        # Try to find code between ```python and ```
        import re
        
        # Pattern 1: ```python ... ```
        pattern1 = r'```python\n(.*?)```'
        match = re.search(pattern1, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: ``` ... ``` (no language specified)
        pattern2 = r'```\n(.*?)```'
        match = re.search(pattern2, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: Just return everything (assume it's all code)
        # Clean up if there are markdown artifacts
        cleaned = response_text.strip()
        if cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned[3:-3].strip()
            if cleaned.startswith('python'):
                cleaned = cleaned[6:].strip()
        
        return cleaned
    
    def _detect_api_key(self) -> Optional[str]:
        """Auto-detect API key from environment"""
        # Check for various API keys
        keys = [
            ('OPENAI_API_KEY', 'openai'),
            ('GEMINI_API_KEY', 'gemini'),
            ('ANTHROPIC_API_KEY', 'claude'),
        ]
        
        for env_var, provider in keys:
            api_key = os.getenv(env_var)
            if api_key:
                logger.info(f"🔑 Detected {provider} API key from environment")
                return api_key
        
        logger.warning("⚠️ No API key found in environment")
        return None
    
    def _detect_model(self) -> str:
        """Auto-detect model based on available API keys"""
        # Check which API keys are available
        if os.getenv('OPENAI_API_KEY'):
            return self.DEFAULT_MODELS['openai']
        elif os.getenv('GEMINI_API_KEY'):
            return self.DEFAULT_MODELS['gemini']
        elif os.getenv('ANTHROPIC_API_KEY'):
            return self.DEFAULT_MODELS['claude']
        
        # Default to OpenAI (will fail if no key)
        return self.DEFAULT_MODELS['openai']
    
    def _set_api_key_env(self) -> None:
        """Set API key in environment based on model"""
        if 'gpt' in self.model_name.lower():
            os.environ['OPENAI_API_KEY'] = self.api_key
        elif 'gemini' in self.model_name.lower():
            os.environ['GEMINI_API_KEY'] = self.api_key
        elif 'claude' in self.model_name.lower():
            os.environ['ANTHROPIC_API_KEY'] = self.api_key
    
    def validate_generated_code(self, code: str) -> bool:
        """
        Validate generated code (syntax check)
        
        Args:
            code: Generated Python code
            
        Returns:
            True if valid
        """
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"❌ Invalid code syntax: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            'model': self.model_name,
            'provider': self._get_provider_from_model(),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def _get_provider_from_model(self) -> str:
        """Detect provider from model name"""
        model_lower = self.model_name.lower()
        
        if 'gpt' in model_lower:
            return 'openai'
        elif 'gemini' in model_lower:
            return 'google'
        elif 'claude' in model_lower:
            return 'anthropic'
        else:
            return 'unknown'

