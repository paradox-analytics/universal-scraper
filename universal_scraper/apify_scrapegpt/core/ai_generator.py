"""
AI Code Generator
Generates BeautifulSoup extraction code using multiple AI providers
"""

import os
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import litellm
from litellm import completion

# Optional: HTML to Markdown conversion (like ScrapeGraphAI)
try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False
    logging.warning("html2text not installed. Install with: pip install html2text")

logger = logging.getLogger(__name__)


class AICodeGenerator:
    """Generates extraction code using AI (OpenAI, Gemini, Claude, etc.)"""
    
    # Default model preferences
    # NOTE: Reverted to GPT-4o-mini - testing showed model quality is not the bottleneck
    # The real issue is DOM pattern detection (giving wrong selectors to the LLM)
    DEFAULT_MODELS = {
        'openai': 'gpt-4o-mini',  # Cost-efficient, issue is DOM detection not model quality
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
        
        logger.info(f" AI Generator initialized: {self.model_name}")
    
    def generate_extraction_code(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str] = None,
        extraction_context: Optional[str] = None,
        structure_analysis: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        field_hints: Optional[Dict[str, Dict[str, Any]]] = None  # NEW: Semantic field mappings
    ) -> Dict[str, Any]:
        """
        Generate BeautifulSoup extraction code with multi-iteration refinement
        
        IMPROVEMENTS:
        1. Uses structure_analysis to guide generation (from ScrapeGraphAI)
        2. Multi-iteration refinement with error feedback (from ScrapeGraphAI)
        3. Tests generated code and fixes errors automatically
        4. Uses semantic field mappings for better field understanding (NEW)
        
        Args:
            cleaned_html: Cleaned HTML to analyze
            fields: Fields to extract
            url: Source URL for context
            extraction_context: User's extraction goal
            structure_analysis: HTML structure analysis to guide generation
            max_iterations: Maximum refinement iterations
            field_hints: Semantic field mappings (from UniversalFieldMapper)
            
        Returns:
            Dict with 'code', 'explanation', 'model_used', 'iterations' keys
        """
        logger.info(f" Generating extraction code for {len(fields)} fields...")
        
        if structure_analysis:
            logger.info(f"   Using structure analysis: {structure_analysis.get('repeating_element')}")
        
        if field_hints:
            logger.info(f"   Using semantic field mappings for {len(field_hints)} fields")
        
        # Try iterative refinement
        best_code = None
        best_result = None
        errors_history = []
        
        for iteration in range(max_iterations):
            logger.info(f"   Iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Generate code
                code = self._generate_code_single_attempt(
                    cleaned_html,
                    fields,
                    url,
                    extraction_context,
                    structure_analysis,
                    previous_errors=errors_history,
                    field_hints=field_hints  # NEW: Pass semantic hints
                )
                
                # Test the code
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(cleaned_html, 'html.parser')
                
                # Execute code in sandbox
                namespace = {'soup': soup, 'BeautifulSoup': BeautifulSoup}
                exec(code, namespace)
                
                if 'extract_data' in namespace:
                    result = namespace['extract_data'](soup)
                    
                    if result and len(result) > 0:
                        #  ENHANCED: Validate that fields aren't mostly/all None (NULL VALUE CHECK)
                        first_item = result[0]
                        if isinstance(first_item, dict):
                            non_null_values = [v for v in first_item.values() if v is not None and v != '']
                            null_fields = [k for k, v in first_item.items() if v is None or v == '']
                            total_fields = len(first_item)
                            null_ratio = len(null_fields) / total_fields if total_fields > 0 else 1.0
                            
                            # Define key fields that should NEVER be null
                            key_fields = ['title', 'name', 'repository', 'author', 'product', 'item']
                            null_key_fields = [f for f in null_fields if f.lower() in key_fields]
                            
                            #  ENHANCED: Trigger if >40% of fields are null (lowered from 50%)
                            # OR if ANY key fields are null
                            # Example: GitHub repository=None -> RETRY (key field)
                            # Example: Craigslist title="text", price=None, location=None -> 67% null -> RETRY
                            should_retry = False
                            retry_reason = ""
                            
                            if null_ratio > 0.4:  # More than 40% of fields are null
                                should_retry = True
                                retry_reason = f"{len(null_fields)}/{total_fields} fields ({null_ratio*100:.0f}%) are NULL"
                            elif null_key_fields:  # Key fields are null
                                should_retry = True
                                retry_reason = f"Key fields are NULL: {', '.join(null_key_fields)}"
                            
                            if should_retry:
                                error_msg = f"Code returned {len(result)} items but {retry_reason}"
                                error_msg += f"\n   Null fields: {', '.join(null_fields)}"
                                error_msg += "\n   This usually means:"
                                error_msg += "\n   - CSS selectors are wrong (check class names, hierarchy)"
                                error_msg += "\n   - Data is in HTML attributes instead of text (use .get('attribute'))"
                                error_msg += "\n   - Data is in a different element (check parent/sibling elements)"
                                
                                # UNIVERSAL FIX: Add specific attribute extraction guidance when null ratio is high
                                if null_ratio > 0.5:
                                    error_msg += "\n\n    HIGH NULL RATIO DETECTED - TRY ATTRIBUTE EXTRACTION:"
                                    error_msg += "\n   - Check data-* attributes: elem.get('data-author'), elem.get('data-score')"
                                    error_msg += "\n   - Check aria-* attributes: elem.get('aria-label'), elem.get('aria-valuetext')"
                                    error_msg += "\n   - Check itemprop attributes: elem.get('itemprop'), elem['content']"
                                    error_msg += "\n   - Check custom attributes: elem.get('score'), elem.get('count')"
                                    error_msg += "\n   - For custom elements like <shreddit-post>, data is usually in attributes!"
                                    error_msg += "\n\n    EXAMPLE FOR CUSTOM ELEMENTS:"
                                    error_msg += "\n   author = elem.get('author') or elem.get('data-author')"
                                    error_msg += "\n   score = elem.get('score') or elem.get('data-score')"
                                
                                if null_key_fields:
                                    error_msg += f"\n   - CRITICAL: Key fields ({', '.join(null_key_fields)}) should NEVER be null!"
                                
                                logger.warning(f"    {error_msg}")
                                errors_history.append(error_msg)
                                best_code = code
                                best_result = (len(result), len(non_null_values))
                                continue  # Try next iteration
                            
                            # If SOME fields are None (but <40% and no key fields), that's okay (partial success)
                            elif len(null_fields) > 0:
                                logger.warning(f"    {len(null_fields)}/{total_fields} fields are null: {', '.join(null_fields[:3])} (acceptable)")
                        
                        #  NEW: Check for single-item extraction when multiple expected (SINGLE ITEM CHECK)
                        # If structure analysis detected many elements but we only got 1, that's suspicious
                        if structure_analysis and len(result) == 1:
                            expected_count = structure_analysis.get('pattern_count', 0)
                            if expected_count > 10:  # If we expected 10+ but got only 1
                                error_msg = f"Code returned only 1 item but structure analysis found {expected_count} repeating elements"
                                error_msg += f"\n   This usually means code used .find() or .find_one() instead of .find_all() or .select()"
                                error_msg += f"\n   Or the selector is too specific (e.g., 'article.featured' instead of 'article')"
                                logger.warning(f"    {error_msg}")
                                errors_history.append(error_msg)
                                best_code = code
                                best_result = (len(result), len(non_null_values) if isinstance(first_item, dict) else len(result))
                                continue  # Try next iteration
                        
                        logger.info(f"    Generated working code ({len(result)} items)")
                        return {
                            'code': code,
                            'explanation': f'Generated with {iteration + 1} iterations',
                            'model_used': self.model_name,
                            'iterations': iteration + 1,
                            'test_result_count': len(result)
                        }
                    else:
                        error_msg = "Code executed but returned 0 items"
                        logger.warning(f"    {error_msg}")
                        errors_history.append(error_msg)
                        best_code = code  # Keep as fallback
                else:
                    error_msg = "Generated code missing extract_data function"
                    logger.warning(f"    {error_msg}")
                    errors_history.append(error_msg)
                    
            except SyntaxError as e:
                error_msg = f"Syntax error: {str(e)}"
                logger.warning(f"    {error_msg}")
                errors_history.append(error_msg)
            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                logger.warning(f"    {error_msg}")
                errors_history.append(error_msg)
                best_code = code if 'code' in locals() else best_code
        
        # If all iterations failed, return best attempt
        if best_code:
            logger.warning(f" All iterations failed, returning best attempt")
            return {
                'code': best_code,
                'explanation': f'Best attempt after {max_iterations} iterations',
                'model_used': self.model_name,
                'iterations': max_iterations,
                'errors': errors_history
            }
        else:
            raise ValueError(f"Code generation failed after {max_iterations} iterations")
    
    def _generate_code_single_attempt(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str],
        extraction_context: Optional[str],
        structure_analysis: Optional[Dict[str, Any]],
        previous_errors: List[str] = None,
        field_hints: Optional[Dict[str, Dict[str, Any]]] = None  # NEW: Semantic field mappings
    ) -> str:
        """
        Generate code for a single attempt (with error feedback and semantic field hints)
        """
        # OPTIONAL: Convert HTML to Markdown (ScrapeGraphAI approach)
        content_for_llm = cleaned_html
        content_format = "HTML"
        
        # Check if HTML has custom elements (tags with hyphens) - they need attributes!
        import re
        has_custom_elements_in_html = bool(re.search(r'<[a-z]+-[a-z-]+', cleaned_html))
        
        # Only convert to Markdown if:
        # 1. HTML2TEXT is available
        # 2. Structure says nested_elements (not attributes/mixed)
        # 3. NO custom elements detected (markdown strips attributes!)
        # 4. NO field_mappings with CSS selectors (they need HTML structure!)
        has_field_mappings = False
        if structure_analysis:
            selectors = structure_analysis.get('key_selectors', {})
            if isinstance(selectors, dict):
                field_mappings = selectors.get('field_mappings', {})
                has_field_mappings = bool(field_mappings)
        
        # CRITICAL FIX: NEVER convert to Markdown for code generation!
        # Code generation ALWAYS needs actual HTML with CSS selectors to work.
        # Markdown conversion is ONLY for LLM direct extraction fallback (which handles it separately).
        # 
        # Previous bug: Converting to Markdown when data_location='nested_elements' caused:
        # - eBay: 0 items (despite correct li.s-card detection)
        # - GitHub: 0 items
        # - Any site with nested elements failed
        #
        # if (HTML2TEXT_AVAILABLE and  # <-- REMOVED THIS ENTIRE BLOCK
        #     structure_analysis and 
        #     structure_analysis.get('data_location') == 'nested_elements' and
        #     not has_custom_elements_in_html and
        #     not has_field_mappings):
        #     ...convert to markdown... <-- THIS WAS THE BUG!
        #
        # Solution: Always keep HTML for code generation!
        logger.info("    Keeping HTML format (required for CSS selectors)")
        
        if has_custom_elements_in_html:
            logger.info("    Keeping HTML format (custom elements detected)")
        elif has_field_mappings:
            logger.info("    Keeping HTML format (CSS selectors from structure analysis)")
        
        # Build prompt with structure analysis, error feedback, and semantic field hints
        prompt = self._build_prompt(
            content_for_llm,
            fields,
            url,
            extraction_context,
            content_format,
            structure_analysis,
            previous_errors,
            field_hints  # NEW: Semantic field mappings
        )
        
        # Generate code
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
        code = self._extract_code_from_response(generated_text)
        
        # Post-process: Fix common CSS selector issues with field names containing spaces
        code = self._fix_css_selectors_with_spaces(code, fields)
        # Post-process: Fix CSS selectors with special characters (like ^ at start of class names)
        code = self._sanitize_css_selectors(code)
        
        return code
    
    def _sanitize_css_selectors(self, code: str) -> str:
        """
        UNIVERSAL fix for CSS selectors with special characters at the start of class names.
        Converts invalid selectors like `.^__asset-list--wuiW8` to attribute selectors.
        """
        import re
        
        # Pattern: Match class selectors starting with special characters (^, ~, @, etc.)
        # Matches: .^__asset-list--wuiW8 or div.^__asset-list--wuiW8
        pattern = r'([\w\s>+~]*?)\.([\^~@\$%&\*\+\|\[\]\(\)\{\}\\\/])([\w\-_]+)'
        
        def replace_special_char_selector(match):
            prefix = match.group(1)  # Everything before the dot (e.g., "div " or "")
            special_char = match.group(2)  # The special character (e.g., "^")
            class_name = match.group(3)  # The class name after special char (e.g., "__asset-list--wuiW8")
            
            # Convert to attribute selector: [class*="^__asset-list--wuiW8"]
            # Using contains (*=) to match the class name anywhere in the class attribute
            return f'{prefix}[class*="{special_char}{class_name}"]'
        
        code = re.sub(pattern, replace_special_char_selector, code)
        
        return code
    
    def _fix_css_selectors_with_spaces(self, code: str, fields: List[str]) -> str:
        """
        UNIVERSAL fix for CSS selectors with spaces (invalid CSS syntax).
        Converts ANY class/ID selector with spaces to attribute selectors.
        
        This is a universal fix that works for ANY field name with spaces or special chars.
        
        Example:
            '.est\\. market\\ value' -> '[data-est-market-value]'
            '.product name' -> '[data-product-name]'
            '#my field' -> '[id="my-field"]'
        """
        import re
        
        # UNIVERSAL Pattern: Find ANY CSS selector with backslash-space pattern
        # This catches invalid selectors like: .field\\. name\\ value, .field\. name\ value
        # Pattern matches: . or # followed by identifier, then backslash-space, then identifier
        
        # Pattern 1: Class selector with literal backslash-space: .field\. name\ value
        # Matches: .identifier\ identifier (where \ is literal backslash)
        pattern1 = re.compile(
            r'\.([a-zA-Z0-9_-]+)\\\s+([a-zA-Z0-9_-]+(?:\s*\\\s*[a-zA-Z0-9_-]+)*)(?=\s|,|\)|\[|$)',
            re.IGNORECASE
        )
        def replace_class_selector(match):
            # Extract parts and normalize to attribute name
            parts = [match.group(1)] + re.split(r'\\\s+', match.group(2))
            normalized = '-'.join(p.lower() for p in parts if p)
            return f'[data-{normalized}]'
        
        code = pattern1.sub(replace_class_selector, code)
        
        # Pattern 2: Class selector with escaped backslashes: .field\\. name\\ value
        # Matches: .identifier\\ identifier (where \\ is escaped backslash in Python string)
        pattern2 = re.compile(
            r'\.([a-zA-Z0-9_-]+)\\\\s+([a-zA-Z0-9_-]+(?:\s*\\\\s+[a-zA-Z0-9_-]+)*)(?=\s|,|\)|\[|$)',
            re.IGNORECASE
        )
        def replace_escaped_class_selector(match):
            parts = [match.group(1)] + re.split(r'\\\\s+', match.group(2))
            normalized = '-'.join(p.lower() for p in parts if p)
            return f'[data-{normalized}]'
        
        code = pattern2.sub(replace_escaped_class_selector, code)
        
        # Pattern 3: ID selector: #field\. name\ value
        pattern3 = re.compile(
            r'#([a-zA-Z0-9_-]+)\\\s+([a-zA-Z0-9_-]+(?:\s*\\\s*[a-zA-Z0-9_-]+)*)(?=\s|,|\)|\[|$)',
            re.IGNORECASE
        )
        def replace_id_selector(match):
            parts = [match.group(1)] + re.split(r'\\\s+', match.group(2))
            normalized = '-'.join(p.lower() for p in parts if p)
            return f'[id="{normalized}"]'
        
        code = pattern3.sub(replace_id_selector, code)
        
        # Pattern 4: Attribute selector with spaces: [data-field\\. name\\ value]
        pattern4 = re.compile(
            r'\[data-([a-zA-Z0-9_-]+)\\\s+([a-zA-Z0-9_-]+(?:\s*\\\s*[a-zA-Z0-9_-]+)*)\]',
            re.IGNORECASE
        )
        def replace_attr_selector(match):
            parts = [match.group(1)] + re.split(r'\\\s+', match.group(2))
            normalized = '-'.join(p.lower() for p in parts if p)
            return f'[data-{normalized}]'
        
        code = pattern4.sub(replace_attr_selector, code)
        
        return code
    
    def _build_prompt(
        self,
        cleaned_content: str,
        fields: List[str],
        url: Optional[str] = None,
        extraction_context: Optional[str] = None,
        content_format: str = "HTML",
        structure_analysis: Optional[Dict[str, Any]] = None,
        previous_errors: Optional[List[str]] = None,
        field_hints: Optional[Dict[str, Dict[str, Any]]] = None  # NEW: Semantic field mappings
    ) -> str:
        """
        Build prompt for AI code generation
        Based on Parsera's proven few-shot approach
        
        NOW SUPPORTS MARKDOWN (ScrapeGraphAI approach):
        If content is Markdown, the LLM can better understand structure
        
        NOW SUPPORTS SEMANTIC FIELD HINTS:
        Field hints provide LLM with semantic understanding of what each field means
        and where to find it in the HTML, dramatically improving accuracy
        """
        
        # Smart content sampling: Find actual content, not just headers
        content_sample = cleaned_content
        sample_size = 15000  # Increased sample size
        
        # Strategy: Look for common content markers
        content_markers = [
            '<shreddit-post',  # Reddit posts
            '<article',        # Articles
            'class="post',     # Generic posts
            'class="product',  # Products
            'class="item',     # List items
            'data-testid="post',
            'data-testid="product',
            '<main',           # Main content area
        ]
        
        # Find the earliest content marker
        earliest_pos = len(cleaned_content)
        for marker in content_markers:
            pos = cleaned_content.lower().find(marker.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        # If we found content markers, start sampling from there (with some context before)
        if earliest_pos < len(cleaned_content):
            # Go back 500 chars to include container context
            start_pos = max(0, earliest_pos - 500)
            content_sample = cleaned_content[start_pos:start_pos + sample_size]
            if len(cleaned_content) > start_pos + sample_size:
                content_sample += f"\n... ({content_format} continues, showing {sample_size} chars from main content area)"
        else:
            # Fallback: try body tag
            body_start = cleaned_content.lower().find('<body')
            if body_start != -1:
                content_sample = cleaned_content[body_start:body_start + sample_size]
                if len(cleaned_content) > body_start + sample_size:
                    content_sample += f"\n... ({content_format} continues)"
            else:
                # Last resort: first 8000 chars
                content_sample = cleaned_content[:8000]
                if len(cleaned_content) > 8000:
                    content_sample += f"\n... ({content_format} continues)"
        
        # Detect custom elements in the HTML (NEW - override structure analysis if needed)
        import re
        
        # Debug: Log content format
        is_html = content_format == "HTML"
        logger.info(f"   Content format: {content_format}, Length: {len(cleaned_content)}, First 100 chars: {cleaned_content[:100]}")
        
        # Only check for custom elements if content is HTML (not Markdown)
        has_custom_elements = False
        if is_html:
            has_custom_elements = bool(re.search(r'<[a-z]+-[a-z-]+', cleaned_content))
            
            if has_custom_elements:
                # Find and log the custom elements
                custom_elem_names = set(re.findall(r'<([a-z]+-[a-z-]+)', cleaned_content))
                logger.warning(f" DETECTED CUSTOM WEB COMPONENTS: {', '.join(list(custom_elem_names)[:5])}")  # Use warning so it definitely shows
                logger.warning("   → USING ATTRIBUTE-FIRST EXTRACTION STRATEGY")
            else:
                logger.info("ℹ  No custom elements detected (HTML checked), using nested element strategy")
        else:
            logger.info(f"ℹ  Content is {content_format}, skipping custom element detection")
        
        # Build structure analysis section (NEW - from ScrapeGraphAI + Sibling Detection)
        structure_section = ""
        if structure_analysis:
            strategy = structure_analysis.get('extraction_strategy', '')
            selectors = structure_analysis.get('key_selectors', {})
            repeating_elem = structure_analysis.get('repeating_element', '')
            data_location = structure_analysis.get('data_location', '')
            element_type = structure_analysis.get('element_type', '')
            confidence = structure_analysis.get('confidence', 0)
            sibling_analysis = structure_analysis.get('sibling_analysis')  # NEW!
            
            # Extract field mappings if available
            field_mappings = selectors.get('field_mappings', {}) if isinstance(selectors, dict) else {}
            element_selector = selectors.get('element_selector', '') if isinstance(selectors, dict) else ''
            
            # Build field mappings section
            field_mappings_str = ""
            if field_mappings:
                mappings_list = [f"  - {field}: {selector}" for field, selector in field_mappings.items()]
                field_mappings_str = "\n".join(mappings_list)
            
            # Build sibling section (CRITICAL FOR STACK OVERFLOW, GITHUB, etc.)
            sibling_section = ""
            if sibling_analysis and sibling_analysis.get('type') != 'container_only':
                sib_type = sibling_analysis.get('type', '')
                parent_selector = sibling_analysis.get('parent_selector', '')
                main_selector = sibling_analysis.get('main_selector', '')
                sibling_selectors = sibling_analysis.get('sibling_selectors', [])
                
                sibling_section = f"""
** CRITICAL: SIBLING-BASED LAYOUT DETECTED! **

This site uses a **{sib_type}** structure where data is in SIBLING elements, not just nested children!

**DETECTED STRUCTURE**:
- Parent Container: `{parent_selector or 'parent element'}`
- Main Container: `{main_selector}` (inside parent)
- Sibling Elements: {', '.join([f'`{s}`' for s in sibling_selectors[:3]])} (inside parent, NOT nested in main!)

** YOU MUST ITERATE OVER PARENT ELEMENTS, NOT THE MAIN CONTAINER!**

**CORRECT CODE PATTERN** (Stack Overflow, GitHub, Indeed use this!):
```python
#  CORRECT - Iterate over PARENTS
parents = soup.select('{parent_selector or "parent-selector"}')
for parent in parents:
    item = {{}}
    
    # Extract from main container
    main = parent.select_one('{main_selector}')
    if main:
        # Extract fields from main container
        pass
    
    # Extract from sibling elements
    {f"sibling = parent.select_one('{sibling_selectors[0]}')" if sibling_selectors else "# sibling = parent.select_one('.sibling-selector')"}
    if sibling:
        # Extract fields from sibling (votes, stars, salary, etc.)
        pass
```

** WRONG CODE PATTERN** (Will miss sibling data!):
```python
#  WRONG - Only looks in main container
containers = soup.select('{main_selector}')
for elem in containers:
    item['votes'] = elem.select_one('.vote-count')  # Won't find it - it's in sibling!
```

**WHY THIS MATTERS**:
- Main container has SOME fields (title, description)
- Sibling elements have OTHER fields (votes, stars, metadata)
- Iterating over main container = you can't access siblings!
- Iterating over parent = you can access both main AND siblings!
"""
            
            structure_section = f"""
** PRE-ANALYZED HTML STRUCTURE** (Follow this precisely!):

1. **Repeating Element**: `{repeating_elem}` (appears multiple times, each = 1 data item)
   - Element Type: {element_type}
   - Confidence: {confidence:.0%}
   
2. **How to Find Elements**:
   - Selector: {element_selector or f"soup.find_all('{repeating_elem}')"}
   
3. **Data Location**: {data_location}
   - {'Data is in HTML ATTRIBUTES (use elem.get())' if data_location == 'attributes' else 'Data is in NESTED ELEMENTS (use elem.select_one())'}

4. **Field Mappings** (use these exact selectors/attributes):
{field_mappings_str if field_mappings_str else "  - No specific mappings provided, analyze the element structure"}

5. **Extraction Strategy**:
{strategy}
{sibling_section}
** CRITICAL**: This analysis is based on actual HTML structure. Trust it and implement exactly as described!
"""
        
        # Build error feedback section (NEW - from ScrapeGraphAI)
        error_section = ""
        if previous_errors and len(previous_errors) > 0:
            errors_str = "\n".join(f"  - {error}" for error in previous_errors[-3:])  # Last 3 errors
            error_section = f"""
**PREVIOUS ATTEMPT ERRORS** (Fix these issues):
{errors_str}

**CRITICAL**: The previous code had these problems. Make sure to fix them!
"""
        
        # Build context section
        context_section = ""
        if extraction_context:
            context_section = f"""
USER'S EXTRACTION GOAL:
{extraction_context}

Use this goal to understand what data to look for.
"""
        
        # Adapt prompt based on content format and detected patterns
        format_note = ""
        if content_format == "Markdown":
            format_note = "NOTE: The content below is in Markdown format (converted from HTML). Look for lists, headings, and structured text patterns."
        
        # Add URGENT warning if custom elements detected
        custom_elements_warning = ""
        if has_custom_elements:
            custom_elements_warning = f"""
 **URGENT - CUSTOM WEB COMPONENTS DETECTED** 
This page uses custom HTML elements (tags with hyphens like <shreddit-post>, <product-card>).
These elements store data in HTML ATTRIBUTES, not nested text!

**YOU MUST**:
1. Use soup.find_all(lambda tag: '-' in tag.name) to find custom elements
2. Use elem.get('attribute-name') to extract data from attributes
3. Try multiple attribute names: elem.get('post-title') or elem.get('data-title') or elem.get('title')
4. DO NOT use .text or .select() on custom elements - they won't work!

Example custom element found in HTML:
{re.search(r'<[a-z]+-[a-z-]+[^>]*>', cleaned_content).group(0) if re.search(r'<[a-z]+-[a-z-]+[^>]*>', cleaned_content) else "See sample below"}

**THIS IS THE MOST IMPORTANT INSTRUCTION - IGNORE IT AND YOUR CODE WILL FAIL**
"""
        
        # NEW: Build semantic field hints section
        field_hints_section = ""
        if field_hints:
            field_hints_section = "\n** SEMANTIC FIELD MAPPINGS** (Critical - Read carefully!):\n\n"
            for field, hint in field_hints.items():
                field_hints_section += f""" Field: '{field}'
   Meaning: {hint['semantic_meaning']}
   Look in: {', '.join(hint['likely_locations'][:3])}
   Strategy: {hint['extraction_strategy'][:150]}...
   Example: {hint['code_example']}

"""
            field_hints_section += """
**CRITICAL INSTRUCTIONS FOR SEMANTIC FIELDS**:
1. DON'T just look for `.{field_name}` or `elem.get('{field_name}')`
2. USE the "Look in" locations and strategies above
3. The semantic meaning tells you WHAT the field represents
4. The example code shows you HOW to extract it
5. Trust the semantic mappings - they're analyzed from the actual domain

"""
        
        prompt = f"""You are an expert web scraping engineer. Generate BeautifulSoup code to extract structured data from HTML.
{custom_elements_warning}{structure_section}{field_hints_section}{error_section}{context_section}
FIELDS TO EXTRACT:
{', '.join(fields)}

{format_note}

CONTENT TO ANALYZE:
```{content_format.lower()}
{content_sample}
```

INSTRUCTIONS:
1. **FIRST**: Check if data is stored in HTML ATTRIBUTES (common in modern sites)
   - Look for: element.get('attribute-name'), data-* attributes, custom elements
   - Example: <shreddit-post author="user" score="42" comment-count="10">
2. Study the HTML structure carefully to find where the target data appears

2.5. ** CRITICAL - FIELDS WITH SPACES OR SPECIAL CHARACTERS **
   **NEVER use class selectors for field names with spaces - this is UNIVERSAL!**
   **NEVER use class selectors starting with special characters (^, ~, @, etc.) - use attribute selectors instead!**
   
    **WRONG** (ANY field with space/special char):
   ```python
   # This is INVALID CSS syntax - spaces separate multiple classes!
   value = elem.select_one('.field\\. name\\ value')  # ← SYNTAX ERROR!
   value = elem.select_one('.product name')  # ← SYNTAX ERROR!
   value = elem.select_one('.est. market value')  # ← SYNTAX ERROR!
   # Class names starting with special characters are also invalid!
   value = elem.select_one('.^__asset-list--wuiW8')  # ← SYNTAX ERROR! (^ is invalid)
   value = elem.select_one('div.^__asset-list--wuiW8')  # ← SYNTAX ERROR!
   ```
   
    **CORRECT** (universal approach for ANY field with spaces or special chars):
   ```python
   # Option 1: Use attribute selector (most reliable - works for ANY field)
   value = elem.select_one('[data-field-name]')  # Universal pattern
   # or
   value = elem.get('data-field-name')
   
   # Option 2: For class names with special characters, use attribute selector with contains
   # If HTML has: <div class="^__asset-list--wuiW8">
   value = elem.select_one('[class*="__asset-list--wuiW8"]')  # Matches class containing this
   # or find by partial class match
   value = elem.select_one('[class*="asset-list"]')
   
   # Option 3: Find the ACTUAL class/attribute name from HTML
   # Look in the HTML sample for how the field is actually stored
   # It might be: class="field-name" or data-field-name="..." or dataFieldName="..."
   # Copy the EXACT name from the HTML - don't guess!
   
   # Option 4: Use text-based search if no attribute exists
   # Find element containing text like "Field Name:" then get sibling/child
   ```
   
   **UNIVERSAL RULE**: If a field name contains spaces, periods, or special characters:
   - ALWAYS use attribute selectors: `[data-field-name]` or `elem.get('data-field-name')`
   - NEVER try to escape spaces in class selectors - it's invalid CSS syntax!
   - Look for the ACTUAL attribute/class name in the HTML (often hyphenated or camelCase)
   - This applies to ANY website, ANY field name with spaces - it's a CSS limitation, not site-specific!

3. ** CRITICAL - DO NOT HALLUCINATE CLASS NAMES! **
   **ALWAYS use class names that ACTUALLY EXIST in the HTML sample above!**
   
    **WRONG** (Stack Overflow example - guessed class name):
   ```python
   votes = elem.select_one('span.vote-count-post')  # ← This class doesn't exist!
   ```
   
    **CORRECT** (checked actual HTML):
   ```html
   <!-- Actual HTML shows: -->
   <span class="s-post-summary--stats-item-number" itemprop="upvoteCount">42</span>
   ```
   ```python
   votes = elem.select_one('span.s-post-summary--stats-item-number')
   # or even better, with attribute selector:
   votes = elem.select_one('span[itemprop="upvoteCount"]')
   ```
   
   **HOW TO AVOID THIS BUG**:
   - Read the HTML sample carefully
   - Copy exact class names from the HTML
   - Use attribute selectors when available ([itemprop], [data-*], [aria-*])
   - Test your selectors mentally against the HTML sample
   - If you're not 100% sure, use a more generic selector + filter
   
   **THIS IS THE #1 CAUSE OF NULL FIELDS - TAKE YOUR TIME TO GET IT RIGHT!**

4. **CRITICAL - CHECK FOR SIBLING-BASED LAYOUTS**:
     Many sites (Stack Overflow, GitHub, Indeed) use SIBLING elements, not nested children!
   
   If you see patterns like this:
   ```html
   <parent>
     <div class="main-container">
       <h3>Title Here</h3>  ← Some fields inside
     </div>
     <div class="metadata">
       <span>42 votes</span>  ← OTHER fields in SIBLING!
     </div>
   </parent>
   ```
   
   **YOU MUST iterate over the PARENT element, not the main container!**
   
    CORRECT approach (parent iteration):
   ```python
   parents = soup.select('parent-selector')  # Find the parent
   for parent in parents:
       item = {{}}
       # Extract from main container
       container = parent.select_one('.main-container')
       if container:
           item['title'] = container.select_one('h3').get_text(strip=True)
       
       # Extract from sibling
       metadata = parent.select_one('.metadata')
       if metadata:
           item['votes'] = metadata.select_one('span').get_text(strip=True)
   ```
   
    WRONG approach (will miss sibling data):
   ```python
   containers = soup.select('.main-container')  # Misses siblings!
   for elem in containers:
       item['votes'] = elem.select_one('span')  # Won't find it - it's in sibling!
   ```
   
   **HOW TO DETECT**: If the HTML sample shows repeating parent elements with multiple child divs
   at the same level (siblings), iterate over parents!

4. **FREQUENCY-BASED DETECTION** (Universal approach):
    Valuable data has HIGH-FREQUENCY patterns!
   
   If you can't find a field in the obvious places, look for high-frequency elements:
   - If the main container appears 15 times, related data elements ALSO appear ~15 times
   - Example: vote counts, prices, ratings appear once per item = same frequency as containers
   
   Strategy:
   1. Count how many times the main pattern repeats (e.g., 15 posts)
   2. Look for OTHER elements that repeat the same number of times
   3. These are likely data fields!
   
   Example: If you see:
   - `div.post-summary` (15x) ← Container
   - `span.vote-count` (15x) ← Probably votes data!
   - `a.post-title` (15x) ← Probably title data!
   - Even if they're in different parts of the DOM tree, match them by frequency!

5. Look for repeating patterns (divs, articles, list items) that contain the data
6. Extract ALL matching items, not just one (use .find_all() or .select(), NOT .find_one())
7. Use exact values from the page, no modifications
8. If a field is missing or not found, use None
9. Handle errors gracefully

** CRITICAL NULL VALUE VALIDATION** (REQUIRED - This is checked!):
After extracting data, you MUST validate that fields are not all None:
```python
# At the end of your extract_data function, ADD THIS VALIDATION:
if items:
    first_item = items[0]
    non_null_count = sum(1 for v in first_item.values() if v is not None and v != '')
    if non_null_count == 0:
        # All fields are None - selectors are wrong!
        # Try alternative approaches: attributes, different selectors, etc.
        # DO NOT return items with all None values
        return []
```

**CRITICAL**: Many modern websites store data in HTML attributes, NOT nested elements!
Always try: elem.get('attr-name') BEFORE trying: elem.select_one('.class').text

**TEST YOUR SELECTORS**: On the FIRST element, verify your selectors find data BEFORE looping through all elements!

EXAMPLE 1 - Extracting from HTML ATTRIBUTES (MOST IMPORTANT - Try this first!):
```python
def extract_data(soup):
    items = []
    
    # Modern sites use custom elements with data in attributes
    # Examples: <shreddit-post author="user" score="42">, <product-card price="99.99">
    # Key: Look for elements with hyphens in the tag name OR many attributes
    
    # Strategy 1: Find ALL custom elements (contain hyphen in tag name)
    custom_elements = soup.find_all(lambda tag: '-' in tag.name)
    
    if custom_elements:
        # Found custom elements! Extract from attributes
        for elem in custom_elements:
            item = {{}}
            
            # Use .get() to extract from attributes - try multiple names
            item['title'] = elem.get('post-title') or elem.get('data-title') or elem.get('title')
            item['author'] = elem.get('author') or elem.get('data-author') or elem.get('user')
            item['upvotes'] = elem.get('score') or elem.get('upvotes') or elem.get('data-score') or elem.get('votes')
            item['comments'] = elem.get('comment-count') or elem.get('comments') or elem.get('data-comments')
            item['link'] = elem.get('href') or elem.get('permalink') or elem.get('content-href')
            
            # Only add if we got at least some data
            if any(item.values()):
                items.append(item)
        
        return items
    
    # Strategy 2: Fall back to elements with data-* attributes
    data_elements = soup.select('[data-title], [data-product], [data-item]')
    if data_elements:
        for elem in data_elements:
            item = {{}}
            # Extract from data-* attributes
            for attr_name, attr_value in elem.attrs.items():
                if attr_name.startswith('data-'):
                    field_name = attr_name.replace('data-', '')
                    item[field_name] = attr_value
            
            if item:
                items.append(item)
        
        return items
    
    # If no attribute-based elements found, return empty (will try nested approach)
    return items
```

** CRITICAL: EXTRACT ALL ITEMS, NOT JUST ONE** (REQUIRED):
- ALWAYS use `.find_all()` or `.select()` (returns multiple elements)
- NEVER use `.find()` or `.find_one()` or `.select_one()` for the main container search
- Example BAD: `container = soup.find('article')`  Only gets first article!
- Example GOOD: `containers = soup.find_all('article')`  Gets all articles!
- Make selectors GENERAL, not specific (e.g., 'article' not 'article.featured')

EXAMPLE 2 - Extracting product listings (nested elements):
```python
def extract_data(soup):
    items = []
    
    # Find ALL product containers (use .find_all() or .select(), NOT .find()!)
    # Try common patterns: .product, .item, article, [data-testid*="product"]
    containers = soup.select('.product-card, article.product, [class*="product"]')
    
    if not containers:
        # Fallback: try finding by common patterns (still use find_all!)
        containers = soup.find_all('div', class_=lambda x: x and ('item' in x.lower() or 'product' in x.lower()))
    
    for container in containers:
        item = {{}}
        
        # Extract name (try multiple selectors)
        name_elem = container.select_one('.name, .title, h2, h3, [class*="name"]')
        item['name'] = name_elem.text.strip() if name_elem else None
        
        # Extract price
        price_elem = container.select_one('.price, [class*="price"]')
        item['price'] = price_elem.text.strip() if price_elem else None
        
        items.append(item)
    
    return items
```

EXAMPLE 3 - Extracting from tables:
```python
def extract_data(soup):
    items = []
    
    # Find table rows
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                item = {{}}
                item['col1'] = cells[0].text.strip()
                item['col2'] = cells[1].text.strip()
                items.append(item)
    
    return items
```

EXAMPLE 4 - Extracting posts/articles (nested elements):
```python
def extract_data(soup):
    items = []
    
    # Find all post containers
    posts = soup.select('[data-testid*="post"], article, .post, [class*="post"]')
    
    for post in posts:
        item = {{}}
        
        # Extract title
        title_elem = post.select_one('h1, h2, h3, [class*="title"], [class*="headline"]')
        item['title'] = title_elem.text.strip() if title_elem else None
        
        # Extract author
        author_elem = post.select_one('[class*="author"], [data-author], .username')
        item['author'] = author_elem.text.strip() if author_elem else None
        
        # Extract link
        link_elem = post.select_one('a[href]')
        item['link'] = link_elem['href'] if link_elem and link_elem.get('href') else None
        
        items.append(item)
    
    return items
```

IMPORTANT NOTES:
- ALWAYS extract ALL matching items (return a list)
- Try multiple selector strategies if first doesn't work
- Look for class names, data attributes, and semantic HTML tags
- **IMPORTANT**: Check for data in HTML ATTRIBUTES first (use elem.get('attr-name'))
- Many modern sites (Reddit, custom web components) store data in attributes, not nested elements
- Use .text.strip() to clean extracted text
- Check if elements exist before accessing (use if elem)
- Return empty list [] if no data found

Now generate the `extract_data(soup)` function for the given HTML and fields.
Only output the Python code, no explanation."""
        
        if url:
            prompt = f"SOURCE URL: {url}\n\n" + prompt
        
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
                logger.info(f" Detected {provider} API key from environment")
                return api_key
        
        logger.warning(" No API key found in environment")
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
            logger.error(f" Invalid code syntax: {str(e)}")
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
    
    def generate_semantic_pattern(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str] = None,
        extraction_context: Optional[str] = None,
        structure_analysis: Optional[Dict[str, Any]] = None,
        field_hints: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate semantic extraction patterns (JSON) instead of code.
        
        This is the NEW universal approach - resilient to layout changes!
        
        Args:
            cleaned_html: Cleaned HTML to analyze
            fields: Fields to extract
            url: Source URL for context
            extraction_context: User's extraction goal
            structure_analysis: HTML structure analysis to guide generation
            field_hints: Semantic field mappings (from UniversalFieldMapper)
            
        Returns:
            Dict with 'pattern', 'explanation', 'model_used' keys
            
        Example output:
        {
            "pattern": {
                "title": {
                    "primary": {"type": "heading", "position": "first"},
                    "fallbacks": [
                        {"type": "link_text"},
                        {"type": "bold_text", "min_length": 10}
                    ]
                },
                "price": {
                    "primary": {"type": "currency", "symbols": ["$", "€"]},
                    "fallbacks": [
                        {"type": "attribute", "name": "data-price"}
                    ]
                }
            },
            "explanation": "Pattern description",
            "model_used": "gpt-4o-mini"
        }
        """
        logger.info(f" Generating semantic pattern for {len(fields)} fields...")
        
        if structure_analysis:
            logger.info(f"   Using structure analysis: {structure_analysis.get('repeating_element')}")
        
        if field_hints:
            logger.info(f"   Using semantic field mappings for {len(field_hints)} fields")
        
        # Build semantic pattern generation prompt
        prompt = self._build_semantic_pattern_prompt(
            cleaned_html,
            fields,
            url,
            extraction_context,
            structure_analysis,
            field_hints
        )
        
        # Call LLM
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if 'gpt' in self.model_name else None
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            result = json.loads(result_text)
            
            # Validate pattern structure
            if 'pattern' not in result:
                raise ValueError("Generated pattern missing 'pattern' key")
            
            logger.info(f"    Generated semantic pattern ({len(result['pattern'])} fields)")
            
            return {
                'pattern': result['pattern'],
                'explanation': result.get('explanation', 'Semantic pattern generated'),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f" Semantic pattern generation failed: {e}")
            raise
    
    def _build_semantic_pattern_prompt(
        self,
        cleaned_html: str,
        fields: List[str],
        url: Optional[str],
        extraction_context: Optional[str],
        structure_analysis: Optional[Dict[str, Any]],
        field_hints: Optional[Dict[str, Dict[str, Any]]]
    ) -> str:
        """Build prompt for semantic pattern generation."""
        
        # Get domain from URL
        domain = "unknown"
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc
        
        # Start with system instructions
        prompt = f"""You are a universal web scraping expert. Your job is to analyze HTML and describe HOW to find data semantically (not with specific CSS selectors).

# YOUR TASK

Analyze this HTML from **{domain}** and generate a semantic extraction pattern for these fields:
{', '.join(fields)}

# IMPORTANT: Generate SEMANTIC STRATEGIES, NOT CSS SELECTORS!

Instead of saying: "Use selector 'h2.title'"
Say: "Find the first heading in each container"

Instead of saying: "Use selector 'span.price'"
Say: "Find text containing currency symbols ($, €)"

# AVAILABLE SEMANTIC STRATEGIES

You can use these strategy types:

## 1. HEADING (for titles, names)
{{"type": "heading", "position": "first"}}  // First h1-h6
{{"type": "heading", "position": "last"}}   // Last h1-h6

## 2. BOLD_TEXT (for emphasized text)
{{"type": "bold_text", "min_length": 10}}   // Bold text at least 10 chars

## 3. LINK_TEXT (for titles, names)
{{"type": "link_text", "return": "text"}}    // Link text
{{"type": "link_text", "return": "href"}}    // Link URL

## 4. ATTRIBUTE (for data-*, aria-*, etc)
{{"type": "attribute", "name": "data-price"}}
{{"type": "attribute", "name": "aria-label"}}

## 5. CURRENCY (for prices)
{{"type": "currency", "symbols": ["$", "€", "£"]}}

## 6. NUMBER (for counts, votes, ratings)
{{"type": "number", "pattern": "\\\\d+"}}

## 7. DATE (for timestamps)
{{"type": "date"}}  // Finds <time> tags or date patterns

## 8. IMAGE (for images)
{{"type": "image", "return": "src"}}   // Image URL
{{"type": "image", "return": "alt"}}   // Image alt text

## 9. TEXT_CONTAINS (for specific patterns)
{{"type": "text_contains", "pattern": "\\\\d+ views"}}

## 10. FIRST_TEXT (fallback)
{{"type": "first_text", "min_length": 5}}

"""

        # Add structure analysis context
        if structure_analysis:
            repeating_element = structure_analysis.get('repeating_element', 'unknown')
            pattern_count = structure_analysis.get('pattern_count', 0)
            data_location = structure_analysis.get('data_location', 'unknown')
            
            prompt += f"""
# STRUCTURE ANALYSIS

Repeating Element: {repeating_element}
Pattern Count: {pattern_count}
Data Location: {data_location}

"""

        # Add field hints (semantic understanding)
        if field_hints:
            prompt += "# FIELD SEMANTICS\n\n"
            for field, hint in field_hints.items():
                semantic_meaning = hint.get('semantic_meaning', '')
                prompt += f"- **{field}**: {semantic_meaning}\n"
            prompt += "\n"
        
        # Add HTML sample (truncated for token limits)
        html_sample = cleaned_html[:15000]  # First 15KB
        prompt += f"""
# HTML SAMPLE

```html
{html_sample}
```

# OUTPUT FORMAT

Return a JSON object with this structure:

{{
  "pattern": {{
    "{fields[0]}": {{
      "primary": {{"type": "...", ...}},
      "fallbacks": [{{"type": "...", ...}}],
      "validation": {{"not_empty": true, "min_length": 5}}
    }},
    ...
  }},
  "explanation": "Brief explanation of the patterns"
}}

# CRITICAL RULES

1. **Use semantic strategies, NOT CSS selectors**
2. **Always provide fallbacks** (at least 2 per field)
3. **Order strategies by reliability** (most reliable first)
4. **Add validation rules** for important fields
5. **Be resilient** - patterns should work even if layout changes

# EXAMPLES OF GOOD PATTERNS

For "title" on a forum:
{{
  "primary": {{"type": "heading", "position": "first"}},
  "fallbacks": [
    {{"type": "link_text"}},
    {{"type": "bold_text", "min_length": 20}},
    {{"type": "first_text", "min_length": 10}}
  ]
}}

For "price" on e-commerce:
{{
  "primary": {{"type": "currency", "symbols": ["$"]}},
  "fallbacks": [
    {{"type": "attribute", "name": "data-price"}},
    {{"type": "text_contains", "pattern": "\\\\$\\\\d+"}}
  ]
}}

For "votes" on Stack Overflow:
{{
  "primary": {{"type": "number", "pattern": "^\\\\d+$"}},
  "fallbacks": [
    {{"type": "attribute", "name": "data-score"}},
    {{"type": "attribute", "name": "data-votes"}}
  ]
}}

Now analyze the HTML and generate semantic patterns for: {', '.join(fields)}
"""

        return prompt

