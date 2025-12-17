"""
Universal Field Mapper - Semantic field understanding for any website

This module maps user-requested fields to their semantic meaning and likely HTML locations
using LLM analysis. Results are heavily cached to maintain cost efficiency.

Architecture:
1. Domain Context (cached by domain): "GitHub = tech repository platform"
2. Field Semantics (cached by domain+fields): "repository = repo name in <h2><a>"
3. Extraction Hints (generated): Specific instructions for code generation

Cost per 100 pages:
- First page: $0.05 (domain + field mapping + code gen)
- Pages 2-100: $0.00 (everything cached)
- Total: $0.05 (vs $10-30 for ScrapeGraphAI)
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, asdict
import litellm
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FieldSemantics:
    """Semantic information about a field in a specific domain context"""
    field_name: str
    semantic_meaning: str
    likely_html_elements: List[str]
    common_attributes: List[str]
    common_class_patterns: List[str]
    extraction_strategy: str
    code_example: str
    confidence: float


@dataclass
class DomainContext:
    """Domain-level understanding of a website"""
    domain: str
    website_type: str  # e.g., "tech_platform", "e-commerce", "social_media"
    entity_type: str   # e.g., "repositories", "products", "posts"
    common_patterns: Dict[str, Any]
    confidence: float


class UniversalFieldMapper:
    """
    Maps user-requested fields to semantic meanings and HTML locations.
    Uses LLM for understanding but caches aggressively for cost efficiency.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        cache_dir: str = "./cache/field_mappings",
        enable_cache: bool = True
    ):
        """
        Initialize the Universal Field Mapper
        
        Args:
            api_key: API key for LLM
            model: Model to use for analysis
            cache_dir: Directory for caching results
            enable_cache: Whether to use caching
        """
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        
        # Create cache directories
        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.domain_cache_dir = self.cache_dir / "domains"
            self.field_cache_dir = self.cache_dir / "fields"
            self.domain_cache_dir.mkdir(exist_ok=True)
            self.field_cache_dir.mkdir(exist_ok=True)
        
        # In-memory caches (for performance)
        self.domain_contexts: Dict[str, DomainContext] = {}
        self.field_semantics: Dict[str, Dict[str, FieldSemantics]] = {}
        
        logger.info(f"  Universal Field Mapper initialized with {model}")
        if enable_cache:
            logger.info(f"   Cache directory: {cache_dir}")
    
    def map_fields(
        self,
        fields: List[str],
        url: str,
        html_sample: str,
        structure_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map user fields to semantic meanings and extraction hints.
        
        This is the main entry point that orchestrates:
        1. Domain context analysis (cached by domain)
        2. Field semantic mapping (cached by domain + fields)
        3. Extraction hint generation (combines above + structure)
        
        Args:
            fields: List of field names to extract
            url: Source URL
            html_sample: Sample of HTML content (first 3-5K chars)
            structure_analysis: Optional structural analysis from DOM detector
        
        Returns:
            Dict mapping field names to extraction hints:
            {
                'repository': {
                    'semantic_meaning': 'Repository name or full path',
                    'likely_locations': ['h2 a', '.repo-name'],
                    'extraction_strategy': 'Look in main heading of each article',
                    'code_example': 'elem.select_one("h2 a").text',
                    'confidence': 0.90
                },
                ...
            }
        """
        domain = self._extract_domain(url)
        
        logger.info(f"  Mapping {len(fields)} fields for {domain}...")
        
        # Step 1: Get or infer domain context (cached by domain)
        domain_context = self._get_domain_context(domain, url, html_sample)
        logger.info(f"   Domain type: {domain_context.website_type} ({domain_context.entity_type})")
        
        # Step 2: Get or infer field semantics (cached by domain + fields)
        field_semantics = self._get_field_semantics(
            fields,
            domain,
            domain_context,
            html_sample
        )
        
        # Step 3: Generate extraction hints (combines semantic + structural)
        extraction_hints = {}
        for field in fields:
            if field in field_semantics:
                semantics = field_semantics[field]
                extraction_hints[field] = self._generate_extraction_hint(
                    semantics,
                    structure_analysis
                )
                logger.info(f"    Mapped '{field}': {semantics.semantic_meaning[:50]}...")
            else:
                # Fallback: no semantic info available
                logger.warning(f"     No semantic mapping for '{field}', using generic")
                extraction_hints[field] = self._generic_extraction_hint(field)
        
        return extraction_hints
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        parsed = urlparse(url)
        # Remove www. prefix if present
        domain = parsed.netloc.replace('www.', '')
        return domain
    
    def _get_domain_context(
        self,
        domain: str,
        url: str,
        html_sample: str
    ) -> DomainContext:
        """
        Get domain context (cached by domain).
        
        This is expensive (LLM call) but cached forever per domain.
        Cost: ~$0.01 per new domain
        """
        # Check in-memory cache
        if domain in self.domain_contexts:
            logger.debug(f"   Using in-memory domain context for {domain}")
            return self.domain_contexts[domain]
        
        # Check disk cache
        if self.enable_cache:
            cache_file = self.domain_cache_dir / f"{domain}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        context = DomainContext(**data)
                        self.domain_contexts[domain] = context
                        logger.debug(f"   Using cached domain context for {domain}")
                        return context
                except Exception as e:
                    logger.warning(f"   Failed to load domain cache: {e}")
        
        # Infer from LLM (expensive!)
        logger.info(f"    Inferring domain context for {domain} (LLM call, ~$0.01)...")
        context = self._infer_domain_context_llm(domain, url, html_sample)
        
        # Cache results
        self.domain_contexts[domain] = context
        if self.enable_cache:
            cache_file = self.domain_cache_dir / f"{domain}.json"
            with open(cache_file, 'w') as f:
                json.dump(asdict(context), f, indent=2)
            logger.debug(f"   Cached domain context to {cache_file}")
        
        return context
    
    def _infer_domain_context_llm(
        self,
        domain: str,
        url: str,
        html_sample: str
    ) -> DomainContext:
        """Use LLM to understand the domain and its patterns"""
        
        prompt = f"""Analyze this website to understand its domain and data structure.

URL: {url}
Domain: {domain}

HTML Sample (first 2000 chars):
{html_sample[:2000]}

Identify:
1. **Website Type**: What kind of website is this?
   - Options: tech_platform, e-commerce, social_media, news, blog, documentation, forum, repository, job_board, other
   
2. **Primary Entity Type**: What are the main data items on this page?
   - Examples: repositories, products, articles, posts, comments, jobs, documentation_pages, users, etc.
   
3. **Common Patterns**: What structural/semantic patterns are common on this domain?
   - Layout patterns (e.g., card-based, list-based, table-based)
   - Naming conventions (e.g., GitHub uses "repository", "stars")
   - Data location patterns (e.g., data in attributes, nested elements, etc.)

Return a JSON object with this structure:
{{
    "website_type": "tech_platform",
    "entity_type": "repositories",
    "common_patterns": {{
        "layout": "card-based list",
        "naming_conventions": ["repository", "stars", "language", "description"],
        "data_locations": ["nested_elements", "text_content"],
        "typical_containers": ["article", "div.item", "li.card"]
    }},
    "confidence": 0.95
}}

Be specific and analyze the actual HTML structure provided.
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web scraping expert who analyzes website structures."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                api_key=self.api_key
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            result_json = self._extract_json(result_text)
            
            return DomainContext(
                domain=domain,
                website_type=result_json.get('website_type', 'unknown'),
                entity_type=result_json.get('entity_type', 'items'),
                common_patterns=result_json.get('common_patterns', {}),
                confidence=result_json.get('confidence', 0.7)
            )
        
        except Exception as e:
            logger.error(f"    Failed to infer domain context: {e}")
            # Return generic fallback
            return DomainContext(
                domain=domain,
                website_type='unknown',
                entity_type='items',
                common_patterns={},
                confidence=0.3
            )
    
    def _get_field_semantics(
        self,
        fields: List[str],
        domain: str,
        domain_context: DomainContext,
        html_sample: str
    ) -> Dict[str, FieldSemantics]:
        """
        Get field semantic mappings (cached by domain + fields).
        
        This is expensive (LLM call) but cached per domain+field combination.
        Cost: ~$0.02 per new domain+field combo
        """
        # Create cache key from domain + sorted fields
        fields_key = ':'.join(sorted(fields))
        cache_key = f"{domain}_{hashlib.md5(fields_key.encode()).hexdigest()[:8]}"
        
        # Check in-memory cache
        if cache_key in self.field_semantics:
            logger.debug(f"   Using in-memory field semantics")
            return self.field_semantics[cache_key]
        
        # Check disk cache
        if self.enable_cache:
            cache_file = self.field_cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        semantics = {
                            k: FieldSemantics(**v) for k, v in data.items()
                        }
                        self.field_semantics[cache_key] = semantics
                        logger.debug(f"   Using cached field semantics")
                        return semantics
                except Exception as e:
                    logger.warning(f"   Failed to load field semantics cache: {e}")
        
        # Infer from LLM (expensive!)
        logger.info(f"    Inferring field semantics (LLM call, ~$0.02)...")
        semantics = self._infer_field_semantics_llm(
            fields,
            domain,
            domain_context,
            html_sample
        )
        
        # Cache results
        self.field_semantics[cache_key] = semantics
        if self.enable_cache:
            cache_file = self.field_cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                serialized = {k: asdict(v) for k, v in semantics.items()}
                json.dump(serialized, f, indent=2)
            logger.debug(f"   Cached field semantics to {cache_file}")
        
        return semantics
    
    def _infer_field_semantics_llm(
        self,
        fields: List[str],
        domain: str,
        domain_context: DomainContext,
        html_sample: str
    ) -> Dict[str, FieldSemantics]:
        """Use LLM to map fields to their semantic meanings and HTML locations"""
        
        prompt = f"""Given this domain context, map the requested fields to their semantic meanings and likely HTML locations.

DOMAIN CONTEXT:
- Domain: {domain}
- Type: {domain_context.website_type}
- Entity: {domain_context.entity_type}
- Patterns: {json.dumps(domain_context.common_patterns, indent=2)}

FIELDS TO MAP:
{', '.join(fields)}

HTML SAMPLE (analyze to find patterns):
{html_sample[:3000]}

**CRITICAL INSTRUCTIONS**:
1. You MUST analyze the actual HTML structure provided above
2. DO NOT guess generic class names like ".description", ".star-count", ".language"
3. LOOK at the actual HTML and provide working CSS selectors you can see
4. Test your selectors against the HTML sample mentally before returning them
5. **PREFER DATA CONTENT over UI ICONS**:
   -  BAD: SVG icons, buttons, action labels (e.g., `<svg aria-label="star">`)
   -  GOOD: Links, text, numbers, data attributes (e.g., `<a href="/stargazers">9,305</a>`)
   - If you see both an icon AND data, choose the DATA (link text, span text, etc.)
6. **EXTRACT NUMBERS FROM LINKS, NOT ICONS**:
   - For "stars": Look for `<a href="...stargazers">NUMBER</a>` (the link text, not the icon)
   - For "forks": Look for `<a href="...forks">NUMBER</a>` (the link text, not the icon)
   - For "views": Look for `<a href="...views">NUMBER</a>` (the link text, not the icon)

For each field, provide:
1. **Semantic Meaning**: What does this field represent in this domain context?
   - Example: "repository" on GitHub = "Repository name or full user/repo path"
   
2. **Likely HTML Elements**: ACTUAL CSS selectors from the HTML above (NOT generic guesses!)
   - Look at the HTML and provide selectors that ACTUALLY exist
   - Examples: ["h2 a", "p.col-9", "span[itemprop='programmingLanguage']"]
   - NOT: [".description", ".star-count"] <- these are WRONG if they don't exist in HTML
   
3. **Common Attributes**: What HTML attributes might contain this data?
   - Examples: ["data-repo", "href", "title", "itemprop"]
   
4. **Common Class Patterns**: What CSS class patterns are actually used in the HTML?
   - Look at the actual classes in the HTML sample
   - Examples: ["col-9", "Box-row", "octicon-star"]
   
5. **Extraction Strategy**: How should we extract this field from the HTML above?
   - Example: "Look for the p tag inside each article element"
   - Be specific about the actual structure you see
   
6. **Code Example**: A working BeautifulSoup snippet based on the actual HTML
   - Example: "elem.select_one('p').text.strip()" if you see a p tag
   - Example: "elem.select_one('span[itemprop=\"programmingLanguage\"]').text.strip()" if you see this
   - DO NOT use selectors like ".description" unless you actually see class="description" in the HTML

7. **Confidence**: How confident are you in this mapping? (0.0-1.0)
   - Set to 0.9+ only if you found the exact elements in the HTML sample
   - Set to 0.5-0.8 if you're making educated guesses

** SPECIAL GUIDANCE FOR TEMPORAL FIELDS** (date, time, posted, published, updated, created, timestamp):
If extracting a temporal field, use this priority order:
1. **<time> tags**: `elem.select_one('time')` or `elem.select_one('time')['datetime']`
2. **datetime attributes**: `elem.select_one('[datetime]')['datetime']`
3. **Relative dates**: Look for text like "2 hours ago", "posted 3d", "yesterday"
4. **Formatted dates**: Look for text like "Nov 12, 2024", "2024-11-12", "12/11/2024"
5. **data-* attributes**: `elem.get('data-time')`, `elem.get('data-timestamp')`

** TEMPORAL EXTRACTION EXAMPLES**:
- `elem.select_one('time').text.strip()` - for <time> tags with text
- `elem.select_one('time')['datetime']` - for <time datetime="..."> attributes
- `elem.select_one('.date, .timestamp, [datetime]').text.strip()` - multiple selectors
- `elem.get('data-time') or elem.select_one('.date').text` - fallback pattern

Return JSON:
{{
    "repository": {{
        "field_name": "repository",
        "semantic_meaning": "Repository name or full user/repo path",
        "likely_html_elements": ["h2 a", "h2.h3 a"],
        "common_attributes": ["href"],
        "common_class_patterns": ["h2", "h3"],
        "extraction_strategy": "Find the h2 a tag in each article element. This contains the repo link.",
        "code_example": "elem.select_one('h2 a').text.strip()",
        "confidence": 0.95
    }},
    ...
}}

REMEMBER: Analyze the ACTUAL HTML structure, don't guess generic class names!
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web scraping expert who maps field names to HTML structures."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                api_key=self.api_key
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            result_json = self._extract_json(result_text)
            
            # Convert to FieldSemantics objects
            semantics = {}
            for field in fields:
                if field in result_json:
                    field_data = result_json[field]
                    semantics[field] = FieldSemantics(
                        field_name=field,
                        semantic_meaning=field_data.get('semantic_meaning', f'Value for {field}'),
                        likely_html_elements=field_data.get('likely_html_elements', []),
                        common_attributes=field_data.get('common_attributes', []),
                        common_class_patterns=field_data.get('common_class_patterns', []),
                        extraction_strategy=field_data.get('extraction_strategy', 'Standard CSS selector extraction'),
                        code_example=field_data.get('code_example', f"elem.select_one('.{field}').text"),
                        confidence=field_data.get('confidence', 0.6)
                    )
                else:
                    # Fallback for missing fields
                    logger.warning(f"   LLM didn't map field '{field}', using generic")
                    semantics[field] = self._generic_field_semantics(field)
            
            return semantics
        
        except Exception as e:
            logger.error(f"    Failed to infer field semantics: {e}")
            # Return generic fallback for all fields
            return {field: self._generic_field_semantics(field) for field in fields}
    
    def _generate_extraction_hint(
        self,
        semantics: FieldSemantics,
        structure_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine semantic understanding with structural analysis
        to generate extraction hints for code generation
        """
        hint = {
            'semantic_meaning': semantics.semantic_meaning,
            'likely_locations': semantics.likely_html_elements,
            'common_attributes': semantics.common_attributes,
            'common_classes': semantics.common_class_patterns,
            'extraction_strategy': semantics.extraction_strategy,
            'code_example': semantics.code_example,
            'confidence': semantics.confidence
        }
        
        # Add structural context if available
        if structure_analysis and structure_analysis.get('best_pattern'):
            pattern = structure_analysis['best_pattern']
            hint['structural_context'] = {
                'repeating_element': pattern.get('selector', 'unknown'),
                'count': pattern.get('count', 0),
                'type': pattern.get('type', 'unknown')
            }
            
            # Combine semantic + structural for better strategy
            hint['combined_strategy'] = f"""
{semantics.extraction_strategy}

Structural context:
- Each {pattern.get('selector', 'item')} contains one {semantics.field_name}
- There are {pattern.get('count', 'multiple')} instances
- Focus extraction within each {pattern.get('selector', 'item')} element
"""
        
        return hint
    
    def _generic_field_semantics(self, field: str) -> FieldSemantics:
        """Fallback generic semantics when LLM fails"""
        return FieldSemantics(
            field_name=field,
            semantic_meaning=f'Value for {field}',
            likely_html_elements=[f'.{field}', f'#{field}', f'[data-{field}]'],
            common_attributes=[f'data-{field}', field],
            common_class_patterns=[field],
            extraction_strategy=f'Look for elements with class or attribute named "{field}"',
            code_example=f"elem.select_one('.{field}').text if elem.select_one('.{field}') else None",
            confidence=0.3
        )
    
    def _generic_extraction_hint(self, field: str) -> Dict[str, Any]:
        """Fallback generic hint when mapping fails"""
        return {
            'semantic_meaning': f'Value for {field}',
            'likely_locations': [f'.{field}', f'#{field}'],
            'common_attributes': [field, f'data-{field}'],
            'common_classes': [field],
            'extraction_strategy': f'Standard CSS selector or attribute lookup',
            'code_example': f"elem.select_one('.{field}').text",
            'confidence': 0.3
        }
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response (may be wrapped in markdown)"""
        import re
        
        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to parse the entire response as JSON
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find any JSON object in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        logger.error("    Failed to extract JSON from LLM response")
        return {}
    
    def clear_cache(self, domain: Optional[str] = None):
        """Clear cached data (for testing or updates)"""
        if domain:
            # Clear specific domain
            if domain in self.domain_contexts:
                del self.domain_contexts[domain]
            
            if self.enable_cache:
                cache_file = self.domain_cache_dir / f"{domain}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"   Cleared cache for {domain}")
        else:
            # Clear all caches
            self.domain_contexts.clear()
            self.field_semantics.clear()
            if self.enable_cache:
                for cache_file in self.domain_cache_dir.glob("*.json"):
                    cache_file.unlink()
                for cache_file in self.field_cache_dir.glob("*.json"):
                    cache_file.unlink()
                logger.info("   Cleared all caches")

