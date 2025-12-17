"""
LLM JSON Source Analyzer (Context-Aware)

Intelligently ranks JSON sources by relevance to user's extraction goal.
Solves the problem of multiple JSON sources (cart, footer, API, etc.)
"""

import json
import logging
from typing import List, Dict, Any, Optional
import litellm
from .context_manager import ExtractionContext

logger = logging.getLogger(__name__)


class LLMJsonAnalyzer:
    """
    Uses LLM to analyze and select the best JSON source
    
    NEW APPROACH (Simplified):
    Instead of ranking all sources, we ask LLM to pick THE BEST ONE.
    This is faster, cheaper, and more accurate than complex ranking.
    
    Critical for choosing the RIGHT JSON when multiple sources exist.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        enable_cache: bool = True
    ):
        """
        Initialize JSON analyzer
        
        Args:
            api_key: OpenAI API key (or any LiteLLM-supported provider)
            model: Model to use for analysis
            enable_cache: Cache analysis results
        """
        self.api_key = api_key
        self.model = model
        self.enable_cache = enable_cache
        self._cache = {}
        
        logger.info(f" JSON Analyzer initialized with {model}")
    
    def rank_sources(
        self,
        json_sources: Dict[str, Any],
        url: str,
        context: ExtractionContext
    ) -> List[Dict[str, Any]]:
        """
        Rank JSON sources based on user's extraction context
        
        Args:
            json_sources: Dict of {source_name: json_data}
            url: Page URL
            context: User's extraction context
        
        Returns:
            List of rankings: [
                {
                    "source": "source_name",
                    "confidence": 0.0-1.0,
                    "reasoning": "Why this source is relevant",
                    "estimated_items": int
                },
                ...
            ]
            Sorted by confidence (highest first)
        """
        
        if not json_sources:
            logger.warning(" No JSON sources to rank")
            return []
        
        # Check cache
        cache_key = self._get_cache_key(json_sources, url, context)
        if self.enable_cache and cache_key in self._cache:
            logger.info(" Using cached JSON source rankings")
            return self._cache[cache_key]
        
        logger.info(f" Analyzing {len(json_sources)} JSON source(s)...")
        
        # PRE-FILTER: Remove obvious non-data sources before LLM analysis
        filtered_sources = self._pre_filter_sources(json_sources, context)
        logger.info(f"   → {len(filtered_sources)} source(s) after pre-filtering")
        
        # Prepare sources for LLM analysis (use filtered sources)
        # HARD LIMIT: Max 15 sources to prevent token overflow
        sources_to_analyze = dict(list(filtered_sources.items())[:15])
        if len(filtered_sources) > 15:
            logger.warning(f" Too many sources ({len(filtered_sources)}), analyzing top 15 only")
        
        sources_summary = []
        for idx, (source_name, source_data) in enumerate(sources_to_analyze.items(), 1):
            # Get AGGRESSIVE summary of this source (prevent token overflow)
            summary = self._summarize_json_source_aggressive(source_name, source_data, context)
            sources_summary.append({
                'index': idx,
                'name': self._sanitize_for_json(source_name, max_length=50),  # Sanitize source name too
                'summary': summary
            })
        
        # Build LLM prompt
        prompt = f"""You are an expert at analyzing JSON data sources for web scraping.

{context.to_llm_prompt_section()}

URL: {url}

JSON SOURCES DISCOVERED:
"""
        
        for source in sources_summary:
            prompt += f"\n{source['index']}. SOURCE: {source['name']}\n{source['summary']}\n"
        
        prompt += f"""

TASK:
Rank these JSON sources by likelihood of containing the TARGET data for: "{context.goal}"

The user wants {context.data_type} data{f" with fields: {', '.join(context.fields)}" if context.fields else ""}.

Consider:
1. Does the source have arrays/lists of items (good sign)?
2. Do field names match what the user wants?
3. Is it likely the PRIMARY data vs metadata/config?
4. How many items might it contain?

Common patterns:
- __NEXT_DATA__ often has page data in props.pageProps
- API responses usually have items in data/results/items arrays
- JSON-LD is usually just metadata
- Window variables can be config or data

Respond in JSON:
{{
    "rankings": [
        {{
            "source": "exact_source_name",
            "confidence": 0.0-1.0,
            "reasoning": "Why this source likely has {context.data_type}",
            "estimated_items": approximate_number_or_0
        }},
        ...
    ]
}}

Rank ALL sources. Order by confidence (highest first).
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON data source analyzer for web scraping. Be analytical and precise. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent JSON
                max_tokens=1500  # Increased to give more room
            )
            
            content = response.choices[0].message.content
            
            # Log raw response for debugging (first 200 chars)
            logger.debug(f"LLM raw response (first 200 chars): {content[:200]}")
            
            # Parse JSON response
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content
            
            # Extract rankings
            rankings = result.get('rankings', [])
            
            if not rankings:
                logger.warning(" LLM returned no rankings")
                return []
            
            # Validate and sort by confidence
            valid_rankings = []
            for rank in rankings:
                if not isinstance(rank, dict):
                    continue
                
                rank.setdefault('confidence', 0.5)
                rank.setdefault('reasoning', 'No reasoning provided')
                rank.setdefault('estimated_items', 0)
                
                valid_rankings.append(rank)
            
            # Sort by confidence (descending)
            valid_rankings.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Log rankings
            logger.info(f" JSON Source Rankings:")
            for i, rank in enumerate(valid_rankings[:5], 1):  # Show top 5
                logger.info(f"   {i}. {rank['source']} (confidence: {rank['confidence']:.2f})")
                logger.info(f"      → {rank['reasoning']}")
            
            # Cache result
            if self.enable_cache:
                self._cache[cache_key] = valid_rankings
            
            return valid_rankings
            
        except json.JSONDecodeError as e:
            logger.error(f" JSON source ranking failed - LLM returned malformed JSON: {e}")
            logger.error(f"   Error at line {e.lineno}, column {e.colno}")
            if 'content' in locals():
                logger.error(f"   Raw response (first 500 chars): {content[:500]}")
            # Fallback: Return all sources with equal low confidence
            fallback = []
            for source_name in json_sources.keys():
                fallback.append({
                    'source': source_name,
                    'confidence': 0.3,
                    'reasoning': f'JSON parse error: {str(e)}',
                    'estimated_items': 0
                })
            return fallback
        except Exception as e:
            logger.error(f" JSON source ranking failed: {type(e).__name__}: {e}")
            # Fallback: Return all sources with equal low confidence
            fallback = []
            for source_name in json_sources.keys():
                fallback.append({
                    'source': source_name,
                    'confidence': 0.3,
                    'reasoning': f'Ranking failed: {str(e)}',
                    'estimated_items': 0
                })
            return fallback
    
    def select_best_source(
        self,
        json_sources: Dict[str, Any],
        url: str,
        context: ExtractionContext
    ) -> Optional[str]:
        """
        SELECT THE BEST JSON SOURCE (Simplified Approach)
        
        Instead of ranking all sources, we ask LLM: "Which ONE source has the target data?"
        This is simpler, faster, and MORE ACCURATE than complex ranking.
        
        Args:
            json_sources: Dict of {source_name: json_data}
            url: Page URL
            context: User's extraction context
        
        Returns:
            Name of the best source, or None if selection fails
        """
        
        if not json_sources:
            logger.warning(" No JSON sources to select from")
            return None
        
        # Pre-filter sources (remove obvious non-data)
        filtered_sources = self._pre_filter_sources(json_sources, context)
        
        if not filtered_sources:
            logger.warning(" All sources filtered out, falling back to first unfiltered source")
            return list(json_sources.keys())[0] if json_sources else None
        
        logger.info(f" Selecting best source from {len(filtered_sources)} candidates (from {len(json_sources)} total)")
        
        # If only one source, return it
        if len(filtered_sources) == 1:
            source_name = list(filtered_sources.keys())[0]
            logger.info(f" Only one source available: {source_name}")
            return source_name
        
        # Check cache
        cache_key = f"select_{url}_{context.goal}_{len(filtered_sources)}"
        if self.enable_cache and cache_key in self._cache:
            logger.info(" Using cached source selection")
            return self._cache[cache_key]
        
        # Limit to top 10 sources to prevent token overflow
        if len(filtered_sources) > 10:
            logger.warning(f" Too many sources ({len(filtered_sources)}), analyzing top 10 only")
            filtered_sources = dict(list(filtered_sources.items())[:10])
        
        # Create simple summaries for LLM
        source_summaries = {}
        for source_name, source_data in filtered_sources.items():
            summary = self._create_simple_summary(source_name, source_data)
            source_summaries[source_name] = summary
        
        # Build selection prompt (MUCH SIMPLER than ranking)
        prompt = f"""Analyze JSON sources from: {url}

USER'S GOAL: {context.goal}
TARGET DATA TYPE: {context.data_type or 'unknown'}
EXPECTED FIELDS: {', '.join(context.fields or [])}

Available JSON sources:
{json.dumps(source_summaries, indent=2)}

TASK: Which ONE source contains the target data?

Respond in JSON:
{{
    "best_source": "exact_source_name",
    "reasoning": "Why this source has the target data",
    "confidence": 0.0-1.0
}}

Pick the MOST RELEVANT source. If none match, pick closest."""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON source selector for web scraping. Pick the BEST source for the user's goal. Be decisive."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            result = json.loads(content) if isinstance(content, str) else content
            
            best_source = result.get('best_source')
            reasoning = result.get('reasoning', 'No reasoning provided')
            confidence = result.get('confidence', 0.0)
            
            if best_source and best_source in filtered_sources:
                logger.info(f" LLM selected: {best_source} (confidence: {confidence:.2f})")
                logger.info(f"   → {reasoning}")
                
                # Cache result
                if self.enable_cache:
                    self._cache[cache_key] = best_source
                
                return best_source
            else:
                logger.warning(f" LLM returned invalid source: {best_source}")
                return list(filtered_sources.keys())[0]
                
        except Exception as e:
            logger.error(f" Source selection failed: {type(e).__name__}: {e}")
            return list(filtered_sources.keys())[0]
    
    def _create_simple_summary(self, source_name: str, source_data: Any) -> Dict[str, Any]:
        """
        Create a simple summary of a JSON source for LLM selection
        
        Much simpler than aggressive summarization - just the facts.
        """
        summary = {
            "name": source_name,
            "type": type(source_data).__name__
        }
        
        if isinstance(source_data, dict):
            # Count arrays
            arrays = [k for k, v in source_data.items() if isinstance(v, list)]
            summary["has_arrays"] = len(arrays) > 0
            summary["array_keys"] = arrays[:5]  # First 5
            summary["top_keys"] = list(source_data.keys())[:10]  # First 10
            
            # Get sample of largest array
            if arrays:
                largest_array_key = max(arrays, key=lambda k: len(source_data[k]))
                largest_array = source_data[largest_array_key]
                summary["largest_array"] = {
                    "key": largest_array_key,
                    "length": len(largest_array),
                    "sample_item": largest_array[0] if largest_array else None
                }
                
        elif isinstance(source_data, list):
            summary["length"] = len(source_data)
            summary["sample_item"] = source_data[0] if source_data else None
        
        return summary
    
    def _pre_filter_sources(
        self,
        json_sources: Dict[str, Any],
        context: ExtractionContext
    ) -> Dict[str, Any]:
        """
        Pre-filter sources to remove obvious non-data before LLM analysis
        NOW CONTEXT-AWARE: Checks if JSON structure could possibly contain target data
        Reduces token count and improves accuracy
        """
        # Analytics/tracking patterns to exclude (check name first)
        ANALYTICS_PATTERNS = [
            'pixel', 'track', 'quota', 'consent', 'cookie', 'gdpr',
            'analytics', 'gtm', 'ga_', 'facebook', 'google_tag',
            'amplitude', 'mixpanel', 'segment', 'hotjar', 'clarity',
            'config', 'settings', 'constants', 'env', 'build'
        ]
        
        # Known analytics/tracking structure signatures
        ANALYTICS_STRUCTURES = [
            {'_framework', '_data', '_paths'},  # Segment/analytics framework
            {'dsn', 'enabled', 'environment'},  # Sentry config
            {'pixelId', 'limitedDataUseEnabled'},  # Pixel tracking
        ]
        
        filtered = {}
        excluded_count = 0
        
        for source_name, source_data in json_sources.items():
            # Check 1: Source name matches analytics patterns
            name_lower = source_name.lower()
            is_analytics_name = any(pattern in name_lower for pattern in ANALYTICS_PATTERNS)
            
            if is_analytics_name:
                logger.debug(f"   ⊗ Excluding analytics source (name): {source_name}")
                excluded_count += 1
                continue
            
            # Check 2: Source is empty or trivial
            if not source_data:
                logger.debug(f"   ⊗ Excluding empty source: {source_name}")
                excluded_count += 1
                continue
            
            # Check 3: Structure matches known analytics signatures
            is_analytics_structure = False
            if isinstance(source_data, dict):
                source_keys = set(source_data.keys())
                for analytics_sig in ANALYTICS_STRUCTURES:
                    if analytics_sig.issubset(source_keys):
                        logger.debug(f"   ⊗ Excluding analytics source (structure): {source_name} - matches {analytics_sig}")
                        excluded_count += 1
                        is_analytics_structure = True
                        break
            
            if is_analytics_structure:
                continue
            
            # Check 4: CONTEXT-AWARE FILTER - Does this JSON look like it could contain target data?
            could_contain_target = self._could_contain_target_data(source_data, context)
            if not could_contain_target:
                logger.debug(f"   ⊗ Excluding: {source_name} - doesn't match context ({context.data_type})")
                excluded_count += 1
                continue
            
            # Check 5: Source has array data (good sign) - skip if no arrays and small
            has_arrays = False
            if isinstance(source_data, dict):
                has_arrays = any(isinstance(v, list) for v in source_data.values())
            elif isinstance(source_data, list):
                has_arrays = len(source_data) > 0
            
            # If source has no arrays and is small, likely config/metadata
            if not has_arrays:
                data_size = len(json.dumps(source_data, default=str))
                if data_size < 500:  # Small non-array = probably config
                    logger.debug(f"   ⊗ Excluding small non-array source: {source_name}")
                    excluded_count += 1
                    continue
            
            # Keep this source
            filtered[source_name] = source_data
        
        if excluded_count > 0:
            logger.info(f"    Pre-filtered out {excluded_count} non-data source(s)")
        
        return filtered if filtered else json_sources  # Fallback to all if nothing passes
    
    def _could_contain_target_data(self, source_data: Any, context: ExtractionContext) -> bool:
        """
        CONTEXT-AWARE filter: Check if JSON structure could possibly contain target data
        
        Uses the extraction context to determine if this JSON is likely to have the data we want.
        This prevents wasting time on analytics/tracking JSON that clearly isn't relevant.
        
        Args:
            source_data: The JSON data to check
            context: User's extraction context
        
        Returns:
            True if this JSON could contain target data, False otherwise
        """
        # Convert to JSON string for analysis
        try:
            json_str = json.dumps(source_data, default=str).lower()
        except:
            return True  # If can't serialize, be safe and keep it
        
        # Get relevant keywords from context
        # Combine data_type + field names + reasoning for maximum coverage
        context_keywords = set()
        
        # Add data type
        if context.data_type:
            context_keywords.add(context.data_type.lower())
        
        # Add field names
        if context.fields:
            context_keywords.update(f.lower() for f in context.fields)
        
        # Extract keywords from reasoning and prompt
        stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been', 'will', 'should', 'would', 'could'}
        
        if context.inference_reasoning:
            # Extract meaningful words (>3 chars, not common stop words)
            words = context.inference_reasoning.lower().split()
            meaningful_words = {w for w in words if len(w) > 3 and w not in stop_words}
            context_keywords.update(meaningful_words)
        
        if context.raw_prompt:
            words = context.raw_prompt.lower().split()
            meaningful_words = {w for w in words if len(w) > 3 and w not in stop_words}
            context_keywords.update(meaningful_words)
        
        if context.goal:
            words = context.goal.lower().split()
            meaningful_words = {w for w in words if len(w) > 3 and w not in stop_words}
            context_keywords.update(meaningful_words)
        
        # Check if ANY context keyword appears in the JSON
        # This is a quick heuristic to filter out completely irrelevant JSON
        matches = sum(1 for keyword in context_keywords if keyword in json_str)
        
        # If we find 2+ matches, it's probably relevant
        # If 0-1 matches, it's probably not relevant (analytics/tracking)
        if matches >= 2:
            logger.debug(f"       Found {matches} context keyword matches - likely relevant")
            return True
        else:
            logger.debug(f"       Only {matches} context keyword matches - likely not relevant")
            return False
    
    def _sanitize_for_json(self, text: str, max_length: int = 100) -> str:
        """
        Sanitize text to be JSON-safe
        Prevents special characters from breaking LLM's JSON response
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Replace problematic characters
        text = text.replace('"', "'")      # Replace double quotes with single
        text = text.replace('\n', ' ')     # Remove newlines
        text = text.replace('\r', ' ')     # Remove carriage returns
        text = text.replace('\t', ' ')     # Remove tabs
        text = text.replace('\\', '/')     # Replace backslashes
        
        # Remove control characters (ASCII 0-31 except space)
        import re
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim and limit length
        text = text.strip()[:max_length]
        
        return text
    
    def _summarize_json_source_aggressive(
        self,
        source_name: str,
        source_data: Any,
        context: ExtractionContext
    ) -> str:
        """
        Create a HIGHLY CONDENSED summary focused on relevance to user's goal
        Prevents token overflow on large JSON sources
        ALL TEXT IS SANITIZED FOR JSON SAFETY
        """
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Type: {type(source_data).__name__}")
        
        # For dicts: focus on arrays and relevance
        if isinstance(source_data, dict):
            # Count and describe arrays (potential item lists)
            relevant_arrays = []
            for key, value in source_data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if array seems relevant to context
                    relevance = self._estimate_field_relevance(key, context)
                    if relevance > 0.3 or len(value) > 10:  # Relevant OR large
                        first_item_type = type(value[0]).__name__ if value else 'empty'
                        # SANITIZE key names
                        safe_key = self._sanitize_for_json(key, max_length=30)
                        relevant_arrays.append(f"{safe_key}({len(value)} {first_item_type})")
            
            if relevant_arrays:
                summary_parts.append(f"Arrays: {', '.join(relevant_arrays[:3])}")
            else:
                # SANITIZE key names for display
                safe_keys = [self._sanitize_for_json(str(k), max_length=20) for k in list(source_data.keys())[:5]]
                summary_parts.append(f"Keys: {', '.join(safe_keys)}")
        
        # For lists: just count and first item type
        elif isinstance(source_data, list):
            summary_parts.append(f"{len(source_data)} items")
            if len(source_data) > 0:
                first_item = source_data[0]
                if isinstance(first_item, dict):
                    # Show only potentially relevant keys
                    keys = list(first_item.keys())
                    relevant_keys = [k for k in keys if self._estimate_field_relevance(k, context) > 0.2]
                    # SANITIZE key names
                    if relevant_keys:
                        safe_keys = [self._sanitize_for_json(k, max_length=20) for k in relevant_keys[:5]]
                        summary_parts.append(f"Keys: {', '.join(safe_keys)}")
                    else:
                        safe_keys = [self._sanitize_for_json(k, max_length=20) for k in keys[:5]]
                        summary_parts.append(f"Keys: {', '.join(safe_keys)}")
        
        # Join and ensure final result is also safe
        result = " | ".join(summary_parts)
        return self._sanitize_for_json(result, max_length=150)
    
    def _estimate_field_relevance(self, field_name: str, context: ExtractionContext) -> float:
        """
        Quick heuristic: does this field name relate to user's goal?
        Returns 0.0-1.0
        """
        field_lower = field_name.lower()
        
        # Check against context fields
        if context.fields:
            for ctx_field in context.fields:
                if ctx_field.lower() in field_lower or field_lower in ctx_field.lower():
                    return 1.0
        
        # Check against data type
        if context.data_type:
            type_words = context.data_type.lower().split()
            for word in type_words:
                if len(word) > 3 and word in field_lower:
                    return 0.8
        
        # Common "data" indicators
        data_indicators = ['item', 'product', 'event', 'listing', 'result', 'entry', 'record']
        if any(ind in field_lower for ind in data_indicators):
            return 0.5
        
        return 0.1
    
    def _summarize_json_source(
        self,
        source_name: str,
        source_data: Any,
        max_depth: int = 3
    ) -> str:
        """
        Create a concise summary of a JSON source for LLM analysis
        """
        summary_parts = []
        
        # Type
        summary_parts.append(f"Type: {type(source_data).__name__}")
        
        # If dict, show keys and structure
        if isinstance(source_data, dict):
            keys = list(source_data.keys())[:10]  # Limit keys shown
            summary_parts.append(f"Keys ({len(source_data)}): {keys}")
            
            # Look for array fields (potential item lists)
            array_fields = []
            for key, value in source_data.items():
                if isinstance(value, list) and len(value) > 0:
                    array_fields.append(f"{key} ({len(value)} items)")
            
            if array_fields:
                summary_parts.append(f"Arrays: {', '.join(array_fields[:5])}")
            
            # Sample first few entries
            if len(source_data) > 0:
                sample = {}
                for i, (k, v) in enumerate(source_data.items()):
                    if i >= 5:  # Limit sample
                        break
                    if isinstance(v, (str, int, float, bool, type(None))):
                        sample[k] = v
                    elif isinstance(v, list):
                        sample[k] = f"[{len(v)} items]"
                    elif isinstance(v, dict):
                        sample[k] = f"{{dict with {len(v)} keys}}"
                
                summary_parts.append(f"Sample: {json.dumps(sample, default=str)[:200]}")
        
        # If list, show length and sample item
        elif isinstance(source_data, list):
            summary_parts.append(f"Length: {len(source_data)}")
            
            if len(source_data) > 0:
                first_item = source_data[0]
                if isinstance(first_item, dict):
                    keys = list(first_item.keys())[:10]
                    summary_parts.append(f"Item keys: {keys}")
                    summary_parts.append(f"Sample item: {json.dumps(first_item, default=str)[:200]}")
                else:
                    summary_parts.append(f"Item type: {type(first_item).__name__}")
        
        # Other types
        else:
            summary_parts.append(f"Value: {str(source_data)[:200]}")
        
        return "\n   ".join(summary_parts)
    
    def _get_cache_key(
        self,
        json_sources: Dict[str, Any],
        url: str,
        context: ExtractionContext
    ) -> str:
        """Generate cache key from sources structure and context"""
        import hashlib
        
        # Create signature from source names and structure
        source_signature = "|".join(sorted(json_sources.keys()))
        
        cache_input = f"{url}|{context.goal}|{source_signature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

