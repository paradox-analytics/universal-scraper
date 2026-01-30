"""
JSON Structure Analyzer - Similar to HTMLStructureAnalyzer
Analyzes JSON structure before extraction to understand schema and field mappings

This ensures consistent extraction quality, similar to how HTML structure analysis
guides code generation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import litellm
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


class JSONStructureAnalyzer:
    """
    Analyzes JSON structure to guide extraction
    Similar to HTMLStructureAnalyzer but for JSON data
    
    Benefits:
    - Understands JSON schema before extraction
    - Maps requested fields to actual JSON keys
    - Caches analysis for reuse
    - Improves extraction accuracy
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", enable_cache: bool = True):
        """
        Initialize JSON structure analyzer
        
        Args:
            api_key: API key for LLM
            model: Model to use for analysis
            enable_cache: Enable persistent caching (uses Apify KV Store on Apify)
        """
        self.api_key = api_key
        self.model = model
        self.enable_cache = enable_cache
        
        # In-memory cache for current run (L1 cache)
        self.analysis_cache = {}  # Cache by domain + structure hash
        self.domain_fields_cache = {}  # NEW: Lightweight cache by domain + fields (for pre-warming)
        
        # Persistent cache (L2 cache) - uses Apify KV Store on Apify, file cache locally
        if self.enable_cache:
            try:
                from .unified_cache import UnifiedPatternCache
                # UnifiedPatternCache automatically detects Apify and uses KV Store
                self.persistent_cache = UnifiedPatternCache(force_local=False)
                logger.info(" JSON Structure Analyzer cache enabled (persists across runs)")
            except Exception as e:
                logger.warning(f"  Failed to initialize persistent cache: {e}, using in-memory only")
                self.persistent_cache = None
        else:
            self.persistent_cache = None
        
        logger.info(f" JSON Structure Analyzer initialized with {model}")
    
    def get_cached_mappings(self, url: str, fields: List[str]) -> Optional[Dict[str, Any]]:
        """
        NEW: Get cached field mappings BEFORE fetching HTML (pre-warming optimization)
        
        For scale (thousands of pages/day): Check if we've analyzed this domain+fields
        combination before. If yes, return cached mappings without needing JSON data.
        
        Args:
            url: Source URL
            fields: Requested fields
            
        Returns:
            Cached field mappings or None if not found
        """
        domain = urlparse(url).netloc
        domain_fields_key = f"{domain}_{'_'.join(sorted(fields))}"
        
        # Check in-memory cache first (L1) - fast path
        if domain_fields_key in self.domain_fields_cache:
            cached_mappings = self.domain_fields_cache[domain_fields_key]
            logger.info(f" Pre-warmed cache hit (memory): Using cached field mappings for {domain} (fields: {', '.join(fields)})")
            return cached_mappings
        
        # Note: Persistent cache check happens in analyze() method (async)
        # This keeps get_cached_mappings synchronous for fast pre-warming checks
        return None
    
    def analyze(
        self,
        json_data: Any,
        url: str,
        fields: List[str],
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze JSON structure to guide extraction
        
        Returns structure analysis with:
        - schema: Understanding of JSON structure
        - field_mappings: Map requested fields to JSON keys/paths
        - extraction_path: Path to data array/items
        - item_structure: Structure of individual items
        - confidence: 0.0-1.0
        
        Args:
            json_data: JSON data to analyze (can be dict, list, or nested)
            url: Source URL (for caching)
            fields: Requested fields to extract
            context: User's extraction goal
            use_cache: Whether to use cached analysis
        """
        domain = urlparse(url).netloc
        
        # NEW: Check lightweight domain+fields cache first (pre-warming)
        domain_fields_key = f"{domain}_{'_'.join(sorted(fields))}"
        
        # Check in-memory cache (L1)
        if use_cache and domain_fields_key in self.domain_fields_cache:
            cached_mappings = self.domain_fields_cache[domain_fields_key]
            logger.info(f" Using pre-warmed field mappings for {domain} (skipped structure analysis)")
            return cached_mappings
        
        # Check persistent cache (L2) - Apify KV Store or file cache
        if use_cache and self.persistent_cache:
            try:
                import asyncio
                cache_key = f"json_structure_{domain_fields_key}"
                # Use asyncio.run if not in async context, or get_event_loop if already running
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Already in async context, create task
                        task = asyncio.create_task(self.persistent_cache.backend.get(cache_key))
                        # Don't await here (sync method), but save task for later
                        # For now, just check sync - we'll improve this later
                        pass
                    else:
                        cached_data = loop.run_until_complete(self.persistent_cache.backend.get(cache_key))
                        if cached_data and isinstance(cached_data, dict):
                            self.domain_fields_cache[domain_fields_key] = cached_data
                            logger.info(f" Using pre-warmed field mappings from persistent cache for {domain} (skipped structure analysis)")
                            return cached_data
                except RuntimeError:
                    # No event loop, create one
                    cached_data = asyncio.run(self.persistent_cache.backend.get(cache_key))
                    if cached_data and isinstance(cached_data, dict):
                        self.domain_fields_cache[domain_fields_key] = cached_data
                        logger.info(f" Using pre-warmed field mappings from persistent cache for {domain} (skipped structure analysis)")
                        return cached_data
            except Exception as e:
                logger.debug(f"Persistent cache check failed: {e}")
        
        # Create cache key from domain + structure sample + fields
        structure_sample = self._sample_json_structure(json_data)
        cache_key_str = f"{domain}_{structure_sample}_{'_'.join(sorted(fields))}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        
        # Check detailed cache (structure-specific)
        if use_cache and cache_key in self.analysis_cache:
            logger.info(f" Using cached JSON structure analysis for {domain}")
            analysis = self.analysis_cache[cache_key]
            # Also cache in lightweight cache for pre-warming
            if domain_fields_key not in self.domain_fields_cache:
                self.domain_fields_cache[domain_fields_key] = analysis
            return analysis
        
        logger.info(f" Analyzing JSON structure for {domain}...")
        logger.info(f"   Requested fields: {', '.join(fields)}")
        
        # PHASE 1: Fast JSON Pattern Detection (no LLM) - like DOM pattern detection
        logger.info("   Phase 1: Fast JSON pattern detection (no LLM)...")
        pattern_analysis = self._detect_json_patterns(json_data, fields)
        
        # If pattern detection has high confidence, return immediately!
        if pattern_analysis['confidence'] >= 0.85:
            logger.info(f"    High-confidence JSON pattern detected (confidence={pattern_analysis['confidence']:.2f})")
            logger.info("    Skipping LLM call (saving time & cost)")
            
            # Cache and return (both caches)
            self.analysis_cache[cache_key] = pattern_analysis
            # Also cache in lightweight cache for pre-warming
            self.domain_fields_cache[domain_fields_key] = pattern_analysis
            
            # Save to persistent cache (async, non-blocking)
            if self.persistent_cache:
                try:
                    import asyncio
                    cache_key_persistent = f"json_structure_{domain_fields_key}"
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.persistent_cache.backend.set(cache_key_persistent, pattern_analysis))
                        else:
                            loop.run_until_complete(self.persistent_cache.backend.set(cache_key_persistent, pattern_analysis))
                    except RuntimeError:
                        asyncio.run(self.persistent_cache.backend.set(cache_key_persistent, pattern_analysis))
                except Exception as e:
                    logger.debug(f"Failed to save to persistent cache: {e}")
            
            return pattern_analysis
        
        # PHASE 2: LLM Analysis (fallback for lower confidence)
        logger.info(f"   Phase 2: LLM analysis (pattern confidence={pattern_analysis['confidence']:.2f}, need LLM)")
        
        # Find the data array (items to extract)
        data_array = self._find_data_array(json_data)
        
        if not data_array:
            logger.warning("    No data array found in JSON")
            return {
                'schema': 'unknown',
                'field_mappings': {},
                'extraction_path': None,
                'item_structure': {},
                'confidence': pattern_analysis.get('confidence', 0.0),
                'error': 'No data array found',
                'source': 'pattern_detection'  # Mark as non-LLM
            }
        
        logger.info(f"    Found data array with {len(data_array)} items")
        
        # Sample first item for analysis
        sample_item = data_array[0] if data_array else {}
        
        # Use pattern analysis hints in LLM prompt
        pattern_hints = ""
        if pattern_analysis.get('field_mappings'):
            pattern_hints = f"\n\n**Pattern Detection Results** (confidence={pattern_analysis['confidence']:.2f}):\n"
            for field, mapping in list(pattern_analysis['field_mappings'].items())[:5]:
                pattern_hints += f"  - {field}: {mapping.get('json_key', 'N/A')} (confidence: {mapping.get('confidence', 0):.2f})\n"
        
        # Build analysis prompt
        context_str = f"\n\n**User's Goal**: {context}" if context else ""
        
        # NEW: Add website-specific field context (critical for correct extraction)
        field_context = self._get_field_context(url, fields)
        
        prompt = f"""Analyze this JSON structure to guide data extraction.

**Source URL**: {url}{context_str}{pattern_hints}

**Requested Fields**: {', '.join(fields)}
{field_context}

**JSON Structure Sample** (first item from data array):
```json
{json.dumps(sample_item, indent=2, default=str)[:2000]}
```

**Your Task**:
1. Understand the JSON schema and structure
2. Map requested fields to actual JSON keys/paths
3. Identify the extraction path to the data array
4. Describe the structure of individual items

**Respond with JSON**:
{{
    "schema": "description of overall JSON structure",
    "extraction_path": "path to data array (e.g., 'props.pageProps.products')",
    "item_structure": {{
        "description": "structure of individual items",
        "keys": ["list", "of", "available", "keys"]
    }},
    "field_mappings": {{
        "requested_field": {{
            "json_key": "actual key in JSON",
            "path": "nested.path.if.needed",
            "confidence": 0.0-1.0,
            "notes": "any special handling needed"
        }}
    }},
    "confidence": 0.0-1.0,
    "reasoning": "why these mappings are correct"
}}

**Important**:
- Map fields semantically (e.g., "title" might be "name", "productName", "title", etc.)
- Handle nested paths (e.g., "price" might be at "pricing.amount")
- **CRITICAL - NESTED OBJECTS**: Many fields are nested objects, not direct values:
  - "color" might be `{"colorName": "Navy"}` → use path "color.colorName"
  - "variant" might be `{"name": "Large"}` → use path "variant.name"
  - "price" might be `{"amount": 52, "currency": "USD"}` → use path "price.amount"
  - When a field value is an object/dict, look for nested keys like: name, value, title, label, displayName, colorName, variantName
  - Always extract the STRING value from nested objects, not the object itself
- Consider variations and aliases
- If a field isn't found, set confidence to 0.0
- **CRITICAL**: For "title" fields, prefer FULL descriptive names over short labels (e.g., "Fancy Feast Gems Mousse..." over "Trial Size")
- Look for the most complete, descriptive value available (check data-name, fullName, productName, etc.)
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing JSON structures for data extraction. Be precise and accurate."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            analysis = json.loads(content)
            
            # Validate and normalize
            analysis.setdefault('field_mappings', {})
            analysis.setdefault('confidence', 0.5)
            analysis.setdefault('extraction_path', None)
            analysis.setdefault('item_structure', {})
            analysis['source'] = 'llm_analysis'  # Mark as LLM-based
            
            # Merge with pattern detection results (LLM can override)
            if pattern_analysis.get('field_mappings'):
                # LLM mappings take precedence, but keep pattern mappings as fallback
                for field, pattern_mapping in pattern_analysis['field_mappings'].items():
                    if field not in analysis['field_mappings']:
                        analysis['field_mappings'][field] = pattern_mapping
            
            logger.info(f"    JSON structure analyzed (confidence: {analysis['confidence']:.2f})")
            logger.info(f"    Extraction path: {analysis.get('extraction_path', 'N/A')}")
            
            # Cache result (both caches for different use cases)
            self.analysis_cache[cache_key] = analysis
            # Also cache in lightweight cache for pre-warming (domain+fields only)
            self.domain_fields_cache[domain_fields_key] = analysis
            
            # Save to persistent cache (async, non-blocking)
            if self.persistent_cache:
                try:
                    import asyncio
                    cache_key_persistent = f"json_structure_{domain_fields_key}"
                    # Save asynchronously (non-blocking)
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context, create task
                            asyncio.create_task(self.persistent_cache.backend.set(cache_key_persistent, analysis))
                        else:
                            loop.run_until_complete(self.persistent_cache.backend.set(cache_key_persistent, analysis))
                    except RuntimeError:
                        # No event loop, create one
                        asyncio.run(self.persistent_cache.backend.set(cache_key_persistent, analysis))
                except Exception as e:
                    logger.debug(f"Failed to save to persistent cache: {e}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"    Failed to parse LLM response as JSON: {e}")
            logger.error(f"   Raw response: {content[:500]}")
            return self._create_fallback_analysis(json_data, fields)
        except Exception as e:
            logger.error(f"    JSON structure analysis failed: {e}")
            return self._create_fallback_analysis(json_data, fields)
    
    def _find_data_array(self, json_data: Any) -> List[Dict]:
        """
        Find the data array in JSON (items to extract)
        """
        if isinstance(json_data, list):
            # If it's already a list, check if items are dicts
            if json_data and isinstance(json_data[0], dict):
                return json_data
            return []
        
        if not isinstance(json_data, dict):
            return []
        
        # Look for arrays of dicts
        arrays = []
        
        def find_arrays(obj, path=""):
            if isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    arrays.append((path, obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    find_arrays(value, f"{path}.{key}" if path else key)
        
        find_arrays(json_data)
        
        if not arrays:
            return []
        
        # Return the largest array (likely the data)
        arrays.sort(key=lambda x: len(x[1]), reverse=True)
        return arrays[0][1]
    
    def _sample_json_structure(self, json_data: Any, max_chars: int = 2000) -> str:
        """
        Create a structure sample for caching (shows structure without full data)
        """
        try:
            if isinstance(json_data, list):
                if json_data:
                    # Sample first item structure
                    sample = json_data[0]
                    return json.dumps(sample, default=str)[:max_chars]
                return "[]"
            
            if isinstance(json_data, dict):
                # Create structure-only sample (keys only, no values)
                structure = {}
                for key, value in list(json_data.items())[:10]:  # First 10 keys
                    if isinstance(value, (dict, list)):
                        structure[key] = type(value).__name__
                    else:
                        structure[key] = type(value).__name__
                
                return json.dumps(structure, default=str)[:max_chars]
            
            return str(type(json_data).__name__)
        except:
            return "unknown"
    
    def _create_fallback_analysis(self, json_data: Any, fields: List[str]) -> Dict[str, Any]:
        """
        Create a fallback analysis when LLM fails
        """
        data_array = self._find_data_array(json_data)
        
        if not data_array or not data_array[0]:
            return {
                'schema': 'unknown',
                'field_mappings': {},
                'extraction_path': None,
                'item_structure': {},
                'confidence': 0.0,
                'error': 'LLM analysis failed and no fallback available'
            }
        
        # Simple field mapping: try exact matches
        sample_item = data_array[0]
        field_mappings = {}
        
        if isinstance(sample_item, dict):
            available_keys = list(sample_item.keys())
            for field in fields:
                # Try exact match first
                if field in available_keys:
                    field_mappings[field] = {
                        'json_key': field,
                        'path': field,
                        'confidence': 0.8,
                        'notes': 'Exact match'
                    }
                else:
                    # Try case-insensitive match
                    field_lower = field.lower()
                    for key in available_keys:
                        if key.lower() == field_lower:
                            field_mappings[field] = {
                                'json_key': key,
                                'path': key,
                                'confidence': 0.7,
                                'notes': 'Case-insensitive match'
                            }
                            break
        
        return {
            'schema': 'fallback',
            'field_mappings': field_mappings,
            'extraction_path': None,
            'item_structure': {
                'keys': list(sample_item.keys()) if isinstance(sample_item, dict) else []
            },
            'confidence': 0.5,
            'source': 'fallback'
        }
    
    def _get_field_context(self, url: str, fields: List[str]) -> str:
        """
        Generate website-specific context about what each field means.
        
        This is CRITICAL: "next token is always the most important for context on what to actually extract,
        and it varies per website" - we need to tell the LLM what "title" means for THIS website.
        
        Args:
            url: Source URL
            fields: Requested fields
            
        Returns:
            Context string explaining field meanings for this website
        """
        domain = urlparse(url).netloc.lower()
        field_contexts = []
        
        # Detect website type from domain
        is_ecommerce = any(site in domain for site in ['chewy', 'amazon', 'ebay', 'etsy', 'shopify', 'product', 'store', 'buy'])
        is_news = any(site in domain for site in ['news', 'article', 'blog', 'medium', 'substack'])
        is_social = any(site in domain for site in ['reddit', 'twitter', 'facebook', 'linkedin', 'instagram'])
        is_forum = any(site in domain for site in ['forum', 'discussion', 'community', 'stackoverflow'])
        
        for field in fields:
            field_lower = field.lower()
            
            # Title field - varies by website type
            if 'title' in field_lower or field_lower == 'name':
                if is_ecommerce:
                    field_contexts.append(
                        f"- **{field}**: FULL product name/description (e.g., 'Fancy Feast Gems Mousse Salmon, Tuna, Chicken & Beef Halo of Savory Gravy Variety Pack Pate Wet Cat Food'). "
                        f"DO NOT extract short labels like 'Trial Size' or 'Variety Pack'. Look for the complete, descriptive product name. "
                        f"Check keys like 'data-name', 'fullName', 'productName', 'name', 'title' and prefer the LONGEST, most descriptive value."
                    )
                elif is_news:
                    field_contexts.append(
                        f"- **{field}**: Full article headline (complete title, not truncated). "
                        f"Look for the most complete headline available."
                    )
                elif is_social or is_forum:
                    field_contexts.append(
                        f"- **{field}**: Post/thread title (complete, not truncated). "
                        f"Extract the full title as it appears."
                    )
                else:
                    field_contexts.append(
                        f"- **{field}**: Complete, descriptive name/title. "
                        f"Prefer full names over short labels. Look for the longest, most descriptive value available."
                    )
            
            # URL field
            elif 'url' in field_lower or 'link' in field_lower:
                field_contexts.append(
                    f"- **{field}**: Complete URL (e.g., 'https://www.chewy.com/dp/12345'). "
                    f"Extract from 'href', 'url', 'link', 'productUrl', or construct from 'data-id' if needed."
                )
            
            # Rating field
            elif 'rating' in field_lower:
                field_contexts.append(
                    f"- **{field}**: Numeric rating (e.g., 4.5, 4.6). Extract only the number, no stars or text."
                )
            
            # Review count field
            elif 'review' in field_lower and 'count' in field_lower:
                field_contexts.append(
                    f"- **{field}**: Number of reviews (e.g., 2391, 1655). Extract only the number."
                )
        
        if field_contexts:
            return "\n\n**FIELD-SPECIFIC CONTEXT** (website-specific meanings):\n" + "\n".join(field_contexts)
        return ""
    
    def _detect_json_patterns(self, json_data: Any, fields: List[str]) -> Dict[str, Any]:
        """
        Fast JSON pattern detection (no LLM) - similar to DOM pattern detection
        
        Uses heuristics to detect:
        - Field mappings (exact matches, case-insensitive, synonyms)
        - Data array location
        - Item structure
        - Confidence score
        
        Returns:
            Dict with pattern analysis (same format as LLM analysis)
        """
        # Find the data array
        data_array = self._find_data_array(json_data)
        
        if not data_array or not data_array[0]:
            return {
                'schema': 'unknown',
                'field_mappings': {},
                'extraction_path': None,
                'item_structure': {},
                'confidence': 0.0,
                'source': 'pattern_detection'
            }
        
        sample_item = data_array[0]
        if not isinstance(sample_item, dict):
            return {
                'schema': 'unknown',
                'field_mappings': {},
                'extraction_path': None,
                'item_structure': {},
                'confidence': 0.0,
                'source': 'pattern_detection'
            }
        
        # Fast field mapping using heuristics
        field_mappings = {}
        available_keys = list(sample_item.keys())
        field_synonyms = {
            'title': ['name', 'productName', 'product_name', 'title', 'label', 'heading'],
            'price': ['price', 'cost', 'amount', 'pricing', 'value'],
            'url': ['url', 'link', 'href', 'uri', 'permalink'],
            'product url': ['productUrl', 'product_url', 'url', 'link', 'href', 'uri', 'permalink', 'productLink', 'product_link'],
            'rating': ['rating', 'score', 'stars', 'reviewRating', 'averageRating'],
            'review count': ['reviewCount', 'review_count', 'reviews', 'numReviews', 'num_reviews', 'reviewCount'],
            'description': ['description', 'desc', 'summary', 'details', 'about'],
            'image': ['image', 'img', 'picture', 'photo', 'thumbnail', 'imageUrl', 'image_url']
        }
        
        confidence_sum = 0.0
        for field in fields:
            field_lower = field.lower()
            best_match = None
            best_confidence = 0.0
            
            # Try exact match
            for key in available_keys:
                if key.lower() == field_lower:
                    best_match = key
                    best_confidence = 0.9
                    break
            
            # Try synonyms
            if not best_match:
                synonyms = field_synonyms.get(field_lower, [])
                for synonym in synonyms:
                    for key in available_keys:
                        if key.lower() == synonym.lower():
                            best_match = key
                            best_confidence = 0.8
                            break
                    if best_match:
                        break
            
            # Try partial match
            if not best_match:
                for key in available_keys:
                    key_lower = key.lower()
                    if field_lower in key_lower or key_lower in field_lower:
                        best_match = key
                        best_confidence = 0.6
                        break
            
            if best_match:
                field_mappings[field] = {
                    'json_key': best_match,
                    'path': best_match,
                    'confidence': best_confidence,
                    'notes': 'Pattern detection (fast)'
                }
                confidence_sum += best_confidence
        
        # Calculate overall confidence
        avg_confidence = confidence_sum / len(fields) if fields else 0.0
        
        # Boost confidence if we found most fields
        if len(field_mappings) >= len(fields) * 0.8:  # 80%+ fields found
            avg_confidence = min(avg_confidence * 1.1, 0.95)
        
        return {
            'schema': 'pattern_detected',
            'field_mappings': field_mappings,
            'extraction_path': None,  # Pattern detection doesn't determine path
            'item_structure': {
                'keys': available_keys,
                'description': f'Detected {len(available_keys)} keys in items'
            },
            'confidence': avg_confidence,
            'source': 'pattern_detection'
        }

