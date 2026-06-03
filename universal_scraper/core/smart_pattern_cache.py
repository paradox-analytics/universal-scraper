"""
Smart Pattern Cache - Stores and retrieves extraction patterns intelligently

This is the brain of the caching system. It:
1. Stores patterns by domain + structure hash
2. Finds matching patterns for new URLs
3. Validates patterns still work (rebuilds if website changed)
4. Tracks pattern performance (success rate, usage)

Cache Hierarchy:
1. Exact match: Same domain + same structure hash → Use cached pattern
2. Domain match: Same domain, different structure → Try pattern, validate, rebuild if needed
3. No match: New domain/structure → Use LLM, generate new pattern
"""

import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from .extraction_pattern import ExtractionPattern, PatternGenerator, PatternExecutor

logger = logging.getLogger(__name__)


class SmartPatternCache:
    """
    Intelligent pattern cache that learns and adapts.

    Key features:
    - Multi-level caching (memory → Redis → file)
    - Pattern validation and auto-rebuild
    - Usage tracking and performance metrics
    - Domain-aware pattern matching
    """

    def __init__(
        self,
        redis_cache=None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        validation_threshold: float = 0.7,  # Min success rate to keep pattern
        max_pattern_age: int = 86400 * 7,   # 7 days max age
    ):
        self.redis_cache = redis_cache
        self.api_key = api_key
        self.model_name = model_name
        self.validation_threshold = validation_threshold
        self.max_pattern_age = max_pattern_age

        # In-memory cache (L1)
        self.memory_cache: Dict[str, ExtractionPattern] = {}

        # Pattern generator and executor
        self.generator = PatternGenerator(api_key, model_name)
        self.executor = PatternExecutor()

        logger.info("🧠 Smart Pattern Cache initialized")

    async def get_pattern(
        self,
        url: str,
        fields: List[str],
        html: str,
        structure_hash: str
    ) -> Tuple[Optional[ExtractionPattern], str]:
        """
        Get a pattern for the given URL and fields.

        Returns:
            Tuple of (pattern, match_type) where match_type is:
            - "exact": Exact domain + structure match
            - "domain": Same domain, validated pattern
            - "none": No pattern found
        """
        domain = urlparse(url).netloc
        normalized_domain = self._normalize_domain(domain)

        # Generate cache keys using normalized domain for consistency
        exact_key = self._make_cache_key(normalized_domain, structure_hash, fields)
        domain_key = self._make_domain_key(normalized_domain, fields)

        # Level 1: Check memory cache (fastest)
        if exact_key in self.memory_cache:
            pattern = self.memory_cache[exact_key]
            if self._is_pattern_valid(pattern):
                logger.info(f"⚡ L1 Cache HIT (memory): {domain}")
                return pattern, "exact"

        # Level 2: Check Redis cache
        if self.redis_cache:
            pattern_data = await self._get_from_redis(exact_key)
            if pattern_data:
                pattern = ExtractionPattern.from_dict(pattern_data)
                if self._is_pattern_valid(pattern):
                    self.memory_cache[exact_key] = pattern  # Promote to L1
                    logger.info(f"⚡ L2 Cache HIT (Redis): {domain}")
                    return pattern, "exact"

            # Try domain-level match
            pattern_data = await self._get_from_redis(domain_key)
            if pattern_data:
                pattern = ExtractionPattern.from_dict(pattern_data)
                if self._is_pattern_valid(pattern):
                    # Validate pattern works on this structure
                    if await self._validate_pattern_on_html(pattern, html, fields):
                        logger.info(f"⚡ Domain Cache HIT: {domain} (different structure, pattern validated)")
                        return pattern, "domain"
                    else:
                        logger.info("⚠️ Domain pattern exists but failed validation, will regenerate")

        logger.info(f"❌ Cache MISS: {domain}")
        return None, "none"

    async def get_pattern_subset(
        self,
        url: str,
        fields: List[str],
        html: str,
        structure_hash: str
    ) -> Tuple[Optional[ExtractionPattern], List[str], List[str]]:
        """
        Find a pattern that matches a subset of requested fields.

        Returns:
            Tuple of (pattern, matched_fields, missing_fields)
            - pattern: Pattern that matches subset of fields (or None)
            - matched_fields: Fields that the pattern can extract
            - missing_fields: Fields that need to be extracted incrementally
        """
        domain = urlparse(url).netloc
        normalized_domain = self._normalize_domain(domain)

        logger.info(f"🔍 Searching for subset patterns: domain={domain} (normalized: {normalized_domain}), fields={fields}")

        # Get all patterns for this domain (try both www. and non-www.)
        all_patterns = await self.get_all_patterns(domain)

        # Also try normalized domain if different
        if domain != normalized_domain:
            normalized_patterns = await self.get_all_patterns(normalized_domain)
            # Merge without duplicates
            existing_domains = {p.domain for p in all_patterns}
            for p in normalized_patterns:
                if p.domain not in existing_domains:
                    all_patterns.append(p)

        logger.info(f"🔍 Found {len(all_patterns)} patterns for domain {domain}")

        if not all_patterns:
            return None, [], fields

        # Find pattern with largest subset match
        best_pattern = None
        best_matched_fields = []
        best_match_count = 0

        requested_fields_set = set(f.lower().strip() for f in fields)

        for pattern in all_patterns:
            if not self._is_pattern_valid(pattern):
                logger.debug(f"  Pattern {pattern.pattern_id[:8]}... invalid (age or success rate)")
                continue

            # Normalize pattern fields for comparison
            pattern_fields_set = set(f.lower().strip() for f in pattern.expected_fields)

            logger.debug(f"  Checking pattern {pattern.pattern_id[:8]}...: pattern_fields={pattern.expected_fields}, requested_fields={fields}")

            # Check if pattern's fields are a subset of requested fields
            if pattern_fields_set.issubset(requested_fields_set):
                matched_count = len(pattern_fields_set)
                logger.info(f"  ✅ Pattern {pattern.pattern_id[:8]}... matches {matched_count} fields: {pattern.expected_fields}")

                if matched_count > best_match_count:
                    # Validate pattern still works on current HTML
                    logger.info("  🔍 Validating pattern on current HTML...")
                    if await self._validate_pattern_on_html(pattern, html, list(pattern_fields_set)):
                        best_pattern = pattern
                        best_matched_fields = list(pattern.expected_fields)  # Keep original casing
                        best_match_count = matched_count
                        logger.info("  ✅ Pattern validated successfully!")
                    else:
                        logger.warning("  ⚠️ Pattern validation failed")
            else:
                logger.debug(f"  ❌ Pattern {pattern.pattern_id[:8]}... doesn't match: {pattern_fields_set} not subset of {requested_fields_set}")

        if best_pattern:
            missing_fields = [f for f in fields if f.lower().strip() not in {mf.lower().strip() for mf in best_matched_fields}]
            logger.info(f"📦 Found subset pattern: {len(best_matched_fields)}/{len(fields)} fields matched")
            logger.info(f"   Matched: {best_matched_fields}")
            logger.info(f"   Missing: {missing_fields}")
            return best_pattern, best_matched_fields, missing_fields

        logger.warning(f"❌ No subset pattern found for domain {domain} with fields {fields}")
        return None, [], fields

    async def _get_pattern_html_elements(self, pattern: ExtractionPattern, html: str) -> List[Any]:
        """
        Get the HTML elements that the pattern would use for extraction.
        This ensures incremental extraction uses the same elements.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Use the same logic as PatternExecutor
        if pattern.item_selector:
            elements = soup.select(pattern.item_selector)
            logger.debug(f"Found {len(elements)} elements using item_selector: {pattern.item_selector}")
            return elements

        # If pattern uses generated code, try to extract item selector from it
        if pattern.extraction_code:
            import re
            # Look for .select() calls in the code
            select_matches = re.findall(r"\.select\(['\"]([^'\"]+)['\"]\)", pattern.extraction_code)
            if select_matches:
                # Use the first selector (usually the item selector)
                item_selector = select_matches[0]
                elements = soup.select(item_selector)
                logger.debug(f"Found {len(elements)} elements using selector from code: {item_selector}")
                return elements

        # Fallback to container selector
        if pattern.container_selector:
            container = soup.select_one(pattern.container_selector)
            if container and pattern.item_selector:
                # Try item_selector within container
                elements = container.select(pattern.item_selector)
                if elements:
                    return elements

        logger.warning(f"Could not find HTML elements for pattern (item_selector={pattern.item_selector})")
        return []

    async def extract_incremental_fields(
        self,
        pattern: ExtractionPattern,
        html: str,
        matched_fields: List[str],
        missing_fields: List[str],
        url: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract data using existing pattern + incremental extraction for new fields.

        Returns:
            Tuple of (items, metadata) where items have all fields (matched + missing)
        """
        from .incremental_extractor import IncrementalFieldExtractor

        # Step 1: Extract existing fields using pattern (FAST)
        logger.info(f"⚡ Extracting {len(matched_fields)} fields using cached pattern...")
        pattern_items, pattern_success = await self.execute_pattern(pattern, html, url)

        if not pattern_success or not pattern_items:
            logger.warning("⚠️ Pattern execution failed, falling back to full extraction")
            return [], {"incremental": False, "reason": "pattern_failed"}

        # Step 2: Extract missing fields incrementally
        logger.info(f"🔍 Extracting {len(missing_fields)} new fields incrementally...")

        # Get HTML elements used by pattern (to ensure we extract from same elements)
        html_elements = await self._get_pattern_html_elements(pattern, html)

        # CRITICAL: Limit to the same count as pattern_items to avoid extracting wrong elements
        # If we found more elements than items, the selector might be too broad (e.g., all links)
        if len(html_elements) > len(pattern_items):
            logger.warning(f"⚠️ Found {len(html_elements)} HTML elements but pattern extracted {len(pattern_items)} items")
            logger.warning(f"   Limiting to first {len(pattern_items)} elements to match pattern count")
            # Filter to elements that are more likely to be product items (have more content)
            # Prefer elements with more text content and nested elements
            scored_elements = []
            for elem in html_elements:
                score = 0
                # Prefer elements with more text content
                text_content = elem.get_text().strip()
                if len(text_content) > 50:  # Has substantial text
                    score += 10
                # Prefer elements with nested structure (divs, spans, etc.)
                nested_elements = elem.find_all(['div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                score += len(nested_elements)
                # Penalize navigation/footer links
                if any(nav_word in text_content.lower() for nav_word in ['categories', 'footer', 'navigation', 'menu', 'legal', 'privacy', 'terms', 'ref=footer']):
                    score -= 20
                scored_elements.append((score, elem))

            # Sort by score (highest first) and take top N
            scored_elements.sort(key=lambda x: x[0], reverse=True)
            html_elements = [elem for score, elem in scored_elements[:len(pattern_items)]]
            logger.info(f"   Selected top {len(html_elements)} elements by content score")

        extractor = IncrementalFieldExtractor(
            api_key=self.api_key,
            model_name=self.model_name
        )

        incremental_results = await extractor.extract_fields(
            pattern=pattern,
            html=html,
            existing_fields=matched_fields,
            new_fields=missing_fields,
            existing_items=pattern_items,
            html_elements=html_elements  # Pass the elements pattern used
        )

        # Step 3: Merge results
        merged_items = []
        for i, pattern_item in enumerate(pattern_items):
            merged_item = pattern_item.copy()

            # Add incremental fields
            if i < len(incremental_results['items']):
                for field in missing_fields:
                    if field in incremental_results['items'][i]:
                        merged_item[field] = incremental_results['items'][i][field]

            merged_items.append(merged_item)

        metadata = {
            "incremental": True,
            "pattern_fields": matched_fields,
            "incremental_fields": missing_fields,
            "pattern_source": "cached",
            "incremental_source": incremental_results.get('source', 'unknown'),
            "pattern_items": len(pattern_items),
            "total_items": len(merged_items)
        }

        total_fields = len(matched_fields) + len(missing_fields)
        logger.info(f"✅ Incremental extraction complete: {len(merged_items)} items with {total_fields} fields")
        return merged_items, metadata

    async def store_pattern(
        self,
        pattern: ExtractionPattern,
        fields: List[str]
    ) -> bool:
        """
        Store a pattern in the cache.

        Stores at multiple levels:
        1. Exact key (domain + structure + fields)
        2. Domain key (domain + fields) - for cross-structure matching
        """
        # Normalize domain for consistent storage and retrieval
        normalized_domain = self._normalize_domain(pattern.domain)

        # Use normalized domain for keys (but keep original in pattern object for reference)
        exact_key = self._make_cache_key(normalized_domain, pattern.structure_hash, fields)
        domain_key = self._make_domain_key(normalized_domain, fields)

        # Store in memory (L1)
        self.memory_cache[exact_key] = pattern

        # Store in Redis (L2)
        if self.redis_cache:
            pattern_data = pattern.to_dict()

            # Store exact match
            await self._set_in_redis(exact_key, pattern_data)

            # Store domain-level (only if this is a good pattern)
            if pattern.success_rate >= self.validation_threshold or pattern.use_count == 0:
                await self._set_in_redis(domain_key, pattern_data)

            logger.info(f"💾 Pattern stored: {pattern.domain} ({pattern.pattern_type.value})")
            return True

        return False

    async def learn_from_extraction(
        self,
        url: str,
        html: str,
        extracted_items: List[Dict],
        fields: List[str],
        structure_hash: str
    ) -> Optional[ExtractionPattern]:
        """
        Learn a new pattern from a successful LLM extraction.

        This is called after Direct LLM extraction succeeds.
        It generates a deterministic pattern that can be reused.
        """
        if not extracted_items:
            return None

        logger.info(f"📚 Learning pattern from {len(extracted_items)} extracted items...")

        # Generate pattern
        pattern = await self.generator.generate_pattern_from_extraction(
            html=html,
            extracted_items=extracted_items,
            fields=fields,
            url=url,
            structure_hash=structure_hash
        )

        if pattern:
            # Validate the pattern produces similar results
            test_items = self.executor.execute(pattern, html)

            if test_items and len(test_items) >= len(extracted_items) * 0.5:
                # Pattern works! Store it
                pattern.success_count = 1
                await self.store_pattern(pattern, fields)
                logger.info(f"✅ Learned and stored pattern: {pattern.pattern_type.value}")
                return pattern
            else:
                logger.warning(f"⚠️ Generated pattern failed validation (got {len(test_items)} items, expected ~{len(extracted_items)})")

        return None

    async def execute_pattern(
        self,
        pattern: ExtractionPattern,
        html: str,
        url: str
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute a pattern and track success/failure.

        Returns:
            Tuple of (items, success)
        """
        try:
            items = self.executor.execute(pattern, html)

            if items and len(items) >= pattern.min_items:
                # Success! Update stats
                pattern.use_count += 1
                pattern.success_count += 1
                pattern.last_used = time.time()

                # Update cache
                await self.store_pattern(pattern, pattern.expected_fields)

                logger.info(f"⚡ Pattern executed successfully: {len(items)} items")
                return items, True
            else:
                # Pattern didn't work well
                pattern.use_count += 1
                pattern.failure_count += 1

                logger.warning(f"⚠️ Pattern underperformed: {len(items) if items else 0} items (expected >= {pattern.min_items})")
                return items or [], False

        except Exception as e:
            logger.error(f"❌ Pattern execution failed: {e}")
            pattern.use_count += 1
            pattern.failure_count += 1
            return [], False

    async def invalidate_pattern(self, domain: str, fields: List[str]) -> bool:
        """
        Invalidate patterns for a domain (force rebuild on next request).
        """
        domain_key = self._make_domain_key(domain, fields)

        # Remove from memory
        keys_to_remove = [k for k in self.memory_cache if domain in k]
        for key in keys_to_remove:
            del self.memory_cache[key]

        # Remove from Redis
        if self.redis_cache:
            await self._delete_from_redis(domain_key)

        logger.info(f"🗑️ Invalidated patterns for {domain}")
        return True

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain (remove www. prefix for consistency)"""
        if not domain:
            return domain
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    async def get_all_patterns(self, domain: Optional[str] = None) -> List[ExtractionPattern]:
        """
        Get all cached patterns, optionally filtered by domain.
        """
        patterns = []
        normalized_domain = self._normalize_domain(domain) if domain else None

        # From memory
        for key, pattern in self.memory_cache.items():
            if domain is None:
                patterns.append(pattern)
            else:
                pattern_domain = self._normalize_domain(pattern.domain)
                if pattern_domain == normalized_domain:
                    patterns.append(pattern)

        # From Redis (if not already in memory)
        if self.redis_cache:
            try:
                # Try multiple domain formats to catch www. vs non-www.
                search_patterns = []
                if domain:
                    normalized = self._normalize_domain(domain)
                    search_patterns = [
                        f"pattern:{domain}*",  # Original format
                        f"pattern:{normalized}*",  # Normalized (no www.)
                        f"pattern:www.{normalized}*",  # With www.
                    ]
                else:
                    search_patterns = ["pattern:*"]

                seen_keys = set()
                for pattern_prefix in search_patterns:
                    async for key in self.redis_cache.redis_client.scan_iter(match=pattern_prefix):
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key

                        # Skip if already processed
                        if key_str in seen_keys:
                            continue
                        seen_keys.add(key_str)

                        # Check if already in patterns list
                        if any(self._make_cache_key(p.domain, p.structure_hash, p.expected_fields) == key_str for p in patterns):
                            continue

                        data = await self._get_from_redis(key_str)
                        if data:
                            pattern = ExtractionPattern.from_dict(data)
                            # Apply domain filter if needed
                            if domain:
                                pattern_domain = self._normalize_domain(pattern.domain)
                                if pattern_domain != normalized_domain:
                                    continue
                            patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Failed to list Redis patterns: {e}")

        return patterns

    def _make_cache_key(self, domain: str, structure_hash: str, fields: List[str]) -> str:
        """Generate exact cache key"""
        fields_hash = hashlib.md5(",".join(sorted(fields)).encode()).hexdigest()[:8]
        return f"pattern:{domain}:{structure_hash[:12]}:{fields_hash}"

    def _make_domain_key(self, domain: str, fields: List[str]) -> str:
        """Generate domain-level cache key"""
        fields_hash = hashlib.md5(",".join(sorted(fields)).encode()).hexdigest()[:8]
        return f"pattern:{domain}:default:{fields_hash}"

    def _is_pattern_valid(self, pattern: ExtractionPattern) -> bool:
        """Check if a pattern is still valid"""
        # Check age
        if time.time() - pattern.created_at > self.max_pattern_age:
            return False

        # Check success rate (only if used enough times)
        if pattern.use_count >= 5 and pattern.success_rate < self.validation_threshold:
            return False

        return True

    async def _validate_pattern_on_html(
        self,
        pattern: ExtractionPattern,
        html: str,
        fields: List[str]
    ) -> bool:
        """
        Validate that a pattern works on new HTML.

        This is used when we have a domain-level pattern but the structure
        might be different. We test if the pattern still extracts data.
        """
        try:
            items = self.executor.execute(pattern, html)

            if not items or len(items) < pattern.min_items:
                return False

            # Check if fields are present
            first_item = items[0]
            present_fields = [f for f in fields if f in first_item and first_item[f]]

            return len(present_fields) >= len(fields) * 0.5

        except Exception:
            return False

    async def _get_from_redis(self, key: str) -> Optional[Dict]:
        """Get pattern from Redis"""
        if not self.redis_cache:
            return None

        try:
            return await self.redis_cache.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def _set_in_redis(self, key: str, data: Dict) -> bool:
        """Set pattern in Redis"""
        if not self.redis_cache:
            return False

        try:
            await self.redis_cache.set(key, data, ttl=self.max_pattern_age)
            return True
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False

    async def _delete_from_redis(self, key: str) -> bool:
        """Delete pattern from Redis"""
        if not self.redis_cache:
            return False

        try:
            await self.redis_cache.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False


class SmartScraper:
    """
    High-level scraper that uses SmartPatternCache for intelligent caching.

    Flow:
    1. Check cache for existing pattern
    2. If found: Execute pattern (instant, no LLM)
    3. If not found: Use Direct LLM extraction
    4. Learn pattern from LLM results
    5. Cache pattern for future use
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        redis_cache=None,
        **kwargs
    ):
        self.api_key = api_key
        self.model_name = model_name

        # Initialize pattern cache
        self.pattern_cache = SmartPatternCache(
            redis_cache=redis_cache,
            api_key=api_key,
            model_name=model_name
        )

        # Initialize Direct LLM extractor (fallback)
        from .direct_llm_extractor import DirectLLMExtractor
        self.direct_llm = DirectLLMExtractor(
            api_key=api_key,
            model_name=model_name,
            enable_cache=False,  # We handle caching at pattern level
            redis_cache=redis_cache
        )

        # Initialize structure hash generator
        from .structural_hash import StructuralHashGenerator
        self.hash_generator = StructuralHashGenerator()

        logger.info("🚀 SmartScraper initialized with pattern caching")

    async def scrape(
        self,
        url: str,
        fields: List[str],
        html: str,
        context: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Smart scrape with automatic pattern caching.

        Args:
            url: URL being scraped
            fields: Fields to extract
            html: HTML content
            context: Optional extraction context

        Returns:
            Tuple of (items, metadata) where metadata includes:
            - source: "pattern_cache" or "direct_llm"
            - pattern_type: Type of pattern used
            - cache_hit: Whether pattern was cached
            - processing_time_ms: Time taken
        """
        start_time = time.time()
        domain = urlparse(url).netloc

        # Generate structure hash
        hash_result = self.hash_generator.generate_hash(html)
        structure_hash = hash_result['hash']

        logger.info(f"🔍 Smart scrape: {domain} ({len(fields)} fields)")
        logger.info(f"   Structure hash: {structure_hash[:16]}...")

        # Step 1: Try to get cached pattern
        pattern, match_type = await self.pattern_cache.get_pattern(
            url=url,
            fields=fields,
            html=html,
            structure_hash=structure_hash
        )

        if pattern:
            # Step 2a: Execute cached pattern (fast path!)
            logger.info(f"⚡ Using cached {pattern.pattern_type.value} pattern ({match_type} match)")

            items, success = await self.pattern_cache.execute_pattern(pattern, html, url)

            if success:
                processing_time = (time.time() - start_time) * 1000
                return items, {
                    "source": "pattern_cache",
                    "pattern_type": pattern.pattern_type.value,
                    "cache_hit": True,
                    "match_type": match_type,
                    "processing_time_ms": processing_time,
                    "pattern_id": pattern.pattern_id,
                    "pattern_success_rate": pattern.success_rate,
                }
            else:
                logger.info("⚠️ Cached pattern failed, falling back to Direct LLM")

        # Step 2b: No pattern or pattern failed - use Direct LLM
        logger.info("🤖 Using Direct LLM extraction (will learn pattern)")

        items = await self.direct_llm.extract(
            html=html,
            fields=fields,
            context=context,
            url=url
        )

        # Step 3: Learn pattern from successful extraction
        if items and len(items) > 0:
            learned_pattern = await self.pattern_cache.learn_from_extraction(
                url=url,
                html=html,
                extracted_items=items,
                fields=fields,
                structure_hash=structure_hash
            )

            pattern_learned = learned_pattern is not None
        else:
            pattern_learned = False

        processing_time = (time.time() - start_time) * 1000

        return items, {
            "source": "direct_llm",
            "pattern_type": learned_pattern.pattern_type.value if pattern_learned else None,
            "cache_hit": False,
            "pattern_learned": pattern_learned,
            "processing_time_ms": processing_time,
        }

