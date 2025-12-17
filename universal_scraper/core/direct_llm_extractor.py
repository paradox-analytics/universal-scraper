"""
Direct LLM Extractor
Uses LLM to extract data directly from HTML (no pattern generation)
This is the foundation for pattern learning - LLM shows us what works, we learn from it

ENHANCED: Now uses Hybrid Markdown Extraction to capture structured data
before markdown conversion (addresses 8 edge cases where pure markdown loses data)

SPEED OPTIMIZED: Parallel chunk processing for 2-3x faster extraction

CACHING: Results are cached by structure hash + fields for reuse across runs
- Local: File-based cache
- Apify: KV Store (persists across runs)
"""
import logging
import json
import re
import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional
import litellm
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document

# Import hybrid extractor for enhanced extraction
try:
    from .hybrid_extractor import HybridMarkdownExtractor, ExtractedContent, create_llm_context
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

logger = logging.getLogger(__name__)


class DirectLLMExtractor:
    """
    Direct LLM extraction - the proven approach used by ScrapeGraphAI, Parsera, etc.
    
    ENHANCED: Now uses hybrid extraction that captures structured data (data attributes,
    form options, hidden inputs) before markdown conversion to prevent data loss.
    
    Quality Modes:
    - conservative: Like ScrapeGraphAI (≥70% fields filled, fewer items, higher quality)
    - balanced: Default (≥50% fields filled, good balance)
    - aggressive: Maximum extraction (≥30% fields filled, more items)
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        max_tokens_per_chunk: int = 4000,  # Match ScrapeGraphAI: smaller chunks + deduplication = better coverage
        quality_mode: str = "balanced",
        use_html2text: bool = True,  # NEW: Convert HTML to text (ScrapeGraphAI approach)
        use_hybrid_extraction: bool = True,  # NEW: Use hybrid extraction (captures data attrs, forms)
        enable_cache: bool = True,  # NEW: Enable result caching
        cache_ttl: int = 3600  # NEW: Cache TTL in seconds (1 hour default)
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.quality_mode = quality_mode
        self.use_html2text = use_html2text
        self.use_hybrid_extraction = use_hybrid_extraction and HYBRID_AVAILABLE
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Initialize cache (uses UnifiedPatternCache infrastructure for Apify compatibility)
        if self.enable_cache:
            try:
                from .unified_cache import UnifiedPatternCache
                # UnifiedPatternCache automatically detects Apify and uses KV Store
                self.result_cache = UnifiedPatternCache(force_local=False)
                logger.info(" Direct LLM result cache enabled (works locally and on Apify)")
            except Exception as e:
                logger.warning(f"  Failed to initialize cache: {e}, caching disabled")
                self.result_cache = None
                self.enable_cache = False
        else:
            self.result_cache = None
        
        # Initialize hybrid extractor if available
        if self.use_hybrid_extraction:
            self.hybrid_extractor = HybridMarkdownExtractor()
            logger.info(" Hybrid extraction enabled (captures data attributes, forms, etc.)")
        else:
            self.hybrid_extractor = None
        
        # Use Langchain's Html2TextTransformer as fallback
        self.html_transformer = Html2TextTransformer()
        
        # Quality thresholds by mode
        self.quality_thresholds = {
            'conservative': 0.50,  # Only items with ≥50% fields
            'balanced': 0.33,      # Default - 1 out of 3 fields (title OR points OR comments)
            'aggressive': 0.10     # Maximum extraction - at least 1 field
        }
        
        if quality_mode not in self.quality_thresholds:
            logger.warning(f"Invalid quality_mode '{quality_mode}', using 'balanced'")
            self.quality_mode = "balanced"
        
        logger.info(f" DirectLLMExtractor initialized (model={model_name}, quality={quality_mode})")
    
    async def extract(
        self,
        html: str,
        fields: List[str],
        context: Optional[str] = None,
        quality_mode: Optional[str] = None,
        url: Optional[str] = None  # NEW: URL for cache key generation
    ) -> List[Dict[str, Any]]:
        """
        Extract data directly from HTML using LLM
        
        Args:
            html: Cleaned HTML content
            fields: List of fields to extract
            context: Optional context about what to extract
            quality_mode: Override quality mode for this extraction
                         ('conservative', 'balanced', 'aggressive')
            url: Optional URL for cache key generation
            
        Returns:
            List of extracted items
            
        ENHANCED (from ScrapeGraphAI + Parsera analysis):
        - Skip chunking for small pages (fits in single LLM context)
        - Conditional retry if quality is poor
        - Chunk overlap with context passing for better continuity
        - Sequential processing for the first few chunks to establish patterns
        - Parallel processing for remaining chunks with established context
        - CACHING: Results cached by structure hash + fields (reuses across runs)
        """
        # Use provided quality_mode or fall back to instance default
        quality_mode = quality_mode or self.quality_mode
        
        logger.info(f" Direct LLM extraction: {len(fields)} fields from {len(html):,} bytes (quality={quality_mode})")
        
        # NEW: Check cache first
        if self.enable_cache and self.result_cache:
            cache_key = self._generate_cache_key(html, fields, url)
            logger.info(f" Checking Direct LLM cache: {cache_key}")
            
            # Try to get cached result
            cached_result = await self.result_cache.backend.get(cache_key)
            if cached_result:
                logger.info(f" Direct LLM cache hit: {cache_key[:50]}...")
                
                # Validate cached result
                if self._validate_cached_result(cached_result, html, fields):
                    logger.info(" Cached result validated - using cache (no LLM call)")
                    return cached_result.get('items', [])
                else:
                    logger.info("  Cached result invalid (structure changed) - re-extracting")
                    # Fall through to LLM extraction
            else:
                logger.info(f" Direct LLM cache miss: {cache_key[:50]}... (not found in cache)")
        
        # LLM extraction (cache miss or validation failed)
        
        #  OPTIMIZATION 1: Skip chunking for small pages (ScrapeGraphAI approach)
        # If content fits in a single LLM context, process it directly for better quality
        SMALL_PAGE_THRESHOLD = self.max_tokens_per_chunk * 4  # ~16K chars = 4K tokens
        
        if len(html) <= SMALL_PAGE_THRESHOLD:
            logger.info(f"    Small page ({len(html):,} bytes) - processing without chunking")
            
            #  ITERATIVE REFINEMENT: The key insight is that autoregressive/iterative 
            # generation works for agentic tasks. Let the model see and refine its output.
            items = await self._extract_with_refinement(html, fields, context, quality_mode, url=url)
            
            # Infer and convert data types
            items = self._infer_and_convert_types(items, fields)
            logger.info(f" Total extracted: {len(items)} items")
            
            # Cache the result
            if self.enable_cache and self.result_cache:
                await self._cache_result(html, fields, items, url)
            
            return items
        
        # For larger pages, use chunking with overlap
        # Chunk HTML with OVERLAP (Parsera-style: 33% overlap helps capture truncated items)
        OVERLAP_FACTOR = 3  # 33% overlap like Parsera
        chunks = self._chunk_html_with_overlap(html, self.max_tokens_per_chunk, overlap_factor=OVERLAP_FACTOR)
        logger.info(f"    Large page - split into {len(chunks)} chunks (with {100//OVERLAP_FACTOR}% overlap)")
        
        all_items = []
        previous_items = []  # Track previous chunk's items for context (Parsera-style)
        
        # For small number of chunks, use sequential extraction with context passing
        # This helps fill in truncated values and ensures continuity
        if len(chunks) <= 4:
            for i, chunk in enumerate(chunks):
                logger.info(f"   Processing chunk {i+1}/{len(chunks)} (with context from {len(previous_items)} previous items)...")
                try:
                    items = await self._extract_from_chunk(
                        chunk, fields, context, quality_mode, 
                        previous_items=previous_items[-5:] if previous_items else None,  # Pass last 5 items
                        url=url  # Pass URL for field context
                    )
                    all_items.extend(items)
                    previous_items.extend(items)
                    logger.info(f"       Chunk {i+1}/{len(chunks)}: {len(items)} items")
                except Exception as e:
                    logger.error(f"       Chunk {i+1}/{len(chunks)} failed: {e}")
        else:
            # For large pages: process first 2 chunks sequentially to establish patterns,
            # then process remaining in parallel batches
            for i in range(min(2, len(chunks))):
                logger.info(f"   Processing chunk {i+1}/{len(chunks)} (establishing patterns)...")
                try:
                    items = await self._extract_from_chunk(
                        chunks[i], fields, context, quality_mode,
                        previous_items=previous_items[-5:] if previous_items else None,
                        url=url  # Pass URL for field context
                    )
                    all_items.extend(items)
                    previous_items.extend(items)
                    logger.info(f"       Chunk {i+1}/{len(chunks)}: {len(items)} items")
                except Exception as e:
                    logger.error(f"       Chunk {i+1}/{len(chunks)} failed: {e}")
            
            # Process remaining chunks in parallel batches
            # OPTIMIZATION: Larger batch size for better parallelism (ScrapeGraphAI uses 10-20)
            # With cleaned HTML, we have fewer chunks, so larger batches are safe
            remaining_chunks = chunks[2:]
            BATCH_SIZE = 10  # Increased from 4 to 10 for better parallelism
            
            for batch_start in range(0, len(remaining_chunks), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(remaining_chunks))
                batch_chunks = remaining_chunks[batch_start:batch_end]
                
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(remaining_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"   Processing batch {batch_num}/{total_batches} (chunks {batch_start+3}-{batch_end+2})...")
                
                # Create parallel extraction tasks (pass previous_items for context)
                tasks = [
                    self._extract_from_chunk(
                        chunk, fields, context, quality_mode,
                        previous_items=previous_items[-3:] if previous_items else None,  # Smaller context for parallel
                        url=url  # Pass URL for field context
                    )
                    for chunk in batch_chunks
                ]
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, result in enumerate(results):
                        chunk_idx = batch_start + i + 3  # +3 for the first 2 sequential chunks
                        if isinstance(result, Exception):
                            logger.error(f"       Chunk {chunk_idx}/{len(chunks)} failed: {result}")
                        else:
                            all_items.extend(result)
                            logger.info(f"       Chunk {chunk_idx}/{len(chunks)}: {len(result)} items")
                except Exception as e:
                    logger.error(f"       Batch extraction failed: {e}")
                    continue
        
        #  ScrapeGraphAI's secret: DEDUPLICATE items from multiple chunks!
        # Even more important with overlapping chunks
        if len(chunks) > 1:
            all_items = self._deduplicate_items(all_items, fields)
            logger.info(f"   After deduplication: {len(all_items)} items")
        
        # Calculate quality for large pages
        if all_items:
            quality = self._calculate_quality(all_items, fields)
            logger.info(f"   Quality: {quality:.1%} field completeness")
            
            #  ITERATIVE REFINEMENT for chunked pages
            # Apply the same refinement logic we use for small pages
            if quality < 0.7 and len(all_items) > 0:
                logger.info(f"    Quality below 70%, applying iterative refinement...")
                
                # Find incomplete fields
                missing_fields = self._find_incomplete_fields(all_items, fields)
                
                if missing_fields:
                    logger.info(f"   Refining {len(missing_fields)} incomplete fields: {missing_fields}")
                    
                    # Refine by showing model the merged items
                    refined_items = await self._refine_extraction(
                        html, fields, all_items, missing_fields, context, quality_mode
                    )
                    
                    if refined_items:
                        new_quality = self._calculate_quality(refined_items, fields)
                        if new_quality > quality:
                            logger.info(f"    Refinement improved quality: {quality:.1%} → {new_quality:.1%}")
                            all_items = refined_items
                            quality = new_quality
                        else:
                            logger.info(f"    Keeping original (refinement quality: {new_quality:.1%})")
        
        # Infer and convert data types for better quality
        all_items = self._infer_and_convert_types(all_items, fields)
        
        logger.info(f" Total extracted: {len(all_items)} items")
        
        # Cache the result
        if self.enable_cache and self.result_cache:
            await self._cache_result(html, fields, all_items, url)
        
        return all_items
    
    async def _extract_single_pass(
        self,
        html: str,
        fields: List[str],
        context: Optional[str],
        quality_mode: str,
        enhanced_prompt: bool = False,
        url: Optional[str] = None  # NEW: URL for field context
    ) -> List[Dict[str, Any]]:
        """
        Extract data in a single LLM pass (no chunking).
        Used for small pages or as a retry mechanism.
        
        Args:
            html: HTML content
            fields: Fields to extract
            context: Optional extraction context
            quality_mode: Quality mode
            enhanced_prompt: If True, use enhanced prompt with more guidance
        """
        # Use hybrid extraction for markdown + metadata
        extracted_content = None
        metadata_context = ""
        
        if self.use_hybrid_extraction and self.hybrid_extractor:
            try:
                extracted_content = self.hybrid_extractor.extract(html)
                content_for_llm = extracted_content.markdown
                metadata_context = extracted_content.get_metadata_summary()
            except Exception as e:
                logger.warning(f"   Hybrid extraction failed: {e}")
                extracted_content = None
        
        # Fallback to html2text
        if extracted_content is None:
            if self.use_html2text:
                doc = Document(page_content=html)
                transformed_docs = self.html_transformer.transform_documents([doc])
                content_for_llm = transformed_docs[0].page_content if transformed_docs else html
            else:
                content_for_llm = html
        
        # Build prompt (optionally enhanced for retry)
        prompt = self._build_extraction_prompt(
            content_for_llm, fields, context, None, metadata_context,
            enhanced_prompt=enhanced_prompt,
            url=url  # Pass URL for field context
        )
        
        # Call LLM
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(enhanced=enhanced_prompt)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1 if not enhanced_prompt else 0.2,  # Slightly higher temp for retry
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            items = []
            if "items" in result:
                items = result["items"]
            elif isinstance(result, list):
                items = result
            
            # Filter quality
            filtered_items = self._filter_quality_items(items, fields, quality_mode)
            
            if len(filtered_items) < len(items):
                logger.debug(f"   Filtered {len(items) - len(filtered_items)} low-quality items")
            
            return filtered_items
            
        except Exception as e:
            logger.error(f" Single-pass extraction failed: {e}")
            return []
    
    async def _extract_with_refinement(
        self,
        html: str,
        fields: List[str],
        context: Optional[str],
        quality_mode: str,
        max_iterations: int = 3,
        quality_threshold: float = 0.7,
        url: Optional[str] = None  # NEW: URL for field context
    ) -> List[Dict[str, Any]]:
        """
        Iterative refinement extraction - the key insight is that autoregressive/iterative
        generation works surprisingly well for agentic tasks.
        
        Instead of single-shot extraction, we:
        1. Extract initial data
        2. Calculate quality
        3. If below threshold, refine by showing model its own output
        4. Repeat until quality met or max iterations
        
        This mimics next-token prediction where each output informs the next step.
        """
        # Initial extraction
        items = await self._extract_single_pass(html, fields, context, quality_mode, url=url)
        
        if not items:
            logger.info(f"   Initial extraction returned 0 items, trying enhanced prompt...")
            items = await self._extract_single_pass(html, fields, context, quality_mode, enhanced_prompt=True, url=url)
            if not items:
                return []
        
        quality = self._calculate_quality(items, fields)
        logger.info(f"   Iteration 1: {len(items)} items, {quality:.1%} quality")
        
        # Iterative refinement loop
        for iteration in range(2, max_iterations + 1):
            if quality >= quality_threshold:
                logger.info(f"    Quality threshold met ({quality:.1%} >= {quality_threshold:.0%})")
                break
            
            # Find what's missing
            missing_fields = self._find_incomplete_fields(items, fields)
            
            if not missing_fields:
                break
            
            logger.info(f"    Iteration {iteration}: Refining {len(missing_fields)} incomplete fields...")
            
            # Refine by showing model its own output
            refined_items = await self._refine_extraction(
                html, fields, items, missing_fields, context, quality_mode
            )
            
            if refined_items:
                new_quality = self._calculate_quality(refined_items, fields)
                
                if new_quality > quality:
                    items = refined_items
                    quality = new_quality
                    logger.info(f"   Iteration {iteration}: {len(items)} items, {quality:.1%} quality (+{(new_quality - quality)*100:.1f}%)")
                else:
                    logger.info(f"   Iteration {iteration}: No improvement ({new_quality:.1%}), stopping")
                    break
            else:
                logger.info(f"   Iteration {iteration}: Refinement failed, keeping previous")
                break
        
        return items
    
    def _find_incomplete_fields(self, items: List[Dict[str, Any]], fields: List[str]) -> List[str]:
        """Find fields that have low fill rates across items"""
        if not items:
            return fields
        
        field_fill_rates = {}
        for field in fields:
            filled = sum(1 for item in items if item.get(field) not in [None, '', 'null', 'None'])
            field_fill_rates[field] = filled / len(items)
        
        # Return fields with < 70% fill rate
        return [f for f, rate in field_fill_rates.items() if rate < 0.7]
    
    async def _refine_extraction(
        self,
        html: str,
        fields: List[str],
        previous_items: List[Dict[str, Any]],
        missing_fields: List[str],
        context: Optional[str],
        quality_mode: str
    ) -> List[Dict[str, Any]]:
        """
        Refine extraction by showing model its previous output.
        This is the autoregressive step - each output informs the next.
        """
        # Use hybrid extraction for markdown
        extracted_content = None
        if self.use_hybrid_extraction and self.hybrid_extractor:
            try:
                extracted_content = self.hybrid_extractor.extract(html)
                content_for_llm = extracted_content.markdown
                metadata_context = extracted_content.get_metadata_summary()
            except:
                extracted_content = None
        
        if extracted_content is None:
            if self.use_html2text:
                doc = Document(page_content=html)
                transformed_docs = self.html_transformer.transform_documents([doc])
                content_for_llm = transformed_docs[0].page_content if transformed_docs else html
            else:
                content_for_llm = html
            metadata_context = ""
        
        # Build refinement prompt - show model its own output
        prompt = self._build_refinement_prompt(
            content_for_llm, fields, previous_items, missing_fields, context, metadata_context
        )
        
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": self._get_refinement_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15,  # Slightly higher for creative gap-filling
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            items = result.get("items", result) if isinstance(result, dict) else result
            
            # Validate that we got the same number of items (refinement should not change count)
            if len(items) != len(previous_items):
                logger.warning(f"  Refinement returned {len(items)} items, expected {len(previous_items)}. Merging to preserve all items.")
                # Merge: keep original items, fill in missing fields from refined items
                # Match items by position/index
                merged_items = []
                for i, original_item in enumerate(previous_items):
                    if i < len(items):
                        # Merge: original item + refined fields
                        merged_item = original_item.copy()
                        for field in missing_fields:
                            if field in items[i] and items[i][field] not in [None, '', 'null', 'None']:
                                merged_item[field] = items[i][field]
                        merged_items.append(merged_item)
                    else:
                        merged_items.append(original_item)
                items = merged_items
            
            return self._filter_quality_items(items, fields, quality_mode)
            
        except Exception as e:
            logger.error(f" Refinement failed: {e}")
            return []
    
    def _build_refinement_prompt(
        self,
        html: str,
        fields: List[str],
        previous_items: List[Dict[str, Any]],
        missing_fields: List[str],
        context: Optional[str],
        metadata_context: str = ""
    ) -> str:
        """Build prompt that shows model its previous output for refinement"""
        
        # Show ALL items, not just first 10 (fixes bug where refinement reduced items from 30 to 10)
        # If too many items, show summary + all items in compact format
        items_to_show = previous_items
        items_preview = ""
        if len(previous_items) > 50:
            # For very large lists, show first 5 and last 5 with summary
            items_preview = f"Total items: {len(previous_items)}\n"
            items_preview += "First 5 items:\n"
            items_preview += json.dumps(previous_items[:5], indent=2)
            items_preview += "\n\n... (showing all items in response) ...\n\n"
            items_preview += "Last 5 items:\n"
            items_preview += json.dumps(previous_items[-5:], indent=2)
            items_to_show = previous_items  # Still pass all items for context
        else:
            items_preview = json.dumps(previous_items, indent=2)
        
        prompt_parts = [
            f"You previously extracted {len(previous_items)} items from a webpage:",
            "",
            "```json",
            items_preview,
            "```",
            "",
            f"However, these fields are often missing or incomplete: {', '.join(missing_fields)}",
            "",
            f"Please re-examine the content and provide a COMPLETE extraction with ALL {len(previous_items)} items and ALL fields filled.",
            f"IMPORTANT: You must return the SAME NUMBER of items ({len(previous_items)}), just with missing fields filled in.",
            "Look for the missing data in:",
            "- Different parts of the content",
            "- Alternative field names or formats",
            "- Data attributes or metadata",
            "",
            "Original content:",
            html[:8000],  # Limit content size
        ]
        
        if metadata_context:
            prompt_parts.extend([
                "",
                "Additional metadata:",
                metadata_context
            ])
        
        prompt_parts.extend([
            "",
            f"Extract items with these fields: {', '.join(fields)}",
            "",
            "Return JSON with 'items' array. Fill ALL fields for each item.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_refinement_system_prompt(self) -> str:
        """System prompt for refinement - emphasizes completing missing data"""
        return """You are refining a previous extraction. Your job is to:

1. Keep ALL items from the previous extraction (CRITICAL: same count)
2. Keep all correctly extracted data
3. Fill in missing or incomplete fields ONLY
4. Look harder for data that was missed
5. Check alternative locations (data attributes, metadata, nearby text)

CRITICAL: Return the EXACT SAME NUMBER of items as the previous extraction.
Do NOT add new items or remove existing items. Only fill in missing fields.

For numeric fields, extract only the number.
If truly not found after careful search, use null.

Return complete JSON with 'items' array containing the same number of items."""
    
    def _calculate_quality(self, items: List[Dict[str, Any]], fields: List[str]) -> float:
        """
        Calculate extraction quality as percentage of fields filled across all items.
        
        Returns:
            Float between 0 and 1 representing field completeness
        """
        if not items or not fields:
            return 0.0
        
        total_fields = len(items) * len(fields)
        filled_fields = 0
        
        for item in items:
            for field in fields:
                value = item.get(field)
                if value is not None and str(value).strip() not in ['', 'null', 'None', 'N/A']:
                    filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    def _chunk_html_with_overlap(self, html: str, max_tokens: int, overlap_factor: int = 3) -> List[str]:
        """
        Split HTML into overlapping chunks (Parsera-style).
        Overlap helps capture items that might be truncated at chunk boundaries.
        """
        # Estimate chars per token (rough: 4 chars per token)
        chars_per_token = 4
        chunk_size = max_tokens * chars_per_token
        overlap_size = chunk_size // overlap_factor
        
        chunks = []
        start = 0
        
        while start < len(html):
            end = start + chunk_size
            
            # Try to break at a sensible boundary (closing tag)
            if end < len(html):
                # Look for </div>, </article>, </tr>, </li> etc near the end
                break_patterns = ['</div>', '</article>', '</tr>', '</li>', '</p>', '</section>']
                best_break = end
                
                for pattern in break_patterns:
                    # Look in the last 20% of the chunk
                    search_start = end - (chunk_size // 5)
                    idx = html.rfind(pattern, search_start, end)
                    if idx > search_start and idx > best_break - chunk_size // 5:
                        best_break = idx + len(pattern)
                        break
                
                end = best_break
            
            chunks.append(html[start:end])
            
            # Move forward with overlap
            start = end - overlap_size
            if start < 0:
                start = 0
            if start >= len(html):
                break
        
        return chunks
    
    async def _extract_from_chunk(
        self,
        html_chunk: str,
        fields: List[str],
        context: Optional[str],
        quality_mode: str,
        previous_items: Optional[List[Dict[str, Any]]] = None,  # Parsera-style context passing
        url: Optional[str] = None  # NEW: URL for field context
    ) -> List[Dict[str, Any]]:
        """
        Extract data from a single HTML chunk using LLM
        
        ENHANCED: Now uses hybrid extraction that captures structured data
        (data attributes, form options, hidden inputs) before markdown conversion.
        
        PARSERA-STYLE: Optionally receives previous_items from earlier chunks
        to help fill in truncated values and maintain extraction continuity.
        """
        # Count potential items in HTML (rough estimate)
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html_chunk, 'html.parser')
            # Try to find repeating patterns (articles, products, posts)
            potential_items = max(
                len(soup.find_all('tr', class_='athing')),  # HackerNews
                len(soup.find_all(class_=lambda x: x and 'item' in x.lower())),  # Generic items
                len(soup.find_all(class_=lambda x: x and 'product' in x.lower())),  # Products
                len(soup.find_all(class_=lambda x: x and 'post' in x.lower())),  # Posts
                len(soup.find_all('article'))  # Article tags
            )
        except:
            potential_items = None
        
        #  HYBRID EXTRACTION: Extract structured data + convert to markdown
        extracted_content = None
        metadata_context = ""
        
        if self.use_hybrid_extraction and self.hybrid_extractor:
            try:
                extracted_content = self.hybrid_extractor.extract(html_chunk)
                content_for_llm = extracted_content.markdown
                metadata_context = extracted_content.get_metadata_summary()
                
                if metadata_context:
                    css_count = len(extracted_content.css_class_data) if hasattr(extracted_content, 'css_class_data') else 0
                    logger.debug(f"   Hybrid: captured {len(extracted_content.data_attributes)} data attrs, "
                               f"{len(extracted_content.form_data)} form fields, {css_count} CSS class data")
            except Exception as e:
                logger.warning(f"   Hybrid extraction failed, falling back: {e}")
                extracted_content = None
        
        # Fallback to standard html2text if hybrid not available or failed
        if extracted_content is None:
            if self.use_html2text:
                doc = Document(page_content=html_chunk)
                transformed_docs = self.html_transformer.transform_documents([doc])
                content_for_llm = transformed_docs[0].page_content if transformed_docs else html_chunk
                logger.debug(f"   Converted HTML to text: {len(html_chunk)} → {len(content_for_llm)} chars")
            else:
                content_for_llm = html_chunk
        
        # Create extraction prompt (now includes metadata context and previous items if available)
        prompt = self._build_extraction_prompt(
            content_for_llm, fields, context, potential_items, metadata_context,
            previous_items=previous_items,  # Parsera-style: pass previous chunk's items
            url=url  # Pass URL for field context
        )
        
        # Call LLM
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4096,  # Increase from default (~2048) to allow longer responses
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract items array
            items = []
            if "items" in result:
                items = result["items"]
            elif isinstance(result, list):
                items = result
            else:
                logger.warning(f"  Unexpected response format: {list(result.keys())}")
                return []
            
            # Post-extraction quality filtering (now more lenient to match ScrapeGraphAI)
            filtered_items = self._filter_quality_items(items, fields, quality_mode)
            
            if len(filtered_items) < len(items):
                logger.info(f"    Filtered {len(items) - len(filtered_items)} low-quality items ({len(filtered_items)} kept, mode={quality_mode})")
            
            return filtered_items
                
        except json.JSONDecodeError as e:
            logger.error(f" Failed to parse LLM response as JSON: {e}")
            logger.error(f"   Response: {content[:200]}")
            return []
        except Exception as e:
            logger.error(f" LLM extraction failed: {e}")
            return []
    
    def _deduplicate_items(
        self,
        items: List[Dict[str, Any]],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate items extracted from multiple chunks (ScrapeGraphAI approach)
        
        Uses smart deduplication:
        - Primary key fields (title, name, id, url) for exact matching
        - Fuzzy matching for near-duplicates
        - Keep the item with most fields filled
        
        Args:
            items: All extracted items (may have duplicates)
            fields: Field names
            
        Returns:
            Deduplicated list of items
        """
        if not items:
            return items
        
        # Identify primary key fields (used for deduplication)
        primary_keys = []
        for field in fields:
            field_lower = field.lower()
            if any(key in field_lower for key in ['title', 'name', 'id', 'url', 'slug', 'repository']):
                primary_keys.append(field)
        
        # If no obvious primary key, use all fields
        if not primary_keys:
            primary_keys = fields
        
        # Track seen items by their primary key values
        seen = {}
        deduplicated = []
        
        for item in items:
            # Create a key from primary key fields
            key_parts = []
            for pk_field in primary_keys:
                value = item.get(pk_field)
                if value is not None:
                    # Normalize the value for comparison
                    key_parts.append(str(value).strip().lower())
            
            if not key_parts:
                # No primary key values found, keep item
                deduplicated.append(item)
                continue
            
            item_key = "|||".join(key_parts)
            
            if item_key in seen:
                # Duplicate found - keep the one with more filled fields
                existing_item = seen[item_key]
                existing_filled = sum(1 for f in fields if existing_item.get(f) not in [None, ''])
                current_filled = sum(1 for f in fields if item.get(f) not in [None, ''])
                
                if current_filled > existing_filled:
                    # Replace with better item
                    seen[item_key] = item
            else:
                # New item
                seen[item_key] = item
        
        # Return deduplicated items
        deduplicated = list(seen.values())
        
        logger.debug(f"   Deduplicated: {len(items)} → {len(deduplicated)} items")
        
        return deduplicated
    
    def _filter_quality_items(
        self,
        items: List[Dict[str, Any]],
        fields: List[str],
        quality_mode: str
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality items post-extraction
        
        Quality criteria (based on quality_mode):
        - conservative: ≥70% fields filled (like ScrapeGraphAI)
        - balanced: ≥50% fields filled (default)
        - aggressive: ≥30% fields filled (maximum extraction)
        
        All modes also check for:
        - Meaningful values (not just "null", "None", "N/A")
        - No obvious navigation/UI text ("Home", "Menu", "Contact", "Subscribe")
        
        Args:
            items: Extracted items from LLM
            fields: Requested fields
            quality_mode: Quality mode ('conservative', 'balanced', 'aggressive')
            
        Returns:
            Filtered high-quality items only
        """
        if not items or not fields:
            return items
        
        # Get threshold for this quality mode
        min_fill_rate = self.quality_thresholds.get(quality_mode, 0.50)
        
        filtered = []
        
        # Only reject OBVIOUS navigation/UI keywords (very conservative to match ScrapeGraphAI)
        # Only full-match exact phrases, not substrings
        nav_keywords = [
            # Only the most obvious UI elements that would NEVER be content
        ]  # Empty for now - ScrapeGraphAI doesn't filter aggressively
        
        for item in items:
            # Count filled fields
            filled_count = 0
            has_nav_text = False
            
            for field in fields:
                value = item.get(field)
                
                if value is not None:
                    value_str = str(value).strip().lower()
                    
                    # Check if value is meaningful
                    if value_str and value_str not in ['null', 'none', 'n/a', 'undefined', '']:
                        filled_count += 1
                        
                        # Check for navigation/UI text (only in text fields, not numbers)
                        if field in ['title', 'name', 'product_name', 'article_title', 'heading']:
                            if any(keyword in value_str for keyword in nav_keywords):
                                has_nav_text = True
                                break
            
            # Calculate fill rate
            fill_rate = filled_count / len(fields) if fields else 0
            
            # Score/Rating validation (for review/rating sites like Metacritic, IMDB, etc.)
            # If a score/rating field exists, validate it's a reasonable value
            has_invalid_score = False
            for field in fields:
                field_lower = field.lower()
                if any(score_term in field_lower for score_term in ['score', 'rating', 'metascore', 'points']):
                    score_value = item.get(field)
                    if score_value is not None:
                        # Check if it's a valid numeric score
                        try:
                            score_num = float(score_value)
                            # Most rating systems use 0-100, 0-10, or 0-5
                            # Accept if in reasonable range
                            if not (0 <= score_num <= 100):
                                # Invalid score range
                                has_invalid_score = True
                                logger.debug(f"   Rejected item with invalid score: {score_num} (field: {field})")
                                break
                        except (ValueError, TypeError):
                            # Score is not numeric (might be "None", "N/A", etc.)
                            # This is okay - some items legitimately don't have scores yet
                            pass
            
            # Quality threshold: based on quality_mode
            if fill_rate >= min_fill_rate and not has_nav_text and not has_invalid_score:
                filtered.append(item)
        
        return filtered
    
    def _infer_and_convert_types(
        self,
        items: List[Dict[str, Any]],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Convert string values to appropriate types (numbers, booleans, etc.)
        
        Improves data quality by returning properly typed values like ScrapeGraphAI.
        
        Args:
            items: Extracted items
            fields: Field names (used to guess type from name)
            
        Returns:
            Items with properly typed values
        """
        # Fields that should be numeric
        numeric_field_patterns = [
            'price', 'cost', 'amount', 'rating', 'score', 'stars',
            'points', 'votes', 'upvotes', 'downvotes', 'likes',
            'count', 'number', 'quantity', 'total', 'sum',
            'comments', 'replies', 'views', 'shares'
        ]
        
        for item in items:
            for key, value in list(item.items()):
                if value is None or value == "":
                    continue
                
                # Check if field name suggests it should be numeric
                is_numeric_field = any(
                    pattern in key.lower()
                    for pattern in numeric_field_patterns
                )
                
                if is_numeric_field and isinstance(value, str):
                    # Try to convert to number
                    # Remove currency symbols, units, and common formatting
                    cleaned = value.strip()
                    
                    # Remove common units and words
                    units_to_remove = [
                        'points', 'point', 'pts', 'pt',
                        'comments', 'comment', 'replies', 'reply',
                        'votes', 'vote', 'upvotes', 'upvote',
                        'views', 'view', 'likes', 'like',
                        'stars', 'star', 'reviews', 'review',
                        'items', 'item', 'results', 'result'
                    ]
                    
                    # Remove units (case-insensitive)
                    for unit in units_to_remove:
                        cleaned = re.sub(rf'\b{unit}\b', '', cleaned, flags=re.IGNORECASE)
                    
                    # Remove currency symbols and common formatting
                    cleaned = re.sub(r'[$€£¥₹,\s]', '', cleaned.strip())
                    
                    try:
                        # Try integer first
                        if '.' not in cleaned:
                            item[key] = int(cleaned)
                        else:
                            item[key] = float(cleaned)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
        
        return items
    
    def _build_extraction_prompt(
        self,
        html: str,
        fields: List[str],
        context: Optional[str],
        expected_count: Optional[int] = None,
        metadata_context: str = "",
        previous_items: Optional[List[Dict[str, Any]]] = None,  # Parsera-style context passing
        enhanced_prompt: bool = False,  # Enhanced prompt for retry
        url: Optional[str] = None  # NEW: URL for website-specific field context
    ) -> str:
        """
        Build extraction prompt for LLM
        
        Args:
            html: HTML to extract from (now markdown if hybrid extraction used)
            fields: Fields to extract
            context: Optional context
            expected_count: Hint about how many items to expect (helps LLM completeness)
            metadata_context: Additional metadata from hybrid extraction (data attrs, forms)
            previous_items: Items extracted from previous chunks (Parsera-style continuity)
            enhanced_prompt: If True, add more detailed guidance (used for retry)
        """
        # Create JSON schema
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            field: {"type": "string", "description": f"The {field} of the item"}
                            for field in fields
                        }
                    }
                }
            },
            "required": ["items"]
        }
        
        # Build the extraction question (ScrapeGraphAI style: simple and direct)
        field_list = ', '.join(fields)
        
        # NEW: Add website-specific field context (critical for correct extraction)
        # "next token is always the most important for context on what to actually extract, and it varies per website"
        field_context = self._get_field_context(url, html[:500] if html else "", fields)
        
        if context:
            user_question = f"{context}\nExtract these fields: {field_list}"
        else:
            user_question = f"Extract all items from this content with these fields: {field_list}"
        
        # Add field context to user question
        if field_context:
            user_question += f"\n\n{field_context}"
        
        # Simple prompt like ScrapeGraphAI (their secret is SIMPLICITY!)
        content_type_desc = "content from a website converted in markdown format" if self.use_html2text else "content from a website"
        
        # Build the prompt
        prompt_parts = [
            f"You are a website scraper and you have just scraped the following {content_type_desc}.",
            "You are now asked to answer a user question about the content you have scraped.",
            "",
            "If you don't find a field value, put null.",
            "Make sure the output format is JSON and does not contain errors."
        ]
        
        # Enhanced guidance for retry (ScrapeGraphAI "reasoning" step inspired)
        if enhanced_prompt:
            prompt_parts.extend([
                "",
                "IMPORTANT - ENHANCED EXTRACTION:",
                "- Look CAREFULLY at the content structure",
                "- For each field, search multiple locations (headings, text, metadata)",
                "- If a value seems partial, look for the complete value nearby",
                "- Extract ALL items visible in the content, even if some fields are missing",
                "- Prefer specific values over generic ones"
            ])
        
        #  PARSERA-STYLE: Add previous items context if available
        # This helps fill in truncated values and maintain extraction continuity
        if previous_items and len(previous_items) > 0:
            # Show last few items from previous chunks
            prev_items_str = json.dumps(previous_items[-3:], indent=2)
            prompt_parts.extend([
                "",
                "CONTEXT: Here are the last items extracted from previous parts of the page.",
                "If you see items that overlap with these, prefer filling in any missing/null values.",
                "If an item appears truncated, try to find its complete data in the current content.",
                f"Previous items: {prev_items_str}"
            ])
        
        # Check if any requested fields are vote-related
        vote_fields = [f for f in fields if any(v in f.lower() for v in ['vote', 'upvote', 'like', 'point', 'score'])]
        if vote_fields:
            prompt_parts.extend([
                "",
                "VOTE/UPVOTE EXTRACTION HINT:",
                "Look for vote counts in these patterns:",
                "- [VOTES: 123] or [UPVOTES: 456] - These are vote counts",
                "- Numbers right after [VOTES: N] or [UPVOTES: N] are the vote values",
                "- Match each product with the vote count that appears near it",
                f"For the '{vote_fields[0]}' field, extract the numeric value from these patterns."
            ])
        
        prompt_parts.extend([
            "",
            f"Output instructions: {json.dumps(schema, indent=2)}",
            "",
            f"User question: {user_question}",
            "",
            "Website content:",
            html
        ])
        
        # Add metadata context if available (from hybrid extraction)
        if metadata_context:
            prompt_parts.extend([
                "",
                "---",
                "ADDITIONAL DATA (extracted from HTML attributes and forms - check these for field values):",
                metadata_context
            ])
        
        prompt_parts.append("")
        prompt_parts.append("Return ONLY valid JSON. No explanations.")
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self, enhanced: bool = False) -> str:
        """
        Get system prompt that guides LLM behavior (ScrapeGraphAI approach: keep it simple!)
        
        Args:
            enhanced: If True, return more detailed prompt for retry attempts
        """
        base_prompt = """You are a professional web scraper. Extract structured data from website content accurately and completely.

IMPORTANT: Extract ALL items from the content. Do not stop early or skip items.

For numeric fields (like prices, ratings, points, comments):
- Return ONLY the number without units or symbols
- Example: "96 points" → 96, "$29.99" → 29.99, "4.5 stars" → 4.5

If a field value is not found, use null.
"""
        
        if enhanced:
            # Enhanced prompt inspired by ScrapeGraphAI's "reasoning" node
            return base_prompt + """
ENHANCED EXTRACTION MODE:
- Analyze the content structure before extracting
- Look for repeating patterns (product cards, list items, table rows)
- Check multiple sources for each field: visible text, data attributes, meta tags
- If a value appears truncated, look for the complete version
- Extract partial data rather than skipping items entirely
- Focus on content quality over quantity
"""
        
        return base_prompt
    
    def _get_field_context(self, url: Optional[str], html_sample: str, fields: List[str]) -> str:
        """
        Generate website-specific context about what each field means.
        
        This is CRITICAL: "next token is always the most important for context on what to actually extract,
        and it varies per website" - we need to tell the LLM what "title" means for THIS website.
        
        Args:
            url: Source URL (preferred for domain detection)
            html_sample: Sample HTML to detect website type (fallback)
            fields: Requested fields
            
        Returns:
            Context string explaining field meanings for this website
        """
        # Detect website type from URL (preferred) or HTML sample
        domain = ""
        if url:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
        
        html_lower = html_sample.lower() if html_sample else ""
        
        # Detect website type from domain (preferred) or HTML sample
        is_ecommerce = any(site in domain for site in ['chewy', 'amazon', 'ebay', 'etsy', 'shopify', 'product', 'store', 'buy']) or \
                      any(indicator in html_lower for indicator in [
                          'product-card', 'add-to-cart', 'product-title', 
                          'product-name', 'buy now', 'shopping', 'e-commerce'
                      ])
        is_news = any(indicator in html_lower for indicator in [
            'article', 'headline', 'news', 'blog-post', 'byline'
        ])
        is_social = any(indicator in html_lower for indicator in [
            'reddit', 'twitter', 'facebook', 'post', 'tweet', 'comment'
        ])
        
        field_contexts = []
        
        for field in fields:
            field_lower = field.lower()
            
            # Title field - varies by website type
            if 'title' in field_lower or field_lower == 'name':
                if is_ecommerce:
                    field_contexts.append(
                        f"**{field}**: Extract the FULL product name/description (e.g., 'Fancy Feast Gems Mousse Salmon, Tuna, Chicken & Beef Halo of Savory Gravy Variety Pack Pate Wet Cat Food'). "
                        f"DO NOT extract short labels like 'Trial Size', 'Variety Pack', or 'Highest Quality'. "
                        f"Look for the complete, descriptive product name - check headings, data attributes (data-name), and nearby text. "
                        f"Prefer the LONGEST, most descriptive value available."
                    )
                elif is_news:
                    field_contexts.append(
                        f"**{field}**: Extract the full article headline (complete title, not truncated)."
                    )
                elif is_social:
                    field_contexts.append(
                        f"**{field}**: Extract the complete post/thread title (not truncated)."
                    )
                else:
                    field_contexts.append(
                        f"**{field}**: Extract the complete, descriptive name/title. Prefer full names over short labels."
                    )
            
            # URL field
            elif 'url' in field_lower or 'link' in field_lower:
                field_contexts.append(
                    f"**{field}**: Extract the complete URL (e.g., 'https://www.chewy.com/dp/12345'). "
                    f"Look for href attributes, link elements, or construct from data-id if needed."
                )
            
            # Rating field
            elif 'rating' in field_lower:
                field_contexts.append(
                    f"**{field}**: Extract only the numeric rating (e.g., 4.5, 4.6). No stars or text."
                )
            
            # Review count field
            elif 'review' in field_lower and 'count' in field_lower:
                field_contexts.append(
                    f"**{field}**: Extract only the number of reviews (e.g., 2391, 1655). No text."
                )
        
        if field_contexts:
            return "\n\nFIELD-SPECIFIC INSTRUCTIONS (website-specific meanings):\n" + "\n".join(field_contexts)
        return ""
    
    def _chunk_html(self, html: str, max_tokens: int) -> List[str]:
        """
        Split HTML into chunks if it's too large
        
        Uses smart chunking:
        - Try to keep complete HTML elements together
        - Split at natural boundaries (closing tags)
        - Ensure each chunk is under max_tokens
        """
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(html) <= max_chars:
            return [html]
        
        chunks = []
        current_chunk = ""
        
        # Split by major HTML elements (to keep structure intact)
        # Look for closing tags of container elements
        split_points = []
        for match in re.finditer(r'</(?:div|article|section|li|tr|tbody|table)>', html):
            split_points.append(match.end())
        
        if not split_points:
            # Fallback: simple character-based chunking
            return [html[i:i+max_chars] for i in range(0, len(html), max_chars)]
        
        last_pos = 0
        for pos in split_points:
            if pos - last_pos > max_chars:
                # Current section is too large, add what we have
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                last_pos = pos
            
            current_chunk += html[last_pos:pos]
            
            if len(current_chunk) >= max_chars:
                chunks.append(current_chunk)
                current_chunk = ""
                last_pos = pos
        
        # Add remaining
        if current_chunk:
            chunks.append(current_chunk)
        elif last_pos < len(html):
            chunks.append(html[last_pos:])
        
        logger.debug(f"   Chunked {len(html):,} bytes into {len(chunks)} chunks")
        return chunks
    
    def _generate_cache_key(self, html: str, fields: List[str], url: Optional[str] = None) -> str:
        """
        Generate cache key for Direct LLM results
        
        Uses structure hash + fields to cache by page structure, not content.
        This allows reuse across different pages with same structure.
        
        Args:
            html: HTML content
            fields: List of fields to extract
            url: Optional URL (for logging/debugging)
            
        Returns:
            Cache key string
        """
        # Generate structure hash (ignores content, focuses on structure)
        try:
            from .structural_hash import StructuralHashGenerator
            hash_gen = StructuralHashGenerator()
            hash_result = hash_gen.generate_hash(html)
            structure_hash = hash_result['hash'][:16]  # Use first 16 chars
        except Exception as e:
            logger.warning(f"  Failed to generate structure hash: {e}, using content hash")
            # Fallback: hash of HTML length + fields
            structure_hash = hashlib.md5(f"{len(html)}:{','.join(sorted(fields))}".encode()).hexdigest()[:16]
        
        # Hash of fields (sorted for consistency)
        fields_str = ','.join(sorted(fields))
        fields_hash = hashlib.md5(fields_str.encode()).hexdigest()[:8]
        
        # Optional URL hash (for URL-specific caching)
        # NOTE: Apify KV Store keys must only contain: a-zA-Z0-9!-_.'()
        # So we use underscores instead of colons
        url_hash = ""
        if url:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            cache_key = f"direct_llm_{url_hash}_{structure_hash}_{fields_hash}"
        else:
            cache_key = f"direct_llm_{structure_hash}_{fields_hash}"
        
        return cache_key
    
    def _validate_cached_result(self, cached_result: Dict, html: str, fields: List[str]) -> bool:
        """
        Validate cached result without calling LLM
        
        Checks:
        1. Structure hash matches (same layout)
        2. Cached items have expected fields
        3. Cached items count is reasonable
        
        Args:
            cached_result: Cached result dict
            html: Current HTML content
            fields: Expected fields
            
        Returns:
            True if cached result is still valid
        """
        try:
            # Check structure hash matches
            cached_structure_hash = cached_result.get('structure_hash')
            if cached_structure_hash:
                try:
                    from .structural_hash import StructuralHashGenerator
                    hash_gen = StructuralHashGenerator()
                    current_hash_result = hash_gen.generate_hash(html)
                    current_structure_hash = current_hash_result['hash'][:16]
                    
                    if current_structure_hash != cached_structure_hash:
                        logger.debug(f"   Structure changed: {cached_structure_hash[:8]}... != {current_structure_hash[:8]}...")
                        return False
                except Exception as e:
                    logger.warning(f"   Failed to validate structure hash: {e}")
                    # Continue with other validations
            
            # Check items exist and have expected fields
            items = cached_result.get('items', [])
            if not items:
                logger.debug("   No items in cache")
                return False
            
            # Check field completeness (sample first 3 items)
            for item in items[:3]:
                for field in fields:
                    if field not in item:
                        logger.debug(f"   Missing field '{field}' in cached item")
                        return False
            
            # Check timestamp (optional TTL check)
            cached_timestamp = cached_result.get('timestamp', 0)
            if cached_timestamp and self.cache_ttl > 0:
                age = time.time() - cached_timestamp
                if age > self.cache_ttl:
                    logger.debug(f"   Cache expired (age: {age:.0f}s > TTL: {self.cache_ttl}s)")
                    return False
            
            logger.debug("   Cached result validated successfully")
            return True
            
        except Exception as e:
            logger.warning(f"   Validation error: {e}")
            return False
    
    async def _cache_result(self, html: str, fields: List[str], items: List[Dict], url: Optional[str] = None) -> None:
        """
        Cache Direct LLM extraction result
        
        Args:
            html: HTML content
            fields: Fields extracted
            items: Extracted items
            url: Optional URL
        """
        if not self.enable_cache or not self.result_cache:
            return
        
        try:
            cache_key = self._generate_cache_key(html, fields, url)
            
            # Generate structure hash for validation
            structure_hash = None
            try:
                from .structural_hash import StructuralHashGenerator
                hash_gen = StructuralHashGenerator()
                hash_result = hash_gen.generate_hash(html)
                structure_hash = hash_result['hash'][:16]
            except Exception:
                pass
            
            # Prepare cache entry
            cache_entry = {
                'items': items,
                'structure_hash': structure_hash,
                'fields': fields,
                'timestamp': time.time(),
                'url': url,
                'item_count': len(items)
            }
            
            # Store in cache (uses UnifiedPatternCache which handles Apify KV Store)
            await self.result_cache.backend.set(cache_key, cache_entry)
            logger.info(f" Cached Direct LLM result: {cache_key[:32]}... ({len(items)} items)")
            
        except Exception as e:
            logger.warning(f"  Failed to cache result: {e}")
    
    def estimate_cost(self, html_length: int, fields_count: int) -> float:
        """
        Estimate extraction cost
        
        Args:
            html_length: Length of HTML in characters
            fields_count: Number of fields to extract
            
        Returns:
            Estimated cost in USD
        """
        # Estimate tokens (input + output)
        input_tokens = (html_length / 4) + (fields_count * 50)  # HTML + prompt
        output_tokens = fields_count * 100  # Rough estimate for JSON output
        
        # GPT-4o-mini pricing (as of Nov 2024)
        # Input: $0.15 / 1M tokens
        # Output: $0.60 / 1M tokens
        input_cost = (input_tokens / 1_000_000) * 0.15
        output_cost = (output_tokens / 1_000_000) * 0.60
        
        return input_cost + output_cost

