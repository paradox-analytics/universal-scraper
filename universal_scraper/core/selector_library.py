"""
Selector Library - Bootstrapping System
Builds site-specific selector library from successful extractions

Purpose: When user scrapes first few pages, reuse that knowledge to reduce
future LLM work. Save successful field-to-selector mappings as training examples.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SelectorPattern:
    """A selector pattern that worked for a field"""
    field_name: str
    selector: str  # CSS selector or XPath
    selector_type: str = "css"  # "css" or "xpath"
    success_count: int = 1
    failure_count: int = 0
    success_rate: float = 1.0
    last_used: Optional[str] = None
    contexts: List[str] = field(default_factory=list)  # URLs or page types where it worked

    def update_success(self):
        """Record a successful use"""
        self.success_count += 1
        self._update_success_rate()
        self.last_used = datetime.now().isoformat()

    def update_failure(self):
        """Record a failed use"""
        self.failure_count += 1
        self._update_success_rate()

    def _update_success_rate(self):
        """Update success rate"""
        total = self.success_count + self.failure_count
        self.success_rate = self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FieldMapping:
    """Field mapping with multiple selector patterns"""
    field_name: str
    patterns: List[SelectorPattern] = field(default_factory=list)
    canonical_selector: Optional[str] = None  # Best selector based on success rate
    alternates: List[str] = field(default_factory=list)  # Alternative selectors

    def add_pattern(self, selector: str, selector_type: str = "css", context: Optional[str] = None):
        """Add or update a selector pattern"""
        # Check if pattern already exists
        for pattern in self.patterns:
            if pattern.selector == selector and pattern.selector_type == selector_type:
                pattern.update_success()
                if context:
                    pattern.contexts.append(context)
                self._update_canonical()
                return

        # New pattern
        pattern = SelectorPattern(
            field_name=self.field_name,
            selector=selector,
            selector_type=selector_type,
            success_count=1,
            success_rate=1.0,
            last_used=datetime.now().isoformat(),
            contexts=[context] if context else []
        )
        self.patterns.append(pattern)
        self._update_canonical()

    def _update_canonical(self):
        """Update canonical selector based on success rates"""
        if not self.patterns:
            return

        # Sort by success rate (descending)
        sorted_patterns = sorted(self.patterns, key=lambda p: p.success_rate, reverse=True)
        self.canonical_selector = sorted_patterns[0].selector
        self.alternates = [p.selector for p in sorted_patterns[1:5]]  # Top 5 alternates

    def get_best_selectors(self, max_count: int = 3) -> List[str]:
        """Get best selectors (canonical + alternates)"""
        selectors = []
        if self.canonical_selector:
            selectors.append(self.canonical_selector)
        selectors.extend(self.alternates[:max_count - 1])
        return selectors


@dataclass
class SiteSelectorLibrary:
    """Site-specific selector library"""
    domain: str
    field_mappings: Dict[str, FieldMapping] = field(default_factory=dict)
    total_scrapes: int = 0
    successful_scrapes: int = 0
    last_updated: Optional[str] = None

    def add_extraction_result(
        self,
        fields: List[str],
        extracted_items: List[Dict[str, Any]],
        html: str,
        url: Optional[str] = None
    ):
        """
        Learn from successful extraction

        Args:
            fields: Requested fields
            extracted_items: Successfully extracted items
            html: HTML content (for selector discovery)
            url: Source URL
        """
        if not extracted_items:
            return

        self.total_scrapes += 1

        # Analyze which selectors worked (if we can infer them)
        # For now, we'll learn from the extraction results
        # In future, we can analyze the HTML to find selectors

        # Mark as successful if we got items
        if len(extracted_items) > 0:
            self.successful_scrapes += 1

        self.last_updated = datetime.now().isoformat()

    def get_field_mapping(self, field_name: str) -> Optional[FieldMapping]:
        """Get field mapping for a field"""
        return self.field_mappings.get(field_name)

    def get_training_examples(self, fields: List[str]) -> Dict[str, List[str]]:
        """
        Get training examples for template generation

        Returns:
            Dict mapping field_name to list of selectors that worked
        """
        examples = {}

        for field_name in fields:
            mapping = self.field_mappings.get(field_name)
            if mapping:
                examples[field_name] = mapping.get_best_selectors(max_count=5)

        return examples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'domain': self.domain,
            'field_mappings': {
                field_name: {
                    'canonical_selector': mapping.canonical_selector,
                    'alternates': mapping.alternates,
                    'patterns': [p.to_dict() for p in mapping.patterns]
                }
                for field_name, mapping in self.field_mappings.items()
            },
            'total_scrapes': self.total_scrapes,
            'successful_scrapes': self.successful_scrapes,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SiteSelectorLibrary':
        """Create from dictionary"""
        library = cls(domain=data['domain'])
        library.total_scrapes = data.get('total_scrapes', 0)
        library.successful_scrapes = data.get('successful_scrapes', 0)
        library.last_updated = data.get('last_updated')

        # Reconstruct field mappings
        for field_name, mapping_data in data.get('field_mappings', {}).items():
            mapping = FieldMapping(field_name=field_name)
            mapping.canonical_selector = mapping_data.get('canonical_selector')
            mapping.alternates = mapping_data.get('alternates', [])

            # Reconstruct patterns
            for pattern_data in mapping_data.get('patterns', []):
                pattern = SelectorPattern(
                    field_name=field_name,
                    selector=pattern_data['selector'],
                    selector_type=pattern_data.get('selector_type', 'css'),
                    success_count=pattern_data.get('success_count', 1),
                    failure_count=pattern_data.get('failure_count', 0),
                    success_rate=pattern_data.get('success_rate', 1.0),
                    last_used=pattern_data.get('last_used'),
                    contexts=pattern_data.get('contexts', [])
                )
                mapping.patterns.append(pattern)

            library.field_mappings[field_name] = mapping

        return library


class SelectorLibrary:
    """
    Manages selector libraries for multiple sites

    Features:
    - Site-specific selector patterns
    - Training examples for template generation
    - Success rate tracking
    - Persistent storage (via UnifiedPatternCache)
    """

    def __init__(self, enable_cache: bool = True):
        """
        Initialize selector library

        Args:
            enable_cache: Enable persistent caching
        """
        self.enable_cache = enable_cache
        self.libraries: Dict[str, SiteSelectorLibrary] = {}  # domain -> library

        # Initialize cache backend
        if self.enable_cache:
            try:
                from .unified_cache import UnifiedPatternCache
                self.cache = UnifiedPatternCache(force_local=False)
                logger.info(f" Selector Library cache enabled ({self.cache.env} backend)")
            except Exception as e:
                logger.warning(f"  Failed to initialize cache: {e}, caching disabled")
                self.cache = None
                self.enable_cache = False
        else:
            self.cache = None

    def _get_domain(self, url: str) -> str:
        """Extract normalized domain from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain

    async def get_library(self, url: str) -> SiteSelectorLibrary:
        """
        Get or create selector library for site

        Args:
            url: Source URL

        Returns:
            SiteSelectorLibrary instance
        """
        domain = self._get_domain(url)

        # Check in-memory cache
        if domain in self.libraries:
            return self.libraries[domain]

        # Check persistent cache
        if self.enable_cache and self.cache:
            try:
                cache_key = f"selector_library_{domain}"
                cached_data = await self.cache.backend.get(cache_key)
                if cached_data:
                    library = SiteSelectorLibrary.from_dict(cached_data)
                    self.libraries[domain] = library
                    logger.debug(f"   Loaded selector library for {domain} from cache")
                    return library
            except Exception as e:
                logger.debug(f"   Cache load failed: {e}")

        # Create new library
        library = SiteSelectorLibrary(domain=domain)
        self.libraries[domain] = library
        return library

    async def learn_from_extraction(
        self,
        url: str,
        fields: List[str],
        extracted_items: List[Dict[str, Any]],
        html: str,
        selectors_used: Optional[Dict[str, str]] = None
    ):
        """
        Learn from successful extraction

        Args:
            url: Source URL
            fields: Requested fields
            extracted_items: Successfully extracted items
            html: HTML content
            selectors_used: Optional dict mapping field_name to selector used
        """
        library = await self.get_library(url)

        # Add extraction result
        library.add_extraction_result(fields, extracted_items, html, url)

        # If selectors were provided, learn them
        if selectors_used:
            for field_name, selector in selectors_used.items():
                if field_name not in library.field_mappings:
                    library.field_mappings[field_name] = FieldMapping(field_name=field_name)

                library.field_mappings[field_name].add_pattern(
                    selector=selector,
                    selector_type="css",  # Default to CSS
                    context=url
                )

        # Save to cache
        if self.enable_cache and self.cache:
            try:
                cache_key = f"selector_library_{library.domain}"
                await self.cache.backend.set(cache_key, library.to_dict())
                logger.debug(f"   Saved selector library for {library.domain}")
            except Exception as e:
                logger.debug(f"   Cache save failed: {e}")

    async def get_training_examples(
        self,
        url: str,
        fields: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get training examples for template generation

        Args:
            url: Source URL
            fields: Fields to extract

        Returns:
            Dict mapping field_name to list of selectors that worked
        """
        library = await self.get_library(url)
        return library.get_training_examples(fields)

    async def get_canonical_selectors(
        self,
        url: str,
        fields: List[str]
    ) -> Dict[str, str]:
        """
        Get canonical selectors for fields

        Args:
            url: Source URL
            fields: Fields to extract

        Returns:
            Dict mapping field_name to canonical selector
        """
        library = await self.get_library(url)
        selectors = {}

        for field_name in fields:
            mapping = library.field_mappings.get(field_name)
            if mapping and mapping.canonical_selector:
                selectors[field_name] = mapping.canonical_selector

        return selectors

    def get_success_rate(self, url: str) -> float:
        """
        Get success rate for site

        Args:
            url: Source URL

        Returns:
            Success rate (0.0-1.0)
        """
        domain = self._get_domain(url)
        library = self.libraries.get(domain)

        if library and library.total_scrapes > 0:
            return library.successful_scrapes / library.total_scrapes

        return 0.0



