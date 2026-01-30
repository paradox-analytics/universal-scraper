"""
Smart HTML Sampler - Dynamically determines optimal HTML sample size
Universal approach that adapts to each website's structure
"""

import logging
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import statistics

logger = logging.getLogger(__name__)


class SmartHTMLSampler:
    """
    Intelligently samples HTML to send to LLMs with minimal waste.
    
    Strategies:
    1. Complete Element Extraction - ensures all fields within elements are visible
    2. Field Coverage Analysis - verifies all important content is included
    3. Statistical Sizing - learns optimal size per website pattern
    4. Content Density Check - stops when reaching low-value content
    """
    
    def __init__(self):
        # Cache optimal sizes per pattern (domain + element type)
        self.optimal_sizes: Dict[str, int] = {}
    
    def extract_optimal_sample(
        self,
        html: str,
        detected_pattern: Optional[str] = None,
        fields: Optional[List[str]] = None,
        domain: Optional[str] = None,
        sibling_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract the optimal HTML sample for LLM analysis.
        
        NEW: Now supports context-block extraction (containers + siblings)
        
        Returns:
            {
                'sample_html': str,
                'sample_size': int,
                'elements_included': int,
                'coverage_complete': bool,
                'strategy_used': str
            }
        """
        
        # Strategy 1: Context-block extraction (NEW - handles siblings)
        if sibling_analysis and sibling_analysis.get('type') != 'container_only':
            result = self._extract_context_blocks(
                html=html,
                sibling_analysis=sibling_analysis,
                fields=fields,
                domain=domain
            )
            if result:
                logger.info(f"    Context-block sample: {result['sample_size']} bytes, {result['elements_included']} blocks")
                return result
        
        # Strategy 2: If we have a detected pattern, use complete elements
        if detected_pattern:
            result = self._extract_complete_elements(
                html=html,
                pattern=detected_pattern,
                fields=fields,
                domain=domain
            )
            if result:
                return result
        
        # Strategy 3: Fallback to intelligent fixed-size sampling
        return self._extract_fixed_sample(html)
    
    def _extract_complete_elements(
        self,
        html: str,
        pattern: str,
        fields: Optional[List[str]],
        domain: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract complete repeating elements with field coverage analysis.
        
        This is the BEST strategy for universal scraping because:
        1. Ensures all fields within elements are visible to LLM
        2. Adapts to element size (small cards vs. large articles)
        3. Learns optimal count per website
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            elements = soup.select(pattern)
            
            if not elements or len(elements) < 2:
                return None
            
            # Analyze element sizes
            element_sizes = [len(str(elem)) for elem in elements[:10]]  # Sample first 10
            avg_size = statistics.mean(element_sizes)
            max_size = max(element_sizes)
            
            logger.debug(f"   Element analysis: avg={avg_size:.0f}b, max={max_size:.0f}b, count={len(elements)}")
            
            # Determine optimal element count
            # Goal: Include enough elements to show pattern, but not waste tokens
            cache_key = f"{domain}:{pattern}" if domain else pattern
            
            if cache_key in self.optimal_sizes:
                target_size = self.optimal_sizes[cache_key]
                element_count = max(2, min(5, int(target_size / avg_size)))
            else:
                # Default strategy: adaptive based on element size
                if avg_size < 2000:
                    element_count = 5  # Small elements (e.g., product cards)
                elif avg_size < 5000:
                    element_count = 3  # Medium elements (e.g., article previews)
                else:
                    element_count = 2  # Large elements (e.g., full articles)
            
            # Extract elements
            sample_elements = elements[:element_count]
            sample_html = ''.join(str(elem) for elem in sample_elements)
            sample_size = len(sample_html)
            
            # Verify field coverage
            coverage_complete = self._verify_field_coverage(
                sample_html=sample_html,
                fields=fields,
                element_count=element_count
            )
            
            # Cache optimal size for future use
            if domain and coverage_complete:
                self.optimal_sizes[cache_key] = sample_size
            
            # Hard limit: 100KB (tokens are expensive!)
            if sample_size > 100000:
                logger.warning(f"   Sample too large ({sample_size:,}b), truncating to 2 elements")
                sample_elements = elements[:2]
                sample_html = ''.join(str(elem) for elem in sample_elements)
                sample_size = len(sample_html)
                element_count = 2
                coverage_complete = False
            
            logger.info(f"    Smart sample: {sample_size:,} bytes from {element_count} complete elements")
            logger.info(f"      Strategy: Complete elements (avg={avg_size:.0f}b per element)")
            logger.info(f"      Coverage: {' Complete' if coverage_complete else ' Partial'}")
            
            return {
                'sample_html': sample_html,
                'sample_size': sample_size,
                'elements_included': element_count,
                'coverage_complete': coverage_complete,
                'strategy_used': 'complete_elements',
                'avg_element_size': avg_size
            }
        
        except Exception as e:
            logger.warning(f"   Failed to extract complete elements: {e}")
            return None
    
    def _extract_context_blocks(
        self,
        html: str,
        sibling_analysis: Dict[str, Any],
        fields: Optional[List[str]],
        domain: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract context blocks (container + siblings) for LLM analysis.
        
        This is the KEY method for fixing sibling-based layouts like Stack Overflow.
        
        Strategy:
        - If type is 'parent_context_block': Extract full parent elements
        - If type is 'sibling_group': Extract container + next siblings
        """
        try:
            from .dom_pattern_detector import escape_css_selector
            
            soup = BeautifulSoup(html, 'html.parser')
            main_selector = sibling_analysis.get('main_selector')
            extraction_type = sibling_analysis.get('type')
            
            if not main_selector:
                return None
            
            # Escape the selector
            escaped_selector = escape_css_selector(main_selector)
            containers = soup.select(escaped_selector)
            
            if not containers:
                logger.warning(f"   No elements found for selector: {main_selector}")
                return None
            
            context_blocks = []
            
            if extraction_type == 'parent_context_block':
                # Extract full parent elements (includes container + all siblings)
                parents_seen = set()
                for container in containers[:10]:  # Sample first 10
                    parent = container.parent
                    if parent and id(parent) not in parents_seen:
                        parents_seen.add(id(parent))
                        context_blocks.append(str(parent))
                
                logger.info(f"    Extracted {len(context_blocks)} parent context blocks")
            
            elif extraction_type == 'sibling_group':
                # Extract container + immediate siblings
                for container in containers[:10]:  # Sample first 10
                    block_parts = [str(container)]
                    
                    # Add next siblings
                    sibling = container.find_next_sibling()
                    sibling_count = 0
                    while sibling and sibling_count < 3:
                        if hasattr(sibling, 'name'):  # It's a Tag
                            block_parts.append(str(sibling))
                            sibling_count += 1
                        sibling = sibling.find_next_sibling()
                    
                    # Wrap in a temporary parent for context
                    context_blocks.append(f"<div class='context-block'>\n" + "\n".join(block_parts) + "\n</div>")
                
                logger.info(f"    Extracted {len(context_blocks)} sibling groups")
            
            else:
                # Fallback: just extract containers
                context_blocks = [str(c) for c in containers[:10]]
                logger.info(f"    Extracted {len(context_blocks)} containers (fallback)")
            
            # Combine context blocks
            sample_html = "\n\n".join(context_blocks)
            sample_size = len(sample_html)
            
            # Limit to 30KB max
            if sample_size > 30000:
                sample_html = sample_html[:30000]
                sample_size = 30000
                logger.info(f"     Truncated sample to 30KB")
            
            return {
                'sample_html': sample_html,
                'sample_size': sample_size,
                'elements_included': len(context_blocks),
                'coverage_complete': True,  # Context blocks include everything
                'strategy_used': f'context_blocks_{extraction_type}'
            }
        
        except Exception as e:
            logger.warning(f"   Failed to extract context blocks: {e}")
            return None
    
    def _verify_field_coverage(
        self,
        sample_html: str,
        fields: Optional[List[str]],
        element_count: int
    ) -> bool:
        """
        Verify that the sample likely contains all requested fields.
        
        Heuristic checks:
        1. Element count is reasonable (2-5)
        2. Sample contains diverse content (not just headers)
        3. For common field names, check for likely patterns
        """
        # If no fields specified, assume coverage
        if not fields:
            return True
        
        # If we have very few elements, coverage might be incomplete
        if element_count < 2:
            return False
        
        # Check for common field patterns
        common_patterns = {
            'price': ['$', '€', '£', 'price', 'cost'],
            'description': ['description', 'summary', 'text', '<p>'],
            'title': ['title', 'heading', '<h1>', '<h2>', '<h3>'],
            'author': ['author', 'by', 'posted'],
            'date': ['date', 'time', 'ago', 'posted'],
            'rating': ['rating', 'star', 'score', ''],
            'url': ['href', 'link'],
            'image': ['img', 'src', 'picture'],
            'stars': ['star', 'fork', 'watch'],  # GitHub-specific
            'language': ['language', 'lang', 'itemprop'],
            'repository': ['repo', 'href', 'project'],
        }
        
        sample_lower = sample_html.lower()
        
        # Check if likely field patterns exist
        field_found = 0
        for field in fields:
            field_lower = field.lower()
            
            # Direct field name in HTML
            if field_lower in sample_lower:
                field_found += 1
                continue
            
            # Pattern matching
            if field_lower in common_patterns:
                patterns = common_patterns[field_lower]
                if any(pattern in sample_lower for pattern in patterns):
                    field_found += 1
                    continue
        
        # Coverage is complete if we found patterns for 70%+ of fields
        coverage_ratio = field_found / len(fields)
        return coverage_ratio >= 0.7
    
    def _extract_fixed_sample(self, html: str) -> Dict[str, Any]:
        """
        Fallback: intelligent fixed-size sampling.
        
        Uses a reasonable default that works for most sites.
        """
        sample_size = min(10000, len(html))
        sample_html = html[:sample_size]
        
        logger.info(f"    Fixed sample: {sample_size:,} bytes (fallback strategy)")
        
        return {
            'sample_html': sample_html,
            'sample_size': sample_size,
            'elements_included': -1,  # Unknown
            'coverage_complete': False,  # Unknown
            'strategy_used': 'fixed_size'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics"""
        return {
            'cached_patterns': len(self.optimal_sizes),
            'optimal_sizes': dict(self.optimal_sizes)
        }


