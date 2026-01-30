"""
DOM Pattern Detector - Fast heuristic-based pattern detection
Finds repeating elements without LLM (inspired by AutoScraper & academic research)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from bs4 import BeautifulSoup, Tag
import re

logger = logging.getLogger(__name__)


def escape_css_selector(selector: str) -> str:
    """
    Escape special characters in CSS selectors
    
    Tailwind CSS uses `:` for arbitrary values (e.g., `h:bg-black-150`)
    which BeautifulSoup treats as pseudo-classes. We need to escape them.
    
    Example:
        'li.h:bg-black-150' -> 'li.h\\:bg-black-150'
        'div[data-test]' -> 'div[data-test]' (unchanged)
    """
    # Split by `.` to process each class separately
    parts = selector.split('.')
    
    # Escape `:` in class names (but not in pseudo-classes like :hover)
    escaped_parts = []
    for i, part in enumerate(parts):
        if i == 0:
            # First part is the tag name, don't escape
            escaped_parts.append(part)
        else:
            # Class names: escape `:` and `/` (common in Tailwind)
            # But only if they're in the middle of a class name (not at start)
            if ':' in part and not part.startswith(':'):
                part = part.replace(':', '\\:')
            if '/' in part:
                part = part.replace('/', '\\/')
            escaped_parts.append(part)
    
    return '.'.join(escaped_parts)


class DOMPatternDetector:
    """
    Detect repeating patterns in HTML using DOM tree analysis
    
    This is a FAST, heuristic-based approach that runs before LLM analysis
    to identify likely data containers.
    """
    
    def __init__(self):
        self.min_pattern_frequency = 5  # Minimum repetitions to consider a pattern
        self.max_depth = 10  # Maximum depth to analyze
    
    def detect_patterns(self, html: str) -> Dict[str, Any]:
        """
        Detect repeating patterns in HTML
        
        Returns:
            Dict with pattern analysis results
        """
        soup = BeautifulSoup(html, 'lxml')
        
        logger.info(" Starting DOM pattern detection...")
        
        # 1. Find repeating tag+class combinations
        element_signatures = self._analyze_element_signatures(soup)
        
        # 2. Find repeating tree structures
        tree_patterns = self._analyze_tree_patterns(soup)
        
        # 3. Find elements with data attributes
        data_attribute_elements = self._analyze_data_attributes(soup)
        
        # 4. Find custom web components
        custom_components = self._analyze_custom_components(soup)
        
        # 5. Identify the most likely data container
        best_pattern = self._identify_best_pattern(
            element_signatures,
            tree_patterns,
            data_attribute_elements,
            custom_components,
            soup
        )
        
        # 6. Analyze sibling patterns (NEW - for context-block extraction)
        sibling_analysis = None
        if best_pattern and best_pattern.get('selector'):
            sibling_analysis = self._analyze_sibling_patterns(
                soup, 
                best_pattern.get('selector'),
                best_pattern.get('count', 0)
            )
            if sibling_analysis:
                logger.info(f"    Context block: {sibling_analysis['type']}")
                if sibling_analysis.get('sibling_selectors'):
                    logger.info(f"    Found {len(sibling_analysis['sibling_selectors'])} consistent siblings")
        
        return {
            'best_pattern': best_pattern,
            'element_signatures': element_signatures,
            'tree_patterns': tree_patterns,
            'data_attributes': data_attribute_elements,
            'custom_components': custom_components,
            'sibling_analysis': sibling_analysis,  # NEW
            'confidence': best_pattern.get('confidence', 0.0) if best_pattern else 0.0
        }
    
    def _analyze_element_signatures(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Analyze repeating tag+class combinations
        
        Example: <li class="s-card"> appears 60 times = likely data container
        """
        signatures = Counter()
        signature_to_element = {}
        
        for elem in soup.find_all():
            if not isinstance(elem, Tag):
                continue
            
            # Create signature: tag.class1.class2
            classes = elem.get('class', [])
            if classes:
                # Track BOTH full signature AND first-class-only signature
                # This helps with elements like <li class="s-card s-card--horizontal">
                full_sig = f"{elem.name}.{'.'.join(classes)}"
                first_class_sig = f"{elem.name}.{classes[0]}"
                
                signatures[full_sig] += 1
                signatures[first_class_sig] += 1
                
                if full_sig not in signature_to_element:
                    signature_to_element[full_sig] = elem
                if first_class_sig not in signature_to_element:
                    signature_to_element[first_class_sig] = elem
            else:
                # Also track elements with IDs
                elem_id = elem.get('id')
                if elem_id:
                    sig = f"{elem.name}#{elem_id}"
                else:
                    sig = elem.name
                
                signatures[sig] += 1
                if sig not in signature_to_element:
                    signature_to_element[sig] = elem
        
        # Filter to frequently repeating patterns
        patterns = []
        for sig, count in signatures.most_common(100):  # Top 100 to catch more patterns
            if count >= self.min_pattern_frequency:
                if sig not in signature_to_element:
                    logger.warning(f"     Signature {sig} not in signature_to_element (count={count})")
                    continue
                
                elem = signature_to_element[sig]
                
                # Calculate data density (how much text content)
                text_length = len(elem.get_text(strip=True))
                
                # Check for data attributes
                has_data_attrs = any(k.startswith('data-') for k in elem.attrs.keys())
                
                patterns.append({
                    'signature': sig,
                    'count': count,
                    'tag': elem.name,
                    'classes': elem.get('class', []),
                    'text_length': text_length,
                    'has_data_attrs': has_data_attrs,
                    'sample_element': str(elem)[:500]  # First 500 chars
                })
        
        logger.info(f"   Found {len(patterns)} repeating element signatures")
        
        # Debug: Log if we found any li.s-card
        for p in patterns:
            if 'li.s-card' in p['signature']:
                logger.info(f"    Found li.s-card: {p['count']} occurrences")
                break
        
        return patterns
    
    def _analyze_tree_patterns(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Analyze repeating tree structures
        
        Example: div > div > span.price appears 60 times = likely price container
        """
        tree_signatures = Counter()
        tree_to_element = {}
        
        for elem in soup.find_all():
            if not isinstance(elem, Tag):
                continue
            
            # Create tree signature: parent > child > grandchild
            tree_sig = self._create_tree_signature(elem, depth=3)
            tree_signatures[tree_sig] += 1
            
            if tree_sig not in tree_to_element:
                tree_to_element[tree_sig] = elem
        
        # Filter to frequently repeating patterns
        patterns = []
        for sig, count in tree_signatures.most_common(15):
            if count >= self.min_pattern_frequency and sig != "None > None > None":
                elem = tree_to_element[sig]
                patterns.append({
                    'tree_signature': sig,
                    'count': count,
                    'sample_element': str(elem)[:300]
                })
        
        logger.info(f"   Found {len(patterns)} repeating tree structures")
        return patterns
    
    def _create_tree_signature(self, elem: Tag, depth: int = 3) -> str:
        """Create a signature based on element ancestry"""
        parts = []
        current = elem
        
        for _ in range(depth):
            if current is None or not isinstance(current, Tag):
                parts.append("None")
                break
            
            # Create signature for this level
            classes = current.get('class', [])
            if classes:
                sig = f"{current.name}.{classes[0]}"  # Use first class only
            else:
                sig = current.name
            
            parts.append(sig)
            current = current.parent
        
        return " > ".join(reversed(parts))
    
    def _analyze_data_attributes(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Find elements with data-* attributes (common in modern websites)
        """
        data_attr_elements = defaultdict(list)
        
        for elem in soup.find_all():
            if not isinstance(elem, Tag):
                continue
            
            # Check for data attributes
            data_attrs = [k for k in elem.attrs.keys() if k.startswith('data-')]
            
            if data_attrs:
                sig = f"{elem.name}[{', '.join(data_attrs[:3])}]"  # First 3 attrs
                data_attr_elements[sig].append({
                    'element': elem.name,
                    'attributes': data_attrs,
                    'sample': str(elem)[:200]
                })
        
        # Convert to list and get most common
        result = []
        for sig, elements in data_attr_elements.items():
            if len(elements) >= self.min_pattern_frequency:
                result.append({
                    'signature': sig,
                    'count': len(elements),
                    'attributes': elements[0]['attributes'],
                    'sample': elements[0]['sample']
                })
        
        logger.info(f"   Found {len(result)} elements with data-* attributes")
        return result
    
    def _analyze_custom_components(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Find custom web components (tags with hyphens like <shreddit-post>)
        """
        custom_tags = Counter()
        custom_to_element = {}
        
        for elem in soup.find_all():
            if not isinstance(elem, Tag):
                continue
            
            # Custom elements have hyphens in tag name
            if '-' in elem.name:
                custom_tags[elem.name] += 1
                if elem.name not in custom_to_element:
                    custom_to_element[elem.name] = elem
        
        patterns = []
        for tag, count in custom_tags.most_common(10):
            elem = custom_to_element[tag]
            
            # Get all attributes (data is usually in attributes for custom components)
            attrs = list(elem.attrs.keys())
            
            patterns.append({
                'tag': tag,
                'count': count,
                'attributes': attrs,
                'sample': str(elem)[:400]
            })
        
        if patterns:
            logger.info(f"    Found {len(patterns)} custom web components!")
        
        return patterns
    
    def _score_element_by_content(self, elem: Tag, soup: BeautifulSoup, count: int) -> float:
        """
        Universal content-based scoring (no keyword ontology needed).
        
        Analyzes INTRINSIC properties that distinguish data from UI:
        - Content density (data has text, UI is empty)
        - Structural complexity (data has nested structure)
        - Semantic HTML (links, headings, time tags)
        - Frequency (data: 10-50, UI: 100+)
        
        Returns:
            score: Higher = more likely data container
        """
        score = 0.0
        
        # ========================================
        # 1. CONTENT DENSITY (Universal Signal)
        # ========================================
        text = elem.get_text(strip=True)
        text_length = len(text)
        
        if 50 < text_length < 500:
            score += 3.0  # Sweet spot for data containers (product cards, articles)
        elif text_length > 500:
            score += 2.0  # Long-form content (articles, detailed descriptions)
        elif 20 < text_length <= 50:
            score += 1.0  # Short content (titles, labels)
        elif text_length <= 20:
            score -= 2.0  # Too short = likely UI button/link
        
        # ========================================
        # 2. SEMANTIC HTML TAGS (Universal)
        # ========================================
        # Data containers have semantic child elements
        has_heading = bool(elem.select('h1, h2, h3, h4, h5, h6'))
        has_link = bool(elem.select('a[href]'))
        has_image = bool(elem.select('img[src]'))
        has_time = bool(elem.select('time, [datetime]'))
        has_paragraph = bool(elem.select('p'))
        
        semantic_score = sum([
            has_heading * 2.0,    # Headings = strong data signal
            has_link * 1.5,       # Links to detail pages
            has_image * 1.0,      # Visual content
            has_time * 2.5,       # Timestamp = very strong signal (articles, posts)
            has_paragraph * 1.0   # Descriptive text
        ])
        
        score += semantic_score
        
        # ========================================
        # 3. FREQUENCY PENALTY (Universal)
        # ========================================
        # Data appears 10-50 times per page
        # UI appears 100+ times (navigation, buttons, icons)
        
        if 10 <= count <= 50:
            score += 2.5  # Perfect range for listings/grids
        elif 5 <= count < 10:
            score += 1.5  # Reasonable for smaller pages
        elif 51 <= count <= 100:
            score += 0.5  # Could be data or UI
        elif count > 200:
            score -= 5.0  # Heavy penalty: almost certainly UI
        elif count > 100:
            score -= 3.0  # Likely UI (navigation, filters)
        elif count < 5:
            score -= 1.0  # Too rare (might be outlier)
        
        # ========================================
        # 4. TEXT-TO-HTML RATIO (Universal)
        # ========================================
        # Data has high text-to-markup ratio
        # UI has low ratio (mostly empty divs/buttons)
        html_length = len(str(elem))
        if html_length > 0:
            ratio = text_length / html_length
            if ratio > 0.3:
                score += 1.5  # High text ratio = data
            elif ratio > 0.15:
                score += 0.5  # Medium ratio
            elif ratio < 0.05:
                score -= 1.5  # Very low ratio = empty UI
        
        # ========================================
        # 5. NESTED STRUCTURE (Universal)
        # ========================================
        # Data containers have rich nested structure
        children = list(elem.children)
        non_text_children = [c for c in children if isinstance(c, Tag)]
        
        if len(non_text_children) >= 3:
            score += 1.0  # Good nesting = likely data
        elif len(non_text_children) >= 5:
            score += 1.5  # Rich nesting = strong data signal
        
        # ========================================
        # 6. LINK DENSITY (Universal)
        # ========================================
        links = elem.find_all('a', recursive=True, limit=15)
        
        if 1 <= len(links) <= 5:
            score += 1.0  # Reasonable links = data (title links, "read more")
        elif len(links) > 10:
            score -= 2.0  # Too many links = navigation/menu
        
        # ========================================
        # 7. DATA ATTRIBUTES (Universal)
        # ========================================
        # Modern sites use data attributes for tracking/metadata
        attrs = elem.attrs
        has_data_attrs = any(k.startswith('data-') for k in attrs.keys())
        has_itemtype = 'itemtype' in attrs or 'itemscope' in attrs
        
        if has_itemtype:
            score += 3.0  # Schema.org markup = very strong signal
        if has_data_attrs:
            score += 0.5  # Data attributes = positive signal
        
        # ========================================
        # 8. UI KEYWORD PENALTY (Universal)
        # ========================================
        # Penalize elements with UI-specific keywords in class/id
        # These indicate decorative/functional elements, not data
        
        UI_KEYWORDS = [
            'tooltip', 'badge', 'icon', 'dropdown', 'menu',
            'popup', 'modal', 'overlay', 'spinner', 'loader',
            'button', 'nav', 'header', 'footer', 'sidebar',
            'ad', 'promo', 'banner', 'cookie', 'notification'
        ]
        
        # Check class and id attributes
        classes = ' '.join(attrs.get('class', [])).lower()
        elem_id = attrs.get('id', '').lower()
        combined = f"{classes} {elem_id}"
        
        ui_keyword_count = sum(1 for keyword in UI_KEYWORDS if keyword in combined)
        
        if ui_keyword_count > 0:
            # Heavy penalty for UI keywords (eBay fix)
            penalty = ui_keyword_count * 2.5  # 2.5 points per keyword
            score -= penalty
            logger.debug(f"   UI keyword penalty: -{penalty:.1f} for {elem.name} (found: {ui_keyword_count} keywords)")
        
        return score
    
    def _get_element_signature(self, elem: Tag) -> str:
        """
        Get a signature string for an element (tag + classes).
        
        Example: <div class="post-meta stats"> → "div.post-meta.stats"
        """
        tag = elem.name
        classes = elem.get('class', [])
        
        if classes:
            # Sort classes for consistency
            classes = sorted(classes)
            return f"{tag}.{'.'.join(classes)}"
        else:
            return tag
    
    def _analyze_sibling_patterns(
        self,
        soup: BeautifulSoup,
        main_selector: str,
        count: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze sibling elements around the main container.
        
        This solves the "Stack Overflow problem" where votes are in a sibling element,
        not nested inside the main container.
        
        Args:
            soup: BeautifulSoup object
            main_selector: CSS selector for the main repeating element
            count: Number of occurrences
            
        Returns:
            Dict with sibling analysis, or None if no consistent siblings
        """
        
        if not main_selector or count < 5:
            return None
        
        try:
            # Escape the selector
            escaped_selector = escape_css_selector(main_selector)
            
            # Find all main containers
            containers = soup.select(escaped_selector, limit=20)
            
            if len(containers) < 5:
                return None
            
            # Analyze siblings for the first 10 containers
            sibling_patterns = defaultdict(int)
            parent_patterns = defaultdict(int)
            
            for container in containers[:10]:
                # Check parent
                parent = container.parent
                if parent and parent.name != 'body':
                    parent_sig = self._get_element_signature(parent)
                    parent_patterns[parent_sig] += 1
                
                # Check next siblings
                next_siblings = []
                sibling = container.find_next_sibling()
                while sibling and len(next_siblings) < 3:
                    if isinstance(sibling, Tag):
                        sig = self._get_element_signature(sibling)
                        sibling_patterns[sig] += 1
                        next_siblings.append(sig)
                    sibling = sibling.find_next_sibling()
                
                # Check previous siblings
                prev_siblings = []
                sibling = container.find_previous_sibling()
                while sibling and len(prev_siblings) < 2:
                    if isinstance(sibling, Tag):
                        sig = self._get_element_signature(sibling)
                        sibling_patterns[sig] += 1
                        prev_siblings.append(sig)
                    sibling = sibling.find_previous_sibling()
            
            # Find consistent patterns (appear in 80%+ of containers)
            threshold = len(containers[:10]) * 0.8
            consistent_siblings = [
                sig for sig, count in sibling_patterns.items()
                if count >= threshold
            ]
            consistent_parent = [
                sig for sig, count in parent_patterns.items()
                if count >= threshold
            ]
            
            # Determine context block type
            if consistent_parent and len(consistent_siblings) >= 1:
                # Parent with consistent siblings - extract full parent blocks
                return {
                    'type': 'parent_context_block',
                    'main_selector': main_selector,
                    'parent_selector': consistent_parent[0] if consistent_parent else None,
                    'sibling_selectors': consistent_siblings,
                    'extraction_strategy': 'group_by_parent',
                    'confidence': 0.9
                }
            elif len(consistent_siblings) >= 1:
                # Siblings without parent - extract container + siblings
                return {
                    'type': 'sibling_group',
                    'main_selector': main_selector,
                    'parent_selector': None,
                    'sibling_selectors': consistent_siblings,
                    'extraction_strategy': 'container_plus_siblings',
                    'confidence': 0.85
                }
            else:
                # No consistent siblings - use original container-only approach
                return {
                    'type': 'container_only',
                    'main_selector': main_selector,
                    'parent_selector': None,
                    'sibling_selectors': [],
                    'extraction_strategy': 'container_only',
                    'confidence': 0.7
                }
        
        except Exception as e:
            logger.debug(f"    Sibling analysis failed: {e}")
            return None
    
    def _identify_best_pattern(
        self,
        element_sigs: List[Dict],
        tree_patterns: List[Dict],
        data_attrs: List[Dict],
        custom_components: List[Dict],
        soup: BeautifulSoup
    ) -> Optional[Dict[str, Any]]:
        """
        Identify the most likely data container pattern
        
        UPDATED Priority (fixed GitHub issue):
        1. Custom components WITH good count (>= 10)
        2. High-frequency semantic repeating elements (article, li, tr, etc.) with >= 10 occurrences
        3. Custom components with low count (< 10) - fallback
        4. Data attributes (but only if high frequency > 50)
        5. Other repeating elements
        """
        
        # Priority 1: Custom components with GOOD count (>= 10)
        high_count_custom = [c for c in custom_components if c['count'] >= 10]
        if high_count_custom:
            best = high_count_custom[0]
            logger.info(f"    Best pattern: Custom component <{best['tag']}> ({best['count']} instances)")
            return {
                'type': 'custom_component',
                'selector': best['tag'],
                'count': best['count'],
                'confidence': 0.95,
                'attributes': best['attributes'],
                'sample': best['sample'],
                'extraction_hint': 'Use elem.get(attribute_name) for data extraction'
            }
        
        # Priority 2: CONTENT-BASED SCORING (Universal - no keyword matching)
        # Score ALL repeating elements using intrinsic properties
        if element_sigs:
            scored = []
            for p in element_sigs:
                # Get sample elements for content analysis
                sig = p['signature']
                if '.' in sig:
                    tag, *classes = sig.split('.')
                    selector_parts = [tag] + classes
                else:
                    selector_parts = [sig]
                
                # Find elements (limit to 5 for performance)
                try:
                    sample_elements = soup.find_all(
                        p['tag'],
                        class_=p['classes'] if p['classes'] else None,
                        limit=5
                    )
                    
                    if not sample_elements:
                        continue
                    
                    # Score using content-based analysis
                    element_scores = [
                        self._score_element_by_content(elem, soup, p['count'])
                        for elem in sample_elements
                    ]
                    avg_score = sum(element_scores) / len(element_scores)
                    
                    scored.append((avg_score, p))
                    
                except Exception as e:
                    logger.debug(f"   Error scoring {sig}: {e}")
                    continue
            
            if not scored:
                logger.warning("     No patterns could be scored")
            else:
                # Sort by score (highest first)
                scored.sort(reverse=True, key=lambda x: x[0])
                best_score, best = scored[0]
                
                # Parse selector
                sig = best['signature']
                if '.' in sig:
                    tag, *classes = sig.split('.')
                    selector = f"{tag}.{'.'.join(classes)}"
                else:
                    selector = sig
                
                # Escape special characters (e.g., `:` in Tailwind classes)
                selector = escape_css_selector(selector)
                
                # Calculate confidence from score
                # Score ranges from ~-10 (UI) to +15 (strong data signal)
                # Map to confidence 0.0-1.0
                confidence = min(0.95, max(0.3, (best_score + 10) / 25))
                
                logger.info(f"    Best pattern: {selector} ({best['count']} instances, score={best_score:.2f}, confidence={confidence:.2f})")
                
                return {
                    'type': 'repeating_element',
                    'selector': selector,
                    'count': best['count'],
                    'confidence': confidence,
                    'tag': best['tag'],
                    'classes': best['classes'],
                    'sample': best['sample_element'],
                    'extraction_hint': 'Use CSS selectors for nested elements'
                }
        
        # Priority 3: Low-count custom components (fallback)
        # If we have custom components but they didn't meet the >= 10 threshold, use them as fallback
        if custom_components:
            best = custom_components[0]
            logger.info(f"    Best pattern: Custom component <{best['tag']}> ({best['count']} instances, low count fallback)")
            return {
                'type': 'custom_component',
                'selector': best['tag'],
                'count': best['count'],
                'confidence': 0.70,  # Lower confidence for low-count custom components
                'attributes': best['attributes'],
                'sample': best['sample'],
                'extraction_hint': 'Use elem.get(attribute_name) for data extraction'
            }
        
        # Priority 4: Data attributes (but only if VERY frequent)
        if data_attrs:
            # Only use data attrs if they have high frequency (> 50)
            high_freq_data = [d for d in data_attrs if d['count'] > 50]
            if high_freq_data:
                best = high_freq_data[0]
                logger.info(f"    Best pattern: Data attributes {best['signature']} ({best['count']} instances)")
                return {
                    'type': 'data_attributes',
                    'selector': best['signature'].split('[')[0],  # Get tag name
                    'count': best['count'],
                    'confidence': 0.85,
                    'attributes': best['attributes'],
                    'sample': best['sample'],
                    'extraction_hint': 'Use elem.get(data-attribute-name) for data extraction'
                }
        
        # Priority 5: Other repeating elements with good data density
        if element_sigs:
            # Score each pattern
            scored_patterns = []
            for pattern in element_sigs:
                score = 0.0
                
                # Frequency score (more = better)
                freq_score = min(pattern['count'] / 100.0, 1.0)  # Cap at 100
                score += freq_score * 0.4
                
                # Data density score (has content = better)
                if pattern['text_length'] > 50:
                    score += 0.3
                elif pattern['text_length'] > 10:
                    score += 0.15
                
                # Has data attributes (better)
                if pattern['has_data_attrs']:
                    score += 0.2
                
                # Semantic tags (article, section, li, tr = better)
                if pattern['tag'] in ['article', 'section', 'li', 'tr']:
                    score += 0.2
                
                # Penalize SVG/icon tags (not data)
                if pattern['tag'] in ['use', 'path', 'svg', 'g', 'circle', 'rect']:
                    score *= 0.1  # Heavy penalty
                
                # Penalize generic tags without classes
                if pattern['tag'] in ['div', 'span'] and not pattern['classes']:
                    score *= 0.3
                
                scored_patterns.append((score, pattern))
            
            # Get best scoring pattern
            scored_patterns.sort(reverse=True, key=lambda x: x[0])
            score, best = scored_patterns[0]
            
            # Parse selector from signature
            sig = best['signature']
            if '.' in sig:
                tag, *classes = sig.split('.')
                selector = f"{tag}.{'.'.join(classes)}"
            elif '#' in sig:
                selector = sig.replace('#', ' ')  # "div#id" -> "div id"
            else:
                selector = sig
            
            # Escape special characters (e.g., `:` in Tailwind classes)
            selector = escape_css_selector(selector)
            
            logger.info(f"    Best pattern: {selector} ({best['count']} instances, score: {score:.2f})")
            
            return {
                'type': 'repeating_element',
                'selector': selector,
                'count': best['count'],
                'confidence': min(score, 0.85),  # Cap at 0.85 for heuristic detection
                'tag': best['tag'],
                'classes': best['classes'],
                'sample': best['sample'],
                'extraction_hint': 'Use CSS selectors for nested elements'
            }
        
        logger.warning("     No strong patterns detected")
        return None

