"""
Smart HTML Sampler - Intelligent HTML sampling for LLM analysis
Shows the LLM the BEST examples to maximize understanding
"""

import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class SmartHTMLSampler:
    """
    Intelligently sample HTML to maximize LLM understanding
    
    Instead of random sampling, this:
    1. Identifies the most important elements (data containers)
    2. Extracts COMPLETE examples of these elements
    3. Provides context (surrounding HTML)
    """
    
    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars
    
    def sample_html(
        self,
        html: str,
        dom_patterns: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an intelligent HTML sample for LLM analysis
        
        Args:
            html: Full HTML content
            dom_patterns: Optional DOM pattern detection results
            
        Returns:
            Intelligently sampled HTML string
        """
        soup = BeautifulSoup(html, 'lxml')
        
        samples = []
        
        # 1. Include <head> section (metadata, important context)
        if soup.head:
            head_sample = str(soup.head)[:1500]
            samples.append("<!-- HEAD SECTION (metadata) -->\n" + head_sample)
        
        # 2. If DOM patterns detected, include examples of the pattern
        if dom_patterns and dom_patterns.get('best_pattern'):
            pattern_samples = self._sample_pattern(soup, dom_patterns['best_pattern'])
            if pattern_samples:
                samples.append("\n\n<!-- REPEATING DATA PATTERN (main content) -->\n" + pattern_samples)
        
        # 3. If custom components detected, include examples
        if dom_patterns and dom_patterns.get('custom_components'):
            component_samples = self._sample_custom_components(soup, dom_patterns['custom_components'])
            if component_samples:
                samples.append("\n\n<!-- CUSTOM WEB COMPONENTS -->\n" + component_samples)
        
        # 4. Fallback: Sample by content density
        if not samples or len(''.join(samples)) < 3000:
            content_samples = self._sample_by_content_density(soup)
            if content_samples:
                samples.append("\n\n<!-- HIGH-DENSITY CONTENT AREAS -->\n" + content_samples)
        
        # Combine and truncate
        combined = '\n\n'.join(samples)
        
        if len(combined) > self.max_chars:
            combined = combined[:self.max_chars] + "\n\n<!-- ... (truncated for brevity) -->"
        
        logger.info(f" Smart sample created: {len(combined):,} chars (from {len(html):,})")
        
        return combined
    
    def _sample_pattern(self, soup: BeautifulSoup, pattern: Dict[str, Any]) -> str:
        """
        Sample the detected repeating pattern
        
        Extract 3-5 COMPLETE examples of the pattern
        """
        selector = pattern.get('selector', '')
        count = pattern.get('count', 0)
        
        if not selector:
            return ""
        
        # Try to select elements using the pattern
        elements = []
        
        # Handle different selector types
        if pattern['type'] == 'custom_component':
            elements = soup.find_all(selector)
        elif pattern['type'] == 'repeating_element':
            # Parse selector like "li.s-card.s-card--horizontal"
            if '.' in selector:
                parts = selector.split('.')
                tag = parts[0]
                classes = parts[1:]
                
                # Find elements with all these classes
                all_elements = soup.find_all(tag)
                elements = [e for e in all_elements if all(c in e.get('class', []) for c in classes)]
            else:
                elements = soup.find_all(selector)
        
        if not elements:
            logger.warning(f"     Could not find elements matching {selector}")
            return ""
        
        # Take 3-5 examples (spaced out for diversity)
        num_samples = min(5, len(elements))
        step = max(1, len(elements) // num_samples)
        sample_elements = [elements[i * step] for i in range(num_samples)]
        
        # Build sample string
        sample_parts = [
            f"<!-- Found {count} instances of: {selector} -->",
            f"<!-- Showing {num_samples} example(s) -->\n"
        ]
        
        for i, elem in enumerate(sample_elements, 1):
            sample_parts.append(f"<!-- Example {i}/{num_samples} -->")
            sample_parts.append(str(elem.prettify()))
            if i < num_samples:
                sample_parts.append("\n<!-- ... -->\n")
        
        return '\n'.join(sample_parts)
    
    def _sample_custom_components(
        self,
        soup: BeautifulSoup,
        custom_components: List[Dict[str, Any]]
    ) -> str:
        """
        Sample custom web components (like <shreddit-post>)
        """
        if not custom_components:
            return ""
        
        samples = []
        
        for component in custom_components[:2]:  # Max 2 component types
            tag = component['tag']
            count = component['count']
            
            elements = soup.find_all(tag)
            if elements:
                samples.append(f"<!-- Custom component: <{tag}> ({count} instances) -->")
                
                # Show 2 examples
                for i, elem in enumerate(elements[:2], 1):
                    samples.append(f"<!-- Example {i} -->")
                    samples.append(str(elem.prettify())[:800])  # Truncate long components
                    samples.append("")
        
        return '\n'.join(samples)
    
    def _sample_by_content_density(self, soup: BeautifulSoup) -> str:
        """
        Fallback: Sample areas with high content density
        
        Find elements with lots of text (likely data containers)
        """
        content_elements = []
        
        # Look for common content containers
        for selector in ['article', 'section', 'main', 'div[class*="content"]', 'div[class*="item"]']:
            elements = soup.select(selector)
            for elem in elements:
                text_length = len(elem.get_text(strip=True))
                if 50 < text_length < 5000:  # Reasonable content size
                    content_elements.append((text_length, elem))
        
        # Sort by text length, take top 3
        content_elements.sort(reverse=True, key=lambda x: x[0])
        top_elements = content_elements[:3]
        
        if not top_elements:
            return ""
        
        samples = ["<!-- High-density content areas -->"]
        for text_len, elem in top_elements:
            samples.append(f"<!-- Content area: {text_len} chars -->")
            samples.append(str(elem.prettify())[:1000])  # Truncate
            samples.append("")
        
        return '\n'.join(samples)
    
    def create_examples_for_llm(
        self,
        soup: BeautifulSoup,
        selector: str,
        num_examples: int = 3
    ) -> str:
        """
        Create explicit examples for LLM prompt
        
        This is used in the AI generator to show concrete examples
        """
        elements = []
        
        # Parse selector
        if '.' in selector:
            parts = selector.split('.')
            tag = parts[0]
            classes = parts[1:]
            all_elements = soup.find_all(tag)
            elements = [e for e in all_elements if all(c in e.get('class', []) for c in classes)]
        elif selector.startswith('<') and selector.endswith('>'):
            # Custom component like "<shreddit-post>"
            tag = selector.strip('<>')
            elements = soup.find_all(tag)
        else:
            elements = soup.find_all(selector)
        
        if not elements:
            return ""
        
        # Create example string
        examples = []
        for i, elem in enumerate(elements[:num_examples], 1):
            examples.append(f"Example {i}:")
            examples.append(str(elem)[:500])  # Truncate long examples
            examples.append("")
        
        return '\n'.join(examples)







