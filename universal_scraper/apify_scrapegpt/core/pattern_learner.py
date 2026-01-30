"""
Pattern Learner
Learns extraction patterns from successful LLM results
This is the KEY INNOVATION that makes our solution cacheable
"""
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re
from collections import Counter

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Learns extraction patterns from successful LLM extraction results
    
    Strategy:
    1. Take successful LLM extraction (HTML + extracted items)
    2. Reverse-engineer: find where each field value appears in HTML
    3. Identify CSS selectors/patterns that consistently work
    4. Generate cacheable extraction pattern
    5. Validate pattern quality
    """
    
    def __init__(self):
        logger.info(" PatternLearner initialized")
    
    async def learn_pattern(
        self,
        html: str,
        extracted_items: List[Dict[str, Any]],
        fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Learn extraction pattern from successful LLM results
        
        Args:
            html: The HTML that was successfully extracted from
            extracted_items: Items extracted by DirectLLMExtractor
            fields: Fields that were extracted
            
        Returns:
            Extraction pattern (CSS selectors + strategies) or None if learning failed
        """
        if not extracted_items or len(extracted_items) < 3:
            logger.warning(f"  Too few items ({len(extracted_items)}) to learn pattern")
            return None
        
        logger.info(f" Learning pattern from {len(extracted_items)} successfully extracted items...")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Step 1: Find repeating containers
        container_selector = self._find_container_pattern(soup, extracted_items, fields)
        if not container_selector:
            logger.warning("  Could not identify container pattern")
            return None
        
        logger.info(f"    Container: {container_selector}")
        
        # Step 2: For each field, learn extraction strategy
        field_patterns = {}
        containers = soup.select(container_selector)
        
        for field in fields:
            pattern = self._learn_field_pattern(
                containers=containers,
                extracted_items=extracted_items,
                field=field
            )
            
            if pattern:
                field_patterns[field] = pattern
                logger.info(f"    {field}: {pattern.get('type', 'unknown')}")
            else:
                logger.warning(f"     Could not learn pattern for {field}")
        
        if not field_patterns:
            logger.warning("  No field patterns learned")
            return None
        
        # Step 3: Generate complete extraction pattern
        extraction_pattern = {
            "version": "1.0",
            "container_selector": container_selector,
            "fields": field_patterns,
            "metadata": {
                "learned_from_items": len(extracted_items),
                "field_count": len(field_patterns),
                "confidence": self._calculate_confidence(field_patterns)
            }
        }
        
        logger.info(f" Pattern learned! Confidence: {extraction_pattern['metadata']['confidence']:.1%}")
        
        return extraction_pattern
    
    def _find_container_pattern(
        self,
        soup: BeautifulSoup,
        extracted_items: List[Dict],
        fields: List[str]
    ) -> Optional[str]:
        """
        Find CSS selector for repeating container elements
        
        Strategy:
        - Look for elements that repeat ~same number as extracted items
        - Prioritize semantic containers (article, li, div with item-like classes)
        - Validate by checking if containers contain field values
        """
        item_count = len(extracted_items)
        
        # Candidate container patterns
        candidates = [
            ('article', None),
            ('li', None),
            ('div', lambda x: x and any(kw in ' '.join(x).lower() for kw in ['item', 'product', 'post', 'card', 'entry', 'result'])),
            ('tr', None),
            ('section', None),
        ]
        
        best_selector = None
        best_score = 0
        
        for tag, class_filter in candidates:
            if class_filter:
                elements = soup.find_all(tag, class_=class_filter)
            else:
                elements = soup.find_all(tag)
            
            if not elements:
                continue
            
            # Check if count is close to expected
            count = len(elements)
            count_match = 1.0 - abs(count - item_count) / max(count, item_count)
            
            if count_match < 0.5:  # Too far off
                continue
            
            # Check if elements contain extracted values
            sample_values = []
            for item in extracted_items[:5]:
                for field in fields:
                    value = item.get(field)
                    if value and isinstance(value, str) and len(value) > 3:
                        sample_values.append(value.lower().strip())
            
            value_matches = 0
            for element in elements[:min(10, len(elements))]:
                element_text = element.get_text().lower()
                for value in sample_values:
                    if value in element_text:
                        value_matches += 1
                        break
            
            value_match_rate = value_matches / min(10, len(elements)) if elements else 0
            
            # Combined score
            score = (count_match * 0.3) + (value_match_rate * 0.7)
            
            logger.debug(f"      {tag}: count={count}, match={count_match:.2f}, values={value_match_rate:.2f}, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                
                # Build selector
                if class_filter and elements:
                    # Use most common class
                    class_names = []
                    for el in elements:
                        class_names.extend(el.get('class', []))
                    
                    if class_names:
                        common_class = Counter(class_names).most_common(1)[0][0]
                        best_selector = f"{tag}.{common_class}"
                    else:
                        best_selector = tag
                else:
                    best_selector = tag
        
        if best_score < 0.4:
            logger.debug(f"      Best score {best_score:.2f} too low")
            return None
        
        return best_selector
    
    def _learn_field_pattern(
        self,
        containers: List,
        extracted_items: List[Dict],
        field: str
    ) -> Optional[Dict[str, Any]]:
        """
        Learn extraction strategy for a single field
        
        Strategy:
        - For each extracted value, find where it appears in the container
        - Identify common patterns (tag, class, position)
        - Generate extraction rule
        """
        if not containers or not extracted_items:
            return None
        
        # Collect (value, container) pairs
        value_locations = []
        
        for i, item in enumerate(extracted_items):
            if i >= len(containers):
                break
            
            value = item.get(field)
            if not value or not isinstance(value, str):
                continue
            
            container = containers[i]
            location = self._find_value_in_container(container, value)
            
            if location:
                value_locations.append(location)
        
        if not value_locations or len(value_locations) < len(extracted_items) * 0.5:
            # Found <50% of values - pattern unreliable
            return None
        
        # Analyze common patterns
        pattern = self._extract_common_pattern(value_locations)
        
        return pattern
    
    def _find_value_in_container(
        self,
        container,
        value: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find where a value appears within a container element
        """
        value_clean = value.lower().strip()
        
        # Search all descendants
        for element in container.find_all():
            element_text = element.get_text().strip().lower()
            
            if value_clean in element_text:
                # Found it!
                return {
                    "tag": element.name,
                    "classes": element.get('class', []),
                    "attrs": {k: v for k, v in element.attrs.items() if k not in ['class', 'style']},
                    "position": self._get_position(container, element),
                    "text_match": value_clean
                }
        
        return None
    
    def _get_position(self, container, element) -> str:
        """
        Get relative position of element within container
        """
        # Find position among siblings of same type
        parent = element.parent
        if not parent:
            return "first"
        
        siblings = [s for s in parent.children if hasattr(s, 'name') and s.name == element.name]
        try:
            idx = siblings.index(element)
            if idx == 0:
                return "first"
            elif idx == len(siblings) - 1:
                return "last"
            else:
                return f"nth-{idx+1}"
        except ValueError:
            return "unknown"
    
    def _extract_common_pattern(
        self,
        locations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract common pattern from multiple value locations
        """
        # Find most common tag
        tags = [loc['tag'] for loc in locations]
        most_common_tag = Counter(tags).most_common(1)[0][0]
        
        # Find most common classes
        all_classes = []
        for loc in locations:
            all_classes.extend(loc['classes'])
        
        common_classes = []
        if all_classes:
            class_counts = Counter(all_classes)
            # Classes that appear in >50% of locations
            threshold = len(locations) * 0.5
            common_classes = [cls for cls, count in class_counts.items() if count >= threshold]
        
        # Build pattern
        pattern = {
            "type": "css_selector",
            "selector": most_common_tag,
            "extract": "text"
        }
        
        if common_classes:
            pattern["selector"] = f"{most_common_tag}.{'.'.join(common_classes[:2])}"
        
        # Add position hint if consistent
        positions = [loc['position'] for loc in locations]
        position_counts = Counter(positions)
        if position_counts:
            most_common_position = position_counts.most_common(1)[0][0]
            if position_counts[most_common_position] / len(positions) > 0.7:
                pattern["position"] = most_common_position
        
        return pattern
    
    def _calculate_confidence(self, field_patterns: Dict[str, Dict]) -> float:
        """
        Calculate confidence score for learned pattern
        """
        if not field_patterns:
            return 0.0
        
        # Simple confidence: fraction of fields we could learn patterns for
        # In reality, this should also consider selector specificity, consistency, etc.
        return len(field_patterns) / max(1, len(field_patterns))
    
    def validate_pattern(
        self,
        pattern: Dict[str, Any],
        html: str,
        expected_items: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate that learned pattern actually works on the source HTML
        
        Args:
            pattern: Learned extraction pattern
            html: Source HTML
            expected_items: Items we expect to extract
            
        Returns:
            True if pattern is valid and produces similar results
        """
        logger.info(" Validating learned pattern...")
        
        soup = BeautifulSoup(html, 'html.parser')
        container_selector = pattern['container_selector']
        containers = soup.select(container_selector)
        
        if not containers:
            logger.warning(f"    No containers found for selector: {container_selector}")
            return False
        
        # Check container count
        expected_count = len(expected_items)
        actual_count = len(containers)
        count_match = 1.0 - abs(actual_count - expected_count) / max(actual_count, expected_count)
        
        logger.info(f"   Container count: {actual_count} (expected {expected_count}, match={count_match:.1%})")
        
        if count_match < 0.5:
            logger.warning(f"    Container count mismatch too large")
            return False
        
        # Try to extract a few items using pattern
        extracted_count = 0
        for i, container in enumerate(containers[:5]):
            item = {}
            for field, field_pattern in pattern['fields'].items():
                selector = field_pattern.get('selector')
                if selector:
                    element = container.select_one(selector)
                    if element:
                        item[field] = element.get_text().strip()
                        extracted_count += 1
            
            if item:
                logger.debug(f"   Sample extraction {i+1}: {list(item.keys())}")
        
        success_rate = extracted_count / (len(pattern['fields']) * min(5, len(containers)))
        logger.info(f"   Extraction success rate: {success_rate:.1%}")
        
        if success_rate < 0.4:
            logger.warning(f"    Pattern validation failed")
            return False
        
        logger.info(f"    Pattern validated!")
        return True




