"""
Intelligent JSON Parser - Automatically extracts data from captured JSON APIs

This module analyzes JSON blobs captured during browser fetching and
intelligently extracts the requested fields without needing CSS selectors
or XPath expressions.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class JSONParser:
    """
    Automatically parses JSON data and extracts requested fields
    using intelligent field matching and data structure analysis
    """
    
    def __init__(self):
        """Initialize JSON Parser"""
        logger.info(" JSON Parser initialized")
    
    def parse_all(
        self,
        json_blobs: List[Dict[str, Any]],
        fields: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse all JSON blobs and extract data matching the requested fields
        
        Args:
            json_blobs: List of JSON objects captured from API calls
            fields: List of field names to extract (e.g., ['product_name', 'price'])
            context: Optional context about what the data represents
            
        Returns:
            List of extracted items with requested fields
        """
        if not json_blobs:
            logger.warning("No JSON blobs to parse")
            return []
        
        logger.info(f" Parsing {len(json_blobs)} JSON blobs for {len(fields)} fields")
        
        all_items = []
        
        for i, blob in enumerate(json_blobs, 1):
            logger.info(f"   Blob {i}: Analyzing structure...")
            
            # Find arrays in the JSON (likely to contain items)
            arrays = self._find_arrays(blob)
            
            if not arrays:
                logger.info(f"   Blob {i}: No arrays found")
                continue
            
            logger.info(f"   Blob {i}: Found {len(arrays)} arrays")
            
            # Try to extract from each array
            for array_path, array_data in arrays:
                if not array_data or not isinstance(array_data, list):
                    continue
                
                # Sample first item to check structure
                sample = array_data[0] if array_data else {}
                if not isinstance(sample, dict):
                    continue
                
                logger.info(f"      Array at '{array_path}': {len(array_data)} items")
                
                # Try to extract fields from this array
                extracted = self._extract_from_array(array_data, fields, context)
                
                if extracted:
                    logger.info(f"       Extracted {len(extracted)} items")
                    all_items.extend(extracted)
        
        logger.info(f" JSON parsing complete: {len(all_items)} items extracted")
        return all_items
    
    def _find_arrays(self, data: Any, path: str = "root") -> List[tuple]:
        """
        Recursively find all arrays in JSON data
        
        Returns list of (path, array) tuples
        """
        arrays = []
        
        if isinstance(data, list):
            # This is an array
            if len(data) > 0:
                arrays.append((path, data))
            
            # Check items in array
            for i, item in enumerate(data[:3]):  # Sample first 3
                if isinstance(item, (dict, list)):
                    arrays.extend(self._find_arrays(item, f"{path}[{i}]"))
        
        elif isinstance(data, dict):
            # Check each key
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    arrays.extend(self._find_arrays(value, f"{path}.{key}"))
        
        return arrays
    
    def _extract_from_array(
        self,
        array: List[Dict],
        fields: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract requested fields from an array of objects
        
        Uses fuzzy field matching to handle different naming conventions:
        - product_name → productName, ProductName, product-name, name
        - price → Price, pricing, cost
        - etc.
        """
        if not array or not isinstance(array[0], dict):
            return []
        
        # Analyze first item to find field mappings
        sample_item = array[0]
        available_keys = list(sample_item.keys())
        
        # Create field mappings
        field_map = {}
        for field in fields:
            matched_key = self._match_field(field, available_keys)
            if matched_key:
                field_map[field] = matched_key
        
        if not field_map:
            logger.info(f"      No field matches found")
            return []
        
        logger.info(f"      Matched {len(field_map)}/{len(fields)} fields: {field_map}")
        
        # Extract data
        results = []
        for item in array:
            extracted_item = {}
            
            for requested_field, json_key in field_map.items():
                value = self._get_nested_value(item, json_key)
                if value is not None:
                    extracted_item[requested_field] = value
            
            # Only include items with at least one field
            if extracted_item:
                results.append(extracted_item)
        
        return results
    
    def _match_field(self, requested_field: str, available_keys: List[str]) -> Optional[str]:
        """
        Match a requested field name to an available JSON key
        
        Handles different naming conventions:
        - Exact match: product_name → product_name
        - CamelCase: product_name → productName
        - PascalCase: product_name → ProductName
        - Kebab-case: product_name → product-name
        - Partial match: product_name → name
        - Synonym match: price → cost, pricing, amount
        """
        # Normalize field name
        field_lower = requested_field.lower()
        field_parts = re.split(r'[_\-\s]', field_lower)
        
        # 1. Exact match (case insensitive)
        for key in available_keys:
            if key.lower() == field_lower:
                return key
        
        # 2. CamelCase/PascalCase match
        # product_name → productName or ProductName
        camel_variations = [
            requested_field,  # product_name
            ''.join(word.capitalize() for word in field_parts),  # ProductName
            field_parts[0] + ''.join(word.capitalize() for word in field_parts[1:])  # productName
        ]
        
        for key in available_keys:
            if key in camel_variations or key.lower().replace('-', '_') == field_lower:
                return key
        
        # 3. Partial match (last part of field name)
        # product_name → name
        last_part = field_parts[-1]
        for key in available_keys:
            if key.lower() == last_part or key.lower().endswith(last_part):
                return key
        
        # 4. Synonym matching
        synonyms = self._get_field_synonyms(requested_field)
        for synonym in synonyms:
            for key in available_keys:
                if key.lower() == synonym.lower():
                    return key
        
        # 5. Contains matching (as last resort)
        for key in available_keys:
            key_lower = key.lower()
            # Check if any part of the requested field is in the key
            if any(part in key_lower for part in field_parts if len(part) > 2):
                return key
        
        return None
    
    def _get_field_synonyms(self, field: str) -> List[str]:
        """
        Get common synonyms for field names
        """
        field_lower = field.lower()
        
        # Common synonym mappings
        synonym_map = {
            'price': ['cost', 'pricing', 'amount', 'value', 'rate'],
            'name': ['title', 'label', 'heading', 'product', 'item'],
            'description': ['desc', 'summary', 'details', 'info', 'about'],
            'image': ['img', 'picture', 'photo', 'thumbnail', 'avatar'],
            'url': ['link', 'href', 'uri', 'path'],
            'date': ['time', 'timestamp', 'created', 'updated'],
            'rating': ['score', 'stars', 'review', 'feedback'],
            'location': ['address', 'place', 'city', 'region'],
            'category': ['type', 'class', 'tag', 'genre'],
            'quantity': ['count', 'amount', 'number', 'qty'],
            'status': ['state', 'condition', 'availability'],
            'id': ['identifier', 'code', 'key', 'sku'],
        }
        
        # Check if the field or any part of it has synonyms
        for key, synonyms in synonym_map.items():
            if key in field_lower or field_lower in key:
                return synonyms
        
        return []
    
    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """
        Get value from nested dict using dot notation
        
        Examples:
            _get_nested_value(data, 'product.name')
            _get_nested_value(data, 'pricing.amount')
        """
        if '.' in key_path:
            keys = key_path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        else:
            return data.get(key_path)
    
    def get_best_array(
        self,
        json_blobs: List[Dict[str, Any]],
        min_items: int = 3
    ) -> Optional[List[Dict]]:
        """
        Find the most likely array containing the main data
        
        Heuristics:
        - Has at least min_items elements
        - Elements are objects (dicts) not primitives
        - Elements have multiple fields (not just id)
        - Array name suggests data (products, items, results, etc.)
        
        Returns the best matching array or None
        """
        logger.info(f" Finding best data array (min {min_items} items)...")
        
        candidates = []
        
        for blob in json_blobs:
            arrays = self._find_arrays(blob)
            
            for path, array in arrays:
                if len(array) < min_items:
                    continue
                
                # Check if array contains dicts
                sample = array[0]
                if not isinstance(sample, dict):
                    continue
                
                # Score this array
                score = len(array)  # More items = better
                
                # Bonus for having multiple fields
                if len(sample) > 3:
                    score += 10
                
                # Bonus for data-suggestive names
                path_lower = path.lower()
                data_keywords = ['product', 'item', 'result', 'entry', 'data', 'list', 'menu', 'strain']
                if any(kw in path_lower for kw in data_keywords):
                    score += 20
                
                candidates.append((score, path, array))
        
        if not candidates:
            logger.info("   No suitable arrays found")
            return None
        
        # Sort by score (highest first)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        best_score, best_path, best_array = candidates[0]
        logger.info(f"    Best array: '{best_path}' ({len(best_array)} items, score={best_score})")
        
        return best_array


# Convenience functions

def parse_json_for_fields(
    json_blobs: List[Dict[str, Any]],
    fields: List[str],
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to parse JSON blobs
    
    Args:
        json_blobs: List of JSON objects from API calls
        fields: Field names to extract
        context: Optional context about the data
        
    Returns:
        List of extracted items
    """
    parser = JSONParser()
    return parser.parse_all(json_blobs, fields, context)




